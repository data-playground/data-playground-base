# airflow/agents/blog_agents.py
"""
All blog pipeline and code intelligence agent functions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL ROUTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent               Provider            Model
─────────────────   ─────────────────   ────────────────────────────
README Writer       Gemini              gemini-2.5-flash
Researcher          Gemini              gemini-2.5-flash (JSON mode)
Code Narrator       Cerebras            qwen-3-235b-a22b-instruct-2507
Ghostwriter         Groq                llama-3.3-70b-versatile
Refiner             Cerebras            llama-3.3-70b
Editor              Gemini              gemini-2.5-flash
Idea Expander       Gemini              gemini-2.5-flash (JSON mode)
Code Commenter      Cerebras            qwen-3-235b-a22b-instruct-2507
Code Improver       Cerebras            qwen-3-235b-a22b-instruct-2507

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING RATIONALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gemini 2.5 Flash:
  Used for low-frequency, quality-sensitive tasks (README Writer, Editor,
  Researcher, Idea Expander). 250 RPD free tier is sufficient when these
  agents run once per article or once per batch. Handles structured JSON
  output natively.

Groq + Llama 3.3 70B:
  Ghostwriter only. Prose generation is a single call per article, never
  per-file. 131K context, 14,400 RPD free, fast inference. Kept here
  because it was already working and prose generation is not a reasoning-
  intensive task where frontier quality matters most.

Cerebras + Qwen3 235B (qwen-3-235b-a22b-instruct-2507):
  Code Narrator, Commenter, Improver. These are high-frequency per-file
  agents. Qwen3 235B is frontier-grade on coding benchmarks (outperforms
  GPT-4.1 and Claude Opus 4 on Artificial Analysis Intelligence Index),
  runs at ~1,400 tokens/sec, and provides 64K context on the free tier
  with 1M tokens/day — vs Gemini's 250 RPD which would exhaust in a
  single large project narration run.

Cerebras + Llama 3.3 70B:
  Refiner only. Mid-frequency (once per article), the task is targeted
  revision rather than deep reasoning. Llama 3.3 70B handles long drafts
  without the TPM throttling that Groq's free tier would impose.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIFFICULTY SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  starter   — 1-2 evenings, 1-2 tools, one clear concept. Reproducible
              in a single sitting. Max 2 integrations.

  weekend   — A full weekend, 2-3 tools, a small system or integration
              pattern. MVP-completable in two days. Max 3 integrations.

  ambitious — Multi-week project. Complex architecture acceptable.
              Include sparingly (1 per 5-idea batch).

Target per 5-idea batch: 2 starter, 2 weekend, 1 ambitious.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDEA TYPE SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  existing_asset — Something already built. Retrospective article:
                   decisions, lessons, what you'd change.

  new_build      — A net-new project to build before writing.
                   Scoped to the difficulty tag.

  tutorial       — Focused how-to. No original project required.
                   Explains one technique or concept using the author's
                   existing experience as the lens. Two tutorials per
                   batch, on different tools/concepts.

Target per 5-idea batch: 1 existing_asset, 2 new_build, 2 tutorial.
"""

import json
import logging
import os
import time

import requests

from services.ai import MODEL_FLASH, call_gemini_json, call_gemini_text

log = logging.getLogger(__name__)


# ── ALLOWED VALUES ────────────────────────────────────────────────────────────

DIFFICULTY_LEVELS = ("starter", "weekend", "ambitious")
PROJECT_TYPES     = ("existing_asset", "new_build", "tutorial")

# Rough token estimate used for large-file pre-flight checks.
# ~4 characters per token is a conservative estimate for mixed
# Python/HTML/CSS/JS code which tokenises less efficiently than prose.
_CHARS_PER_TOKEN = 4
LARGE_FILE_THRESHOLD_TOKENS = 40_000   # ~160K characters


# ── KEY HELPERS ───────────────────────────────────────────────────────────────

def _groq_key() -> str:
    # from gcp_secrets import get_key
    return os.environ.get("GROQ_API")

def _cerebras_key() -> str:
    # from gcp_secrets import get_key
    return os.environ.get("CEREBRAS_API")


# ── PROVIDER CALL HELPERS ─────────────────────────────────────────────────────

def _groq_llama(system: str, prompt: str, temperature: float = 0.7) -> str:
    """
    Calls Llama 3.3 70B via Groq for prose generation.
    Used by: Ghostwriter.
    Free tier: 14,400 RPD, 131K context, ~6,000 TPM (fine for single-call tasks).
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {_groq_key()}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  8192,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# Default backoff schedule for 429 responses (seconds).
# Cerebras resets its RPM window every 60 seconds, so the max wait
# is capped there. If Retry-After header is present it overrides this.
_CEREBRAS_BACKOFF = [75, 150, 300, 600]

# Add this constant near the top with the other model IDs
_CEREBRAS_INTER_REQUEST_SLEEP = 65  # seconds — slightly over 1 full minute window

def _cerebras(model, system, prompt, temperature=0.3, max_tokens=4096):
    log.info("_cerebras() v2 — retry loop active, backoff=%s", _CEREBRAS_BACKOFF)

    import time

    from cerebras.cloud.sdk import APIStatusError, Cerebras, RateLimitError

    client = Cerebras(api_key=_cerebras_key(), max_retries=0).with_raw_response
    last_exc = None

    for attempt, wait in enumerate(_CEREBRAS_BACKOFF):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            remaining_day = resp.headers.get("x-ratelimit-remaining-tokens-day", "?")
            remaining_min = resp.headers.get("x-ratelimit-remaining-tokens-minute", "?")
            reset_min     = resp.headers.get("x-ratelimit-reset-tokens-minute", "?")

            if resp.status_code == 200:
                log.info(
                    "Cerebras %s OK — tokens remaining: %s/day, %s/min (reset in %ss)",
                    model, remaining_day, remaining_min, reset_min,
                )
                content = resp.json()["choices"][0]["message"]["content"]
                
                # Return content AND remaining tokens so the DAG can decide how long to sleep
                # We pack it as a tuple; callers that don't need it just take [0]
                try:
                    remaining_min_int = int(remaining_min)
                except (ValueError, TypeError):
                    remaining_min_int = 0
                return content, remaining_min_int

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                actual_wait = float(retry_after) if retry_after else wait
                log.warning(
                    "Cerebras 429 on attempt %d/%d. Waiting %.1fs.",
                    attempt + 1, len(_CEREBRAS_BACKOFF), actual_wait,
                )
                last_exc = RuntimeError(f"Cerebras 429 on attempt {attempt + 1}")
                time.sleep(actual_wait)
                continue

            if resp.status_code == 503:
                log.warning("Cerebras 503 on attempt %d/%d. Waiting %ds.",
                            attempt + 1, len(_CEREBRAS_BACKOFF), wait)
                last_exc = RuntimeError(f"Cerebras 503 on attempt {attempt + 1}")
                time.sleep(wait)
                continue

            resp.raise_for_status()

        except RateLimitError as exc:
            response = getattr(exc, "response", None)
            retry_after = 60  # safe default
            if response is not None:
                try:
                    retry_after = int(getattr(response, "headers", {}).get("retry-after", 60))
                except (ValueError, TypeError):
                    pass
            log.warning(
                "Cerebras 429 RateLimitError on attempt %d/%d. Waiting %ds.",
                attempt + 1, len(_CEREBRAS_BACKOFF), retry_after,
            )
            last_exc = exc
            time.sleep(retry_after)
            continue

        except APIStatusError as exc:
            if exc.status_code == 503:
                log.warning("Cerebras APIStatusError 503 on attempt %d/%d. Waiting %ds.",
                            attempt + 1, len(_CEREBRAS_BACKOFF), wait)
                last_exc = exc
                time.sleep(wait)
                continue
            log.error("Cerebras non-retriable APIStatusError: %s", exc)
            raise

        except Exception as exc:
            log.error("Cerebras unexpected error: %s", exc)
            raise

    raise RuntimeError(
        f"Cerebras {model} unavailable after {len(_CEREBRAS_BACKOFF)} retries. "
        f"Last error: {last_exc}"
    )


# ── CEREBRAS MODEL IDs ────────────────────────────────────────────────────────
# Pinned to specific versioned IDs to avoid silent quality regressions when
# Cerebras rotates default model aliases. Update these when migrating.

_CEREBRAS_QWEN3   = "qwen-3-235b-a22b-instruct-2507"   # Code Narrator, Commenter, Improver
_CEREBRAS_LLAMA33 = "llama-3.3-70b"                     # Refiner


# ── FILE TYPE DETECTION ───────────────────────────────────────────────────────

def _detect_file_type(file_name: str) -> str:
    """
    Maps a file name to a broad category used to select Narrator/Commenter
    focus and commenting style conventions.

    Returns one of:
        "python_router"   — FastAPI router files (endpoints, request/response handling)
        "python_dag"      — Airflow DAG files (task dependencies, scheduling, retry logic)
        "python_agent"    — LLM agent files (prompt strategy, model calls, output parsing)
        "python_model"    — SQLAlchemy models / Pydantic schemas
        "python_service"  — Service/utility files (GitHub service, GCP helpers, etc.)
        "python_general"  — Any other Python file
        "html_template"   — Jinja2 HTML templates
        "css"             — CSS stylesheets
        "javascript"      — JS files
        "shell"           — Bash/shell scripts
        "config"          — YAML, TOML, INI, .env files
        "sql"             — SQL files
        "other"           — Anything else
    """
    name = file_name.lower()

    # Specific Python sub-types based on naming conventions
    if name.endswith(".py"):
        if "router" in name or name in ("ats.py", "jobs.py", "blog.py",
                                         "finance.py", "staging.py",
                                         "explorer.py", "dashboard.py",
                                         "code_intelligence.py"):
            return "python_router"
        if "dag" in name:
            return "python_dag"
        if "agent" in name:
            return "python_agent"
        if "model" in name:
            return "python_model"
        if "service" in name or "secret" in name or "database" in name:
            return "python_service"
        return "python_general"

    if name.endswith((".html", ".jinja", ".jinja2")):
        return "html_template"
    if name.endswith(".css"):
        return "css"
    if name.endswith((".js", ".ts", ".jsx", ".tsx")):
        return "javascript"
    if name.endswith((".sh", ".bash")):
        return "shell"
    if name.endswith((".yml", ".yaml", ".toml", ".ini", ".env")):
        return "config"
    if name.endswith(".sql"):
        return "sql"

    return "other"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for mixed code/prose."""
    return len(text) // _CHARS_PER_TOKEN


# ── BLUEPRINT JSON SCHEMA ─────────────────────────────────────────────────────

_BLUEPRINT_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "project_type": {
            "type": "STRING",
            "description": (
                "existing_asset — something already built, article covers lessons and decisions. "
                "new_build — a net-new project to build before writing. "
                "tutorial — a focused how-to requiring no original project."
            ),
        },
        "difficulty": {
            "type": "STRING",
            "description": (
                "starter — 1-2 evenings, 1-2 tools, one clear concept. "
                "weekend — a full weekend, 2-3 tools, a small system or integration. "
                "ambitious — multi-week, complex architecture. Use sparingly."
            ),
        },
        "title_concept": {
            "type": "STRING",
            "description": (
                "Clear, human-readable title. No buzzword stacking. "
                "Should describe what the reader will learn or build, "
                "not just what sounds impressive."
            ),
        },
        "the_build": {
            "type": "STRING",
            "description": (
                "What gets built, described plainly and MVP-scoped to the difficulty level. "
                "Starter: max 2 integrated tools. Weekend: max 3. "
                "Describe the smallest useful version, not the fully productionised one."
            ),
        },
        "the_narrative": {
            "type": "STRING",
            "description": (
                "The story arc: what problem prompted this, what was tried, "
                "what was learned. Be honest about friction and tradeoffs — "
                "that is what readers relate to."
            ),
        },
        "the_selling_point": {
            "type": "STRING",
            "description": (
                "One sentence: why would a reader care? What skill, pattern, "
                "or insight do they take away?"
            ),
        },
    },
    "required": [
        "project_type", "difficulty", "title_concept",
        "the_build", "the_narrative", "the_selling_point",
    ],
}

# Idea Expander uses a richer schema that also captures difficulty
_EXPANDER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "project_type": {
            "type": "STRING",
            "description": "existing_asset, new_build, or tutorial",
        },
        "difficulty": {
            "type": "STRING",
            "description": "starter, weekend, or ambitious",
        },
        "title_concept":     {"type": "STRING"},
        "the_build":         {"type": "STRING"},
        "the_narrative":     {"type": "STRING"},
        "the_selling_point": {"type": "STRING"},
    },
    "required": [
        "project_type", "difficulty", "title_concept",
        "the_build", "the_narrative", "the_selling_point",
    ],
}


# ── DISTRIBUTION HELPERS ──────────────────────────────────────────────────────

def _summarize_recent_difficulties(existing_titles: list[str] | None) -> str:
    """
    Stub hook — in production this is replaced by the DAG calling
    _get_recent_difficulty_summary() with a live DB connection.
    Returns empty string when called without DB context (e.g. from tests).
    """
    return ""


def _validate_difficulty_distribution(blueprints: list[dict]) -> list[dict]:
    """
    Post-generation guard. Ensures every blueprint has a valid difficulty
    value and that no more than one ambitious idea slipped through.

    If the model returned more than one ambitious idea, extras are demoted
    to 'weekend'. Logs a warning so the prompt can be tuned if this fires
    frequently.
    """
    for bp in blueprints:
        if bp.get("difficulty") not in DIFFICULTY_LEVELS:
            bp["difficulty"] = "weekend"
            log.warning(
                "Blueprint missing valid difficulty, defaulted to 'weekend': %s",
                bp.get("title_concept"),
            )
        if bp.get("project_type") not in PROJECT_TYPES:
            bp["project_type"] = "new_build"
            log.warning(
                "Blueprint missing valid project_type, defaulted to 'new_build': %s",
                bp.get("title_concept"),
            )

    ambitious = [bp for bp in blueprints if bp["difficulty"] == "ambitious"]
    if len(ambitious) > 1:
        log.warning(
            "Researcher returned %d ambitious ideas (max 1). Demoting extras to 'weekend'.",
            len(ambitious),
        )
        for bp in ambitious[1:]:
            bp["difficulty"] = "weekend"

    return blueprints


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 0 — README WRITER
# Provider: Gemini 2.5 Flash
# Frequency: Once per project — low frequency, quality matters
# ═════════════════════════════════════════════════════════════════════════════

def agent_readme_writer(
    project_name: str,
    file_summaries: list[dict],
    description: str = "",
    is_docker_project: bool = False,
) -> str:
    """
    Generates a project-level README from file narrations.

    The README is written primarily for a hiring manager or technical
    recruiter reading it on GitHub — it should communicate what was built,
    why, and what it demonstrates technically. It is not a contributor guide.

    Args:
        project_name:     Human-friendly project name.
        file_summaries:   List of {"path": str, "narration": str} dicts
                          from Code Narrator output.
        description:      Optional author note about the project's purpose.
        is_docker_project: True if the project runs via Docker Compose.
                          Adds a Docker-specific "How to Run" section.

    Returns:
        Full README.md content as a Markdown string.
    """
    system = """
You are a Senior Engineer writing a GitHub README for a personal portfolio project.
Your primary reader is a hiring manager or senior engineer evaluating the author's work.
A secondary reader is a technical person who wants to understand what was built.

This is NOT a contributor guide. Focus on:
  1. Making the project's purpose and technical ambition immediately clear
  2. Showing what problems it solves and why those problems were interesting
  3. Demonstrating the author's technical decision-making

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
README STRUCTURE — follow this order exactly:

## [Project Name]
One sharp paragraph: what it is, what problem it solves, and why it exists.
No fluff. No "In today's world...". Get to the point.

## Why I Built This
2-3 sentences on the genuine motivation. What friction was this replacing?
What were you trying to learn or prove? Be honest — "I was tired of X"
is more compelling than "I wanted to explore Y".

## What It Does
Concrete description of the key capabilities. Use a short bullet list.
Focus on what a user or viewer would actually experience or observe.

## Technical Architecture
A brief architecture narrative followed by an ASCII diagram if helpful.
Call out non-obvious decisions: why this database, why this framework,
why async vs sync, why this particular AI model or provider.

## Key Technical Decisions
2-4 decisions worth highlighting to a technical reader. Each one should
answer: what were the alternatives, why did you choose this, what did
you learn. This is what separates a portfolio project from a tutorial clone.

## Stack
Flat list: Language · Framework · Database · Key Libraries · Infrastructure.
Keep it scannable, not exhaustive.

## Prerequisites & Setup
Only if genuinely non-trivial. For Docker projects, show the exact commands.
Skip this section for simple scripts — link to the relevant docs instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE: Direct, confident, first-person where appropriate.
Do not oversell. A hiring manager can smell inflated language.
The architecture and decisions should speak for themselves.
"""

    setup_context = (
        "This project runs via Docker Compose. "
        "Include the exact docker compose commands in Prerequisites & Setup."
        if is_docker_project else
        "Include only genuinely non-trivial setup steps."
    )

    summary_block = "\n\n".join(
        f"**{s['path']}**\n{s['narration']}" for s in file_summaries
    )

    prompt = (
        f"Project name: {project_name}\n"
        f"Author description: {description or 'Not provided — infer from the file summaries.'}\n"
        f"Setup context: {setup_context}\n\n"
        f"File narrations:\n{summary_block}\n\n"
        f"Write the README now. First-person is fine. Be direct."
    )
    return call_gemini_text(system, prompt, model=MODEL_FLASH)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 1 — RESEARCHER
# Provider: Gemini 2.5 Flash (JSON mode)
# Frequency: 5 ideas/day — well within 250 RPD free tier
# ═════════════════════════════════════════════════════════════════════════════

def agent_researcher(
    interests: str,
    existing_projects: str,
    file_narrations: list[dict] | None = None,
    existing_titles: list[str] | None = None,
    difficulty_context: str = "",
) -> list[dict]:
    """
    Generates 5 blog post blueprints with explicit difficulty tagging
    and a balanced mix of idea types.

    Distribution target per batch:
      Difficulty : 2 starter, 2 weekend, 1 ambitious
      Type       : 1 existing_asset, 2 new_build, 2 tutorial
                   (tutorials must be on different tools/concepts)

    At least 2 ideas must be in domains unrelated to the author's existing
    projects to push net-new work.

    Args:
        interests:          Comma-separated topics the author cares about.
        existing_projects:  Brief description of current projects.
        file_narrations:    Optional [{"path": str, "narration": str}] from
                            Code Narrator — gives richer existing_asset context.
        existing_titles:    Already-generated idea titles to avoid duplicating.
        difficulty_context: Plain-English summary of recent difficulty breakdown
                            from the DAG (used to rebalance if skewed).

    Returns:
        List of 5 blueprint dicts, each including 'difficulty' and
        'project_type' fields. Validated by _validate_difficulty_distribution.
    """
    system = """
You are a senior Developer Advocate and Content Strategist helping a data
professional build a consistent, readable blog portfolio.

Your job is to generate blog post blueprints that are REALISTIC TO BUILD AND WRITE.
A great blog post is not about the most complex system — it is about a clear
problem, an honest build story, and a genuine lesson. Senior thinking shows
in the quality of explanation, not in the number of integrated services.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIFFICULTY — produce EXACTLY this distribution:

  • 2 ideas tagged difficulty: "starter"
      Completable in 1–2 evenings. Uses 1–2 tools. Teaches exactly one
      clear concept. The build described must be something a reader could
      reproduce in a single sitting. MAXIMUM 2 integrations.
      Scope examples: a Python script that reads a CSV and produces a chart,
      a single FastAPI endpoint calling one external API, a SQL pattern
      explained with a real dataset.

  • 2 ideas tagged difficulty: "weekend"
      Completable over a weekend. Uses 2–3 tools. Teaches a small system
      or integration pattern. MAXIMUM 3 integrations.
      Scope examples: a small Airflow DAG pulling from one API into a local DB,
      a basic HTMX + FastAPI feature end-to-end, a simple dashboard over a
      local dataset.

  • 1 idea tagged difficulty: "ambitious"
      Multi-week project. Complex architecture acceptable. Include to keep
      the backlog interesting — but describe the MVP first, not the fully
      productionised version. Even ambitious posts start somewhere small.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDEA TYPE — produce EXACTLY this mix:

  • 1 idea type: "existing_asset"
      Something already built. The article is retrospective: decisions made,
      what worked, what didn't, what you'd change. No new build required.

  • 2 ideas type: "new_build"
      A net-new project to build before writing the article.
      Scope the build to the difficulty tag. Do not describe a 5-service
      architecture for a starter idea.

  • 2 ideas type: "tutorial"
      Focused how-tos. No original project required. Explain one technique,
      tool, or concept using the author's existing experience as the lens.
      These are low-friction to write and get high search traffic.
      CONSTRAINT: The two tutorials MUST cover different tools or concepts.
      Do not generate two SQL posts, two Airflow posts, etc.
      Examples: "How I use SQL window functions for X",
      "Async SQLAlchemy patterns that tripped me up",
      "Debugging Airflow DAGs without losing your mind."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOMAIN SPREAD:
  • At least 2 of the 5 ideas must be in domains OUTSIDE the author's
    existing projects: sports analytics, music, food, movies, general Python,
    career/job market, data visualization, etc.
  • No more than 2 ideas in the same domain within a single batch.
  • Mix technical tools and subject matter across the batch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TITLE QUALITY:
  • Clear and human-readable. A reader understands the topic immediately.
  • No buzzword stacking. Avoid strings like "Multi-Agent Autonomous
    Orchestration Layer" or "Leveraging Generative AI for..."
  • Good: "Building a Simple NBA Stats Tracker with Python and SQLite"
  • Bad:  "Orchestrating Multi-Dimensional Basketball Intelligence via
           Vertex AI Embeddings"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUILD DESCRIPTION (the_build):
  • Describe the MVP — the smallest version that makes the article writable.
  • Starter: name the 1–2 tools, what goes in and what comes out.
  • Weekend: describe the flow plainly, not in enterprise architecture terms.
  • Ambitious: still start with the MVP — follow-up posts can add complexity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NARRATIVE (the_narrative):
  • The most important field for a readable post.
  • What problem prompted this? What did you try first? What surprised you?
  • Honest friction and tradeoffs are what readers relate to.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ideas must be meaningfully distinct from each other AND from the existing
titles provided. Do not suggest near-duplicates.
"""

    narration_block = ""
    if file_narrations:
        narration_block = (
            "\n\nRecent code context (use for existing_asset and tutorial ideas):\n"
            + "\n\n".join(
                f"{n['path']}: {n['narration'][:400]}" for n in file_narrations[:10]
            )
        )

    existing_block = ""
    if existing_titles:
        existing_block = (
            "\n\nAlready in backlog — do NOT suggest similar topics:\n"
            + "\n".join(f"- {t}" for t in existing_titles[:30])
        )

    # difficulty_context is passed in from the DAG based on recent DB history.
    # Fall back to the stub if called without DB context (tests, BYOI path).
    if not difficulty_context:
        difficulty_context = _summarize_recent_difficulties(existing_titles)

    recent_note = f"\n{difficulty_context}" if difficulty_context else ""

    prompt = (
        f"Generate exactly 5 blog blueprints.\n\n"
        f"Follow the difficulty distribution (2 starter, 2 weekend, 1 ambitious) "
        f"and type mix (1 existing_asset, 2 new_build, 2 tutorial) precisely. "
        f"The two tutorials must cover different tools or concepts.\n\n"
        f"Author interests: {interests}\n\n"
        f"Existing projects (for existing_asset context):\n{existing_projects}\n"
        f"{recent_note}"
        f"{existing_block}"
        f"{narration_block}\n\n"
        f"REMINDER: At least 2 of the 5 ideas must be in domains unrelated to the "
        f"existing projects. At least 1 starter must be reproducible in a single "
        f"evening with minimal setup."
    )

    schema = {"type": "ARRAY", "items": _BLUEPRINT_ITEM_SCHEMA}
    raw = call_gemini_json(prompt, schema=schema, system=system, model=MODEL_FLASH)
    blueprints = json.loads(raw)
    return _validate_difficulty_distribution(blueprints)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 2 — CODE NARRATOR
# Provider: Cerebras + Qwen3 235B
# Frequency: High — once per file, per project sync
# Context: 64K free tier — covers all real files comfortably
# ═════════════════════════════════════════════════════════════════════════════

# File-type-specific focus instructions injected into the Narrator prompt.
# Each entry covers: what to emphasise, what to skip, what makes this
# type interesting from a blog/portfolio perspective.
_NARRATOR_FOCUS = {
    "python_router": """
FOCUS FOR FASTAPI ROUTER FILES:
  - The endpoint contract: what comes in (path params, query params, form data,
    request body), what goes out (response model, status codes, HTML vs JSON).
  - Error handling strategy: what gets a 404 vs 422 vs 500, and why.
  - The HTMX interaction model if present: which endpoints return HTML fragments
    vs full pages, and what that means for the frontend swap behaviour.
  - Database session management: how get_db() is used, whether commits happen
    in the right places, any N+1 risks in relationship loading.
  - Any non-obvious business logic embedded in the router that should probably
    be a service layer.
  SKIP: Basic CRUD patterns that are self-explanatory from the code.
""",
    "python_dag": """
FOCUS FOR AIRFLOW DAG FILES:
  - The task dependency graph: what runs in what order, and why that order matters.
  - Retry logic and failure handling: retries, retry_delay, on_failure_callback.
  - How the DAG gets its configuration: conf dict, environment variables, or DB reads.
  - The boundary between DAG orchestration logic and task business logic — is
    business logic correctly in PythonOperator callables, or leaking into the DAG?
  - Any XCom usage or state sharing between tasks.
  - Schedule interval and what triggers this DAG (scheduled vs sensor vs manual).
  SKIP: Boilerplate Airflow imports and standard DAG constructor arguments.
""",
    "python_agent": """
FOCUS FOR LLM AGENT FILES:
  - The prompt strategy for each agent function: what goes in the system instruction
    vs the user prompt, and what structural constraints are imposed (JSON schema,
    output format, word count, etc.).
  - Model selection rationale where evident: why this provider and model for this task.
  - Output parsing and validation: how structured output is extracted, what happens
    when the model doesn't follow format.
  - Any retry or fallback logic for API failures.
  - The most interesting or non-obvious prompt engineering decision in the file.
  - What this agent feeds into downstream — which other agents consume its output.
  SKIP: The text of every prompt in full — summarise the strategy, not the words.
""",
    "python_model": """
FOCUS FOR SQLALCHEMY MODEL / PYDANTIC SCHEMA FILES:
  - Relationship definitions: back_populates, lazy loading strategy, cascade rules.
  - Enum definitions and their use: what values are allowed and what state machine
    they represent (if any).
  - Any computed properties (@property) and what they derive.
  - Foreign key constraints and ondelete behaviour.
  - The mapping between ORM models and Pydantic response schemas — what gets
    exposed vs kept internal.
  - Any non-obvious column decisions: why nullable, why a specific type, why a
    server_default vs Python default.
  SKIP: Standard column definitions that are self-explanatory (id, created_at, etc.).
""",
    "python_service": """
FOCUS FOR SERVICE / UTILITY FILES:
  - What external system this wraps and why a dedicated service layer exists.
  - The authentication and credential handling approach.
  - Error handling: what exceptions are caught, what gets re-raised, what gets
    logged and swallowed.
  - Any retry or rate-limit handling.
  - The interface contract: what calling code needs to provide, what it gets back.
  - Any non-obvious implementation choices (e.g. why base64, why async vs sync).
  SKIP: Standard library usage that is self-explanatory.
""",
    "python_general": """
FOCUS FOR GENERAL PYTHON FILES:
  - The primary purpose: what problem does this solve and for whom.
  - Key data flows: what comes in, what is transformed, what goes out.
  - Notable architectural choices and why they matter.
  - External dependencies and how they are used.
  - Any clever patterns, workarounds, or non-obvious logic.
  - Potential gotchas for a developer modifying this file.
""",
    "html_template": """
FOCUS FOR JINJA2 HTML TEMPLATE FILES:
  - The HTMX interaction model: which elements have hx-get/hx-post/hx-target,
    what triggers them, and what they swap and where.
  - Template variables: what context the router must provide for this template
    to render correctly.
  - Partial vs full-page templates: is this a fragment returned by HTMX swap,
    or a full page extending base.html?
  - Any conditional rendering logic worth noting ({% if %} blocks that affect
    significant UI state).
  - JavaScript embedded in the template: what it does and why it's inline
    rather than in a separate file.
  SKIP: Standard HTML structure and Jinja2 syntax that is self-explanatory.
""",
    "css": """
FOCUS FOR CSS FILES:
  - The theming system if present: CSS custom properties (variables), theme
    switching mechanism (data-theme attribute), what changes between themes.
  - Layout approach: grid, flexbox, or a mix — what each section uses and why.
  - Any non-obvious specificity decisions or cascade dependencies.
  - Responsive breakpoints and what changes at each.
  - Animation and transition patterns.
  - Any utility class system or naming convention in use.
  SKIP: Standard property values that are self-explanatory.
""",
    "javascript": """
FOCUS FOR JAVASCRIPT FILES:
  - The overall purpose: what user interaction or data flow does this enable.
  - HTMX integration if present: where JS supplements HTMX rather than
    replacing it, and why.
  - Event handling patterns: what listens to what, and any non-obvious
    delegation or timing decisions.
  - State management: where state lives (DOM, module scope, localStorage),
    and what drives UI updates.
  - Any async patterns (fetch, Promise chains, async/await).
  - Non-obvious browser compatibility decisions.
  SKIP: Standard DOM manipulation that is self-explanatory.
""",
    "shell": """
FOCUS FOR SHELL SCRIPT FILES:
  - What this script automates and when it is intended to run (cron, manual, CI).
  - Credential and secret handling: how sensitive values are sourced.
  - Error handling: what happens on failure, whether the script is idempotent.
  - External dependencies: what tools must be installed and available.
  - Any non-obvious flag or command choices.
""",
    "config": """
FOCUS FOR CONFIGURATION FILES:
  - What this configures and how it fits into the broader system.
  - Non-obvious or non-default settings and why they were chosen.
  - Any environment-specific branching.
  - Secrets handling: what is parameterised vs hardcoded, what must be
    provided at runtime.
""",
    "sql": """
FOCUS FOR SQL FILES:
  - What data problem this query or schema element solves.
  - Performance considerations: indexes used, join strategy, potential
    full-table scans.
  - Any non-obvious SQL patterns (window functions, CTEs, subqueries).
  - Data integrity constraints and why they were chosen.
""",
    "other": """
FOCUS FOR THIS FILE TYPE:
  - Primary purpose: what problem does this solve.
  - Key decisions and non-obvious choices.
  - How this file fits into the broader system.
  - Anything a developer needs to know before modifying it.
""",
}


def agent_code_narrator(
    code_content: str,
    file_name: str,
    readme_context: str = "",
) -> str:
    """
    Reads a single file and produces a technical narration optimised for
    consumption by other AI agents (Researcher, Ghostwriter, Idea Expander,
    README Writer).

    This is NOT a human-facing document — it is structured intelligence
    for agents that need to understand what this file does and what is
    interesting about it.

    The narration adapts its focus based on file type, detected via
    _detect_file_type(). A FastAPI router gets different emphasis than an
    Airflow DAG, a CSS file, or a shell script.

    Two deliberate editorial lenses are always applied, regardless of file type:
      1. The most elegant or clever thing in the file — what is worth writing
         about from a portfolio or blog perspective.
      2. The most fragile or risky thing in the file — what a developer needs
         to understand before touching it.

    Args:
        code_content:   Raw file content as a string.
        file_name:      File name with extension (e.g. "finance.py").
                        Used for file type detection and context.
        readme_context: Optional existing README or project description
                        for orientation.

    Returns:
        Markdown narration string, under 700 words.
        Consumed by: Researcher, Ghostwriter, Idea Expander, README Writer.
    """
    file_type = _detect_file_type(file_name)
    type_focus = _NARRATOR_FOCUS.get(file_type, _NARRATOR_FOCUS["other"])

    system = f"""
You are a Principal Engineer reviewing a file for two audiences simultaneously:
  1. Other AI agents that need structured intelligence about this file to do
     their jobs (generating blog ideas, writing articles, producing READMEs).
  2. A hiring manager reading a portfolio project and trying to understand the
     author's technical decisions.

Your output is a technical narration — NOT a blog post, NOT a tutorial.
Write it as a dense, analytical briefing. Assume the reader is a senior engineer.
Do NOT explain basic syntax or standard library usage.

{type_focus}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY EDITORIAL LENSES — always include both, regardless of file type:

  🟢 THE MOST INTERESTING THING
     What is elegant, clever, or non-obvious in this file?
     What decision or pattern here would be worth writing a blog post about?
     Be specific — name the function, pattern, or line range.

  🔴 THE MOST FRAGILE THING
     What is the highest-risk thing in this file for a developer who doesn't
     know the codebase? What could break silently, what assumption is buried,
     what would a new contributor get wrong?
     Be specific and honest — this is the most useful part of the narration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT:
  Use Markdown. Structure with short headers. Keep under 700 words.
  Prioritise signal over completeness — a sharp 400-word narration is
  better than an exhaustive 700-word one.
"""

    project_ctx = f"Project context: {readme_context}\n" if readme_context else ""
    prompt = (
        f"File: {file_name}\n"
        f"{project_ctx}"
        f"Code:\n{code_content}"
    )

    content, _ = _cerebras(_CEREBRAS_QWEN3, system, prompt, temperature=0.25)
    return content


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 3 — GHOSTWRITER
# Provider: Groq + Llama 3.3 70B
# Frequency: Once per article — single call, never per-file
# ═════════════════════════════════════════════════════════════════════════════

def agent_ghostwriter(
    blueprint: dict,
    author_notes: str,
    code_narrative: str,
) -> str:
    """
    Writes a 70-80% complete first draft of a blog post, calibrated to
    the difficulty level of the blueprint.

    Starter posts get a tighter, more accessible draft. Weekend posts get
    balanced depth. Ambitious posts get full technical coverage.

    Args:
        blueprint:      Dict with title_concept, the_build, the_narrative,
                        the_selling_point, and ideally 'difficulty' and
                        'project_type'.
        author_notes:   HITL evidence from the author: code snippets,
                        lessons learned, actual results, personal context.
        code_narrative: Code Narrator output for the relevant file(s).

    Returns:
        Draft blog post in Markdown. Intentionally 70-80% complete —
        leaves room for the Refiner and the author's own voice.
    """
    difficulty   = blueprint.get("difficulty", "weekend")
    project_type = blueprint.get("project_type", "new_build")

    # Calibrate word count and tone to difficulty
    difficulty_guidance = {
        "starter": (
            "STARTER POST: Accessible and focused. Target 800–1,200 words. "
            "One core concept, explained clearly enough that a reader could "
            "follow along and reproduce the result in a single evening. "
            "Do not pad the post with background theory — get to the build fast."
        ),
        "weekend": (
            "WEEKEND PROJECT POST: Balanced depth and readability. Target "
            "1,200–1,800 words. Walk through the key implementation decisions "
            "without turning into a reference manual. A motivated reader should "
            "be able to build a similar thing over the weekend."
        ),
        "ambitious": (
            "AMBITIOUS POST: Technical depth is appropriate and expected. Target "
            "1,500–2,200 words. Cover the architecture, the tradeoffs, and the "
            "non-obvious lessons. Assume a senior reader who wants to understand "
            "not just what you built but why you made each significant decision."
        ),
    }.get(difficulty, "Target 1,200–1,800 words.")

    # Tutorial posts get a specific structural instruction
    tutorial_guidance = (
        "\nTUTORIAL STRUCTURE: Since this is a tutorial, each step needs a "
        "'why' not just a 'how'. Before showing the code, explain what problem "
        "that code solves. After showing it, explain what could go wrong and "
        "what the reader should verify. The reader should finish the post knowing "
        "not just how to do the thing but when and why to use this approach."
        if project_type == "tutorial" else ""
    )

    system = f"""
You are an expert Technical Ghostwriter for a data engineering blog.
Write a draft that is engaging, technically credible, and authentic.

{difficulty_guidance}{tutorial_guidance}

VOICE AND STRUCTURE:
  - Tone: a senior engineer writing for peers. Direct. No fluff.
  - Use clear headings, bold for key terms, fenced code blocks for all code.
  - Avoid clichés: no "In conclusion", "In today's world", "game-changer",
    "seamlessly", "leverage", "unlock", "harness".
  - Preferred structure: Hook → Problem → Build → Key Insight → Result → Takeaway.
    The Hook should be a specific situation or frustration, not a generic claim.
  - Leave room for the author's voice — do not over-polish. This is a 70-80%
    draft, not a finished article.
  - If the author provided notes about what actually happened (bugs, surprises,
    decisions that didn't work), use those. Real friction is more compelling than
    smooth narratives.
"""

    prompt = (
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        f"Author notes and results:\n{author_notes or 'None provided — use the blueprint narrative.'}\n\n"
        f"Technical code breakdown:\n{code_narrative or 'None provided.'}"
    )

    return _groq_llama(system, prompt, temperature=0.7)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 4 — REFINER
# Provider: Cerebras + Llama 3.3 70B
# Frequency: Once per article, after human review of draft_v1
# ═════════════════════════════════════════════════════════════════════════════

def agent_refiner(original_draft: str, user_feedback: str) -> str:
    """
    Integrates the author's review feedback into draft_v1 to produce draft_v2.

    The Refiner is a precise editor, not a rewriter. It applies the author's
    instructions exactly and preserves everything not explicitly addressed.

    Uses Cerebras + Llama 3.3 70B rather than Groq to avoid TPM throttling
    on long drafts. The task is instruction-following, not deep reasoning,
    so Llama 3.3 70B is appropriate.

    Args:
        original_draft: draft_v1 from the Ghostwriter.
        user_feedback:  Author's notes from the HITL review step in the
                        LifeOS UI. Can range from "make the intro punchier"
                        to specific section-by-section instructions.

    Returns:
        Complete revised draft in Markdown (draft_v2).
    """
    system = """
You are a professional Technical Editor. You receive a blog post draft and
specific feedback from the author. Your job is precise integration — apply
what the author asked for and preserve everything else.

Rules:
  - If the author asks to change a section, change it and nothing else.
  - If they ask to add something, add it in the most natural place.
  - If they ask to remove something, remove it cleanly.
  - Preserve the author's voice throughout — do not rewrite sentences
    they didn't ask you to touch, even if you could improve them.
  - Do not add sections the author didn't ask for.
  - Output the COMPLETE revised draft in Markdown. Nothing else — no
    preamble, no "Here is the revised draft:", no summary of changes.
"""
    prompt = (
        f"Original draft:\n{original_draft}\n\n"
        f"Author feedback:\n{user_feedback}"
    )

    content, _ = _cerebras(_CEREBRAS_QWEN3, system, prompt, temperature=0.3)
    return content


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 5 — EDITOR & SEO SPECIALIST
# Provider: Gemini 2.5 Flash
# Frequency: Once per article — quality matters, RPD is sufficient
# ═════════════════════════════════════════════════════════════════════════════

def agent_editor(draft_content: str) -> str:
    """
    Performs final polish, formatting consistency, and SEO metadata generation.

    This is the last AI pass before the article is marked READY_TO_PUBLISH.
    Quality matters here — the Editor sets the title, meta description, and
    tags that determine search discoverability.

    Args:
        draft_content: draft_v2 from Refiner, or draft_v1 if Refiner was
                       skipped (e.g. author was happy with the first draft).

    Returns:
        Structured string with a YAML-like metadata block followed by the
        polished article. The blog router parses this with _parse_editor_output.

        Format:
            ---
            Title: [final SEO title, max 80 chars]
            Meta Description: [150-char description]
            Tags: [tag1, tag2, tag3, tag4, tag5]
            ---

            [Full polished Markdown article]
    """
    system = """
You are a Managing Editor and SEO Specialist for a technical blog with a
technical but non-specialist audience (data engineers, developers, analysts).

YOUR TASKS:

1. NARRATIVE POLISH
   Fix narrative inconsistencies, repetitive phrasing, and awkward transitions.
   Do not rewrite the author's voice — improve clarity, not style.
   Remove filler phrases: "It's worth noting that", "As we can see",
   "In order to", "It is important to".

2. MARKDOWN FORMATTING
   Ensure all code blocks have language tags (```python, ```sql, ```bash).
   Ensure heading hierarchy is consistent (H2 for major sections, H3 for sub-sections).
   Ensure bold is used for terms being introduced, not for emphasis.
   Ensure no raw URLs — all links should be [descriptive text](url).

3. SEO TITLE (max 80 characters)
   Should be specific and searchable. Prefer concrete nouns over abstract ones.
   Good: "Building a Job Tracker with FastAPI, Gemini AI, and MariaDB"
   Bad:  "How I Used AI to Transform My Job Search"

4. META DESCRIPTION (max 150 characters)
   One sentence. What will the reader learn or be able to do after reading?
   Include the primary keyword naturally.

5. TAGS (5–7 tags)
   Mix specific (FastAPI, Apache Airflow, MariaDB) with broad (Python, SQL,
   data engineering). At least one tag should be a searchable topic, not
   just a technology name.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — follow exactly, no deviation:

---
Title: [Title here]
Meta Description: [Description here]
Tags: [tag1, tag2, tag3, tag4, tag5]
---

[Full polished Markdown article here]
"""
    prompt = f"Polish this draft and generate SEO metadata:\n\n{draft_content}"
    return call_gemini_text(system, prompt, model=MODEL_FLASH)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 6 — IDEA EXPANDER (BYOI — Bring Your Own Idea)
# Provider: Gemini 2.5 Flash (JSON mode)
# Frequency: Low — triggered manually from the LifeOS UI
# ═════════════════════════════════════════════════════════════════════════════

def agent_idea_expander(user_idea: str) -> dict:
    """
    Transforms rough user notes into a structured blog blueprint, including
    difficulty and project_type assignment.

    The user input is typically a stream-of-consciousness paragraph — the
    first sentence is treated as the most likely topic statement. The Expander
    finds the most interesting technical decision or result buried in the input
    and builds the blueprint around that, while honouring the opening framing.

    For existing_asset ideas (something already built), difficulty is inferred
    conservatively — the build is done, only the writing remains, so the
    effective difficulty is lower than the project's original complexity.

    Args:
        user_idea: Free-form text from the BYOI modal. Typically a paragraph
                   of notes, code snippets, or a stream of consciousness
                   describing a project or technique.

    Returns:
        Single blueprint dict including 'difficulty' and 'project_type'.
    """
    system = """
You are a senior Developer Advocate helping an author structure a blog idea.
The author has provided rough notes — a stream of consciousness. Your job is
to extract the strongest idea and shape it into a compelling article blueprint.

READING THE INPUT:
  - Treat the first sentence as the most likely topic statement and primary frame.
  - Scan the full input for the most interesting technical decision, surprising
    result, or non-obvious insight — this becomes the core of the_narrative.
  - If the first sentence and the most interesting insight conflict, honour the
    first sentence as the intended topic but surface the interesting insight in
    the_narrative as "the thing worth writing about."

INFERRING project_type:
  - If the author describes something already built and running: existing_asset.
  - If the author describes something they want to build: new_build.
  - If the author describes a technique or pattern they want to explain without
    a new project: tutorial.

INFERRING difficulty:
  - For existing_asset: err toward starter or weekend. The build is done.
    The article just needs writing. Match to the complexity of explaining it,
    not the complexity of building it.
  - For new_build: match to the realistic build time for someone of the
    author's evident skill level.
  - For tutorial: usually starter or weekend. Tutorials rarely need an
    ambitious tag.
  - Only assign ambitious if the idea genuinely requires multi-week work
    that doesn't exist yet.

FILLING THE BLUEPRINT:
  - title_concept: Specific and human-readable. Reflects the actual topic.
  - the_build: MVP-scoped to the difficulty. Not the full vision.
  - the_narrative: The story — problem, approach, key moment, result.
    Include the friction or surprise if the input mentions it.
  - the_selling_point: One sentence. Why does a technical reader care?
"""

    prompt = (
        f"Structure this idea into a blog blueprint:\n\n{user_idea}\n\n"
        f"Remember: the first sentence defines the topic. Find the most "
        f"interesting insight and make it the narrative's core."
    )

    raw = call_gemini_json(prompt, schema=_EXPANDER_SCHEMA, system=system, model=MODEL_FLASH)
    return json.loads(raw)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 7 — CODE COMMENTER
# Provider: Cerebras + Qwen3 235B
# Frequency: High — once per file, per project
# ═════════════════════════════════════════════════════════════════════════════

# Commenting conventions per file type.
# The Commenter follows the existing convention for each file type rather than
# inventing a new style. For Python, the project already uses # ── SECTION ──
# dividers extensively — the Commenter adopts and extends that pattern.
_COMMENTER_CONVENTIONS = {
    "python_router": """
PYTHON FASTAPI ROUTER CONVENTIONS:
  Section dividers: use the existing # ── SECTION NAME ─────────── style for
  logical groups of endpoints (e.g. # ── CRUD ENDPOINTS ──────────────────────).
  Docstrings: Google-style with Args, Returns, Raises. Be specific about what
  the endpoint returns in different success cases (200 HTML vs 200 JSON vs 204).
  Inline: comment the non-obvious — SQLAlchemy join strategies, HTMX-specific
  response decisions, status code choices that aren't self-evident.
""",
    "python_dag": """
PYTHON AIRFLOW DAG CONVENTIONS:
  Section dividers: # ── TASK: task_name ───────── before each task function.
  Docstrings: explain what the task does, what it reads from the DB or conf,
  and what state it writes back. Include the trigger context (manual vs scheduled).
  Inline: comment retry logic, XCom usage, and any non-obvious Airflow API usage.
""",
    "python_agent": """
PYTHON AGENT FILE CONVENTIONS:
  Section dividers: # ── AGENT N — AGENT NAME ─── (already present — extend this).
  Docstrings: explain the agent's role in the pipeline, what it consumes, what
  it produces, and which downstream agents use its output.
  Inline: comment prompt engineering decisions — why a constraint exists, why
  a specific output format was chosen, what would break if it changed.
  Do NOT reproduce full prompts in comments — summarise the strategy.
""",
    "python_model": """
PYTHON MODEL/SCHEMA CONVENTIONS:
  Section dividers: # ── TABLE NAME ─────────────── before each model class.
  Docstrings: explain the model's role in the system and its key relationships.
  Inline: comment non-obvious column choices (why nullable, why server_default,
  why a specific Enum), relationship loading strategies (lazy="selectin" and why),
  and any @property methods.
""",
    "python_service": """
PYTHON SERVICE FILE CONVENTIONS:
  Section dividers: # ── OPERATION NAME ──────────── for logical groups.
  Docstrings: explain what external system is wrapped, the authentication
  approach, and error handling contract.
  Inline: comment authentication header construction, base64 encoding steps,
  SHA usage for conflict detection, and any non-obvious API constraints.
""",
    "python_general": """
PYTHON GENERAL CONVENTIONS:
  Section dividers: # ── SECTION NAME ─────────────── for logical groups of 5+
  related lines that share a purpose.
  Docstrings: Google-style with Args, Returns, Raises.
  Inline: comment non-obvious logic, non-obvious imports, and any "why not X"
  decisions that future developers might question.
""",
    "html_template": """
JINJA2 HTML TEMPLATE CONVENTIONS:
  Block comments: {# ── SECTION NAME ── #} before major structural blocks
  (header, body sections, modals, partials).
  Inline: comment HTMX attributes that aren't self-explanatory — what triggers
  the request, what it targets, what it swaps, and what the expected response is.
  Comment conditional rendering blocks that control significant UI state.
  Do NOT comment standard HTML structure — only what is non-obvious.
""",
    "css": """
CSS CONVENTIONS:
  Section dividers: /* ── SECTION NAME ─────────────────────────────────────── */
  (match the style already present in the file if any exist).
  Inline: comment non-obvious property values, specificity decisions, and
  the purpose of custom properties (CSS variables) where the name alone is
  insufficient. Comment z-index stacking contexts.
  Do NOT comment standard property-value pairs where the purpose is obvious
  from the property name.
""",
    "javascript": """
JAVASCRIPT CONVENTIONS:
  Section dividers: // ── SECTION NAME ────────────────── for logical groups.
  JSDoc comments: /** @param, @returns */ for exported functions.
  Inline: comment async/await patterns, HTMX integration points, event
  delegation decisions, and any browser compatibility workarounds.
  Comment state management — where state lives and what drives UI updates.
  Do NOT comment standard DOM methods where the purpose is obvious.
""",
    "shell": """
SHELL SCRIPT CONVENTIONS:
  Section headers: # =============================================================
                   # SECTION NAME
                   # =============================================================
  Inline: comment non-obvious flags (e.g. --single-transaction on mysqldump),
  credential sourcing patterns, and error-checking logic.
  Comment every non-trivial pipe or redirection.
""",
    "config": """
CONFIG FILE CONVENTIONS:
  Section comments: explain what each configuration block controls.
  Inline: comment non-default values and explain why they differ from defaults.
  Comment any values that must be environment-specific.
""",
    "other": """
GENERAL CONVENTIONS:
  Add section comments before logical groups of related lines.
  Add function/block docstrings where purpose isn't immediately obvious.
  Inline: comment non-obvious logic only. Obvious means obvious to a senior
  developer familiar with the language and tools in this file.
""",
}


def agent_code_commenter(code_content: str, file_name: str) -> str:
    """
    Adds comments and docstrings to a file without changing any logic.

    This agent adapts its commenting style to the file type, following
    the conventions already established in the codebase rather than
    imposing a generic style.

    Key principles:
      - Group commenting over line-by-line: a section divider above a group
        of related lines is better than commenting each line individually.
      - "Obvious" is calibrated to a senior developer who knows the language
        and the tools (FastAPI, SQLAlchemy, Airflow, HTMX). Do not comment
        session.commit() or standard ORM patterns.
      - Google-style docstrings for Python functions and classes, even where
        type hints already provide type information — the docstring explains
        intent, not types.
      - Follow the file's existing commenting conventions and style.

    This agent does NOT: change logic, rename variables, reorder code,
    refactor, or optimise. It is a pure documentation pass.

    Args:
        code_content: Raw file content as a string.
        file_name:    File name with extension. Used for style detection.

    Returns:
        Complete file content with comments added. Same logic, more readable.
        No preamble — just the file content.
    """
    file_type = _detect_file_type(file_name)
    conventions = _COMMENTER_CONVENTIONS.get(file_type, _COMMENTER_CONVENTIONS["other"])

    system = f"""
You are a Senior Engineer performing a documentation pass on a codebase.
Your ONLY job is to add comments and docstrings. You MUST NOT change any
logic, rename anything, reorder code, or refactor anything whatsoever.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE-TYPE CONVENTIONS:
{conventions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMENTING PHILOSOPHY:

  GROUP OVER LINE:
    A single section divider above 5 related lines is better than
    5 individual inline comments. Ask: "Does this group of lines share
    a single purpose?" If yes, one comment above the group is enough.

  CALIBRATE "OBVIOUS" CORRECTLY:
    Obvious means obvious to a senior developer who knows this language
    AND the specific tools in this file (FastAPI, SQLAlchemy async,
    Airflow, HTMX, Jinja2). Do not comment:
      - session.commit(), await db.refresh(), return result
      - Standard ORM column definitions (id, created_at, etc.)
      - Standard HTML tags and structure
      - CSS property values whose purpose is evident from the name

  WHAT ALWAYS GETS COMMENTED:
    - Non-obvious "why" decisions (why this loading strategy, why this
      status code, why async here)
    - Workarounds and their reason ("# MariaDB requires full ENUM list")
    - Any logic that could silently break if a collaborator changes it
    - Group boundaries where the next block does something meaningfully
      different from the previous block

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT: The COMPLETE file with comments added.
No preamble. No explanation. No "Here is the commented version:".
Just the file, starting from the first line.
"""

    prompt = f"Add comments to {file_name} following the conventions above:\n\n{code_content}"

    content, _ = _cerebras(_CEREBRAS_QWEN3, system, prompt, temperature=0.2)
    return content


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 8 — CODE IMPROVER
# Provider: Cerebras + Qwen3 235B
# Frequency: High — once per file, per project
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for mixed code."""
    return len(text) // _CHARS_PER_TOKEN


def agent_code_improver(
    code_content: str,
    file_name: str,
    narration: str = "",
) -> str:
    """
    Analyzes a file and produces a structured improvement report with both
    formal structure (for scannability) and conversational explanation
    (for context and motivation).

    This agent NEVER rewrites the code — it proposes changes for human review.

    For large files (estimated >40K tokens), it always includes a mandatory
    File Structure Recommendation section suggesting how to split the file
    into smaller, single-responsibility modules. This is framed as a genuine
    software quality concern, not a tooling workaround.

    Stack-specific checks always performed:
      - SQLAlchemy: N+1 query risks, missing await on async operations,
        relationship loading strategy appropriateness
      - FastAPI: missing error handling, response model mismatches,
        business logic that belongs in a service layer
      - Airflow: DB logic in DAG files instead of helpers, missing retry
        configuration, XCom anti-patterns
      - General: hardcoded strings that should be constants or config,
        missing test coverage (flagged as a suggestion, not a blocker)

    Args:
        code_content: Raw file content as a string.
        file_name:    File name with extension.
        narration:    Code Narrator output for context (optional but recommended
                      — helps the Improver understand intent before critiquing).

    Returns:
        Markdown improvement report. Each suggestion has a formal header
        (for filtering) followed by a conversational explanation (for context).
    """
    token_estimate = _estimate_tokens(code_content)
    is_large = token_estimate > LARGE_FILE_THRESHOLD_TOKENS

    large_file_instruction = ""
    if is_large:
        large_file_instruction = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠ LARGE FILE DETECTED (~{token_estimate:,} tokens / {len(code_content):,} characters)

You MUST include a final section:

## File Structure Recommendation
**Category:** Maintainability
**Severity:** Medium

Frame this as a software quality concern (which it genuinely is), not a
tooling limitation. A file this large is almost certainly doing too many
things. Propose a concrete split:
  - Name each new module specifically (e.g. "routers/finance_categories.py")
  - Describe its single responsibility in one sentence
  - List which specific functions or classes move there
Do NOT say "split into smaller files" — be specific about what goes where.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    system = f"""
You are a Principal Engineer doing a thorough code review.
Produce an improvement report — do NOT rewrite the code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORT FORMAT — each suggestion uses this structure:

## [Short, specific title — e.g. "N+1 Query Risk in get_pipeline_summary"]

**Lines:** [line range or function name]
**Category:** Performance | Readability | Correctness | Security | Maintainability | Testing
**Severity:** Low | Medium | High

*What's happening:* [1-2 sentences of formal description — what the code does
and why it is a problem. Precise and scannable.]

*Why it matters here:* [1-3 sentences of conversational context — explain it
as you would in a PR comment to a colleague. Reference the specific codebase
context: what does this connect to, what could go wrong in production, what
would a future developer misunderstand? Be direct, not diplomatic.]

*Suggestion:*
```[language]
# Concrete alternative — the smallest change that fixes the issue.
# If the fix requires more context, show the before/after clearly.
```
[Optional: one sentence on the tradeoff if the fix has a cost.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS CHECK FOR (in addition to general issues):

  SQLAlchemy async:
    - N+1 risks: relationships loaded inside loops, missing joinedload/selectinload
    - Missing await on async session operations
    - lazy="select" on relationships that will always be needed (should be selectin)
    - Commits happening before refresh when fresh data is needed

  FastAPI:
    - Endpoints doing too much (should delegate to a service layer)
    - Missing HTTPException for edge cases that currently fail silently
    - Response model mismatches (ORM object returned where Pydantic expected)
    - Form data vs JSON body inconsistencies

  Airflow:
    - Business logic or DB queries in the DAG file instead of a task helper
    - Missing retries or retry_delay on tasks that call external APIs
    - XCom used to pass large objects (should use DB or object storage)
    - Tasks that are not idempotent (re-running would cause duplicate data)

  General:
    - Hardcoded strings that should be constants or environment config
    - Functions that do more than one thing (violation of single responsibility)
    - Missing test coverage — flag specific functions that are high-risk and
      untested. Frame as: "This function has no tests. Given [reason], a test
      for [specific case] would catch [specific failure mode]."
    - Silent exception swallowing (bare except, logging but not re-raising)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORDERING: High severity first, then Medium, then Low.
LIMIT: Maximum 10 suggestions (excluding the File Structure section if present).
FOCUS: Real issues only. Skip style nitpicks unless they cause actual friction.
       "This could be more Pythonic" is not a suggestion. "This will fail
       silently when X happens" is.
{large_file_instruction}
"""

    context_block = f"Context from Code Narrator:\n{narration[:600]}\n\n" if narration else ""
    prompt = (
        f"File: {file_name}\n"
        f"{context_block}"
        f"Code:\n{code_content}"
    )

    content, remaining_tokens = _cerebras(_CEREBRAS_QWEN3, system, prompt, temperature=0.2)
    return content, remaining_tokens
