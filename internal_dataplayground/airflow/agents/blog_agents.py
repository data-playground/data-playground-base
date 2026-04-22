# airflow/agents/blog_agents.py
"""
All blog pipeline agent functions.

Models used:
  - README Writer   : Gemini Flash (fast, instruction-following)
  - Researcher      : Gemini Pro with structured output (strict JSON schema)
  - Code Narrator   : Claude 3.5 Sonnet via GitHub Models (best code comprehension)
  - Ghostwriter     : Llama 3.3 70B via Groq (fast, natural prose)
  - Editor          : Gemini Flash (formatting + SEO)
  - Refiner         : Gemini Flash (targeted revision)
  - Idea Expander   : Gemini Pro with structured output
  - Code Commenter  : Gemini Flash (instruction following, no logic changes)
  - Code Improver   : Gemini Pro (deep reasoning for code review)

All API keys are pulled from GCP Secret Manager via get_key().
Never hardcode credentials here.
"""

import json
import logging
import os
import requests
import time

log = logging.getLogger(__name__)


# ── KEY HELPERS ───────────────────────────────────────────────────────────────

def _gemini_key() -> str:
    from gcp_secrets import get_key
    return get_key("Gemini-API")

def _groq_key() -> str:
    from gcp_secrets import get_key
    return get_key("Groq-API")

def _github_token() -> str:
    from gcp_secrets import get_key
    return get_key("GitHub-Models-Token")
    # Note: this is a separate token for GitHub Models inference,
    # distinct from your GitHub-PAT used for repo API calls.


# ── SHARED CALL HELPERS ───────────────────────────────────────────────────────

def _gemini_flash(system: str, prompt: str) -> str:
    """Calls Gemini Flash for fast, instruction-following tasks."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-3.1-flash-lite-preview:generateContent?key={_gemini_key()}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def _gemini_flash_2_5(system: str, prompt: str) -> str:
    """Calls Gemini Flash for fast, instruction-following tasks."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={_gemini_key()}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def _gemini_pro_json(system: str, prompt: str, schema: dict, retries: int = 3) -> str:
    """Calls Gemini Pro with structured JSON output."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-3-flash-preview:generateContent?key={_gemini_key()}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            if "503" in str(exc) or "Service Unavailable" in str(exc):
                if attempt < retries - 1:
                    wait = 5 ** attempt  # 1s, 2s, 4s
                    log.warning("Gemini 503, retrying in %ds (attempt %d/%d)", wait, attempt+1, retries)
                    time.sleep(wait)
                    continue
            raise  # re-raise non-503 errors immediately
    raise RuntimeError("Gemini unavailable after retries")


def _claude_sonnet(system: str, prompt: str) -> str:
    """Calls Claude 3.5 Sonnet via GitHub Models for code comprehension tasks."""
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_github_token()}",
    }
    payload = {
        "model": "Anthropic-Claude-3-5-Sonnet",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _groq_llama(system: str, prompt: str, temperature: float = 0.7) -> str:
    """Calls Llama 3.3 70B via Groq for fast natural prose generation."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {_groq_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 8192,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── BLUEPRINT JSON SCHEMA (shared by Researcher and Idea Expander) ─────────────

_BLUEPRINT_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "project_type": {
            "type": "STRING",
            "description": "existing_asset or new_build",
        },
        "title_concept": {"type": "STRING"},
        "the_build": {
            "type": "STRING",
            "description": "Technical architecture and tools used",
        },
        "the_narrative": {
            "type": "STRING",
            "description": "Core story — what problem this solves and why it matters",
        },
        "the_selling_point": {
            "type": "STRING",
            "description": "Why a hiring manager or senior engineer cares",
        },
    },
    "required": [
        "project_type", "title_concept",
        "the_build", "the_narrative", "the_selling_point",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 0 — README WRITER
# Scope: project-level (receives narrations from all files, not raw code)
# ─────────────────────────────────────────────────────────────────────────────

def agent_readme_writer(
    project_name: str,
    file_summaries: list[dict],
    description: str = "",
) -> str:
    """
    Generates a project-level README from file narrations.

    Args:
        project_name: human-friendly project name
        file_summaries: [{"path": "routers/finance.py", "narration": "..."}, ...]
        description: optional human note about what the project does

    Returns:
        Full README.md content as a string
    """
    system = """
You are a Senior DevOps Engineer. Write a clean, standard Markdown README.md
for a code repository based on per-file technical summaries provided to you.

Do NOT explain low-level code logic — treat each file as a black box and focus on:
1. Project Title & one-paragraph Overview
2. Architecture diagram (ASCII is fine)
3. Key Modules — one bullet per file summary provided
4. Prerequisites (Python version, libraries, tools)
5. Environment Variables required
6. How to run / deploy (Docker commands if applicable)
7. Folder structure

Keep it professional and concise. No fluff.
"""
    summary_block = "\n\n".join(
        f"**{s['path']}**\n{s['narration']}" for s in file_summaries
    )
    prompt = (
        f"Project: {project_name}\n"
        f"Description: {description or 'Not provided'}\n\n"
        f"File summaries:\n{summary_block}"
    )
    return _gemini_flash(system, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — RESEARCHER
# Scope: project-level (uses narrations as context, not raw code)
# ─────────────────────────────────────────────────────────────────────────────

def agent_researcher(
    interests: str,
    existing_projects: str,
    file_narrations: list[dict] | None = None,
    existing_titles: list[str] | None = None,   # ← add this
) -> list[dict]:
    """
    Generates 3 blog post blueprints following the 1-and-2 Rule:
      - 1 idea based on an existing completed asset
      - 2 ideas for net-new pipelines to build

    Args:
        interests: comma-separated topics the author cares about
        existing_projects: brief description of current projects
        file_narrations: optional list of {"path": str, "narration": str}
                         from code_files — gives the agent richer context

    Returns:
        List of 3 blueprint dicts
    """
    system = """
You are a senior Developer Advocate and Content Strategist.
Generate compelling, technically rigorous blog post blueprints for a senior
data professional's portfolio.

Strictly follow the 1-and-2 Rule:
  - Exactly 1 idea using an existing_asset (something already built)
  - Exactly 2 ideas for new_build (end-to-end pipelines to be built)

Ideas must be specific, actionable, and demonstrate senior-level thinking.
Avoid generic topics. Each idea must be meaningfully distinct.
Ideas must be specific and meaningfully distinct from each other AND from the existing titles provided.
"""
    narration_block = ""
    if file_narrations:
        narration_block = "\n\nRecent code context:\n" + "\n\n".join(
            f"{n['path']}: {n['narration'][:400]}" for n in file_narrations[:10]
        )

    existing_block = ""
    if existing_titles:
        existing_block = "\n\nAlready covered — do NOT suggest similar topics:\n" + \
            "\n".join(f"- {t}" for t in existing_titles[:30])

    prompt = (
        f"Generate the next 3 blog blueprints.\n"
        f"Author interests: {interests}\n"
        f"Existing projects: {existing_projects}"
        f"{existing_block}"
        f"{narration_block}"
    )

    schema = {
        "type": "ARRAY",
        "items": _BLUEPRINT_ITEM_SCHEMA,
    }

    raw = _gemini_pro_json(system, prompt, schema)
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — CODE NARRATOR
# Scope: file-level (reads one script, produces a technical summary)
# ─────────────────────────────────────────────────────────────────────────────

def agent_code_narrator(code_content: str, file_name: str, readme_context: str = "") -> str:
    """
    Reads a single script and produces a technical narration for agent consumption.
    Output is used by: Ghostwriter, README Writer, Researcher, Idea Expander.

    Args:
        code_content: raw source code
        file_name: e.g. "finance.py" — used for context
        readme_context: optional existing README or project description

    Returns:
        Markdown narration (NOT a blog post — a technical summary for agents)
    """
    system = """
You are a Principal Data Engineer reviewing code written by a colleague.
Your job is to produce a structured technical narration of the code for use
by other AI agents (not for human readers directly).

Focus on:
- What this file does in 1-2 sentences (the purpose)
- Key data flows: what comes in, what goes out, what is transformed
- Notable architectural choices and why they matter
- Any clever patterns, workarounds, or non-obvious logic
- External dependencies (APIs, DBs, third-party libs) and how they're used
- Potential gotchas or things a developer should know before modifying this file

Do NOT explain basic syntax. Assume the reader is a senior engineer.
Format as Markdown. Keep it under 600 words.
"""
    prompt = (
        f"File: {file_name}\n"
        f"{'Project context: ' + readme_context + chr(10) if readme_context else ''}"
        f"Code:\n{code_content}"
    )
    return _gemini_flash_2_5(system, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — GHOSTWRITER
# Scope: article-level (takes blueprint + narration + author notes)
# ─────────────────────────────────────────────────────────────────────────────

def agent_ghostwriter(
    blueprint: dict,
    author_notes: str,
    code_narrative: str,
) -> str:
    """
    Writes a 70-80% complete first draft of a blog post.
    Intentionally leaves room for human review and the Refiner agent.

    Args:
        blueprint: dict with title_concept, the_build, the_narrative, the_selling_point
        author_notes: HITL notes from the author (code snippets, lessons, results)
        code_narrative: output from Code Narrator for the relevant file

    Returns:
        Draft blog post in Markdown
    """
    system = """
You are an expert Technical Ghostwriter for a data engineering blog.
Write a 70-80% complete draft — engaging, technically credible, and authentic.

Rules:
- Match the tone of a senior engineer writing for peers: direct, no fluff
- Use clear headings, bold for key terms, and fenced code blocks for any code
- Avoid clichés: no "In conclusion", "In today's world", "game-changer"
- Structure: Hook → Problem → Build → Key Insight → Result → Takeaway
- Leave room for the author's voice — don't over-polish
- Target length: 1200-1800 words
"""
    prompt = (
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        f"Author's notes and results:\n{author_notes or 'None provided'}\n\n"
        f"Technical code breakdown:\n{code_narrative or 'None provided'}"
    )
    return _groq_llama(system, prompt, temperature=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — REFINER
# Scope: article-level (takes draft_v1 + author review notes → draft_v2)
# ─────────────────────────────────────────────────────────────────────────────

def agent_refiner(original_draft: str, user_feedback: str) -> str:
    """
    Integrates the author's review feedback into the draft.
    Produces draft_v2 — a significantly improved version ready for final editing.

    Args:
        original_draft: draft_v1 from Ghostwriter
        user_feedback: author's notes from the HITL review step in LifeOS UI

    Returns:
        Revised draft in Markdown (draft_v2)
    """
    system = """
You are a professional Technical Editor. You receive a blog post draft and
specific feedback from the author. Integrate the feedback precisely:
- If the author asks to change a section, change it
- If they ask to add something, add it
- If they ask to remove something, remove it
- Preserve the author's voice — do not over-polish
- Output the complete revised draft in Markdown, nothing else
"""
    prompt = (
        f"Original draft:\n{original_draft}\n\n"
        f"Author feedback:\n{user_feedback}"
    )
    return _gemini_flash(system, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — EDITOR & SEO SPECIALIST
# Scope: article-level (takes draft_v2 → final article + SEO metadata)
# ─────────────────────────────────────────────────────────────────────────────

def agent_editor(draft_content: str) -> str:
    """
    Performs final polish, formatting, and SEO optimization.

    Args:
        draft_content: draft_v2 from Refiner (or draft_v1 if skipped)

    Returns:
        Structured string with SEO header block + final article.
        Format:
            ---
            Title: [final title]
            Meta Description: [150-char description]
            Tags: [tag1, tag2, tag3]
            ---
            [full polished Markdown article]
    """
    system = """
You are a strict Managing Editor and SEO Specialist for a technical publication.

Tasks:
1. Fix any narrative inconsistencies or repetitive phrasing
2. Ensure Markdown formatting is flawless (especially code blocks and headings)
3. Generate a highly clickable, professional final title (max 80 chars)
4. Generate a 150-character SEO meta description
5. Generate 5-7 comma-separated tags relevant to the content

Your output MUST follow this exact format with no deviation:
---
Title: [Title here]
Meta Description: [Description here]
Tags: [tag1, tag2, tag3, tag4, tag5]
---

[Full polished Markdown article here]
"""
    prompt = f"Polish this draft:\n\n{draft_content}"
    return _gemini_flash(system, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6 — IDEA EXPANDER (BYOI — Bring Your Own Idea)
# Scope: idea-level (takes user raw input → structured blueprint)
# ─────────────────────────────────────────────────────────────────────────────

def agent_idea_expander(user_idea: str) -> dict:
    """
    Takes rough user notes, code snippets, or raw ideas from the LifeOS UI
    and structures them into a full blog blueprint.

    Args:
        user_idea: free-form text from the BYOI modal

    Returns:
        Single blueprint dict matching _BLUEPRINT_ITEM_SCHEMA
    """
    system = """
You are a senior Developer Advocate and Content Strategist.
The user has provided rough notes, code, or a starter idea for a blog post.
Analyze the input and structure it into a complete, compelling article blueprint.
Be specific about the technical stack mentioned. Infer project_type from context:
if the user describes something already built, use existing_asset; otherwise new_build.
"""
    prompt = f"Expand this idea into a full blueprint:\n\n{user_idea}"

    raw = _gemini_pro_json(system, prompt, _BLUEPRINT_ITEM_SCHEMA)
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 7 — CODE COMMENTER
# Scope: file-level (adds comments + docstrings, NO logic changes)
# ─────────────────────────────────────────────────────────────────────────────

def agent_code_commenter(code_content: str, file_name: str) -> str:
    """
    Adds comments and docstrings to a Python file without changing any logic.

    IMPORTANT: The output should be pushed to GitHub only after human review.
    This agent NEVER restructures, refactors, or renames anything.

    Args:
        code_content: raw source code
        file_name: e.g. "finance.py"

    Returns:
        Complete file content with comments added — same logic, more readable
    """
    system = """
You are a Senior Python Engineer performing a documentation pass.
Your ONLY job is to add comments and docstrings. Do NOT change any logic,
rename variables, reorder code, or refactor anything whatsoever.

Rules:
- Add a module-level docstring at the top explaining the file's purpose and role
- Add a Google-style docstring to every function and class that lacks one
  (Args, Returns, Raises sections where relevant)
- Add inline comments on non-obvious logic: complex queries, tricky conditionals,
  math, API call structures, async patterns
- Do NOT comment obvious lines like `return result`, `session.commit()`,
  simple assignments, or standard library usage
- Preserve all existing comments exactly as-is
- Output the COMPLETE file with comments added. No preamble, no explanation.
"""
    prompt = f"Add comments and docstrings to {file_name}:\n\n{code_content}"
    return _gemini_flash(system, prompt)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 8 — CODE IMPROVER
# Scope: file-level (analyzes code, proposes changes — does NOT apply them)
# ─────────────────────────────────────────────────────────────────────────────

def agent_code_improver(
    code_content: str,
    file_name: str,
    narration: str = "",
) -> str:
    """
    Analyzes a Python file and produces a structured improvement report.
    Does NOT rewrite the code — only proposes changes for human review.

    Args:
        code_content: raw source code
        file_name: e.g. "finance.py"
        narration: optional Code Narrator output for context

    Returns:
        Markdown improvement report with one section per suggestion
    """
    system = """
You are a Principal Python Engineer doing a thorough code review.
Produce a structured improvement report — do NOT rewrite the code.

For each suggestion, provide exactly:
## [Short title of the issue]
**Lines:** [line range or function name]
**Category:** Performance | Readability | Correctness | Security | Maintainability
**Severity:** Low | Medium | High
**Problem:** [1-2 sentences explaining the issue]
**Suggestion:** [concrete alternative with a short code snippet in a fenced block]

Focus on real issues. Skip style nitpicks unless they cause actual friction.
Order by severity (High first). Maximum 10 suggestions per file.
"""
    prompt = (
        f"File: {file_name}\n"
        f"{'Context: ' + narration[:500] + chr(10) if narration else ''}"
        f"Code:\n{code_content}"
    )
    
    return _gemini_flash(system, prompt)

