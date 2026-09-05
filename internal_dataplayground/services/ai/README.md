# services/ai — AI Service Layer

GOVERNANCE.md §2.3's shared home for all LLM provider calls in this
repository. This document is the "model-routing rationale... moved from
blog_agents.py's header comment, since it applies project-wide" that
GOVERNANCE.md §2.3's target-state block calls for, plus guidance on the
package's two-tier interface (provider-specific functions vs. the
generic dispatcher added in WO#16) and the one documented SDK exception.

**This is a copy, not a move.** The original MODEL ROUTING table and
ROUTING RATIONALE section below still live in
`airflow/agents/blog_agents.py`'s module docstring too, unchanged and
undeleted, per WO#16 Step 5's explicit instruction — removing
institutional knowledge from its original, well-known location was
judged riskier than a little redundancy. A future cleanup pass can
consolidate to one location once people are used to checking this one.

---

## Model Routing

| Agent | Provider | Model |
|---|---|---|
| README Writer | Gemini | gemini-2.5-flash |
| Researcher | Gemini | gemini-2.5-flash (JSON mode) |
| Code Narrator | Cerebras | qwen-3-235b-a22b-instruct-2507 |
| Ghostwriter | Groq | llama-3.3-70b-versatile |
| Refiner | Cerebras | llama-3.3-70b *(documented — see discrepancy note below)* |
| Editor | Gemini | gemini-2.5-flash |
| Idea Expander | Gemini | gemini-2.5-flash (JSON mode) |
| Code Commenter | Cerebras | qwen-3-235b-a22b-instruct-2507 |
| Code Improver | Cerebras | qwen-3-235b-a22b-instruct-2507 |
| Recipe Extractor | Gemini | gemini-2.5-flash |
| Recipe Extractor (image) | Gemini, vision | gemini-2.5-flash *(added WO#16)* |
| Ingredient Normalizer | Gemma | gemma-4-31b-it |
| Recipe Discoverer | Gemini | gemini-2.5-flash |
| Job Fit Scorer | Gemini | gemini-2.5-flash-lite |
| Meal Planner / Workout Scheduler | Gemini | gemini-2.5-flash |
| Workout Plan Generator | Gemini | gemini-2.5-flash |
| Media Recommendation Explainer | Gemini | gemini-2.5-flash |
| Finance CSV Categoriser | google-genai SDK (not this package) | gemma-3-27b-it — see "SDK Exceptions" |

> **Known, pre-existing, intentionally-not-"corrected" discrepancy:** the
> Refiner row documents "Llama 3.3 70B," matching `blog_agents.py`'s own
> module docstring — but `agent_refiner()`'s actual code has always
> called the Qwen3 model constant, not Llama 3.3. This was confirmed and
> deliberately preserved (not silently fixed) during the WO#15 Cerebras
> migration, per that work order's own HARD BOUNDARIES: migrate the code
> exactly as it behaves, flag the documentation/code mismatch as a
> separate product decision. It remains open — whoever owns this should
> decide whether to fix the docstring or the code, not have it decided
> implicitly by whichever future change happens to touch the function.

## Routing Rationale

*(Copied verbatim from `blog_agents.py`'s module docstring — this
applies project-wide, not just to the blog pipeline agents originally
documented alongside it.)*

**Gemini 2.5 Flash:** Used for low-frequency, quality-sensitive tasks
(README Writer, Editor, Researcher, Idea Expander). 250 RPD free tier is
sufficient when these agents run once per article or once per batch.
Handles structured JSON output natively.

**Groq + Llama 3.3 70B:** Ghostwriter only. Prose generation is a single
call per article, never per-file. 131K context, 14,400 RPD free, fast
inference. Kept here because it was already working and prose
generation is not a reasoning-intensive task where frontier quality
matters most.

**Cerebras + Qwen3 235B (qwen-3-235b-a22b-instruct-2507):** Code
Narrator, Commenter, Improver. These are high-frequency per-file agents.
Qwen3 235B is frontier-grade on coding benchmarks (outperforms GPT-4.1
and Claude Opus 4 on Artificial Analysis Intelligence Index), runs at
~1,400 tokens/sec, and provides 64K context on the free tier with 1M
tokens/day — vs Gemini's 250 RPD which would exhaust in a single large
project narration run.

**Cerebras + Llama 3.3 70B:** Refiner only, per the file's documentation
— see the discrepancy note above; the code as-shipped actually uses
Qwen3. Mid-frequency (once per article), the task is targeted revision
rather than deep reasoning. Llama 3.3 70B (as documented) would handle
long drafts without the TPM throttling Groq's free tier would impose.

---

## When to use `call_ai_text` / `call_ai_json` vs. a provider-specific function

Two tiers of interface exist in this package:

**Provider-specific functions** — `call_gemini_text`, `call_gemini_json`,
`call_gemma_json`, `call_gemini_vision_json`, `call_groq_text`,
`call_cerebras_text`. The original interface, built up across
WO#11–15 (vision added WO#16). Every existing caller in this codebase
(`job_agents.py`, `recipe_agents.py`, `weekly_agents.py`,
`domains/workout/routers/workout_plan_ai_generator.py`,
`domains/media/routers/media_recommend.py`, and every function in
`airflow/agents/blog_agents.py`) uses these directly, and continues to
— WO#16 does not retrofit any of them onto the generic dispatcher below.
Use a provider-specific function when you already know which provider
you want (the normal case — provider/model choice is usually a
deliberate part of an agent's design, per the routing table above), or
when you need something the generic dispatcher doesn't expose
(Cerebras's `remaining_tokens` value; Gemini's vision support).

**`call_ai_text()` / `call_ai_json()`** (added WO#16) — a thin,
provider-agnostic dispatcher for new code where the provider is a
runtime parameter rather than a hardcoded design choice (e.g. a future
settings-driven "let the user pick their AI provider" feature). Notable
limitations, intentional and documented rather than designed around:

- `call_ai_text(provider="cerebras", ...)` silently discards the
  `remaining_tokens` value `call_cerebras_text()` returns. If you need
  it, call `call_cerebras_text()` directly.
- `call_ai_json()` only supports `provider="gemini"` today — Groq and
  Cerebras have no real JSON-mode caller anywhere in this codebase to
  generalize their JSON behavior from yet.

If you're writing a new agent function and you already know it should
use Gemini, Groq, or Cerebras specifically — use that provider's
function directly. The generic dispatcher exists for genuine
provider-agnostic use cases, not as a default first choice.

---

## SDK Exceptions

**`domains/finance/routers/finance_upload.py`'s `_categorise_batch()`**
calls the `google-genai` Python SDK directly
(`client.models.generate_content(model="gemma-3-27b-it", ...)`) rather
than going through `services/ai/`. This is the one AI-provider call site
in the codebase that has never been migrated into this package.

**Decision (WO#16):** left as an intentional, documented exception —
not converted to the raw-REST pattern used by every other provider call
in `services/ai/`. The SDK handles response parsing (`response.text`)
and model selection differently enough from the raw-REST
`requests.post(...)` + manual JSON-path-extraction pattern used
elsewhere that converting it doesn't reduce duplication — it just
changes which duplication exists (a raw-REST reimplementation of what
the SDK already does correctly, for no functional gain). `google-genai`
remains a real, currently-used dependency in `requirements.txt` and
`Dockerfile.airflow` because of this one caller.

**Revisit only if a second SDK-based caller appears.** At that point the
SDK-calling pattern itself — auth via `genai.Client(api_key=...)`,
`.models.generate_content(...)`, `response.text` parsing — would be
worth its own `services/ai/providers/` entry (SDK-based, structurally
distinct from the three existing raw-REST provider modules), the same
way Gemini/Groq/Cerebras each earned their own module once real call
sites existed to generalize from. Until then, building that abstraction
from a single caller would be guessing — the same reasoning this whole
migration series has applied consistently (see WO#11 postmortem §1.2,
WO#15 postmortem §6.3).

---

## Provider Coverage

| Provider | Text | JSON (schema) | JSON (no schema) | Gemma variant | Vision | Retry/backoff | Notes |
|---|---|---|---|---|---|---|---|
| Gemini | ✅ `call_gemini_text` | ✅ `call_gemini_json` | ✅ `call_gemini_json(schema=None)` | ✅ `call_gemma_json` | ✅ `call_gemini_vision_json` *(WO#16)* | ✅ via `post_with_retry` (429/503, fixed backoff) | |
| Groq | ✅ `call_groq_text` | — | — | — | — | ❌ none (matches the original `_groq_llama()`, which had none — WO#14 deliberately did not add any) | |
| Cerebras | ✅ `call_cerebras_text` — returns `(content, remaining_tokens)` | — | — | — | — | ✅ production-tuned: `_CEREBRAS_BACKOFF = [75, 150, 300, 600]`s, dual 429/503 handling (raw-response + SDK-exception paths), `Retry-After` header override | Only provider whose text function returns a tuple — see "When to use" above |
| google-genai SDK *(finance_upload.py only)* | via SDK, not this package | — | — | uses a Gemma model (`gemma-3-27b-it`) via the SDK | — | Whatever the SDK does internally — not inspected as part of this migration series | See "SDK Exceptions" above |

**Generic dispatcher:** `call_ai_text(provider, model, prompt, system=None, **kwargs)`,
`call_ai_json(provider, model, prompt, schema=None, system=None, **kwargs)`
— see "When to use" section above.
