# Work Order #14 — AI Service Layer: Groq Provider + Ghostwriter Migration

*First work order to add a second provider to `services/ai/`. Unlike
WO#11–13 (all Gemini-shaped), Groq uses the OpenAI-compatible
`chat/completions` API shape — different payload structure, different
auth header, different response parsing path. `blog_agents.py`'s
`_groq_llama()` (used only by the Ghostwriter agent) is the sole caller
being migrated here. `blog_agents.py` itself remains otherwise untouched —
its Cerebras and Gemini logic stays in place until their own dedicated
work orders (see WO#13's closing note).*

---

## ROLE
You are a senior refactoring engineer adding a second provider to a shared
service layer. Your job is to extract Groq's existing, working call logic
into `services/ai/providers/groq.py` with zero behavior change, and update
exactly one caller to use it. Resist designing a "unified" text-completion
interface across Gemini and Groq in this pass — their shapes differ enough
(system-instruction-as-message vs. system-instruction-as-field,
temperature/max_tokens handling) that forcing a shared abstraction now
would be premature. That generalization, if ever needed, comes after
Cerebras is also migrated (WO#15) and there are three real shapes to learn
from, not two.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Only `_groq_llama()` and its one caller (`agent_ghostwriter()`) move
  in this work order.** `_gemini_flash()`, `_gemini_flash_json()`,
  `_cerebras()`, and every other function in `blog_agents.py` remain
  completely untouched — this is a single-function extraction from a
  larger file, not a full migration of that file.
- Do not create `services/ai/providers/cerebras.py` in this work order —
  that's WO#15.
- No behavior changes: same model (`llama-3.3-70b-versatile`), same
  `temperature`/`max_tokens` defaults, same endpoint, same header
  structure as `_groq_llama()` currently has.
- `blog_agents.py`'s `_groq_key()` helper is used only by `_groq_llama()`.
  Once that function is deleted from `blog_agents.py` (moved to the new
  provider module), `_groq_key()` should be deleted from `blog_agents.py`
  too — do not leave a dead, unused private key-fetcher behind. Confirm no
  other function in `blog_agents.py` calls `_groq_key()` before deleting it
  (grep first, don't assume).

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline to confirm
   it is not a regression you introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue.

## WORKING METHOD
Execute steps in order. Verify incrementally, not only at the end. If an
acceptance criterion needs a live API call that isn't available here,
verify request construction via a mocked HTTP layer instead, state the
substitution explicitly, and mark ⚠️.

## OUTPUT FORMAT
1. **Files created**
2. **Files moved**
3. **Files edited** (path — description; flag/explain anything beyond the
   literal instructions)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Notes**

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.

---

## SCOPE

**New file to create:**
- `services/ai/providers/groq.py`

**File to edit (adds the re-export):**
- `services/ai/__init__.py`

**File to edit (the one function being extracted, and its one caller):**
- `airflow/agents/blog_agents.py` — specifically `_groq_llama()`,
  `_groq_key()`, and `agent_ghostwriter()` only. No other function in this
  file changes.

**File to edit (uses `services.ai.keys`):**
- `services/ai/keys.py` (already exists from WO#11 — add Groq's key
  lookup, do not restructure the file)

**Not in scope, referenced only to confirm no breakage:**
- `airflow/dags/life_os_blog_creator.py` (the DAG that calls
  `agent_ghostwriter()` — should need zero changes, since it calls
  `blog_agents.py`'s public function, not `_groq_llama` directly; confirm
  rather than assume)

---

## STEPS

1. **Add Groq's key lookup to `services/ai/keys.py`.**
   `get_provider_key("groq")` should return
   `os.environ.get("GROQ_API")`, matching `blog_agents.py`'s current
   `_groq_key()` behavior exactly — same environment variable name, same
   direct read (no GCP Secret Manager fallback needed here unless
   `gcp_secrets.get_key("GROQ_API")` is confirmed to produce the identical
   value, same caution as WO#11 Step 1 applied to the Gemini key).

2. **Create `services/ai/providers/groq.py`.** Extract `_groq_llama()`'s
   logic verbatim:
   ```python
   MODEL_LLAMA_70B = "llama-3.3-70b-versatile"

   def call_groq_text(
       system: str,
       prompt: str,
       model: str = MODEL_LLAMA_70B,
       temperature: float = 0.7,
       max_tokens: int = 8192,
   ) -> str:
       """
       Calls Groq's OpenAI-compatible chat/completions endpoint.
       Used for prose generation (currently: the Ghostwriter agent only).
       """
   ```
   Same endpoint (`https://api.groq.com/openai/v1/chat/completions`), same
   `messages` array structure (`system` role + `user` role, not a
   Gemini-style `systemInstruction` field), same header structure using
   `services.ai.keys.get_provider_key("groq")`. This function does not
   need the shared retry/backoff logic from `services/ai/base.py` unless
   `_groq_llama()` already had retry logic — check the original function
   first; if it currently has no retry/backoff (a single `requests.post`
   with `raise_for_status()`), keep it that way. Do not add retry logic
   that wasn't there before — that would be a behavior change, not a
   relocation.

3. **Export `call_groq_text` and `MODEL_LLAMA_70B`** from
   `services/ai/__init__.py`.

4. **Update `blog_agents.py`:**
   - Delete the `_groq_llama()` function body.
   - Delete `_groq_key()`, after confirming via grep that no other
     function in this file references it.
   - Update `agent_ghostwriter()`'s call from `_groq_llama(system, prompt,
     temperature=0.7)` to `from services.ai import call_groq_text` at the
     top of the file, then `call_groq_text(system, prompt,
     temperature=0.7)` at the call site — same arguments, same order.

---

## ACCEPTANCE CRITERIA

- [ ] `services.ai.call_groq_text` produces an identical HTTP request
  (URL, headers including `Authorization: Bearer <key>`, JSON body with
  `model`, `messages`, `temperature`, `max_tokens`) to what
  `blog_agents._groq_llama()` produced pre-migration, for the same
  inputs — verify via mocked HTTP layer, mark ⚠️ if live verification
  isn't available
- [ ] `agent_ghostwriter()` still builds its system/prompt content
  identically (difficulty-calibrated guidance, tutorial-structure
  addendum, blueprint/notes/narrative interpolation) and passes it to the
  new `call_groq_text` correctly
- [ ] `blog_agents.py` no longer defines `_groq_llama` or `_groq_key` —
  confirm via grep within the file
- [ ] Every other function in `blog_agents.py`
  (`agent_readme_writer`, `agent_researcher`, `agent_code_narrator`,
  `agent_refiner`, `agent_editor`, `agent_idea_expander`,
  `agent_code_commenter`, `agent_code_improver`, and the `_cerebras`/
  `_gemini_flash`/`_gemini_flash_json` helpers they use) are **completely
  unchanged** — `git diff` on `blog_agents.py` should show changes
  isolated to the `_groq_llama`/`_groq_key` removal and
  `agent_ghostwriter`'s one call-site update, nothing else; paste or
  describe the diff scope in your report
- [ ] `life_os_blog_creator.py` (the DAG calling `agent_ghostwriter`)
  requires zero changes — confirm by reviewing its import (`from
  agents.blog_agents import agent_ghostwriter`, a public-function import
  unaffected by this internal refactor)
- [ ] `grep -r "_groq_llama\|_groq_key"` across the whole repo returns zero
  hits outside `services/ai/providers/groq.py`'s new implementation and
  git history

---

## For the next work order (not part of this one)

**Work Order #15** should add `services/ai/providers/cerebras.py` and
migrate `blog_agents.py`'s `_cerebras()` function — the most complex
migration in this whole series, since the current implementation has
real, hard-won retry/backoff logic (`_CEREBRAS_BACKOFF` schedule,
token-remaining-aware sleep timing used by `life_os_code_improve.py`'s
DAG, handling of both `RateLimitError`/`APIStatusError` SDK exceptions and
raw HTTP status codes) that must be preserved exactly, not simplified —
this is production-tuned behavior addressing a real rate-limit problem,
not incidental complexity to clean up. `_cerebras()` is also called by
five different `blog_agents.py` functions (Narrator, Refiner, Commenter,
Improver, and indirectly by the DAGs that orchestrate them) and returns a
`(content, remaining_tokens)` tuple that callers depend on — the migrated
`services.ai.call_cerebras_text()` (or similar name) must preserve that
tuple return shape exactly, since `life_os_code_improve.py` uses the
`remaining_tokens` value to decide how long to sleep between files. This
one deserves careful, unhurried treatment — do not batch it with anything
else.
