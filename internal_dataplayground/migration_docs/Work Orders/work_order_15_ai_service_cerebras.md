# Work Order #15 — AI Service Layer: Cerebras Provider Migration

*The most complex migration in this series. `_cerebras()` has production-
tuned retry/backoff logic addressing a real rate-limit problem — this is
not incidental complexity to simplify, it is the point of the function.
Four different `blog_agents.py` functions call it (Narrator, Refiner,
Commenter, Improver), and one of those (`agent_code_improver`) exposes a
`(content, remaining_tokens)` tuple that a DAG (`life_os_code_improve.py`)
depends on for adaptive sleep timing between files. Every detail in HARD
BOUNDARIES below exists because getting this migration wrong would degrade
real production behavior, not just cause a test failure.*

---

## ROLE
You are a senior refactoring engineer relocating a rate-limit-hardened
provider client with zero tolerance for behavioral drift. Treat this more
like moving a piece of safety-critical code than a routine refactor — read
`_cerebras()`'s full implementation twice before writing anything, and if
anything about its retry logic is unclear or seems inconsistent, stop and
report rather than guessing at the "obviously correct" simplification.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Preserve the exact retry/backoff schedule.** `_CEREBRAS_BACKOFF = [75,
  150, 300, 600]` (seconds) must move unchanged. The distinct handling of
  429 (`RateLimitError` exception path AND raw HTTP 429 response path — the
  current code has both, via the SDK's raw-response mode) versus 503
  (`APIStatusError` exception path AND raw HTTP 503 response path) must
  both be preserved, including the `Retry-After` header override on 429
  when present.
- **Preserve the `(content: str, remaining_tokens: int) -> tuple` return
  signature exactly.** `agent_code_improver()` unpacks and returns this
  tuple to its caller, and `life_os_code_improve.py`'s DAG task
  (`task_improve_files`) uses the `remaining_tokens` value to decide
  whether to sleep `INTER_REQUEST_DELAY_SEC` (65s) between files. Do not
  change this to a different return shape (e.g. a dataclass or dict) even
  if it seems like a cleaner design — that would require updating the DAG
  too, which is out of scope (DAGs never get edited as part of an AI
  service migration; see the standing DAG/FastAPI boundary rule).
- **Preserve both `_CEREBRAS_QWEN3` (`"qwen-3-235b-a22b-instruct-2507"`)
  and `_CEREBRAS_LLAMA33` (`"llama-3.3-70b"`) as separate exported
  constants**, even though only `_CEREBRAS_QWEN3` currently appears to be
  used by any caller (see the Notes requirement in Step 4 about this
  discrepancy — do not silently drop `_CEREBRAS_LLAMA33` just because it
  looks unused; report the finding instead).
- **Do not "fix" the apparent Refiner model mismatch.** `blog_agents.py`'s
  module docstring documents the Refiner as using "Cerebras + Llama 3.3
  70B," but `agent_refiner()`'s actual code calls `_cerebras(_CEREBRAS_QWEN3,
  ...)` — i.e., it actually uses Qwen3, not Llama 3.3, contradicting its own
  documentation. This is a pre-existing discrepancy, not something
  introduced by this migration. Per the standing pre-existing-bug handling
  rule: do not change which model `agent_refiner()` calls — migrate it
  exactly as it currently behaves (using `_CEREBRAS_QWEN3`), and report the
  documentation/code mismatch under Notes for a separate decision (should
  the docstring be corrected, or should the code actually switch to Llama
  3.3? That's a product decision, not a refactoring one).
- **Do not touch the four callers' surrounding logic** — only their
  `_cerebras(...)` call sites change. `agent_code_narrator()`'s
  `readme_context` construction, `agent_code_commenter()`'s conventions
  lookup, `agent_code_improver()`'s large-file instruction logic, etc. all
  stay exactly as they are.
- `_cerebras_key()` (the private Cerebras API key fetcher currently in
  `blog_agents.py`) — confirm via grep it's used only by `_cerebras()`
  before deleting it from `blog_agents.py`, same discipline as WO#14's
  `_groq_key()` handling.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline to confirm
   it is not a regression you introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue.

(The Refiner model mismatch noted in HARD BOUNDARIES is a specific,
pre-identified instance of this rule — handle it exactly per that
instruction, don't rediscover it as a "new" finding.)

## WORKING METHOD
Execute steps in order. Verify incrementally, not only at the end. If an
acceptance criterion needs a live API call that isn't available here,
verify request construction AND retry-path logic via a mocked HTTP/SDK
layer instead, state the substitution explicitly, and mark ⚠️. Given the
stakes here, prefer testing more retry-path branches over fewer if time
allows — a 429-with-Retry-After-header test and a 503 test are both worth
doing, not just one.

## OUTPUT FORMAT
1. **Files created**
2. **Files moved**
3. **Files edited** (path — description; flag/explain anything beyond the
   literal instructions)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Notes** (must include the `_CEREBRAS_LLAMA33` usage finding and the
   Refiner model mismatch finding, per HARD BOUNDARIES, even though both
   are pre-identified — confirm them explicitly rather than omitting them
   as "already known")

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.
Given this migration's risk profile, if ANY acceptance criterion related to
retry-path behavior fails and cannot be immediately root-caused, prefer
rolling back over attempting a quick fix — this is not a domain where "ship
now, patch later" is acceptable, since a broken retry path fails silently
in production (requests just stop retrying correctly) rather than loudly.

---

## SCOPE

**New file to create:**
- `services/ai/providers/cerebras.py`

**File to edit (adds the re-export):**
- `services/ai/__init__.py`

**File to edit (adds Cerebras key lookup):**
- `services/ai/keys.py`

**File to edit (the function being extracted, and its four callers):**
- `airflow/agents/blog_agents.py` — specifically `_cerebras()`,
  `_cerebras_key()`, `_CEREBRAS_BACKOFF`, `_CEREBRAS_QWEN3`,
  `_CEREBRAS_LLAMA33`, and the four call sites inside
  `agent_code_narrator()`, `agent_refiner()`, `agent_code_commenter()`,
  and `agent_code_improver()`. No other function in this file changes.
  **Leave the large commented-out prior `_cerebras()` implementation
  block** (the one preceded by `# def _cerebras(...)` with everything
  inside it commented out) **exactly as-is for now** — deleting genuinely
  dead commented-out code is legitimate cleanup, but it's a separate,
  trivial, easily-reviewable change and should not be bundled into a
  work order this sensitive. Note it under "Notes" as a one-line cleanup
  candidate for later instead.

**Not in scope, referenced only to confirm no breakage:**
- `airflow/dags/life_os_code_improve.py` (reads the `remaining_tokens`
  tuple element from `agent_code_improver()` — must keep working via the
  preserved return shape; this DAG itself is never edited)
- `airflow/dags/life_os_code_narrate.py`, `life_os_code_comment.py`
  (call `agent_code_narrator()`/`agent_code_commenter()` respectively —
  neither depends on the tuple return, both discard the second element
  already at the `blog_agents.py` layer, so these DAGs need zero changes
  regardless)
- `airflow/dags/life_os_blog_finalizer.py` (calls `agent_refiner()` — same,
  needs zero changes)

---

## STEPS

1. **Add Cerebras's key lookup to `services/ai/keys.py`.**
   `get_provider_key("cerebras")` should return
   `os.environ.get("CEREBRAS_API")`, matching `blog_agents.py`'s current
   `_cerebras_key()` behavior exactly.

2. **Create `services/ai/providers/cerebras.py`.** Extract `_cerebras()`'s
   full implementation verbatim — every branch of the retry loop, both
   exception-path and raw-response-path handling for 429/503, the
   `Retry-After` header check, the token-remaining header parsing and
   `int()` conversion with its `except (ValueError, TypeError)` fallback to
   `0`, and the final `RuntimeError` raised after all retries are
   exhausted with its `last_exc` detail. Function signature:
   ```python
   MODEL_QWEN3 = "qwen-3-235b-a22b-instruct-2507"
   MODEL_LLAMA33 = "llama-3.3-70b"

   def call_cerebras_text(
       model: str,
       system: str,
       prompt: str,
       temperature: float = 0.3,
       max_tokens: int = 4096,
   ) -> tuple[str, int]:
       """
       Calls a Cerebras-hosted model with production-tuned retry/backoff
       for rate limiting. Returns (content, remaining_tokens_this_minute).
       Callers that don't need the remaining-token count can discard it:
       content, _ = call_cerebras_text(...)
       """
   ```
   Import the `Cerebras`/`APIStatusError`/`RateLimitError` SDK classes the
   same way `_cerebras()` currently does (deferred import inside the
   function body, matching the existing pattern — do not move this import
   to module level unless you confirm that's safe, since the existing code
   deliberately imports it lazily).

3. **Export `call_cerebras_text`, `MODEL_QWEN3`, and `MODEL_LLAMA33`**
   from `services/ai/__init__.py`.

4. **Update `blog_agents.py`:**
   - Delete `_cerebras()`'s live implementation (the one actually being
     called — NOT the commented-out prior version above it, see SCOPE
     note).
   - Delete `_cerebras_key()`, `_CEREBRAS_BACKOFF`, `_CEREBRAS_QWEN3`, and
     `_CEREBRAS_LLAMA33` from `blog_agents.py`, after confirming via grep
     that nothing else in the file depends on them beyond the four call
     sites being updated.
   - Update all four call sites
     (`agent_code_narrator`, `agent_refiner`, `agent_code_commenter`,
     `agent_code_improver`) from `_cerebras(MODEL, system, prompt,
     temperature=...)` to `from services.ai import call_cerebras_text` at
     the top of the file, then `call_cerebras_text(MODEL, system, prompt,
     temperature=...)` at each call site — same model constant reference
     (now `services.ai.MODEL_QWEN3` etc.), same arguments, same order.
     **`agent_refiner()` keeps calling with the Qwen3 model constant**, per
     HARD BOUNDARIES (do not "correct" it to Llama 3.3).
   - Confirm `agent_code_improver()`'s `content, remaining_tokens =
     call_cerebras_text(...)` unpacking and its `return content,
     remaining_tokens` still match the function's existing return
     signature exactly (this function already returns a tuple today — see
     its current signature `-> str` in the docstring, which is actually
     inaccurate since it returns a tuple; do not fix that docstring typo
     either, note it under Notes alongside the other pre-existing
     discrepancies).

---

## ACCEPTANCE CRITERIA

- [ ] `services.ai.call_cerebras_text` produces an identical HTTP
  request (model, messages, temperature, max_tokens) to what
  `blog_agents._cerebras()` produced pre-migration, for the same inputs
- [ ] **429 retry path:** simulate a 429 response with a `Retry-After`
  header — confirm the function sleeps for exactly that duration (not the
  `_CEREBRAS_BACKOFF` default) before retrying, matching pre-migration
  behavior
- [ ] **429 retry path (no header):** simulate a 429 response without
  `Retry-After` — confirm it falls back to the `_CEREBRAS_BACKOFF` schedule
  value for that attempt number
- [ ] **503 retry path:** simulate a 503 response — confirm it uses the
  `_CEREBRAS_BACKOFF` schedule (not the `Retry-After` logic, which is
  429-specific)
- [ ] **Exhausted retries:** simulate all four backoff attempts failing —
  confirm a `RuntimeError` is raised with the same message format as
  pre-migration, including the `last_exc` detail
- [ ] **Return shape:** confirm `call_cerebras_text(...)` returns a
  2-tuple `(str, int)` in every success path, matching what
  `agent_code_improver()` expects to unpack
- [ ] `agent_code_narrator()`, `agent_refiner()`, `agent_code_commenter()`
  all correctly discard the second tuple element (`content, _ = ...`) and
  return only `content`, matching their pre-migration signatures
  (`-> str`)
- [ ] `agent_code_improver()` still returns the full `(content,
  remaining_tokens)` tuple to its own caller — confirm this by tracing its
  return statement, and if feasible, confirm `life_os_code_improve.py`'s
  `task_improve_files()` still correctly reads
  `remaining_tokens` from it (this DAG itself isn't edited, but its
  contract with `agent_code_improver()` must still hold)
- [ ] `blog_agents.py` no longer defines `_cerebras`, `_cerebras_key`,
  `_CEREBRAS_BACKOFF`, `_CEREBRAS_QWEN3`, or `_CEREBRAS_LLAMA33` as live
  (non-commented) code — confirm via grep within the file, and confirm
  the separate, still-commented-out prior implementation block is
  untouched (still present, still commented out, not deleted)
- [ ] `agent_refiner()` still calls with the Qwen3 model — confirm
  explicitly, per HARD BOUNDARIES (this is a "confirm it's unchanged," not
  a "confirm it's correct," criterion)
- [ ] `grep -r "_cerebras\b" airflow/agents/blog_agents.py` (word-boundary
  match, to exclude the commented-out block if you want to distinguish)
  returns hits only inside comments and the four updated call sites'
  surrounding context — no live function definition remains

---

## For the next work order (not part of this one)

With Gemini (WO#11–13), Groq (WO#14), and Cerebras (WO#15) all now in
`services/ai/providers/`, three real provider shapes exist. **Work Order
#16** can now responsibly attempt what WO#11 and #12 deliberately deferred:
designing the provider-agnostic `call_ai_text()` / `call_ai_json()`
wrapper interface from GOVERNANCE.md §2.3, informed by all three real
shapes rather than guessed at from one or two. That work order should also
finally tackle `finance_upload.py`'s SDK-based Gemma call (deciding
whether it becomes a fourth raw-REST-style provider function or stays
SDK-based as a documented, intentional exception) and the vision-support
gap in `recipe_agents.py::agent_extract_recipe_from_image()` flagged back
in WO#12. Once all of that lands, `blog_agents.py` will have every one of
its provider calls routed through `services/ai/`, at which point it's
worth a final, separate pass specifically to remove the shim-like leftover
structure in that file (the now-empty section headers, the dead commented
block flagged in this work order's SCOPE note) — again, as its own small,
easily-reviewable cleanup change, not bundled into a functional migration.
