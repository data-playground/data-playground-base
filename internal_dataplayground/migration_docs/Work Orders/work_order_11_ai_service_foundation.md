# Work Order #11 — AI Service Layer: Foundation + First Migration

*First of several work orders implementing GOVERNANCE.md §2.3 (the AI
Service Layer). This one builds the shared `services/ai/` package and
migrates exactly ONE existing caller (`gemini_client.py`, already the
cleanest of the six duplicate implementations) into it, to prove the
pattern before the remaining five call sites are migrated one at a time in
later work orders. Do not attempt to migrate more than the one caller
specified here — see HARD BOUNDARIES.*

---

## ROLE
You are a senior refactoring engineer consolidating duplicated
infrastructure code. Your job is to build a clean, shared implementation
and prove it works against exactly one real caller — not to chase down and
convert every duplicate at once. Resist the urge to "since I'm in here,
let me also fix the other five." Flag them as future work in NOTES, per
usual, but do not touch them.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Do not touch `blog_agents.py`, `recipe_agents.py`, `weekly_agents.py`,
  `workout_plans.py`'s `_call_gemini_for_plan`, `media_recommend.py`'s
  `_gemini_explain`, or `finance_upload.py`'s Gemini SDK call.** These are
  the remaining five duplicate implementations tracked in GOVERNANCE.md
  §2.3 — each gets its own dedicated work order later. This work order
  only builds the shared layer and migrates `job_agents.py` /
  `gemini_client.py`, nothing else.
- No behavior changes to the actual API calls — same model IDs, same
  retry/backoff timing, same error handling semantics as
  `gemini_client.py` currently has. This is a relocation-and-generalization
  of existing working code, not a rewrite or improvement of its retry
  logic.
- `job_agents.py`'s actual scraping/scoring logic
  (`search_linkedin_jobs`, `get_job_details`, `score_job_batch`, etc.) does
  not change — only its import of `call_gemini_json` changes, from
  `agents.gemini_client` to `services.ai`.
- Do not create `services/ai/providers/groq.py` or
  `services/ai/providers/cerebras.py` in this work order — only
  `providers/gemini.py` is needed for this first migration.
  `services/ai/base.py` and `services/ai/keys.py` should be written
  generically enough to support those providers later without rework, but
  do not build the provider modules themselves yet.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline (the code
   as it existed before your changes) to confirm it is not a regression you
   introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket:
   which endpoint/file, the exact error, and root cause if you found one.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue —
   explain the distinction in the one-line reason.

## WORKING METHOD
Execute steps in the order listed. After each step that changes running
behavior, pause and self-verify against the relevant acceptance criteria
before continuing. Do not defer all verification to the end.

If an acceptance criterion requires a resource not listed in SCOPE (e.g. a
live Gemini API key), do not skip it silently — perform the closest
achievable check (e.g. confirm the request payload/headers are constructed
identically to the pre-migration version, using a mocked HTTP layer), state
the substitution explicitly, and mark ⚠️.

## OUTPUT FORMAT
End with a report in exactly this structure:
1. **Files created**
2. **Files moved** (old path → new path, if any)
3. **Files edited** (path — description; flag and explain anything beyond
   the literal instructions)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Notes** (future migration candidates, risks, anything noticed but not
   acted on)

## ROLLBACK
This work order operates on files already tracked in git. If acceptance
criteria fail and cannot be quickly fixed, the safe rollback is `git
checkout` on every file listed in "Files created / moved / edited" above —
do not attempt a partial manual revert.

---

## SCOPE

**New package to create:**
```
services/ai/
    __init__.py
    base.py
    providers/
        __init__.py
        gemini.py
    keys.py
```

**Existing files to reference (read-only, for extracting logic — do not
edit in place, their content moves into the new package):**
- `airflow/agents/gemini_client.py` (this file's logic becomes
  `services/ai/providers/gemini.py` + `services/ai/base.py`; the original
  file itself is deleted once the migration is verified — see Step 5)
- `gcp_secrets.py` (read-only — `services/ai/keys.py` wraps this, does not
  replace it)

**File to edit (the one caller being migrated):**
- `airflow/agents/job_agents.py`

**Not in scope, referenced only to confirm no breakage:**
- Any DAG that calls `job_agents.py` functions (e.g. `life_os_job_scout.py`,
  `life_os_job_scout_ats.py`) — these call `job_agents.py`'s public
  functions (`search_linkedin_jobs`, `score_job_batch`, etc.), not
  `gemini_client.py` directly, so they should need zero changes. Confirm
  this rather than assuming it.

---

## STEPS

1. **Create `services/ai/keys.py`.** Provide one function:
   ```python
   def get_provider_key(provider: str) -> str:
       """
       Returns the API key/secret for a given provider name
       ("gemini", "groq", "cerebras", ...). Wraps gcp_secrets.get_key()
       with the naming convention each provider currently uses
       (e.g. "gemini" -> GEMINI_API env var / GCP secret name).
       """
   ```
   For the `"gemini"` case specifically, it should produce the same value
   `gemini_client.py`'s current `_gemini_key()` does
   (`os.environ.get("GEMINI_API")`) — call `gcp_secrets.get_key("GEMINI_API")`
   if that function's env-var-first/GCP-fallback behavior is compatible,
   or fall back to the direct `os.environ.get` pattern if not, but do not
   change what value is actually returned for `"gemini"` — only
   *centralize* how it's fetched. If `gcp_secrets.get_key()`'s behavior
   differs in a way that would change the returned value, do not force it
   — use the direct env-var read instead and note the discrepancy under
   Notes (this is a case where "getting the right value" matters more than
   "using the intended helper").

2. **Create `services/ai/base.py`.** Extract the generic retry/backoff
   HTTP-call machinery currently inside `gemini_client.py`'s
   `_post_with_retry()` — same 429/503 handling, same wait schedule. This
   becomes the shared low-level function that `providers/gemini.py` (and,
   later, `groq.py`/`cerebras.py`) build on. Keep the function signature
   generic (model URL, payload, retries) rather than Gemini-specific.

3. **Create `services/ai/providers/gemini.py`.** Move
   `gemini_client.py`'s `MODEL_FLASH`, `MODEL_FLASH_LITE` constants,
   `call_gemini_text()`, and `call_gemini_json()` here, updated to use
   `services.ai.base`'s shared retry function and `services.ai.keys` for
   the API key instead of their current private `_gemini_key()`. Keep the
   function names and signatures **identical** to what `job_agents.py`
   already calls (`call_gemini_json(system, prompt, schema, model=...,
   retries=...)`) — this is what makes Step 4 a zero-risk one-line import
   change rather than a call-site rewrite.

4. **Create `services/ai/__init__.py`** re-exporting `call_gemini_text`
   and `call_gemini_json` from `providers/gemini.py` at the top level (per
   the target public contract in GOVERNANCE.md §2.3 —
   `from services.ai import call_ai_text, call_ai_json` is the eventual
   goal, but for this first work order it's acceptable to re-export the
   Gemini-specific names as-is; the provider-agnostic wrapper naming
   (`call_ai_text`/`call_ai_json`) is deferred to the work order that adds
   a second provider, since designing that interface properly needs at
   least two real providers to generalize from correctly — don't guess at
   it with only one).

5. **Update `airflow/agents/job_agents.py`.** Change its import from
   `from agents.gemini_client import call_gemini_json, MODEL_FLASH_LITE`
   to `from services.ai import call_gemini_json` and `from
   services.ai.providers.gemini import MODEL_FLASH_LITE` (or re-export
   `MODEL_FLASH_LITE` from `services/ai/__init__.py` too, if that reads
   more naturally — your call, just be consistent and explain the choice).
   No other line in `job_agents.py` changes.

6. **Delete `airflow/agents/gemini_client.py`** only after confirming
   `job_agents.py` works correctly against the new `services/ai/` package
   (i.e., after Acceptance Criteria pass) — do not delete it speculatively
   before verification.

---

## ACCEPTANCE CRITERIA

- [ ] `services.ai.call_gemini_json` produces an identical HTTP request
  (same URL, same headers, same payload structure) to what
  `agents.gemini_client.call_gemini_json` produced pre-migration, for the
  same inputs — verify via a mocked HTTP layer since a live API call isn't
  available here; mark ⚠️ with this explanation if full live verification
  isn't possible
- [ ] `job_agents.py::score_job_batch()` still builds its system/prompt
  content identically and calls the new `services.ai.call_gemini_json`
  with the same arguments it previously passed to
  `agents.gemini_client.call_gemini_json`
- [ ] Retry behavior on a simulated 429 and a simulated 503 response
  matches the original `_post_with_retry()` timing/attempt-count logic
  (mock the HTTP responses to trigger both paths)
- [ ] `airflow/agents/gemini_client.py` no longer exists after this work
  order (confirm it was deleted only post-verification, per Step 6)
- [ ] No other file in the repo references `agents.gemini_client` or
  `from agents.gemini_client import ...` after this change — `grep -r
  "gemini_client"` across the repo should return zero hits outside this
  work order's own files and git history
- [ ] DAGs that transitively depend on `job_agents.py` (via
  `life_os_job_scout.py`, `life_os_job_scout_ats.py`) require **zero**
  changes — confirm by grepping for any direct `gemini_client` reference
  in those DAG files (there shouldn't be one; they only call
  `job_agents.py`'s public functions)

---

## For the next work order (not part of this one)

**Work Order #12** should migrate the second caller — recommend
`recipe_agents.py`, since it's structurally the most similar to
`gemini_client.py` (raw REST, `_gemini_flash`/`_gemini_flash_json`
functions with near-identical bodies) and will validate the pattern again
before tackling the more divergent cases (`_groq_llama`'s different
provider, `_cerebras`'s more complex retry/backoff, and the two inline
router-level calls in `workout_plans.py` and `media_recommend.py`). Only
after at least two providers exist in `services/ai/providers/` should the
`call_ai_text()`/`call_ai_json()` provider-agnostic wrapper interface from
GOVERNANCE.md §2.3 actually be designed and added to `services/ai/__init__.py`
— attempting to generalize the interface from a single provider's shape
risks guessing wrong about what needs to be configurable.
