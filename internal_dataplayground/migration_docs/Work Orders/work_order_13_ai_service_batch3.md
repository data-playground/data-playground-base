# Work Order #13 — AI Service Layer: Migrate `weekly_agents.py`, `workout_plans.py`, `media_recommend.py`

*Third migration batch. Two of these three callers (`workout_plans.py`'s
`_call_gemini_for_plan` and `media_recommend.py`'s `_gemini_explain`) use
JSON mode **without** schema enforcement, and the second also has **no**
system instruction at all — neither shape fits `call_gemini_json()` as it
exists after WO#11/#12 (which currently requires both `system` and
`schema`). This work order generalizes `call_gemini_json()` to make both
parameters optional, then migrates all three callers through it. This is
the point flagged in WO#12's closing note where enough real call shapes
exist to generalize the interface with confidence — do this generalization
carefully, since WO#11 and WO#12's already-migrated callers must keep
working identically through the same function.*

---

## ROLE
You are a senior refactoring engineer consolidating duplicated
infrastructure code. Your job is to generalize the shared service function
just enough to fit three more real callers — not to redesign it
speculatively for hypothetical future needs. Flag anything else in NOTES;
do not act on it.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Do not touch `blog_agents.py`, `finance_upload.py`'s Gemini SDK call,
  or `agent_extract_recipe_from_image()` in `recipe_agents.py`.** Still
  excluded, same as WO#11/#12.
- **`_groq_llama` and `_cerebras` (both in `blog_agents.py`) remain
  untouched** — this work order only extends the Gemini provider, not a
  second/third provider. Building `services/ai/providers/groq.py` and
  `cerebras.py` is separate future work.
- **The generalization to `call_gemini_json()` must be strictly additive
  and backward-compatible.** `job_agents.py` (WO#11) and `recipe_agents.py`
  (WO#12) already call it with both `system` and `schema` provided
  positionally/by-keyword in the current signature — after this change,
  those exact same call sites must produce byte-identical requests to what
  they produce today. Verify this explicitly (see ACCEPTANCE CRITERIA) —
  do not just assume adding optional parameters is safe without checking.
- Do not change `MODEL_FLASH`/`MODEL_FLASH_LITE` constants or retry timing.

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

**Existing service-layer files to extend (built in WO#11, extended in
WO#12):**
- `services/ai/providers/gemini.py`

**Files to edit (the three callers being migrated):**
- `airflow/agents/weekly_agents.py`
- `routers/workout_plans.py` (or `domains/workout/routers/workout_plans.py`
  if WO#8 has already run — locate wherever it currently is; this work
  order does not depend on WO#8's status)
- `routers/media_recommend.py` (or
  `domains/media/routers/media_recommend.py` if WO#9 has already run —
  same locate-wherever-it-is note)

**Not in scope, verified only for backward-compatibility confirmation:**
- `airflow/agents/job_agents.py` (WO#11's migrated caller — must still
  work identically after this work order's signature change)
- `airflow/agents/recipe_agents.py` (WO#12's migrated callers — same)

---

## STEPS

1. **Generalize `call_gemini_json()` in `services/ai/providers/gemini.py`.**
   Change its signature from requiring `system` and `schema` to:
   ```python
   def call_gemini_json(
       prompt: str,
       schema: dict | None = None,
       system: str | None = None,
       model: str = MODEL_FLASH,
       retries: int = 3,
   ) -> str:
   ```
   Behavior:
   - If `system` is provided, include the `systemInstruction` key in the
     payload exactly as before. If `system` is `None`, omit that key
     entirely from the payload (matching `media_recommend.py`'s current
     request shape, which has no `systemInstruction` at all).
   - If `schema` is provided, include `responseSchema` in
     `generationConfig` exactly as before. If `schema` is `None`, omit
     `responseSchema` but still set `responseMimeType: "application/json"`
     (matching `workout_plans.py`'s and `media_recommend.py`'s current
     request shape, which both use JSON mode without schema enforcement).
   - **Update the two already-migrated call sites** (`job_agents.py`,
     `recipe_agents.py`) to use keyword arguments explicitly (`system=...,
     schema=...`) if they currently rely on positional argument order that
     the new signature would break. Check the current call sites carefully
     — this is the step most likely to silently break WO#11/#12's work if
     rushed.

2. **Update `weekly_agents.py`.** Both `agent_plan_meals()` and
   `agent_schedule_workouts()` use `_gemini_flash_json(system, prompt,
   schema)` — replace with `from services.ai import call_gemini_json` and
   update call sites to `call_gemini_json(prompt, schema=schema,
   system=system)` (both `system` and `schema` are provided here, so this
   is the "full" case, structurally identical to WO#11/#12's migrations).
   Delete the now-unused private `_gemini_flash_json()` function.

3. **Update `workout_plans.py`'s `_call_gemini_for_plan()`.** This
   function currently builds its own `requests.post(...)` call with
   `os.environ.get("GEMINI_API")` fetched directly. Replace its body with
   a call to `services.ai.call_gemini_json(prompt, schema=None,
   system=system, model=MODEL_FLASH)` (import `MODEL_FLASH` from
   `services.ai` or `services.ai.providers.gemini`). The function itself
   can stay as a thin wrapper (`_call_gemini_for_plan(prompt, system)` →
   internally calls the service layer) if that minimizes the diff at its
   call site in `generate_plan()`, or `generate_plan()` can call
   `services.ai.call_gemini_json` directly and `_call_gemini_for_plan` can
   be deleted — your choice, but state which you did and why in your
   report. Remove the now-unused direct `os.environ.get("GEMINI_API")`
   read and the local `import requests as req` if nothing else in this
   file needs it.

4. **Update `media_recommend.py`'s `_gemini_explain()`.** This function
   builds its own `requests.post(...)` call with `os.environ.get('GEMINI_API')`
   fetched directly, no system instruction, no schema. Replace with
   `services.ai.call_gemini_json(prompt, schema=None, system=None)`
   (explicit `system=None` for clarity even though it's the default, since
   this is the one caller where the *absence* of a system instruction is
   a meaningful, intentional detail worth being explicit about in the
   code). Remove the now-unused direct `os.environ.get('GEMINI_API')` read
   and local `import requests as req` if nothing else in this file needs
   it. Leave the rest of `_gemini_explain()` (prompt construction, response
   parsing via `re.sub` + `json.loads`) completely unchanged — that's
   domain-specific response handling, not a provider-call concern.

---

## ACCEPTANCE CRITERIA

- [ ] **Backward compatibility (required, check first):** `job_agents.py`'s
  `score_job_batch()` call to `call_gemini_json` still produces an
  identical request payload to its WO#11 post-migration behavior — confirm
  via mocked HTTP layer
- [ ] **Backward compatibility (required, check first):** `recipe_agents.py`'s
  three `call_gemini_json` call sites (from WO#12) still produce identical
  request payloads to their WO#12 post-migration behavior
- [ ] `weekly_agents.py::agent_plan_meals()` and `agent_schedule_workouts()`
  produce identical requests (with `systemInstruction` and
  `responseSchema` both present) to their pre-migration behavior
- [ ] `workout_plans.py`'s plan generator produces a request with
  `systemInstruction` present but **no** `responseSchema` key, only
  `responseMimeType: "application/json"` — matching pre-migration exactly
- [ ] `media_recommend.py`'s `_gemini_explain()` produces a request with
  **no** `systemInstruction` key and **no** `responseSchema` key, only
  `responseMimeType: "application/json"` — matching pre-migration exactly
- [ ] `GET /workout/plans` → `POST /workout/plans/generate` route (or its
  `domains/workout/` equivalent) still reaches the plan generator
  correctly end-to-end (mocked HTTP response); mark ⚠️ if live API
  verification isn't available
- [ ] `POST /media/recommend/generate` with `MEDIA_RECOMMEND_AI=true` still
  reaches `_gemini_explain` correctly end-to-end (mocked HTTP response);
  mark ⚠️ if live API verification isn't available
- [ ] `grep -r "os.environ.get(.GEMINI_API.)"` in `workout_plans.py` and
  `media_recommend.py` returns zero hits (confirming the direct env-var
  reads were removed, not just bypassed)
- [ ] `weekly_agents.py` no longer defines `_gemini_flash_json` — confirm
  via grep within the file

---

## For the next work order (not part of this one)

With `job_agents.py`, `recipe_agents.py`, `weekly_agents.py`,
`workout_plans.py`, and `media_recommend.py` all now routed through
`services/ai/providers/gemini.py`, the remaining known duplicates
(GOVERNANCE.md §2.3) are: `blog_agents.py`'s `_groq_llama` and `_cerebras`
(two genuinely different providers, needing their own
`services/ai/providers/groq.py` and `cerebras.py` — recommend these as two
separate work orders, #14 and #15, since Cerebras's retry/backoff logic in
particular is more complex than anything migrated so far and deserves
focused attention on its own), `finance_upload.py`'s `google-genai`
SDK-based call (a different calling convention entirely — worth a short
design discussion on whether to keep it SDK-based as a documented exception
or convert it to the raw-REST pattern, before writing that work order,
rather than assuming one approach), and the vision-support gap in
`recipe_agents.py`'s `agent_extract_recipe_from_image()` flagged back in
WO#12. Once Groq and Cerebras both have provider modules, `blog_agents.py`
itself — explicitly untouched through WO#11–13 — finally becomes eligible
for its own migration work order, since by then every provider it uses
will have a home in `services/ai/providers/`.
