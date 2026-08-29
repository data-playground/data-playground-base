# Work Order #13 (REWRITTEN) — AI Service Layer: Batch Migration —
`weekly_agents.py`, `workout_plan_ai_generator.py`, `media_recommend.py`,
`blog_agents.py` (Gemini functions) + TMDB Duplication Reconciliation

**This replaces the original WO#13 draft.** Rewritten after WO#8's and
WO#9's postmortems surfaced two things the original draft couldn't have
known: (1) `routers/workout_plans.py` no longer exists — WO#8's
authorized follow-up split it into `workout_plans_crud.py` and
`workout_plan_ai_generator.py`, and only the latter is this WO's concern;
(2) WO#9's own follow-on work created `airflow/agents/media_agents.py`, a
second, independent TMDB-calling implementation that didn't exist when any
AI Service Layer work order was drafted.

This WO also now closes a real gap flagged in WO#11's own postmortem:
`blog_agents.py`'s Gemini-shaped functions (`_gemini_flash`,
`_gemini_flash_json`) were never assigned to any drafted work order —
WO#14 only takes `_groq_llama`, WO#15 only takes `_cerebras`, and WO#16
explicitly excludes blog_agents.py's remaining functions. Left alone, the
AI Service Layer program would finish exactly as planned and still leave
GOVERNANCE §2.3's target state unmet for this file. This WO closes that
gap by extracting blog_agents.py's Gemini pieces now, ahead of WO#14
(Groq) and WO#15 (Cerebras) — the same single-provider-at-a-time
extraction pattern those two already use, just done first.

**Read this whole document before starting — Task 1, Step 5 and Task 2 are
the parts most likely to go wrong if rushed.** Task 2 is *not* an AI
Service Layer migration in the usual sense — see its own preamble below
before assuming the fix pattern is "move it into `services/ai/`."

---

## ROLE
You are a senior refactoring engineer consolidating duplicated
infrastructure code and closing two pieces of real, previously-disclosed
technical debt. You are not redesigning prompts, changing retry behavior,
or "improving" anything about how these agents work. You are also not
merging two things GOVERNANCE.md deliberately keeps separate — see Task
2's HARD BOUNDARIES before touching `media_agents.py`.

## HARD BOUNDARIES

**General:**
- Only read/edit files explicitly listed in SCOPE below.
- Do not touch `finance_upload.py`'s Gemini SDK call or
  `recipe_agents.py::agent_extract_recipe_from_image()` — both remain
  WO#16's concern.
- Do not create `services/ai/providers/groq.py` or
  `services/ai/providers/cerebras.py` — still out of scope (WO#14, WO#15).
- **The `call_gemini_json()` generalization must be strictly additive and
  backward-compatible** with the WO#11 (`job_agents.py`) and WO#12
  (`recipe_agents.py`) call sites, exactly as the original WO#13 required
  — and must *also* not break whatever new call sites this WO itself adds
  for blog_agents.py's functions in Step 5. Verify both directions
  explicitly (see ACCEPTANCE CRITERIA); do not assume adding optional
  parameters is automatically safe.
- Do not change `MODEL_FLASH`/`MODEL_FLASH_LITE`/`MODEL_GEMMA` constants
  or retry timing.

**Task 1 — blog_agents.py specifically:**
- **Do NOT touch `blog_agents.py`'s Groq logic** (`_groq_llama`,
  `_groq_key`, `agent_ghostwriter()`'s call site) — that is WO#14's job,
  which runs after this one.
- **Do NOT touch `blog_agents.py`'s Cerebras logic** (`_cerebras`,
  `_cerebras_key`, `_CEREBRAS_BACKOFF`, `_CEREBRAS_QWEN3`,
  `_CEREBRAS_LLAMA33`, or the four callers that use it:
  `agent_code_narrator`, `agent_refiner`, `agent_code_commenter`,
  `agent_code_improver`) — that is WO#15's job. Do not touch the
  still-commented-out prior `_cerebras()` implementation block either —
  that's WO#19's job, later still.
- **Confirm — do not assume — which functions call `_gemini_flash` vs.
  `_gemini_flash_json` before writing anything.** Based on WO#11's
  postmortem and process of elimination against WO#14/#15's known
  targets, the expected Gemini-based functions are `agent_researcher`,
  `agent_editor`, `agent_idea_expander`, and `agent_readme_writer` — but
  this is an inference, not a confirmed fact from having read the file.
  Grep for `_gemini_flash(` and `_gemini_flash_json(` call sites as your
  first step and treat whatever the grep returns as ground truth. If it
  returns functions not in this expected list, or omits one that is,
  proceed with what's actually there and note the discrepancy — do not
  force the file to match this document's guess.
- Once every live Gemini call site is migrated, delete
  `_gemini_flash()`, `_gemini_flash_json()`, and `_gemini_key()` from
  `blog_agents.py` — but only after confirming via grep that no remaining
  (Groq/Cerebras) function still calls them, same discipline WO#14 uses
  for `_groq_key()` and WO#15 uses for `_cerebras_key()`.
- **The `life_os_weekly_synthesis.py` fix is not optional and must land in
  the same change as the `blog_agents.py` edit**, not a follow-up. Per
  WO#11's postmortem: this DAG imports `_gemini_flash` directly, as a
  private function, inside a task function body — invisible at DAG-parse
  time. Deleting `_gemini_flash` from `blog_agents.py` without also fixing
  this import breaks the DAG's synthesis fallback path silently, only at
  runtime, only when that task actually executes.
- Leave every other function in `blog_agents.py`
  untouched — including its Groq/Cerebras logic, its module docstring
  (routing table), and its section-header comments. This is a partial-file
  edit, not a full migration of the file.

**Task 2 — `media_agents.py` / TMDB duplication:**
- **This is not a `services/ai/` consolidation and must not become one.**
  `media_agents.py` calls TMDB, not an LLM provider — it has nothing to
  do with GOVERNANCE §2.3. The reason two implementations exist at all is
  GOVERNANCE §2.2's *absolute* rule that DAGs never import
  `services/`/`routers/`/`models.py`. That rule is not up for
  reconsideration in this work order.
- **Do not make `media_agents.py` import `services/tmdb_service.py`, and
  do not make `services/tmdb_service.py` import anything from
  `airflow/agents/`.** Either direction violates GOVERNANCE §2.2. If you
  find yourself wanting to do this to "reduce duplication," stop — that
  is exactly the workaround GOVERNANCE §2.2 and the WO#9 postmortem's own
  §3.9 explicitly rule out. The correct fix is verify-and-document, or
  (only if real drift is found) fix `media_agents.py` in isolation.
- If the diff in Task 2, Step 8 finds behavioral drift between the two
  implementations, **do not silently reconcile it.** Treat it as a
  pre-existing-bug finding per the standing rule below — report it, mark
  the criterion ⚠️, and let it be ticketed separately.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline to confirm
   it is not a regression you introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue.

(A drift finding in Task 2 is a specific, pre-identified instance where
this rule applies — handle it exactly per this instruction, not as
something to quietly patch.)

## WORKING METHOD
Complete Task 1 fully (including verification) before starting Task 2 —
they are independent in substance but structurally different in kind
(one is a `services/ai/` migration, one explicitly is not), and keeping
them sequential keeps the diff and the report easy to review as two
distinct pieces, mirroring the precedent already set by WO#19.

Within Task 1, do Step 5 (blog_agents.py) and Step 6
(`life_os_weekly_synthesis.py`) as one atomic unit — verify both together,
not Step 5 alone followed by Step 6 as an afterthought.

If an acceptance criterion needs a live API call that isn't available
here, verify request construction via a mocked HTTP layer instead, state
the substitution explicitly, and mark ⚠️.

## OUTPUT FORMAT
1. **Files created**
2. **Files moved**
3. **Files edited** (path — description; flag/explain anything beyond the
   literal instructions)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Task 2 finding** (required, separate from the acceptance criteria
   table): state explicitly whether `media_agents.py` and
   `services/tmdb_service.py` agree behaviorally, what was compared, and
   which outcome from Task 2 Steps 8–9 applies
6. **Notes**

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.
Task 1 and Task 2 are independently revertable — a Task 2 problem does not
require reverting Task 1's work, and vice versa.

---

## SCOPE

### Task 1 — AI Service Layer batch + blog_agents.py Gemini extraction

**Files to edit:**
- `services/ai/providers/gemini.py` (extend — generalize `call_gemini_json`)
- `airflow/agents/weekly_agents.py`
- `domains/workout/routers/workout_plan_ai_generator.py` — **corrected
  path.** If, at execution time, this file does not exist and
  `routers/workout_plans.py` (or `domains/workout/routers/workout_plans.py`)
  is found instead, WO#8's follow-up split has not actually landed in
  whatever environment this WO is running against — stop and report the
  discrepancy rather than guessing which file to edit.
- `domains/media/routers/media_recommend.py` — **corrected, no longer
  conditional.** WO#9 has run; this is the real path.
- `airflow/agents/blog_agents.py` (Gemini-shaped functions only — see
  HARD BOUNDARIES)
- `airflow/dags/life_os_weekly_synthesis.py` (one import line + one call
  site — see Step 6)

**Not in scope, referenced only to confirm no breakage:**
- `airflow/agents/job_agents.py` (WO#11's migrated caller)
- `airflow/agents/recipe_agents.py` (WO#12's migrated callers)
- `airflow/dags/life_os_blog_creator.py`, `life_os_blog_finalizer.py`,
  `life_os_blog_scout.py`, `life_os_idea_expander.py`,
  `life_os_readme_writer.py`, `life_os_code_narrate.py`,
  `life_os_code_comment.py`, `life_os_code_improve.py` — every DAG that
  calls one of blog_agents.py's public functions. None of these call
  `_gemini_flash`/`_gemini_flash_json` directly (only
  `life_os_weekly_synthesis.py` does), so none should need changes — but
  confirm this rather than assume it, the same way WO#11 confirmed it for
  `job_agents.py`'s own DAG consumers.

### Task 2 — TMDB duplication reconciliation

**Files to read (comparison source, do not edit unless Step 9 applies):**
- `services/tmdb_service.py`

**File to edit (comparison target):**
- `airflow/agents/media_agents.py`

**Not in scope:** everything else in `domains/media/`, all of Task 1's
files.

---

## STEPS

### Task 1

1. **Generalize `call_gemini_json()` in `services/ai/providers/gemini.py`.**
   Same generalization the original WO#13 specified:
   ```python
   def call_gemini_json(
       prompt: str,
       schema: dict | None = None,
       system: str | None = None,
       model: str = MODEL_FLASH,
       retries: int = 3,
   ) -> str:
   ```
   Omit `systemInstruction` from the payload when `system` is `None`;
   omit `responseSchema` (but keep `responseMimeType: "application/json"`)
   when `schema` is `None`. **Update the WO#11 and WO#12 call sites to use
   explicit keyword arguments** if they currently rely on positional order
   the new signature would break — check both files' current call sites
   carefully, this is the step most likely to silently break prior work if
   rushed. (`call_gemma_json`, added in WO#12, has its own independent
   signature and is unaffected by this change — do not touch it.)

2. **Update `weekly_agents.py`.** Both `agent_plan_meals()` and
   `agent_schedule_workouts()` use `_gemini_flash_json(system, prompt,
   schema)` — replace with `from services.ai import call_gemini_json` and
   `call_gemini_json(prompt, schema=schema, system=system)`. Delete the
   now-unused private `_gemini_flash_json()`.

3. **Update `workout_plan_ai_generator.py`'s plan generator.** Replace
   `_call_gemini_for_plan()`'s body with a call to
   `services.ai.call_gemini_json(prompt, schema=None, system=system,
   model=MODEL_FLASH)`. Either keep `_call_gemini_for_plan` as a thin
   wrapper or have `generate_plan()` call the service directly and delete
   the wrapper — your choice, state which and why. Remove the now-unused
   direct `os.environ.get("GEMINI_API")` read and any now-unused
   `import requests`.

4. **Update `media_recommend.py`'s `_gemini_explain()`.** Replace with
   `services.ai.call_gemini_json(prompt, schema=None, system=None)`
   (explicit `system=None` — the absence of a system instruction is a
   meaningful, intentional detail for this one caller). Remove the
   now-unused direct env-var read and `import requests` if nothing else in
   the file needs them. Leave prompt construction and response parsing
   (`re.sub` + `json.loads`) unchanged.

5. **Confirm and migrate `blog_agents.py`'s Gemini call sites.** Grep for
   `_gemini_flash(` and `_gemini_flash_json(` across the file first —
   treat the result as ground truth over this document's expected list
   (`agent_researcher`, `agent_editor`, `agent_idea_expander`,
   `agent_readme_writer`). For each confirmed call site, replace with
   `call_gemini_text(...)` (no schema) or `call_gemini_json(prompt,
   schema=..., system=...)` (schema present), matching the same argument
   content, not necessarily the same argument order (import
   `call_gemini_text`/`call_gemini_json` from `services.ai` at the top of
   the file). Delete `_gemini_flash()`, `_gemini_flash_json()`, and
   `_gemini_key()` once grep confirms zero remaining references anywhere
   else in the file (i.e., no Groq/Cerebras function secretly depends on
   them — expected not to, but confirm).

6. **Fix `life_os_weekly_synthesis.py`'s private import, in the same
   change as Step 5.** Change:
   ```python
   from agents.blog_agents import _gemini_flash
   ...
   synthesis_text = _gemini_flash(system_prompt, prompt)
   ```
   to:
   ```python
   from services.ai import call_gemini_text
   ...
   synthesis_text = call_gemini_text(system_prompt, prompt)
   ```
   Confirm the argument order matches — `_gemini_flash(system, prompt)`
   and `call_gemini_text(system, prompt, model=..., retries=...)` are
   positionally compatible per WO#11's own postmortem finding, but verify
   this against the real signatures rather than trusting the finding
   blindly.

### Task 2

7. **Read both implementations.** `services/tmdb_service.py`'s
   watch-providers function (used by `media_search.py`'s
   `add_from_search()`) and `airflow/agents/media_agents.py`'s
   `get_tmdb_watch_providers(tmdb_id, media_type)` (used by the
   `life_os_refresh_streaming_availability` DAG).

8. **Diff them for behavioral drift**, specifically:
   - Does the service also restrict to **US-region, `flatrate`
     (subscription) providers only** — not `rent`/`buy`, not other
     regions — matching `MediaItem`'s own docstring
     ("available for streaming in the US") and `media_agents.py`'s
     documented behavior?
   - Same response-shape handling for "confirmed nothing streams this"
     (a real, meaningful `None`/empty result) vs. "the request itself
     failed" (should raise, not return empty)?
   - Same provider-ID extraction and normalization?

   **If they agree:** proceed to Step 9. **If they disagree:** stop, do
   not reconcile — go straight to the pre-existing-bug handling rule
   above and report the exact drift under Notes and in the required Task
   2 finding section. Mark the relevant acceptance criterion ⚠️.

9. **If they agree, document the split as intentional and permanent** —
   do not merge them. Add a short comment to both files, cross-referencing
   each other, mirroring the pattern WO#16 uses for `finance_upload.py`'s
   SDK exception:
   ```python
   # NOTE: This duplicates services/tmdb_service.py's watch-providers
   # logic rather than importing it. This is intentional — DAGs never
   # import from services/ (GOVERNANCE.md §2.2, absolute rule). Confirmed
   # behaviorally equivalent (US flatrate-only extraction) as of WO#13.
   # If either implementation changes, re-diff the other.
   ```

---

## ACCEPTANCE CRITERIA

**Task 1 — backward compatibility (required, check first):**
- [ ] `job_agents.py`'s `score_job_batch()` call to `call_gemini_json`
  still produces an identical request payload to its WO#11 post-migration
  behavior
- [ ] `recipe_agents.py`'s three `call_gemini_json` call sites (WO#12)
  still produce identical request payloads to their WO#12 post-migration
  behavior

**Task 1 — new migrations:**
- [ ] `weekly_agents.py::agent_plan_meals()` and
  `agent_schedule_workouts()` produce identical requests (`systemInstruction`
  and `responseSchema` both present) to their pre-migration behavior
- [ ] `workout_plan_ai_generator.py`'s plan generator produces a request
  with `systemInstruction` present, no `responseSchema`, only
  `responseMimeType: "application/json"`
- [ ] `media_recommend.py`'s `_gemini_explain()` produces a request with
  no `systemInstruction`, no `responseSchema`, only `responseMimeType:
  "application/json"`
- [ ] Every confirmed `blog_agents.py` Gemini call site (per Step 5's
  grep) produces an identical request to its pre-migration behavior —
  list which functions were actually found and migrated, since this
  document's expected list is an inference, not a confirmed fact
- [ ] `blog_agents.py` no longer defines `_gemini_flash`,
  `_gemini_flash_json`, or `_gemini_key` — confirm via grep, and confirm
  its Groq/Cerebras logic is **completely untouched** (`git diff` should
  show changes isolated to the confirmed Gemini call sites and the three
  deleted helpers, nothing else)
- [ ] `life_os_weekly_synthesis.py` no longer imports `_gemini_flash` from
  `blog_agents.py`; its synthesis generation still builds the same
  system/prompt content and reaches `call_gemini_text` correctly
- [ ] `GET /workout/plans` → `POST /workout/plans/generate` (or wherever
  the split router now serves this) and `POST /media/recommend/generate`
  still reach their respective generators correctly end-to-end (mocked);
  mark ⚠️ if live API verification isn't available
- [ ] `grep -r "os.environ.get(.GEMINI_API.)"` in
  `workout_plan_ai_generator.py` and `media_recommend.py` returns zero
  hits

**Task 2:**
- [ ] Behavioral comparison in Step 8 completed and its outcome stated
  explicitly in the required Task 2 finding section (agree/disagree, what
  was compared)
- [ ] If agreement: both files carry the cross-referencing documentation
  comment from Step 9
- [ ] If disagreement: reported under Notes with full detail, **not**
  fixed in this WO, and the relevant criterion marked ⚠️
- [ ] Confirm neither file was made to import from the other, or from
  `services/`/`airflow/agents/` across the DAG boundary, in either
  direction

---

## For the next work order (not part of this one)

**WO#14 (Groq + Ghostwriter) now runs against a smaller `blog_agents.py`**
than it was originally drafted against — this WO's Step 5 will have
already removed the file's Gemini-shaped functions and helpers. WO#14's
own "confirm every other function in blog_agents.py is unchanged" grep
scope should be read as covering whatever remains after this WO, not the
full original file. No functional change needed to WO#14's own steps —
`_groq_llama`/`agent_ghostwriter` are independent of the Gemini pieces —
just a scope-awareness note for whoever executes it. See the separate
`work_order_14-20_amendments.md` for the precise text change.

After WO#13 (this), WO#14, and WO#15 all run, `blog_agents.py` will have
had every one of its provider-specific functions migrated into
`services/ai/` — the file itself will still exist (its module docstring
routing table is deliberately duplicated into `services/ai/README.md` by
WO#16 rather than deleted, per that WO's own Step 5 rationale), but should
contain little beyond the public `agent_*` function shells and their
non-provider logic. Worth a final "does this file need a shell-level
cleanup pass" check once WO#15 lands — not scoped here, since WO#15 hasn't
run yet and its own diff should be seen before deciding.
