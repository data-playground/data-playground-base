# Work Order #29 — Workout Domain Router Split (3 files: `workout_settings.py`, `workout_plan_ai_generator.py`, `workout_log.py`)

---

## ROLE
You are a senior refactoring engineer splitting oversized router files by
responsibility. Location-and-organization refactor only — no behavior
change, no schema change. Part C of this work order is a **request for
sign-off, not an instruction to split** — read it before starting any of the
three parts.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes anywhere in Parts A or B.
- **Part C is different from Parts A/B: do not split `workout_log.py`
  unilaterally.** WO#8's own HARD BOUNDARIES explicitly said its
  two-router-in-one-file structure (`router` + `body_metrics_router`) must
  not be merged or further split during that migration. The only clean
  split available now would reverse that precedent. Confirm the line count,
  report the situation, and **stop and ask for explicit sign-off** before
  touching this file at all — see Part C below.
- Do not touch `_call_gemini_for_plan`'s routing through
  `services.ai.call_gemini_json` in `workout_plan_ai_generator.py` — that's
  WO#13's migration, unrelated to this split.
- Only touch the files named in each Part's SCOPE.

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.
(`GET /workout/exercises`'s documented pre-existing bug, if still present,
is a specific instance of this — see Part C's acceptance criteria.)

## WORKING METHOD
Run `scripts/check_router_line_limits.py` for all three files first;
confirm real current line counts before proceeding with any part.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results, grouped by Part
5. Notes

## ROLLBACK
`git checkout` on every file touched.

---

## Part A — `workout_settings.py`

### SCOPE
- `domains/workout/routers/workout_settings.py` (split)
- New: `domains/workout/routers/workout_locations.py`

### STEPS
1. Run the lint script; confirm the real current line count.
2. Move the Locations & Equipment CRUD block (`_fetch_locations`,
   `_location_list_ctx`, `create_location`, `update_location`,
   `delete_location`, `set_default_location`, `add_equipment`,
   `update_equipment`, `delete_equipment`) into `workout_locations.py`.
3. Keep the settings page itself (`workout_settings`) and Custom Exercises
   CRUD (`create_custom_exercise`, `search_exercises`) in
   `workout_settings.py`.
4. Do not fix, only report: `search_exercises()`'s docstring documents two
   already-fixed bugs (undefined `rows` NameError, enum
   JSON-serialization) — leave that docstring as historical record, don't
   trim it for length.

---

## Part B — `workout_plan_ai_generator.py`

### SCOPE
- `domains/workout/routers/workout_plan_ai_generator.py` (split)
- New: `domains/workout/routers/workout_plan_prompts.py`

### STEPS
1. Run the lint script; confirm the real current line count.
2. Move `_build_exercise_history_context` and the large inline
   system-prompt/prompt-template string construction currently embedded in
   `generate_plan()` into `workout_plan_prompts.py` as named
   constants/functions (e.g. `PLAN_SYSTEM_INSTRUCTION`,
   `build_plan_prompt(...)`).
3. Keep `generate_plan`, `save_generated_plan`, `_fuzzy_match_exercise`, and
   the thin `_call_gemini_for_plan` wrapper in
   `workout_plan_ai_generator.py`.

---

## Part C — `workout_log.py` (sign-off request, not a split instruction)

### SCOPE
Read-only investigation this time.

### STEPS
1. This file exports **two** separate `APIRouter` instances from one module
   (`router` and `body_metrics_router`), and WO#8's own HARD BOUNDARIES
   explicitly said not to merge or further split that structure during
   that migration. The only clean split available now — separating
   `body_metrics_router` into its own file — would reverse that precedent.
2. Confirm the real current line count via the lint script.
3. If still over 300: report the situation (two-router-in-one-file
   structure; the only real split available is separating the
   body-metrics half into its own file) and **stop, requesting explicit
   owner sign-off** before proceeding — do not split this file as part of
   executing this work order.
4. **Only if sign-off is given in a follow-up:** the split would be
   `workout_log.py` (keeps `router` / `/workout/sessions/*` —
   `start_session`, `log_set`, `end_session`, `session_detail`,
   `delete_set`) + `workout_body_metrics.py` (the `body_metrics_router` —
   `log_body_metric`, `get_body_metrics`), with both still registered as
   separate `app.include_router()` calls in `main.py`, exactly as today.

---

## ACCEPTANCE CRITERIA (all three parts)
- [ ] Parts A and B: all resulting files under 300 lines.
- [ ] Part C: explicitly held pending sign-off in this run, OR (only if
  sign-off was obtained out-of-band before this ran) both resulting files
  under 300 lines with both routers still registered separately in
  `main.py`.
- [ ] Every workout endpoint (session start/log-set/end/detail/delete-set,
  body-metrics log/get, locations/equipment CRUD, custom-exercise
  create/search, plan generate/save) reachable at its original path,
  identical behavior.
- [ ] `GET /workout/exercises` — confirm whether the documented
  pre-existing bug is still present; if it's already been fixed elsewhere,
  update the criterion to reflect that rather than assuming the old bug
  report is still accurate.

## For the next work order (not part of this one)
Parts A and B can run fully in parallel with every other Track C work
order, Track A, Track B, and Track D. Part C is blocked on owner sign-off
and should be tracked separately once that's given.
