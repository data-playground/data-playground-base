# Work Order #8 — Domain Migration: `workout`

*Larger model set than prior work orders (16 ORM classes, no separate
Pydantic response schemas), but no separate CSS/route complexity beyond
that. One router file (`workout_log.py`) exports two distinct `APIRouter`
instances that are both included separately in `main.py` — that detail must
be preserved exactly. `dashboard.py` does NOT read from this domain (unlike
every prior work order), so this migration has no dashboard-consumer
verification step — its only external consumer is the not-yet-migrated
`planning` domain.*

---

## ROLE
You are a senior refactoring engineer performing a structural code migration.
Your job is NOT to improve, optimize, or modernize the code you move — only
to relocate it correctly and verify it still behaves identically. Resist the
urge to "clean up while you're in there." Flag improvement opportunities as
a NOTES section at the end instead of acting on them.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below. If you believe you
  need to touch a file outside this list to complete the task, STOP and
  report why — do not proceed and do not guess.
- No schema changes. No renamed tables, columns, or endpoints. No behavior
  changes. This is a location refactor only.
- If a step's instructions conflict with what you find in the actual code
  (e.g. a file/class isn't where the work order says it is), stop and report
  the discrepancy rather than improvising a fix.
- You may create empty `__init__.py` package markers as needed to support
  new import paths — this is expected scaffolding, not a scope expansion,
  and does not need to be flagged as a deviation (a one-line mention in the
  report's "Files created" section is enough).
- **`routers/workout_log.py` defines and exports TWO separate `APIRouter`
  instances from one file**: `router` (prefix `/workout/sessions`) and
  `body_metrics_router` (prefix `/workout/body-metrics`). `main.py`
  currently includes both separately:
  ```python
  app.include_router(workout_log.router)
  app.include_router(workout_log.body_metrics_router)
  ```
  **Preserve this exact two-router-from-one-file structure and both
  separate `include_router` calls** — do not merge them into one router or
  split them into two files. This is existing, intentional design, not
  something to "clean up."
- The AI plan generator inside `routers/workout_plans.py`
  (`_call_gemini_for_plan`) is one of the six known duplicate AI-client
  implementations tracked in GOVERNANCE.md §2.3. **Do not touch it, do not
  route it through any service layer** — it's explicitly out of scope for a
  location-only refactor, same treatment as `finance_upload.py`'s Gemini
  call in WO#5 and `blog_agents.py` in WO#2.

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

**Known pre-existing bug already on record for this domain** (found
independently, not by a prior work order, but worth carrying into your
verification pass): `routers/workout_settings.py::search_exercises()`
returns a list comprehension over a variable named `rows`, which is never
defined in that function (the query result is stored in `result`, and
`result.all()` or similar is never called into a `rows` variable). This
means `GET /workout/exercises` (the exercise autocomplete endpoint used by
the session log panel) is currently broken with a `NameError`. **Do not fix
this** — reproduce it against the pre-migration baseline to confirm, mark
the relevant acceptance criterion ⚠️ referencing this known issue, and
still report it under Notes with the exact detail above so it can be
ticketed (or cross-reference this work order in that ticket, since it's
already partially diagnosed here).

## WORKING METHOD
Execute steps in the order listed. After each step that changes running
behavior (not pure file moves), pause and self-verify against the relevant
acceptance criteria before continuing to the next step. Do not defer all
verification to the end.

If an acceptance criterion requires a resource not listed in SCOPE (e.g. a
config file, external service, or database not provided), do not skip the
criterion silently. Perform the closest verification achievable with what
you have, state explicitly what the substitute check was and why, and mark
the result ⚠️ rather than ✅ or leaving it blank. (Note: `workout_plans.py`'s
Gemini-based plan generator cannot be verified against the live API in this
environment — verify the route is reachable and correctly builds its prompt
context before the API call, mark ⚠️ with this explanation.)

## OUTPUT FORMAT
End with a report in exactly this structure:
1. **Files created** (list — include any scaffolding `__init__.py` files here)
2. **Files moved** (old path → new path)
3. **Files edited** (path — one-line description of the change; if a change
   goes beyond what a step literally asked for but was necessary to make
   that step work, say so explicitly and explain why)
4. **Acceptance criteria results** (checklist, ✅/❌/⚠️ per item, with a
   one-line reason for any ❌ or ⚠️ — including substitute-verification
   explanations per the rule above)
5. **Notes / things I noticed but did not act on** (optional — improvement
   ideas, pre-existing bugs found per the rule above, inconsistencies, risks
   spotted while working, explicitly out of scope for this task)

## ROLLBACK
This work order operates on files already tracked in git. If acceptance
criteria fail and cannot be quickly fixed, the safe rollback is `git
checkout` on every file listed in "Files created / moved / edited" above —
do not attempt a partial manual revert.

---

## SCOPE

**Models to extract from `models.py`:**
`LocationType`, `EquipmentType`, `MuscleGroup`, `ExerciseEquipmentType`,
`PlanOrigin`, `WorkoutGoal`, `WeightUnit`, `WorkoutLocation`, `Equipment`,
`Exercise`, `WorkoutPlan`, `WorkoutPlanDay`, `WorkoutPlanExercise`,
`WorkoutSession`, `WorkoutSet`, `BodyMetric`

*(No separate Pydantic response schemas exist for this domain in the
current codebase — do not invent any; move only what's listed.)*

**Routers:**
- `routers/workout.py`
- `routers/workout_log.py` (exports **two** routers — see HARD BOUNDARIES)
- `routers/workout_plans.py`
- `routers/workout_settings.py`

**Templates:**
- `templates/workout.html`
- `templates/workout_history.html`
- `templates/workout_progress.html`
- `templates/workout_plans.html`
- `templates/workout_plan_preview.html`
- `templates/workout_settings.html`
- `templates/partials/workout/active_session_header.html`
- `templates/partials/workout/body_metric_saved.html`
- `templates/partials/workout/custom_exercise_list.html`
- `templates/partials/workout/equipment_list.html`
- `templates/partials/workout/exercise_history.html`
- `templates/partials/workout/location_list.html`
- `templates/partials/workout/plan_generate_error.html`
- `templates/partials/workout/plan_list.html`
- `templates/partials/workout/plan_saved.html`
- `templates/partials/workout/session_detail.html`
- `templates/partials/workout/session_summary.html`
- `templates/partials/workout/set_logged_row.html`

**Static:**
- `static/css/workout.css`
- `static/js/workout.js`

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:**
- `routers/weekly_plan.py` (planning domain, not yet migrated — imports
  `WorkoutPlan`, `WorkoutPlanDay`, `WorkoutSession`, `WeightUnit` from
  `models` at module level; `WeeklyPlanDay.workout_session_id` also has a
  real `relationship("WorkoutSession", ...)`, not just a bare FK column —
  a genuine cross-domain relationship, same category as the `Recipe` one
  in WO#7. All of this must keep resolving via the shim.)

*(Note: unlike every prior domain migration, `routers/dashboard.py` does
NOT import anything from this domain — confirmed by reviewing its current
import list. No dashboard verification step is needed for this work
order.)*

---

## STEPS

1. **Create `domains/workout/models.py`.** Move the seven enums
   (`LocationType`, `EquipmentType`, `MuscleGroup`, `ExerciseEquipmentType`,
   `PlanOrigin`, `WorkoutGoal`, `WeightUnit`) first, then the nine ORM
   classes (`WorkoutLocation`, `Equipment`, `Exercise`, `WorkoutPlan`,
   `WorkoutPlanDay`, `WorkoutPlanExercise`, `WorkoutSession`, `WorkoutSet`,
   `BodyMetric`), preserving current relative order. Import `Base` from
   `core.base_model`. All relationships between these 9 classes
   (`WorkoutLocation.equipment`, `WorkoutPlan.days`,
   `WorkoutPlanDay.exercises`, `WorkoutSession.sets`, etc.) are same-module
   string references — no special handling needed.

2. **In `models.py`:** delete the sixteen moved definitions and replace
   with a re-export shim: `from domains.workout.models import
   LocationType, EquipmentType, MuscleGroup, ExerciseEquipmentType,
   PlanOrigin, WorkoutGoal, WeightUnit, WorkoutLocation, Equipment,
   Exercise, WorkoutPlan, WorkoutPlanDay, WorkoutPlanExercise,
   WorkoutSession, WorkoutSet, BodyMetric`. Tag it `# TODO: remove after
   all cross-references are updated`.

3. **Move routers:**
   - `routers/workout.py` → `domains/workout/routers/workout.py`
   - `routers/workout_log.py` → `domains/workout/routers/workout_log.py`
     (both `router` and `body_metrics_router` stay in this one file)
   - `routers/workout_plans.py` → `domains/workout/routers/workout_plans.py`
   - `routers/workout_settings.py` → `domains/workout/routers/workout_settings.py`

   Update each file's model imports to pull from `domains.workout.models`
   instead of `models`. Update each file's `templates =
   Jinja2Templates(directory="templates")` to `from core.templating import
   templates`. Leave `workout_plans.py`'s `_call_gemini_for_plan` and its
   `os.environ.get("GEMINI_API")` call completely untouched.

4. **Move templates**, preserving the `partials/workout/` subfolder
   structure (note: two levels deep — `partials/workout/`, not just
   `partials/`, unlike most other domains), into
   `domains/workout/templates/` per the SCOPE list above.

5. **Move static assets** into `domains/workout/static/css/workout.css` and
   `domains/workout/static/js/workout.js`. Update every `<link
   rel="stylesheet" href="/static/css/workout.css">` reference (present in
   all six workout templates) to `/static/workout/css/workout.css`, and the
   `<script src="/static/js/workout.js"></script>` reference (in
   `workout.html`) to `/static/workout/js/workout.js`.

6. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/workout/templates/` as an additional search root, alongside
   the roots already added in WO#1–7.

7. **In `main.py`:**
   - Update the imports/includes for all four router modules to their new
     paths (`from domains.workout.routers import workout, workout_log,
     workout_plans, workout_settings`).
   - **Both `workout_log.router` and `workout_log.body_metrics_router`
     must still be included as two separate `app.include_router()` calls**
     — confirm this explicitly in your report, since it's the one detail
     in this work order most likely to be silently collapsed into one call
     by mistake.
   - Add the new static mount: `app.mount("/static/workout",
     StaticFiles(directory="domains/workout/static"), name="workout_static")`.
     Register it **before** the general `/static` mount, per the ordering
     rule in GOVERNANCE.md §2.6.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /workout` renders identically — start-session card or active
  session banner (depending on state), today's plan section, body weight
  card, recent sessions list
- [ ] `GET /workout/history` and `GET /workout/progress` render identically
- [ ] `GET /workout/plans` and `GET /workout/plans/{id}/save`'s resulting
  preview flow render identically
- [ ] `GET /workout/settings` renders identically — locations/equipment
  panel, custom exercises panel
- [ ] `POST /workout/sessions/start`, `POST
  /workout/sessions/{id}/sets`, `PATCH /workout/sessions/{id}/end`, `GET
  /workout/sessions/{id}`, and `DELETE
  /workout/sessions/{id}/sets/{set_id}` (all from the `router` half of
  `workout_log.py`) all still work and return their respective partials
  correctly
- [ ] `POST /workout/body-metrics` and `GET /workout/body-metrics` (from
  the `body_metrics_router` half of `workout_log.py`) still work correctly
  — this specifically confirms the two-router-in-one-file split registered
  correctly in `main.py`
- [ ] `GET /workout/exercises` — **expected to fail** per the known
  pre-existing bug documented above (`NameError: name 'rows' is not
  defined`). Confirm it fails identically pre- and post-migration, mark
  this criterion ⚠️ referencing that bug, do not fix it.
- [ ] `POST /workout/locations`, equipment add/delete, and `POST
  /workout/exercises` (custom exercise creation, a *different* endpoint
  from the broken `GET /workout/exercises` above — confirm you're not
  conflating the two) all still work correctly
- [ ] `POST /workout/plans/generate` reaches
  `_call_gemini_for_plan` correctly (mocked/stubbed per WORKING METHOD);
  mark ⚠️ with explanation
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  same table count before/after, `models.WorkoutSession is
  domains.workout.models.WorkoutSession`, `models.Exercise is
  domains.workout.models.Exercise`, no `InvalidRequestError` on mapper
  configuration
- [ ] `routers/weekly_plan.py`'s existing `WorkoutPlan` / `WorkoutPlanDay`
  / `WorkoutSession` / `WeightUnit` imports (via `from models import ...`)
  still resolve, and `WeeklyPlanDay.workout_session` relationship still
  resolves correctly (test by loading a weekly plan with a linked workout
  session if feasible in this environment; otherwise mark ⚠️ and explain
  what couldn't be tested)
- [ ] `grep -r "from models import"` for each of the sixteen moved class
  names across the repo returns only the shim's own lines in `models.py`
  plus the `weekly_plan.py` references noted above — nothing else should
  need updating
- [ ] Confirm explicitly (git diff or direct statement) that
  `routers/dashboard.py` required **zero** changes for this migration,
  consistent with it having no workout-domain imports to begin with

---

## For the next work order (not part of this one)

Per GOVERNANCE.md §3.3, **Work Order #9 = `media`** is next (routers:
`media`, `media_search`, `media_recommend`, `media_settings`). Advance
notice for that one: it depends on the external `ml-service` Docker
container (`services/ml_service_client.py`, not moving) for embeddings and
similarity search — same "can't verify the live external call, verify the
route reaches it correctly" treatment as this work order's Gemini plan
generator. It also has the `MEDIA_RECOMMEND_AI` environment-variable
toggle read once at router import time in `media_recommend.py` — worth
calling out explicitly in that work order so the verification step doesn't
mistake stale toggle state for a migration bug.
