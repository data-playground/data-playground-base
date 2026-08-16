# Work Order #10 — Domain Migration: `planning`

*Last of the real domains on the backlog (GOVERNANCE.md §3.3). Saved for
last deliberately — this domain accumulates cross-domain references into
`recipes` (WO#7), `workout` (WO#8), and `journal` (WO#6), all of which are
now migrated, which means this work order can finally point those imports
at their true destinations instead of the root `models.py` shim. It also
requires a genuine file split, not just relocation: `weekly_plan.py` is
already past the 300-line rule (GOVERNANCE.md §1.2), and moving it as-is
would just relocate an oversized file rather than fix it. Read this whole
work order before starting — the split in Step 4 and the import cleanup in
Step 7 are the parts most likely to go wrong if rushed.*

---

## ROLE
You are a senior refactoring engineer performing a structural code migration.
Your job is NOT to improve, optimize, or modernize the code you move — only
to relocate it correctly and verify it still behaves identically. Resist the
urge to "clean up while you're in there." Flag improvement opportunities as
a NOTES section at the end instead of acting on them.

**Exception to the "don't improve" rule for this work order specifically:**
this domain requires two changes that go beyond pure relocation, both
explicitly authorized below and only these two:
1. Splitting `weekly_plan.py` into two files by responsibility (Step 4).
2. Repointing this domain's own cross-domain imports (into `recipes` and
   `workout`) from the root `models.py` shim to the real
   `domains.recipes.models` / `domains.workout.models` paths, since both of
   those domains are already migrated (Step 7).
Do not treat either of these as license to make *other* improvements while
you're in the file — everything else in ROLE/HARD BOUNDARIES still applies
at full strength.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below. If you believe you
  need to touch a file outside this list to complete the task, STOP and
  report why — do not proceed and do not guess.
- No schema changes. No renamed tables, columns, or endpoints. No behavior
  changes beyond the two explicitly authorized exceptions above. This is
  still fundamentally a location refactor, not a feature or logic change.
- If a step's instructions conflict with what you find in the actual code
  (e.g. a file/class isn't where the work order says it is), stop and report
  the discrepancy rather than improvising a fix.
- You may create empty `__init__.py` package markers as needed to support
  new import paths — this is expected scaffolding, not a scope expansion,
  and does not need to be flagged as a deviation (a one-line mention in the
  report's "Files created" section is enough).
- **`airflow/agents/weekly_agents.py` is explicitly OUT OF SCOPE and must
  NOT be moved.** It contains `agent_plan_meals` and
  `agent_schedule_workouts`, both Gemini-based, imported directly by the
  AI-generation half of `weekly_plan.py` (see Step 4). It has no `models`
  imports, so it needs no internal edits either — leave every import of it
  completely untouched.
- **`UserIntent.to_ai_context()` is a plain data-formatting method, not an
  AI call itself** — it builds a prompt-context string that
  `weekly_agents.py`'s functions consume as a parameter. Do not confuse it
  with the actual AI provider calls; it stays exactly where it is, moving
  with `UserIntent` in Step 1.

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
behavior (not pure file moves), pause and self-verify against the relevant
acceptance criteria before continuing to the next step. Do not defer all
verification to the end.

If an acceptance criterion requires a resource not listed in SCOPE (e.g. a
config file, external service, or database not provided), do not skip the
criterion silently. Perform the closest verification achievable with what
you have, state explicitly what the substitute check was and why, and mark
the result ⚠️ rather than ✅ or leaving it blank. (Note: `weekly_agents.py`'s
Gemini-based meal/workout scheduling calls cannot be exercised live in this
environment — verify the split router correctly reaches both agent
functions with the right arguments, using a mocked/stubbed response, mark
⚠️ with this explanation, same pattern as every prior AI-dependent work
order.)

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
`FitnessGoal`, `WeeklyPlanStatus`, `PlanDayStatus`, `PlanMealType`,
`PlanMealStatus`, `UserIntent`, `WeeklyPlan`, `WeeklyPlanDay`,
`WeeklyPlanMeal`, `ShoppingList`

*(No separate Pydantic response schemas exist for this domain in the
current codebase — do not invent any; move only what's listed.)*

**Routers:**
- `routers/intent.py`
- `routers/weekly_plan.py` — **to be split into two files, see Step 4**

**Templates:**
- `templates/intent.html`
- `templates/weekly_plan_hub.html`
- `templates/weekly_plan_new.html`
- `templates/weekly_plan_review.html`
- `templates/weekly_plan_view.html`
- `templates/shopping_list.html`
- `templates/partials/plan_day_card.html`
- `templates/partials/plan_meal_row.html`
- `templates/partials/intent_saved.html`

**Static:** none — this domain's pages use inline `extra_css` blocks only,
consistent with `recipes` (WO#7) and `code_intel` (WO#2).

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`
- `domains/journal/routers/journal.py` (edited in place — internal import
  path only, file does NOT move again; see Step 8)

**Not in scope, referenced only for the import cleanup in Step 7:**
- `domains/recipes/models.py` (already migrated in WO#7 — this domain will
  import from it directly instead of the root `models.py` shim)
- `domains/workout/models.py` (already migrated in WO#8 — same treatment)

*(Note: `routers/dashboard.py` does NOT import anything from this domain —
confirmed by reviewing its current import list, same situation as `media`
in WO#9. No dashboard verification step is needed for this work order.)*

---

## STEPS

1. **Create `domains/planning/models.py`.** Move the five enums
   (`FitnessGoal`, `WeeklyPlanStatus`, `PlanDayStatus`, `PlanMealType`,
   `PlanMealStatus`) first, then `UserIntent`, `WeeklyPlan`,
   `WeeklyPlanDay`, `WeeklyPlanMeal`, `ShoppingList`, preserving current
   relative order. Import `Base` from `core.base_model`. All relationships
   among these five classes (`WeeklyPlan.days`, `WeeklyPlanDay.meals`,
   `WeeklyPlanMeal.recipe`/`swap_recipe`, `WeeklyPlanDay.workout_session`,
   `ShoppingList.weekly_plan`) reference either same-module classes or
   classes in already-migrated domains via string names — resolution
   requires no special handling as long as `domains.recipes.models` and
   `domains.workout.models` are imported by the time mappers configure,
   which they already are via their own shims in root `models.py`.

2. **In `models.py`:** delete the ten moved definitions (5 enums + 5
   classes) and replace with a re-export shim: `from domains.planning.models
   import FitnessGoal, WeeklyPlanStatus, PlanDayStatus, PlanMealType,
   PlanMealStatus, UserIntent, WeeklyPlan, WeeklyPlanDay, WeeklyPlanMeal,
   ShoppingList`. Tag it `# TODO: remove after all cross-references are
   updated`.

3. **Move `routers/intent.py`** to `domains/planning/routers/intent.py`.
   Update its `from models import FitnessGoal, UserIntent` to `from
   domains.planning.models import FitnessGoal, UserIntent`. Update its
   `templates = Jinja2Templates(directory="templates")` to `from
   core.templating import templates`.

4. **Split `routers/weekly_plan.py` into two files, by responsibility:**

   - **`domains/planning/routers/weekly_plan.py`** — keeps: `_get_monday()`
     helper, `plan_hub()`, `plan_new_form()`, `confirm_plan()`,
     `plan_view()`, `override_day()`, `update_meal_status()`,
     `_generate_shopping_list()` helper, `shopping_list_view()`,
     `_sync_plan_status()`. This file keeps the `router = APIRouter(prefix="/plan",
     tags=["Weekly Plan"])` declaration and every route except the one
     moving below.

   - **`domains/planning/routers/weekly_plan_generator.py`** — contains
     only the `generate_plan()` handler (currently `POST /plan/generate`),
     which is the one function in this file that calls out to
     `airflow.agents.weekly_agents` (`agent_plan_meals`,
     `agent_schedule_workouts`). This file needs its own `router =
     APIRouter(prefix="/plan", tags=["Weekly Plan"])` declaration (same
     prefix as the other file — this is safe since `/plan/generate` does
     not collide with any route in the other file; confirm this explicitly
     during verification, don't just assume it).

   Both files import their needed models from `domains.planning.models`
   (post-Step-7-cleanup, see below) and both update `templates =
   Jinja2Templates(directory="templates")` to `from core.templating import
   templates`.

5. **Move templates**, preserving the `partials/` subfolder structure, into
   `domains/planning/templates/` per the SCOPE list above.

6. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/planning/templates/` as an additional search root, alongside
   the roots already added in WO#1–9.

7. **Repoint this domain's own cross-domain imports to their real
   locations** (the authorized exception from ROLE):
   - In whichever of the two split files uses them, change
     `Recipe, RecipeMealType` and `Ingredient, PantryItem` (plus the local
     `from models import RecipeIngredient` inside
     `_generate_shopping_list()`) from `from models import ...` to `from
     domains.recipes.models import ...`.
   - Change `WorkoutPlan, WorkoutPlanDay, WorkoutSession, WeightUnit` from
     `from models import ...` to `from domains.workout.models import ...`.
   - Do NOT change how `UserIntent`, `WeeklyPlan`, etc. (this domain's own
     classes) are imported within these same files — those already point
     at `domains.planning.models` per Steps 3–4, this step is specifically
     about the *other* domains' classes these files also happen to use.

8. **Update `domains/journal/routers/journal.py`'s local import.** WO#6
   deliberately left `save_entry()`'s local `from models import
   WeeklyPlanDay as _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS`
   import pointing at the root `models.py` shim, specifically because
   `planning` hadn't migrated yet at that time. Now that it has, update
   this import to `from domains.planning.models import WeeklyPlanDay as
   _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS`. This is the only
   change to this file — do not touch anything else in
   `journal.py`.

9. **In `main.py`:**
   - Update the `intent` router import/include to its new path (`from
     domains.planning.routers import intent`).
   - Update the `weekly_plan` import to include **both** split routers:
     `from domains.planning.routers import weekly_plan,
     weekly_plan_generator`, with two separate `app.include_router(...)`
     calls, one for each module's `router`.
   - This domain has no static assets, so **no new `StaticFiles` mount is
     needed** — do not add one.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /intent` renders identically — goal grid, workout/nutrition
  targets, cooking preferences, health notes
- [ ] `POST /intent` still saves correctly and returns `intent_saved.html`
- [ ] `GET /intent/context` still returns the correct JSON context string
- [ ] `GET /plan` (hub) renders identically — current week card, day strip,
  adherence bars, side panel with goals + past weeks
- [ ] `GET /plan/new` renders identically — day availability grid, intent
  preview, active-plan-or-warning state
- [ ] `POST /plan/generate` — now served by
  `weekly_plan_generator.router` — still reaches `agent_schedule_workouts`
  and `agent_plan_meals` correctly and returns the review page; mark ⚠️
  per WORKING METHOD since live Gemini calls aren't available here
- [ ] `POST /plan/confirm` still correctly creates `WeeklyPlan`,
  `WeeklyPlanDay`, `WeeklyPlanMeal` rows, creates `WorkoutSession` stubs for
  workout days (confirming the `domains.workout.models` import from Step 7
  works), and calls `_generate_shopping_list` correctly (confirming the
  `domains.recipes.models` import from Step 7 works)
- [ ] `GET /plan/{id}` renders `weekly_plan_view.html` identically
- [ ] `PATCH /plan/{id}/day/{date_str}` still returns `plan_day_card.html`
  correctly for all three override types (rest/skip/note)
- [ ] `PATCH /plan/meal/{meal_id}` still returns `plan_meal_row.html`
  correctly, including the swapped/off-plan/eaten/skipped status paths
- [ ] `GET /plan/{id}/shopping` renders `shopping_list.html` identically —
  needs-to-buy grouped by category, already-in-pantry section
- [ ] **Router split verification (required):** confirm both
  `weekly_plan.router` and `weekly_plan_generator.router` share the
  `/plan` prefix without any path collision — list every route each file
  registers and confirm no duplicates
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  same table count before/after, `models.WeeklyPlan is
  domains.planning.models.WeeklyPlan`, `models.UserIntent is
  domains.planning.models.UserIntent`, no `InvalidRequestError` on mapper
  configuration
- [ ] `domains/journal/routers/journal.py`'s updated local import (Step 8)
  still resolves correctly — trigger a journal save while a confirmed
  weekly plan day exists for today, if feasible in this environment, to
  confirm plan-day linking still works end-to-end across both migrated
  domains; otherwise mark ⚠️ and explain what couldn't be tested
- [ ] `grep -r "from models import"` for each of the ten moved planning
  class names, PLUS a fresh check that `Recipe`/`Ingredient`/`PantryItem`/
  `RecipeIngredient`/`WorkoutPlan`/`WorkoutPlanDay`/`WorkoutSession`/
  `WeightUnit` are no longer imported from root `models` anywhere in this
  domain's two router files (confirming Step 7's repointing worked) —
  across the whole repo, results should show only the shim's own lines in
  `models.py` for the ten planning classes, and zero remaining root-`models`
  references to the eight recipes/workout classes from within
  `domains/planning/`
- [ ] Confirm explicitly (git diff or direct statement) that
  `routers/dashboard.py` required **zero** changes for this migration

---

## Wrap-Up Note (applies after this work order, not part of it)

With `habits`, `blog`+`code_intel`, `jobs`, `explorer`, `finance`,
`journal`, `recipes`+`pantry`, `workout`, `media`, and `planning` all
migrated, every domain on the backlog (GOVERNANCE.md §3.3) is complete.
`dashboard` remains the only intentional top-level exception (GOVERNANCE.md
§2.2). Two follow-up items to schedule as their own separate, small work
orders once this one is reviewed and accepted — neither should be folded
into this one:

1. **Shim removal pass (GOVERNANCE.md §2.4).** With every domain now
   migrated, go through each domain's shim in root `models.py` and check
   whether `dashboard.py` is still the *only* consumer. Where true, update
   `dashboard.py`'s import for that domain to point directly at
   `domains.<name>.models` and delete that domain's shim entirely. This is
   a mechanical, low-risk cleanup best done as one pass across all domains
   at once now that the whole backlog is settled, rather than piecemeal.
2. **Root `models.py` should now be nearly empty** except shims — worth a
   final check that nothing was accidentally left behind (a stray class,
   an orphaned import) before treating the migration project itself as
   closed out.

The DAG relocation question (GOVERNANCE.md §2.5) and the AI service layer
consolidation (GOVERNANCE.md §2.3) remain separate, larger efforts not
addressed by any of these ten work orders — both should be scoped as their
own dedicated project phases when you're ready to take them on.
