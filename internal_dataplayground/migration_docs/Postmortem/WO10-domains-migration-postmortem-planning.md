# Work Order #10 — Planning Domain Migration: Post-Mortem & Final-Cleanup Requirements

**Domain:** `planning` (`weekly_plan` + `intent`)
**Status:** Original scope complete, with one gap (§1.4). Five follow-up changes
explicitly agreed to and delivered after the original migration was reported —
see Part 2. **This is the last domain in GOVERNANCE.md §3.3's backlog** — as of
this document, every domain-folder migration work order (WO#1–WO#10) has been
executed. Part 4 of this document is the final-cleanup work order for the whole
program, written to be actionable *now*, not deferred.

**Companion documents:** `migration_docs/GOVERNANCE.md` (all §-references below
point there unless stated otherwise), and every prior domain postmortem
(WO#1 habits, WO#2 blog/code_intel, WO#3 jobs, WO#4 explorer, WO#5 finance,
WO#6 journal, WO#7 recipes/pantry, WO#8 workout, WO#9 media) — this document
carries forward and consolidates the final-cleanup sections those all deferred.

---

## SECTION 0 — How to read this document

Per GOVERNANCE.md §4.5 ("Bugs Found During Migration Are Not Migration Work")
and §4.6 ("What 'Done' Means for a Domain Migration"), and per explicit
instruction for this document specifically: **Part 1 and Part 2 are kept
strictly separate.**

- **Part 1** covers only what the original WO#10 — as written and scoped —
  actually delivered. Judge "did the migration succeed" against Part 1 alone.
- **Part 2** covers five items of follow-up work, each individually proposed
  or requested and explicitly authorized in conversation **after** WO#10's
  original delivery was reported. None of it was bundled into the original
  diff. One item (the `weekly_agents.py` bugfix) is a genuine, authorized
  exception to the original HARD BOUNDARIES — flagged explicitly with its
  authorization quote, not silently absorbed.
- **Part 3** is a short "what does done mean" checklist for this domain
  specifically, per GOVERNANCE.md §4.6.
- **Part 4** is the final-cleanup work order for the **entire** domain-folder
  migration program (WO#1–WO#10), since this domain closes out the backlog.
  Unlike every prior postmortem's own final-cleanup section (which had to say
  "wait until every domain is done"), this one is written to be executable
  immediately — with the caveat in §4.1 about confirming that premise first.

---

# PART 1 — POST-MORTEM: THE ORIGINAL WORK ORDER (AS SCOPED AND DELIVERED)

## 1.1 Summary

WO#10 moved the `planning` domain (10 model definitions — 5 enums + 5 ORM
classes — 2 routers, 9 templates) out of the flat `models.py` / `routers/` /
`templates/` layout and into `domains/planning/`, following the pattern
established by WO#1–#9. Unlike most prior domains, this work order required
one genuine structural change beyond pure relocation, explicitly authorized
in the work order text itself: splitting `weekly_plan.py` by responsibility,
isolating the one handler (`generate_plan`) that calls out to
`airflow.agents.weekly_agents` into its own module,
`weekly_plan_generator.py`. It also required completing an import-repointing
step no prior domain needed in the same way: because `recipes` (WO#7) and
`workout` (WO#8) were already migrated by the time planning ran, this
domain's own cross-domain imports could be pointed directly at
`domains.recipes.models` / `domains.workout.models` instead of the root
shim — and `domains/journal/routers/journal.py`'s local import (deliberately
left pointing at the root shim by WO#6, since planning hadn't migrated yet)
could finally be corrected too.

**One real gap at original delivery:** `templates/shopping_list.html` was
listed in this work order's own SCOPE, but its content was never included
among the materials provided for this engagement. It could not be moved,
and the corresponding acceptance criterion (`GET /plan/{id}/shopping`) was
marked ❌, not ⚠️, since a required deliverable was simply missing outright —
see §1.4 and §1.6.

## 1.2 Scope & objective (as originally written)

**In scope, and delivered:**
- **Models → `domains/planning/models.py`:** `FitnessGoal`,
  `WeeklyPlanStatus`, `PlanDayStatus`, `PlanMealType`, `PlanMealStatus` (5
  enums); `UserIntent`, `WeeklyPlan`, `WeeklyPlanDay`, `WeeklyPlanMeal`,
  `ShoppingList` (5 ORM classes). No separate Pydantic response schemas
  exist for this domain — none were invented.
- **Routers:** `routers/intent.py` → `domains/planning/routers/intent.py`.
  `routers/weekly_plan.py` → split into `domains/planning/routers/weekly_plan.py`
  (everything except AI generation) and `domains/planning/routers/weekly_plan_generator.py`
  (the `POST /plan/generate` handler only).
- **Templates:** `intent.html`, `weekly_plan_hub.html`, `weekly_plan_new.html`,
  `weekly_plan_review.html`, `weekly_plan_view.html`, `shopping_list.html`,
  `partials/plan_day_card.html`, `partials/plan_meal_row.html`,
  `partials/intent_saved.html` → `domains/planning/templates/`. **8 of these
  9 were actually moved** — `shopping_list.html` could not be (§1.4).
- **Static:** none — this domain has no CSS/JS assets of its own (confirmed;
  all styling is inline `extra_css` blocks, same as `recipes` and `code_intel`).
- **Config edits:** `models.py` (shim swap), `main.py` (import repoint + two
  `include_router` calls for the split), `core/templating.py` (new
  `ChoiceLoader` entry), `domains/journal/routers/journal.py` (one local
  import repointed — completing what WO#6 deliberately deferred).

**Explicitly out of scope, and respected:**
- `airflow/agents/weekly_agents.py` — HARD BOUNDARIES stated this "must NOT
  be moved." It was not moved, and (in the original pass) not edited either.
  See Part 2, §2.6 for why this changed later, under separate authorization.
- No schema changes, no renamed endpoints, no behavior changes beyond the
  one explicitly authorized router split.

## 1.3 What shipped (original pass)

**Files created:**
- `domains/planning/__init__.py`, `domains/planning/routers/__init__.py`
- `domains/planning/models.py`
- `domains/planning/routers/weekly_plan_generator.py`

**Files moved:**
- `routers/intent.py` → `domains/planning/routers/intent.py`
- `routers/weekly_plan.py` → `domains/planning/routers/weekly_plan.py`
  (with `generate_plan()` extracted into the new generator file above)
- 8 of 9 templates (all except `shopping_list.html`) → `domains/planning/templates/`

**Files edited (shared files, targeted diffs):**
- `models.py` — the 10-name Weekly Planning block replaced with a re-export
  shim from `domains.planning.models`, tagged
  `# TODO: remove after all cross-references are updated` — identical
  structure to every other domain's shim.
- `main.py` — `from routers import intent, weekly_plan` replaced with
  `from domains.planning.routers import intent, weekly_plan,
  weekly_plan_generator # WO10`; a third `include_router` call added for
  the split generator router. No new static mount (domain has none).
- `core/templating.py` — `FileSystemLoader("domains/planning/templates")`
  appended to the `ChoiceLoader`.
- `domains/journal/routers/journal.py` — `save_entry()`'s local import
  changed from `from models import WeeklyPlanDay as _WPD, WeeklyPlan as
  _WP, WeeklyPlanStatus as _WPS` to `from domains.planning.models import
  ...`. **This is the only line changed in this file** — completes the
  fix WO#6 explicitly deferred (see that postmortem's §7, "don't fix this
  early").
- Within the moved router files: cross-domain imports repointed per the
  work order's one authorized exception beyond pure relocation —
  `Recipe`/`Ingredient`/`PantryItem`/`RecipeIngredient` → `domains.recipes.models`;
  `WorkoutPlan`/`WorkoutPlanDay`/`WorkoutSession`/`WeightUnit` →
  `domains.workout.models`.

## 1.4 The one real gap at original delivery: `templates/shopping_list.html`

**Symptom:** the file's content was never included among the materials
shared for this engagement, despite being explicitly listed in the work
order's own SCOPE section.

**Handling:** per the work order's own WORKING METHOD clause (substitute
checks, mark ⚠️ and explain), this was evaluated — but since the file wasn't
merely *unverifiable*, it was *entirely absent*, it was marked ❌ rather than
⚠️ on the one acceptance criterion it blocks (`GET /plan/{id}/shopping`).
Nothing else was guessed at or fabricated to paper over the gap. The router
function (`shopping_list_view()`) still moved correctly and still calls
`templates.TemplateResponse("shopping_list.html", ...)` unchanged, so once
supplied, the fix was expected to be a pure drop-in with zero router code
changes — which is exactly what happened (Part 2, §2.3).

## 1.5 Verification methodology used in the original pass

No live MariaDB or Airflow was available (standing constraint across this
entire series). Unlike some earlier postmortems in this series, this pass
did not rely on static reasoning alone — real dependencies were installed
and real checks were executed:

- **Real `sqlalchemy`/`jinja2`/`pydantic` installed** (not simulated) —
  `pip install sqlalchemy pydantic jinja2`.
- **Real `sqlalchemy.orm.configure_mappers()` run** against
  `domains.planning.models` + `domains.recipes.models` +
  `domains.workout.models` (the real, provided sources — not stubs) — zero
  `InvalidRequestError`. Both genuine cross-domain relationships were
  inspected via `sqlalchemy.inspect()` and confirmed to resolve to the
  correct classes: `WeeklyPlanDay.workout_session` →
  `domains.workout.models.WorkoutSession`; `WeeklyPlanMeal.recipe` /
  `.swap_recipe` → `domains.recipes.models.Recipe`.
- **Full `import models` executed end-to-end**, using minimal stand-in stub
  packages for the six domains whose real source wasn't part of this
  engagement (jobs, finance, blog, code_intel, habits, media) plus a stub
  `domains/journal/models.py` (journal's real model source has never been
  shared in this entire migration series — see Part 4, §4.9). This
  confirmed the shim wiring for real: `models.WeeklyPlan is
  domains.planning.models.WeeklyPlan` (and the other 9 names), via direct
  identity assertions, not inference.
- **AST-based check** confirming none of the 5 moved ORM classes remain
  *defined* (only imported) anywhere in `models.py`.
- **AST-based route enumeration**, not manual reading, to prove the
  `weekly_plan.py` / `weekly_plan_generator.py` split shares the `/plan`
  prefix safely: 7 routes vs. 1 route, zero overlap.
- **Real Jinja2 `ChoiceLoader` compilation** of all 8 moved templates (the
  9th, `shopping_list.html`, couldn't be — see §1.4).
- **Real template *rendering*** (not just compilation) against real
  `UserIntent` instances and realistic mock `WeeklyPlanDay`/`WeeklyPlanMeal`
  objects for `intent.html`, `partials/intent_saved.html`,
  `weekly_plan_hub.html` (no-plan branch), `partials/plan_meal_row.html`,
  and `partials/plan_day_card.html` (including its nested `{% include %}`
  of `plan_meal_row.html`) — this is the class of check that caught real
  context-mismatch bugs in the habits and media migrations; none were found
  here.

## 1.6 Original acceptance criteria results

| Criterion | Result | Reason |
|---|---|---|
| `GET /intent` renders identically | ✅ | Rendered for real against a real `UserIntent` instance |
| `POST /intent` saves + returns `intent_saved.html` | ⚠️ | Router logic diff-confirmed unchanged; template rendered for real; live DB write not exercised |
| `GET /intent/context` returns correct JSON | ⚠️ | Router logic unchanged; not exercised live |
| `GET /plan` (hub) renders identically | ✅ | Rendered for real (no-plan branch) |
| `GET /plan/new` renders identically | ⚠️ | Compiles cleanly; not rendered with full context in this pass |
| `POST /plan/generate` reaches both agents | ⚠️ | No live Gemini access; `weekly_agents.py` not shared at this point in the engagement; import/wiring confirmed only |
| `POST /plan/confirm` creates rows + reaches both cross-domain imports | ✅ | The two things the repointing exists to prove verified via real `configure_mappers()` + `inspect()` |
| `GET /plan/{id}` renders identically | ⚠️ | Compiles cleanly; not rendered with a full `plan` object |
| `PATCH /plan/{id}/day/{date_str}` returns `plan_day_card.html` correctly | ✅ | Rendered for real against a realistic mock, including the nested include |
| `PATCH /plan/meal/{meal_id}` returns `plan_meal_row.html` correctly | ✅ | Rendered for real against a realistic mock |
| `GET /plan/{id}/shopping` renders identically | ❌ | `shopping_list.html` content never provided — see §1.4 |
| Router-split verification | ✅ | AST-enumerated: 7 vs. 1 routes, zero overlap |
| `Base.metadata` identity check | ✅ | Real `configure_mappers()`, real identity assertions, AST-confirmed no stray class definitions |
| `journal.py`'s updated import resolves | ⚠️ | Confirmed statically; live save-with-linked-plan-day not exercised |
| `grep` sweep (10 planning + 8 recipes/workout classes) | ✅ | Zero stray references found |
| `dashboard.py` required zero changes | ✅ | Checked directly against the real provided source |

**Per GOVERNANCE.md §4.6, the original WO#10 was *not* fully "done"** at
this point — one required file was missing, and several ⚠️ items reflected
this sandbox's standing inability to hit a live DB. It was reported
honestly as such rather than rounded up.

---

# PART 2 — POST-MORTEM: FOLLOW-UP WORK (AGREED AFTER THE ORIGINAL DELIVERY)

**Everything in this part was proposed and/or requested, and explicitly
authorized, in conversation after Part 1's delivery was reported.** None of
it was part of WO#10's original acceptance criteria. It's recorded
separately, per GOVERNANCE.md §4.5, so a reviewer can judge "did the
relocation succeed" independently of "were these follow-up changes any
good" — and because one item is a genuine, flagged exception to the
original HARD BOUNDARIES.

## 2.1 Timeline / provenance

| Step | What happened | Authorization |
|---|---|---|
| 1 | Original WO#10 delivered — models split, routers split+moved, 8/9 templates moved, config wired, one gap flagged (§1.4) | Standing work-order template, issued as WO#10 |
| 2 | Reviewer noted the deliverable files themselves hadn't actually been produced as downloadable output (only described in a text report) | — |
| 3 | Files copied to the output directory and presented for the first time | Direct request: *"I don't see the files"* |
| 4 | Reviewer supplied `shopping_list.html` and `weekly_agents.py` (both previously missing/out of scope) and gave five explicit instructions in one message | See §2.2–§2.6 below, one per item |
| 5 | All five items executed and individually re-verified | — |
| 6 | Reviewer asked for the deliverable repackaged as a single, well-organized zip | Direct request |
| 7 | Zip delivered; reviewer then asked for the `weekly_agents.py` bug to be fixed **before** finalizing the zip | *"keep going with providing the zip, but fix the `agent_plan_meals()` before that"* |
| 8 | Bug fixed, re-verified against the same reproduction that found it, zip rebuilt and delivered | — |
| 9 | This document produced | Explicit request, including the requirement to separate original-WO from follow-up work |

## 2.2 Follow-up item 1 — Three-way router split (`weekly_plan_shopping.py`)

**What was proposed:** in the original report's Notes, `weekly_plan.py` was
flagged as still 534 lines after the AI-generation split — over
GOVERNANCE.md §1.2's 300-line ceiling — with a specific, named follow-up
recommendation: extract `_generate_shopping_list()` + `shopping_list_view()`
into their own `weekly_plan_shopping.py`.

**Authorization:** *"Go ahead and make the split you recommended."*

**What was done:** new file `domains/planning/routers/weekly_plan_shopping.py`
(155 lines), containing `_generate_shopping_list()` and
`GET /{plan_id}/shopping` with its own `router = APIRouter(prefix="/plan",
tags=["Weekly Plan"])`. `weekly_plan.py` now imports
`_generate_shopping_list` from this sibling module for use inside
`confirm_plan()`, rather than duplicating it. `main.py` updated with a
third `include_router` call for this new router.

**Verification, real and re-run, not assumed carried-over:**
- `python -m py_compile` on all three router files.
- Real `sqlalchemy.orm.configure_mappers()` re-run after the split — still
  clean.
- `wp._generate_shopping_list is wps._generate_shopping_list` asserted
  directly (proves single source of truth, not accidental duplication).
- AST-based route enumeration re-run across all **three** files:
  `weekly_plan.py` → 6 routes, `weekly_plan_generator.py` → 1 route,
  `weekly_plan_shopping.py` → 1 route, **8 total, zero collisions** (down
  from 7 in `weekly_plan.py` alone before this split — the missing route is
  `GET /{plan_id}/shopping`, now in its own file, confirming nothing was
  lost).
- **Real line-count result:** `weekly_plan.py` 534 → 413 lines. **Still
  over the 300-line ceiling** — this further split was scoped exactly to
  what was recommended and authorized; a deeper split wasn't requested and
  wasn't done. See Part 4, §4.8 for the full, program-wide file-size
  ceiling audit this feeds into.

## 2.3 Follow-up item 2 — `templates/shopping_list.html` supplied and moved in

**Authorization:** *"I added again `templates/shopping_list.html`."*

**What was done:** the file was moved into
`domains/planning/templates/shopping_list.html` verbatim — no content
changes, since the router code that serves it (`shopping_list_view()`,
already moved and unchanged in the original pass) never needed to change
either.

**Verification, real:**
- Compiled through the real Jinja2 `ChoiceLoader`.
- **Rendered for real, both branches** — the empty-list state (`"No
  shopping list yet"`) and the populated state (a real category-grouped
  item plus a pantry-section item, confirming the `need_grouped` /
  `pantry_items` template logic still works against the moved file).

This closes the ❌ from §1.6 — `GET /plan/{id}/shopping` can now be marked
✅ for template correctness (still ⚠️ for the live-DB write path, same as
every other endpoint in this domain, per the standing sandbox limitation).

## 2.4 Follow-up item 3 — Import hygiene

**Authorization:** *"Do the hygiene as you see fit"* — in response to two
previously-flagged, not-yet-fixed findings from the original report:
`RecipeMealType` (already dropped in the original pass, since it was never
referenced by name) and `WorkoutPlanDay` (kept in the original pass "out of
caution," despite being unused).

**What was done:**
- `WorkoutPlanDay` dropped from `weekly_plan.py`'s import line — confirmed
  unused by name anywhere in the file.
- As a direct consequence of the §2.2 split: `ShoppingList`, `Ingredient`,
  and `PantryItem` are no longer referenced in `weekly_plan.py` either
  (they moved with the functions that used them) — dropped from its import
  lines too.

**Verification, real, not just diff-reading:** re-imported the module and
inspected its live namespace (`dir(wp)`) to confirm `ShoppingList`,
`Ingredient`, `PantryItem`, and `WorkoutPlanDay` are genuinely absent —
not just missing from a source diff that could theoretically still bind
the name transitively via `import *` or similar.

## 2.5 Follow-up item 4 — Stale docstring fix

**What was flagged in the original report:** `weekly_plan.py`'s own module
docstring listed `POST /plan/{id}/shopping/regenerate → Regenerate
shopping list` as an endpoint, but no such route existed anywhere in the
code — a pre-existing doc/code mismatch, the same category of issue as the
habits postmortem's §4.3 finding.

**Authorization:** *"Is it something in one of the new files I shared? If
not, then feel free to fix now"* — i.e., conditional authorization: fix it
if the newly-shared files (`shopping_list.html`, `weekly_agents.py`) don't
reference or imply it should exist.

**What was checked, and the result:** both newly-shared files were grepped
for `regenerate` / `shopping` — zero matches in either. Nothing in the
actual application logic calls for this endpoint. Per the conditional
authorization, the docstring was corrected (the phantom line removed, and
a short note added explaining why) rather than a new endpoint being
speculatively built to match a stale comment.

## 2.6 Follow-up item 5 — `weekly_agents.py` bug found and fixed (explicit HARD BOUNDARIES exception)

**What was requested:** *"I shared the agent file now. Revisit and let me
know if any issues still exist"* — initially a request to *report*, not to
fix, consistent with `weekly_agents.py` still being outside this work
order's SCOPE.

**What was found, verified with real, executed reproductions (not
inferred):**
1. **Call-site signature check** — `weekly_plan_generator.py`'s calls to
   `agent_plan_meals()` and `agent_schedule_workouts()` were checked via
   `inspect.signature()` against the real function definitions: every
   keyword argument matches exactly, zero missing or extra parameters.
2. **A real, previously-unverifiable bug** — `agent_plan_meals()` computes
   `workout_day_names` / `rest_day_names` via
   `date(week_start.year, week_start.month, week_start.day + d - 1)`. This
   is raw day-offset arithmetic with no month-boundary handling. Reproduced
   with a mocked HTTP layer (so no network dependency): **swept every
   Monday of 2026 and found 11 of 52 weeks (~21%, roughly monthly) raise
   `ValueError: day is out of range for month`** whenever a plan week's
   day 7 (Sunday) spills into the next calendar month. The calling
   router's `try/except Exception` catches this, so the user-visible
   effect isn't a 500 — it's a **silently empty meal plan** for the entire
   week. `agent_schedule_workouts()` was checked and confirmed unaffected
   (it works from ISO date strings via `date.fromisoformat()`, not
   day-offset construction).

**Authorization to fix (not just report):** *"ok... keep going with
providing the zip, but fix the `agent_plan_meals()` before that"*.

**Fix applied, in `airflow/agents/weekly_agents.py`:**
```diff
-    workout_day_names = [
-        date(week_start.year, week_start.month, week_start.day + d - 1).strftime("%A")
-        for d in workout_days
-    ]
-    rest_day_names = [
-        date(week_start.year, week_start.month, week_start.day + d - 1).strftime("%A")
-        for d in rest_days
-    ]
+    workout_day_names = [
+        (week_start + timedelta(days=d - 1)).strftime("%A")
+        for d in workout_days
+    ]
+    rest_day_names = [
+        (week_start + timedelta(days=d - 1)).strftime("%A")
+        for d in rest_days
+    ]
```
`timedelta` was already imported at the top of the file — no new import
needed. Scoped to exactly these two list comprehensions; nothing else in
the 224-line file was touched (confirmed via direct inspection of the
diff).

**Re-verification, real:**
- The exact same full-year sweep re-run post-fix: **0 of 52 weeks crash**.
- A **correctness** check, not just an absence-of-crash check: rendered
  the actual prompt text for a month-crossing week (Jan 26 – Feb 1, 2026)
  and confirmed all 7 weekday names (Monday through Sunday) appear
  correctly in the generated `WORKOUT DAYS` / `REST DAYS` prompt lines.

**Explicit deviation log entry** (per the pattern established in WO#8's
own postmortem §4.4):

| Deviation | Standing rule | Authorization |
|---|---|---|
| Editing `airflow/agents/weekly_agents.py`, explicitly out of SCOPE and marked "must NOT be moved" under WO#10's original HARD BOUNDARIES | WO#10 HARD BOUNDARIES: "Only read/edit files explicitly listed in SCOPE" | *"fix the `agent_plan_meals()` before that"* |

**Important scoping note:** the file was *edited in place*, not *moved* —
it still lives at its original real path, `airflow/agents/weekly_agents.py`,
and was never relocated into `domains/planning/`. This is consistent with
GOVERNANCE.md §2.5 (DAG/agent relocation is a separately deferred
initiative) and with the AI Service Layer program (GOVERNANCE.md §2.3,
WO#11/WO#12) — `weekly_agents.py` is tracked there as **WO#13 (planned, not
started)**, one of the remaining files slated for eventual migration onto
`services/ai/`. **Whoever picks up that future WO#13 needs to know this fix
already landed** and must not silently regress it by reintroducing the old
`date(...)` construction from an out-of-date snapshot — flagged again in
Part 4, §4.9.

## 2.7 Updated acceptance criteria (supersedes §1.6 where noted)

| Criterion | §1.6 result | Result after Part 2 | What changed |
|---|---|---|---|
| `GET /plan/{id}/shopping` renders identically | ❌ | ✅ (template) / ⚠️ (live DB) | `shopping_list.html` supplied and moved in (§2.3), rendered for real, both branches |
| Router-split verification | ✅ (2-way) | ✅ (3-way) | Re-verified after the shopping split; 8 routes, zero collisions |
| `weekly_plan.py` size vs. GOVERNANCE §1.2 | 534 lines (over) | 413 lines (still over) | Shopping split reduced it but didn't bring it under the ceiling — see Part 4, §4.8 |
| `POST /plan/generate` reaches both agents correctly | ⚠️ (agent file not shared) | ⚠️ (agent file now shared, call sites verified exact, but a real bug was found and fixed inside it — live Gemini call still not exercised) | See §2.6 |
| All other criteria from §1.6 | — | Unchanged | Not affected by the follow-up round |

## 2.8 Verification methodology used in the follow-up round

Same standing constraints as Part 1 (no live MariaDB/Airflow), same
standard of "install real dependencies, execute real checks" rather than
static reasoning:
- `fastapi` and `python-multipart` installed for real (Part 1 had only
  needed `sqlalchemy`/`jinja2`/`pydantic`) so the actual router *modules*
  — not just the model layer — could be imported and introspected.
- A minimal `database.py` stand-in (`async def get_db(): yield None`) was
  added purely so router files import cleanly for static verification —
  explicitly not a real DB connection, noted here so it isn't mistaken for
  one.
- Every check in §2.2–§2.6 above was executed against the real, current
  state of the files after each change — not assumed to still hold from
  Part 1.

---

# PART 3 — What "done" means for this domain (GOVERNANCE.md §4.6 checklist)

- [x] Models, routers, and templates all live under `domains/planning/`.
- [x] `main.py` and `core/templating.py` reference the new paths.
- [x] A legacy shim exists in root `models.py` for external consumers.
- [x] Every acceptance criterion is ✅ or an explained ⚠️ (per §2.7 — the
      one ❌ from the original pass was closed in the follow-up round).
- [x] No unrelated behavior changed **except** the one explicitly authorized,
      logged exception (§2.6's `weekly_agents.py` bugfix) — logged, not
      silently absorbed.

**This domain is ready to be marked done**, on the understanding that the
remaining ⚠️ items (live DB writes, live Gemini calls) require the real
deployment environment to close out — the same standing caveat every prior
domain in this series has carried.

---

# PART 4 — REQUIREMENTS FOR FINAL CLEANUP (execute now — every domain has migrated)

**This section is different in kind from every prior postmortem's own
final-cleanup section.** WO#1 through WO#9 each had to say "wait until
every remaining domain migrates" — because more domains were still
pending. **Planning was the last one.** Per GOVERNANCE.md §3.3 and the
`00_MASTER_INDEX.md` work-order list, WO#10 (`planning`) was explicitly
scoped to run last, after WO#7 (`recipes`) and WO#8 (`workout`), because of
the cross-domain relationships this domain depends on. With it done, this
section can be treated as a real, executable work order — not a forward
placeholder — subject to the readiness check in §4.1.

## 4.1 Confirm readiness first

Before executing anything below, re-run the standing check every prior
postmortem in this series specifies:
```bash
grep -rn "from models import" --include="*.py" . | grep -v "^./models.py:"
```
**Expected result, if this document's premise holds:** exactly one real
consumer, `routers/dashboard.py`.

**A note on why this needs re-confirming, not just assuming:** the WO#9
(media) postmortem's own §3.7 found a real discrepancy — at the time that
document was written, `main.py` still showed `recipes`, `pantry`, and
`workout` imported via the old flat `routers/` paths, contradicting an
earlier assumption that they'd already migrated. **That discrepancy is
resolved as of this document** — the `main.py` provided for WO#10 already
showed `domains.recipes.routers` (tagged `# WO7`) and
`domains.workout.routers` (tagged `# WO8`) fully wired in, confirmed
directly by reading that file, not inferred. Still: **re-run the grep
above against the real repository before proceeding** — this document's
confidence is bounded by the files it was given, the same limitation every
prior postmortem in this series has flagged about itself.

## 4.2 `models.py` final state

**Confirmed via a real AST parse of the current `models.py`** (not
estimated): the file is **249 lines**, contains **zero remaining
`ClassDef` nodes**, and consists entirely of **10 shim `ImportFrom`
blocks** — one per domain:

```
domains.jobs.models
domains.finance.models
domains.blog.models
domains.code_intel.models
domains.habits.models
domains.journal.models
domains.recipes.models
domains.workout.models
domains.media.models
domains.planning.models
```

Every domain's real classes now live entirely in `domains/*/models.py`.
`models.py` itself is, functionally, already just a registry — this makes
the choice between the two options the blog/code_intel postmortem (Part 2,
§2) first laid out simpler than when that document was written, since
there's no longer any question of "some domains still have real classes
directly in `models.py`":

- **Option 1 — delete `models.py` entirely.** Requires moving the
  import-triggers-registration guarantee somewhere else (e.g.
  `database.py`'s `init_db()`) — see §4.4's critical relationship-risk
  section before choosing this.
- **Option 2 (recommended, same reasoning as every prior postmortem) —
  reduce `models.py` to a pure import-registry**, e.g.:
  ```python
  # models.py — model registry.
  # Every ORM model lives in its own domain's models.py. This file's only
  # job is to guarantee every domain module gets imported at least once
  # before the first query, so SQLAlchemy's mapper registry has every
  # class available for string-based relationship() resolution.
  from domains.habits import models as _habits_models          # noqa: F401
  from domains.blog import models as _blog_models              # noqa: F401
  from domains.code_intel import models as _code_intel_models  # noqa: F401
  from domains.jobs import models as _jobs_models               # noqa: F401
  from domains.finance import models as _finance_models         # noqa: F401
  from domains.journal import models as _journal_models         # noqa: F401
  from domains.recipes import models as _recipes_models         # noqa: F401
  from domains.workout import models as _workout_models         # noqa: F401
  from domains.media import models as _media_models             # noqa: F401
  from domains.planning import models as _planning_models       # noqa: F401
  ```
  This is now a **10-line file**, not a re-derivation — every domain is
  already known and named above.

**Whichever option is chosen, `routers/dashboard.py` must be updated in the
same pass** — see §4.3, the one confirmed real consumer.

## 4.3 `routers/dashboard.py` — full per-domain breakdown, now complete

Every prior postmortem in this series confirmed `dashboard.py`'s status for
its own domain in isolation. **This is the first point in the whole
program where the full picture can be assembled in one place.**
`dashboard.py`'s real, complete import block (confirmed verbatim against
the actual provided source, not paraphrased):

```python
from models import (
    Job, ApplicationLog, ApplicationStatus,
    StagingJob, StagingJobStatus,
    Transaction, BlogIdea, BlogIdeaStatus,
    Habit, HabitLog, HabitSettings,
    JournalEntry, WeeklySynthesis,
)
```

| Domain | Shim size | Names `dashboard.py` actually imports | Fully orphaned? | Evidence |
|---|---|---|---|---|
| Jobs | 13 | `Job`, `ApplicationLog`, `ApplicationStatus`, `StagingJob`, `StagingJobStatus` (5) | No | Direct inspection |
| Finance | 9 | `Transaction` (1) | No | Direct inspection |
| Blog | 6 | `BlogIdea`, `BlogIdeaStatus` (2) | No | Direct inspection |
| Code Intel | 12 | *(none)* | **Yes** | First surfaced in WO#9's postmortem §3.2; not independently re-verified here |
| Habits | 7 | `Habit`, `HabitLog`, `HabitSettings` (3) | No | Direct inspection |
| Journal | 2 | `JournalEntry`, `WeeklySynthesis` (both) | No | Direct inspection |
| Recipes | 17 | *(none)* | **Yes** | Confirmed directly against the real `dashboard.py` source in this engagement |
| Workout | 16 | *(none)* | **Yes** | Confirmed in WO#8's own postmortem §7.1 ("zero references... confirmed by grepping its import block"); reconfirmed here |
| Media | 15 | *(none)* | **Yes** | Confirmed in WO#9's postmortem §1.9/§3.2 |
| Planning | 10 | *(none)* | **Yes** | Confirmed directly against the real `dashboard.py` source during WO#10 |

**Practical implication:** when the shim-removal pass runs, **5 of 10
domains (Code Intel, Recipes, Workout, Media, Planning) can have their
shim deleted with zero corresponding edit to `dashboard.py`**. The other 5
(Jobs, Finance, Blog, Habits, Journal) need `dashboard.py`'s import block
split into direct per-domain imports:

```python
from domains.jobs.models import Job, ApplicationLog, ApplicationStatus, StagingJob, StagingJobStatus
from domains.finance.models import Transaction
from domains.blog.models import BlogIdea, BlogIdeaStatus
from domains.habits.models import Habit, HabitLog, HabitSettings
from domains.journal.models import JournalEntry, WeeklySynthesis
```

Per the batching principle established in the habits postmortem
("do this for all domains in the same pass, not incrementally") — do this
in one shim-removal pass across all 10 domains, not five separate edits.

## 4.4 Critical: cross-domain SQLAlchemy relationship risk — full inventory

Per the standing warning first raised in the blog/code_intel postmortem
(Part 2, §3): any `relationship()` using a **string** class name only
resolves correctly if every module defining a referenced class has been
imported by *something* before the first query touches that relationship.
**This is the single most important thing to get right before choosing
Option 1 in §4.2** (deleting `models.py` outright) — if chosen, the
import-triggers-registration guarantee must move somewhere else
(`database.py`'s `init_db()` is the natural home) before the shim imports
disappear.

**Full inventory of genuine cross-domain relationships, compiled from every
postmortem in this series plus direct verification in WO#10:**

| # | Location | Relationship | Crosses | Verification status |
|---|---|---|---|---|
| 1 | `WeeklyPlanDay.workout_session` | `relationship("WorkoutSession", ...)` | `planning` ↔ `workout` | **Verified for real in WO#10** — real `configure_mappers()` + `inspect()`, resolves to `domains.workout.models.WorkoutSession` |
| 2 | `WeeklyPlanMeal.recipe` / `.swap_recipe` | `relationship("Recipe", ...)` | `planning` ↔ `recipes` | **Verified for real in WO#10** — resolves to `domains.recipes.models.Recipe` |
| 3 | `BlogIdea.code_file` / `.code_project`, `CodeFile.blog_ideas`, `CodeProject.blog_ideas` | `relationship("CodeFile"/"BlogIdea", ...)` | `blog` ↔ `code_intel` | Documented in WO#2's postmortem; not independently re-verified in this document (blog/code_intel's real source wasn't part of this engagement) |
| — | `WeeklyPlanDay.journal_entry_id` | **Plain FK only, no `relationship()`** | `planning` ↔ `journal` | Confirmed in WO#10: no special handling needed — a bare FK column doesn't require the referenced class to be import-registered, only a real `relationship()` does |

**Before finalizing whichever `models.py` end-state is chosen (§4.2), run
this across every domain's real source** (not the stubs used for
verification in this engagement):
```bash
grep -rn 'relationship(' domains/*/models.py
```
and confirm every string-named class's owning module is covered by
whichever registry mechanism is chosen. Also grep for any
`secondary=<bare_table_object>` pattern (rather than
`secondary="table_name_string"`) — that would indicate a *direct* Python
object dependency, not just a string-resolved one, and would need an
actual cross-domain import, not just registration. (`domains/recipes/models.py`'s
`Recipe.tags` uses `secondary=recipe_tags_junction`, the bare `Table`
object — but this is a **same-domain** reference, `RecipeTag` and `Recipe`
both live in `domains/recipes/models.py`, so it's a non-issue; flagged here
only so the grep-and-check step doesn't skip past it as if it were a
cross-domain case.)

## 4.5 `core/templating.py` final state

Confirmed current (all 10 domains + root):
```python
templates.env.loader = ChoiceLoader([
    FileSystemLoader("templates"),               # shared/core only: base.html, dashboard.html, 404.html, 500.html
    FileSystemLoader("domains/habits/templates"),
    FileSystemLoader("domains/blog/templates"),
    FileSystemLoader("domains/code_intel/templates"),
    FileSystemLoader("domains/jobs/templates"),
    FileSystemLoader("domains/explorer/templates"),
    FileSystemLoader("domains/finance/templates"),
    FileSystemLoader("domains/journal/templates"),
    FileSystemLoader("domains/recipes/templates"),
    FileSystemLoader("domains/workout/templates"),
    FileSystemLoader("domains/media/templates"),
    FileSystemLoader("domains/planning/templates"),
])
```
This is now the **final-form list** — no more domains remain to add.
Nothing further needed here beyond confirming this survives whatever final
review happens. Also still outstanding, carried forward from two prior
postmortems (blog/code_intel, explorer) and never investigated:
`templates/desktop.ini`, a stray Windows Explorer artifact — resolve or
delete now that no domain migration is still "in flight" to justify
continuing to defer it.

## 4.6 `main.py` final state

Confirmed current: every router import is `from domains.X.routers import
Y` **except** `from routers import dashboard` — which is correct and
intentional (§4.7). The planning domain specifically registers 4 router
modules (`intent`, `weekly_plan`, `weekly_plan_generator`,
`weekly_plan_shopping` — the last one added in Part 2, §2.2) across 3
`include_router` calls plus intent's own. No static mount for planning
(it has none).

**Action needed:** the *old*, flat `routers/intent.py` and
`routers/weekly_plan.py` files should be deleted from the real repository
now that their replacements are confirmed working — this sandboxed
engagement never had filesystem delete access to the real repo (same
standing limitation the finance postmortem's §5.3 already documented for
every domain before it). This applies to **every** domain's superseded
flat-layout originals, not just planning's — see §4.7.

## 4.7 Fate of `routers/`, `templates/`, `static/`

With planning's migration, the flat `routers/` folder should now contain
**nothing except `dashboard.py` and `_helpers.py`** (the shared
`html_error()` helper, referenced across multiple domains' postmortems as
genuinely cross-cutting, non-domain-specific code). Confirm this — audit
for any file left behind unexpectedly — before deciding `_helpers.py`'s
final home (leave it in a slim `routers/` package, or relocate to
`core/http_helpers.py` alongside `core/templating.py` and
`core/base_model.py`, per the finance postmortem's own §5.5
recommendation, never acted on).

The flat `templates/` and `templates/partials/` directories should now
contain only genuinely shared content: `base.html`, `dashboard.html`,
`404.html`, `500.html`, any shared partials actually used by more than one
domain, plus the stray `desktop.ini` (§4.5). The flat `static/` directory
should contain only genuinely global CSS/JS.

**Audit checklist:**
```bash
ls routers/                    # expect: dashboard.py, _helpers.py, maybe __init__.py — nothing else
ls templates/                  # expect: base.html, dashboard.html, 404.html, 500.html, desktop.ini, __pycache__ or similar noise only
ls templates/partials/         # expect: only genuinely shared partials, if any
ls static/css/ static/js/      # expect: only genuinely global assets
```
Anything beyond that is either a domain's file that was never deleted from
its old flat location (delete it, its replacement is confirmed working),
or something genuinely shared that was missed by every prior domain
audit (investigate before deleting).

## 4.8 GOVERNANCE.md §1.2 file-size ceiling — full program audit

Consolidated from every postmortem in this series that flagged an
over-ceiling router, plus WO#10's own finding:

| File | Lines | Status | Notes |
|---|---|---|---|
| `domains/planning/routers/weekly_plan.py` | 413 | Over (was 534 before the WO#10 follow-up split) | See Part 2, §2.2. A further split (e.g. day-override + meal-status into their own file) would need its own explicit go-ahead, per the established pattern of this series (propose → authorize → execute → document) |
| `domains/workout/routers/workout_plan_ai_generator.py` | 373 | Over | Flagged in WO#8's postmortem §7.2; not yet revisited |
| `domains/workout/routers/workout_log.py` | 357 | Over | Flagged in WO#8's postmortem §7.2; not yet revisited |
| `domains/workout/routers/workout_settings.py` | 378 | Over | Flagged in WO#8's postmortem §7.2; not yet revisited |

**Recommendation:** per `00_MASTER_INDEX.md`'s own "Deferred / Not Yet
Scoped" section, item A.1 already calls for a lint script (WO#19) to get
real, current line counts across the whole `domains/*/routers/` tree
before scoping any further splits — do that first rather than guessing at
split boundaries file-by-file. This table should be treated as a snapshot,
not a final list; some of these numbers may have drifted since their
source postmortems were written.

## 4.9 New findings from WO#10 that final cleanup must carry forward

None of these were introduced as bugs by this migration — they're facts
discovered during it that the next agent shouldn't have to rediscover:

1. **`domains/journal/models.py`'s real source has never been shared in
   this entire migration series** — not in WO#6 (journal's own migration),
   not here. Every verification pass that needed it (including WO#10's
   full-`import models` check) used a minimal hand-built stand-in. This is
   a standing gap, not new debt from this migration, but it's worth
   surfacing plainly: **whoever does the final full-program verification
   pass (§4.10) needs journal's real `models.py` to do it for real.**
2. **The `airflow/agents/weekly_agents.py` bugfix (Part 2, §2.6) must not
   be silently lost.** This file is tracked in the AI Service Layer
   program (GOVERNANCE.md §2.3, WO#11/WO#12) as **WO#13 (planned, not
   started)** — one of the remaining files slated for eventual migration
   onto `services/ai/`. If that future work order is scoped from an
   out-of-date snapshot of `weekly_agents.py` (predating this fix), it
   could silently reintroduce the month-boundary bug. Flag this explicitly
   in whatever scopes WO#13.
3. **The planning domain now has a 3-router split under one shared `/plan`
   prefix** (`weekly_plan.py`, `weekly_plan_generator.py`,
   `weekly_plan_shopping.py`) — the same pattern `workout` (WO#8) already
   established for `workout_plans_crud.py` /
   `workout_plan_ai_generator.py`, now with a third file. There's no
   registration-order constraint between the three (their path patterns
   don't overlap — verified via AST route enumeration, not just eyeballed)
   but worth knowing this shape exists before any future refactor of
   `main.py`'s include-router block.

## 4.10 `00_MASTER_INDEX.md` is stale and needs a refresh pass

As provided in this engagement, `00_MASTER_INDEX.md`'s work-order table
marks **every** domain migration work order (#1–#10) as "📝 Drafted, not
yet executed" — including habits (#1), which that document's own
cross-reference (the habits postmortem) describes as "Complete and
confirmed working in production." This is a real inconsistency between two
documents in this same migration-docs set, not something to silently paper
over. **Whoever does the final cleanup pass should also update
`00_MASTER_INDEX.md`'s status column for all ten domain-migration rows** to
reflect what the individual postmortems actually show (all ten complete,
per this document and its nine predecessors), and should treat this as a
reminder that status tracked in two places drifts — worth deciding whether
`00_MASTER_INDEX.md` should be regenerated from the postmortems going
forward rather than maintained by hand alongside them.

## 4.11 Final verification checklist (run once, now that every domain is in)

- [ ] `grep -rn "from models import" --include="*.py" .` returns only
      `routers/dashboard.py` as a real consumer (§4.1).
- [ ] `models.py` reduced per the chosen option in §4.2; `dashboard.py`
      updated per §4.3's per-domain breakdown in the same pass.
- [ ] `grep -rn 'relationship(' domains/*/models.py` re-run across every
      domain's **real** source (not stubs); every string-named class's
      module confirmed covered by the registry mechanism chosen (§4.4).
- [ ] `Base.metadata` table-identity check run against the **full, real**
      app (all 10 domains' real `models.py` files, not the stand-ins used
      in this and prior engagements) — same table count before/after,
      `models.X is domains.Y.models.X` for every class, `configure_mappers()`
      clean. **This is the first point in the whole program where this
      check can finally be run for real** rather than against a partial
      stub environment.
- [ ] Every router's `Jinja2Templates(...)` instantiation confirmed
      replaced with `from core.templating import templates` — spot-check a
      few, since no single engagement in this series has had every
      domain's real router source at once to check exhaustively.
- [ ] Every domain's static mount registered before the general `/static`
      mount in `main.py` — re-confirm the full list, not just planning's
      (which has none).
- [ ] Old flat-layout files for every migrated domain deleted from the real
      repository (`routers/intent.py`, `routers/weekly_plan.py`, and the
      equivalent originals for all nine other domains) — §4.6/§4.7.
- [ ] `routers/`, `templates/`, `templates/partials/`, `static/` audited
      per §4.7's checklist; `templates/desktop.ini` finally resolved.
- [ ] Full regression pass — every endpoint this document and its nine
      predecessors' postmortems list — run against a real MariaDB and a
      real running app, not a sandboxed stand-in. This is the first point
      in the whole program where a genuine end-to-end `uvicorn main:app`
      boot across every domain simultaneously is even possible.
- [ ] `00_MASTER_INDEX.md` status column refreshed (§4.10).
- [ ] The `weekly_agents.py` fix (Part 2, §2.6) confirmed still present
      before/after any future WO#13 (AI Service Layer) work touches that
      file (§4.9, item 2).
- [ ] `GOVERNANCE.md` updated to mark the domain-folder migration program
      (§3.3's backlog) as complete, pointing at this document (and its
      nine predecessors) for institutional memory of how it actually went.

---

# PART 5 — Reference: files touched across this entire engagement (planning domain, final state)

**New:**
- `domains/planning/__init__.py`, `domains/planning/routers/__init__.py`
- `domains/planning/models.py`
- `domains/planning/routers/weekly_plan_generator.py` (original pass)
- `domains/planning/routers/weekly_plan_shopping.py` (follow-up, §2.2)

**Moved (then edited in place per the diffs above):**
- `domains/planning/routers/intent.py` (from `routers/intent.py`)
- `domains/planning/routers/weekly_plan.py` (from `routers/weekly_plan.py`;
  split twice — once in the original pass, once in the follow-up round)
- `domains/planning/templates/intent.html`, `weekly_plan_hub.html`,
  `weekly_plan_new.html`, `weekly_plan_review.html`, `weekly_plan_view.html`,
  `shopping_list.html` (the last one moved in the follow-up round, §2.3)
- `domains/planning/templates/partials/plan_day_card.html`,
  `plan_meal_row.html`, `intent_saved.html`

**Edited in place (shared files, targeted diffs, not full-file
replacement, per the standing rule since WO#1):**
- `models.py` — planning shim added (original pass)
- `main.py` — planning router imports + 3 `include_router` calls (original
  pass, extended in the follow-up round for the third split router)
- `core/templating.py` — `domains/planning/templates` added to `ChoiceLoader`
- `domains/journal/routers/journal.py` — one local import repointed
  (original pass)
- `airflow/agents/weekly_agents.py` — month-boundary bugfix (follow-up
  round, §2.6 — the one explicit HARD BOUNDARIES exception in this whole
  engagement)

**Endpoints covered by this domain (11 total, across 4 router modules):**
`GET /intent`, `POST /intent`, `GET /intent/context`, `GET /plan`,
`GET /plan/new`, `POST /plan/confirm`, `GET /plan/{id}`,
`PATCH /plan/{id}/day/{date_str}`, `PATCH /plan/meal/{meal_id}`,
`POST /plan/generate`, `GET /plan/{id}/shopping`.
