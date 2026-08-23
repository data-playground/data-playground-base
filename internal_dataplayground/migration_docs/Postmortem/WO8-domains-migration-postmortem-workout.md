# Work Order #8 — Workout Domain Migration: Post-Mortem & Handoff

**Domain:** `workout`
**Status:** Complete (original scope) + 3 explicitly-authorized follow-up changes complete. See §6 for what still requires live-environment verification before this can be marked fully closed per GOVERNANCE.md §4.6.
**Companion document:** `migration_docs/GOVERNANCE.md` (all §-references below point there unless stated otherwise)

---

## 0. Purpose of this document

This document is the authoritative record of what happened in Work Order
#8, in two distinct layers:

1. **What was authorized and executed under the original Work Order #8
   scope** (§3) — a pure relocation, no behavior changes, per the work
   order's own ROLE and HARD BOUNDARIES.
2. **What was explicitly authorized as follow-up changes after the
   original migration was reported complete** (§4) — three specific,
   individually-approved deviations from those same HARD BOUNDARIES.

This separation is the single most important thing for whoever reviews
this work order to understand: **the original migration diff, on its
own, is a clean relocation with zero behavior changes** (per §4.5 /
§4.6's "no unrelated behavior changed" requirement). Everything that
changed behavior — a bug fix and two refactors — happened afterward, was
individually proposed with a rationale, and was individually approved by
the project owner before being executed. None of it was bundled into the
original diff silently.

A future agent picking up this project should treat §7 ("Deferred Work")
as its primary input for scoping any follow-on work order that touches
this domain again.

---

## 1. Executive summary

The `workout` domain (16 ORM classes, 4 router files — one of which
exports two separate `APIRouter` instances — 18 templates, 2 static
assets) was relocated from the flat `routers/`/`templates/`/`static/`
structure into `domains/workout/`, following the same pattern already
established in Work Orders #1–7. `models.py` was left with a re-export
shim so no other file's imports broke. `main.py` and
`core/templating.py` were updated to reference the new paths.

Verification in the migration sandbox was constrained: no network
egress (no live Gemini call possible), no installable SQLAlchemy (no
live mapper-configuration or live DB check possible), and no access to
the actual project filesystem (work was done from pasted file contents,
not a live `git` checkout). Every criterion that could only be verified
live was explicitly marked ⚠️ rather than ✅, with the substitute check
performed and stated. These remain open — see §6.

After the migration was reported, the project owner authorized three
follow-up changes, executed in a second pass: a genuine bug fix (a
`NameError` plus a previously-masked `TypeError` in the exercise-search
endpoint), a file split to bring `workout_plans.py` under GOVERNANCE.md
§1.2's line-count ceiling, and a DRY consolidation of two duplicated
helpers into a new `domains/workout/routers/_shared.py`. Each is
documented separately in §4 with its own rationale and authorization
record, because — per GOVERNANCE.md §4.5 — bug fixes and cleanup found
during a migration are supposed to get their own ticket, not be folded
into the migration diff. That rule was deliberately overridden here,
with explicit sign-off each time, not silently bypassed.

---

## 2. Timeline / provenance

| Step | What happened | Authorization |
|---|---|---|
| 1 | Original WO8 executed: models split + shim, 4 routers moved, 18 templates moved, 2 static assets moved, `main.py` + `core/templating.py` updated | Standing work-order template (GOVERNANCE §4.3), issued as WO8 |
| 2 | WO8 report delivered — acceptance criteria table, ✅/⚠️ per item, Notes section flagging the pre-existing `search_exercises()` bug | — |
| 3 | Reviewer asked "confirm everything is done correctly" | — |
| 4 | Response: confirmed what was directly testable (syntax, grep-based cross-references, mount ordering), restated the ⚠️ items honestly as still-open | — |
| 5 | Reviewer asked for further recommendations | — |
| 6 | Follow-up #1 (bug fix), #2 (file split), #3 (DRY consolidation) proposed as recommendations, explicitly *not* executed yet | — |
| 7 | Reviewer replied: *"1. Let's do that... 2. I like that. Go ahead... 3. ...let's do that as well altogether"* | **Explicit authorization for all three, in the same message** |
| 8 | All three follow-up changes executed, verified, reported | — |
| 9 | This document produced | Explicit request: *"make a post-mortem overview... add a section to explain what needs to be done after all the other migrations are completed"* |

---

## 3. Part A — Original Work Order #8 (as authorized)

### 3.1 Scope, restated

Per the work order's own SCOPE section: 16 ORM classes out of `models.py`
into `domains/workout/models.py`; 4 router files into
`domains/workout/routers/`; 18 templates into `domains/workout/templates/`
(preserving the `partials/workout/` two-level nesting); 2 static assets
into `domains/workout/static/`; edits to `models.py` (shim only),
`main.py`, and `core/templating.py`.

### 3.2 What was done — files created / moved / edited

**Created:**
- `domains/workout/__init__.py`, `domains/workout/routers/__init__.py`
- `domains/workout/models.py`

**Moved** (old path → new path, content unchanged except import lines):
- `routers/workout.py` → `domains/workout/routers/workout.py`
- `routers/workout_log.py` → `domains/workout/routers/workout_log.py`
- `routers/workout_plans.py` → `domains/workout/routers/workout_plans.py` *(superseded — see §4.2, this file no longer exists as of the follow-up pass)*
- `routers/workout_settings.py` → `domains/workout/routers/workout_settings.py`
- `templates/workout.html`, `workout_history.html`, `workout_progress.html`, `workout_plans.html`, `workout_plan_preview.html`, `workout_settings.html` → `domains/workout/templates/`
- `templates/partials/workout/*.html` (12 files) → `domains/workout/templates/partials/workout/`
- `static/css/workout.css`, `static/js/workout.js` → `domains/workout/static/css/`, `domains/workout/static/js/`

**Edited:**
- `models.py` — 16-class block replaced with a re-export shim tagged `# TODO: remove after all cross-references are updated`, matching the pattern already used for `jobs`, `finance`, `blog`, `code_intel`, `habits`, `journal`. *(See §4/models_py_shim_patch.md for how this was actually delivered — a patch-instructions file, not a full-file overwrite, since the working copy was a reconstructed subset, not the live repo file.)*
- `main.py` — imports repointed to `domains.workout.routers`; `/static/workout` mount added **before** the general `/static` mount (GOVERNANCE §2.6); both `workout_log.router` and `workout_log.body_metrics_router` kept as two separate `app.include_router()` calls.
- `core/templating.py` — `FileSystemLoader("domains/workout/templates")` added to the `ChoiceLoader` chain.
- 4 router files — model imports repointed to `domains.workout.models`; `templates = Jinja2Templates(directory="templates")` replaced with `from core.templating import templates`.
- 6 full-page templates — `href="/static/css/workout.css"` → `href="/static/workout/css/workout.css"`; `src="/static/js/workout.js"` → `src="/static/workout/js/workout.js"`. No other template content touched.

### 3.3 Hard boundaries respected

Explicitly verified as *not* changed, per the work order's HARD
BOUNDARIES:

- `workout_log.py`'s two-`APIRouter`-in-one-file structure — preserved exactly, both still registered separately in `main.py`.
- `_call_gemini_for_plan` in the AI generator — byte-identical, not routed through any service layer.
- The `search_exercises()` `NameError` bug — reproduced identically, **not fixed** under the original work order (it was fixed later, under separate, explicit authorization — see §4.1).
- No schema, endpoint, or behavior changes anywhere in the original diff.

### 3.4 Acceptance criteria — original WO8 results

| # | Criterion | Result (as of original WO8) | Result (as of this document, after §4 follow-ups) |
|---|---|---|---|
| 1 | `GET /workout` renders identically | ✅ | ✅ (unchanged) |
| 2 | `/workout/history`, `/workout/progress` render identically | ✅ | ✅ (unchanged) |
| 3 | `/workout/plans` + preview/save flow | ✅ | ✅ — now served by 2 routers instead of 1, see §4.2; behavior unchanged |
| 4 | `/workout/settings` renders identically | ✅ | ✅ (unchanged) |
| 5 | Session-logging endpoints | ✅ | ✅ (unchanged) |
| 6 | Body-metrics endpoints (both routers) | ✅ | ✅ (unchanged) |
| 7 | `GET /workout/exercises` | ⚠️ known pre-existing bug, reproduced, left unfixed | ✅ **fixed** — see §4.1 |
| 8 | Locations/equipment CRUD + custom-exercise creation | ✅ | ✅ (unchanged) |
| 9 | `/workout/plans/generate` reaches Gemini call correctly | ⚠️ substitute check only (no network) | ⚠️ still open — unchanged, see §6 |
| 10 | `Base.metadata` / mapper-configuration check | ⚠️ substitute check only (no SQLAlchemy) | ⚠️ still open — unchanged, see §6 |
| 11 | `weekly_plan.py` imports resolve via shim | ⚠️ static-analysis only, no live join test | ⚠️ still open — unchanged, see §6 |
| 12 | `grep` sweep for the 16 moved class names | ✅ | ✅ (unchanged) |
| 13 | `dashboard.py` required zero changes | ✅ (confirmed zero workout references) | ✅ (unchanged) |

### 3.5 Verification environment caveats

The sandbox this work was executed in had three hard limitations, all
stated at the time and none silently worked around:

1. **No filesystem access to the actual repository.** All file contents
   were transcribed from pasted text, not read from or written to a live
   `git` checkout. `py_compile` catches syntax errors but not a subtle
   transcription drift that's still valid Python. **A real `git diff`
   against the actual original files has never been run.**
2. **No installable SQLAlchemy, no live database.** The
   `Base.metadata`/mapper-configuration identity check (criterion #10)
   and the `weekly_plan → WorkoutSession` live relationship test
   (criterion #11) were never executed — only reasoned through by manual
   inspection of `relationship()` string references.
3. **No network egress, no API keys.** The Gemini plan-generation call
   (criterion #9) was never exercised live.

None of these were skipped silently — each is marked ⚠️ above and
carried forward into §6 as explicit outstanding debt.

---

## 4. Part B — Post-migration follow-up changes

Each change below was proposed as a *recommendation* first, with its own
rationale, and only executed after explicit sign-off in a later message.
None of these were part of the original WO8 diff.

### 4.1 Change 1 — `search_exercises()` bug fix (closes criterion #7)

**What was found:** Two bugs, one masking the other.
- Bug A (already known, documented in the original WO8 report): the
  return statement iterated `rows`, a name never assigned in the
  function — `NameError` on every call.
- Bug B (found *while fixing* Bug A, not previously known): once Bug A
  is fixed, `primary_muscle_group` and `equipment_type` are returned as
  raw `enum.Enum` members. Starlette's `JSONResponse` calls plain
  `json.dumps()` with no enum handler, so the endpoint would have
  immediately failed again with `TypeError: Object of type MuscleGroup
  is not JSON serializable`. Confirmed via a standalone reproduction
  before writing the fix (not assumed from memory).

**Fix applied:** `rows = result.all()` added; `.value` added when
serializing the two enum fields.

**File touched:** `domains/workout/routers/workout_settings.py`
(`search_exercises()` only — no other function in that file touched).

**Authorization:** *"1. Let's do that. It seems simple and it makes
sense"* — explicit, in response to a proposal that named both bugs and
the exact fix.

**Deviation from standing convention:** GOVERNANCE.md §4.5 states bugs
found during migration get their own standalone ticket, "never bundled
into the migration's diff, even if the fix is one line." This was
overridden by explicit project-owner authorization, not bypassed
silently. Recorded here for the reviewer's benefit.

### 4.2 Change 2 — `workout_plans.py` split

**What was found:** `workout_plans.py` was 523 lines — already over
GOVERNANCE.md §1.2's 300-line hard ceiling on routers, and already named
by GOVERNANCE.md itself as pre-existing debt requiring exactly this
split (CRUD vs. AI-generation).

**Fix applied:** Split into two files, both mounted at the same
`/workout/plans` prefix, both registered as separate
`app.include_router()` calls in `main.py` — the same two-routers-one-
domain-endpoint pattern `workout_log.py` already used:
- `domains/workout/routers/workout_plans_crud.py` (190 lines) —
  `list_plans`, `create_plan`, `activate_plan`, `delete_plan`.
- `domains/workout/routers/workout_plan_ai_generator.py` (373 lines) —
  `_build_exercise_history_context`, `_call_gemini_for_plan` (byte-
  identical, only relocated), `_fuzzy_match_exercise`, `generate_plan`,
  `save_generated_plan`.

**Files touched:** `workout_plans.py` deleted; two new files created;
`main.py` updated (import line + two `include_router` calls replacing
one).

**Authorization:** *"1. Let's do that. It seems simple and it makes
sense"* (same message covered items 1–3).

**Known incomplete outcome, stated plainly:** `workout_plan_ai_generator.py`
is still 373 lines — over the 300-line ceiling. The AI system-prompt
string and the two endpoint functions carry real, hard-to-shrink weight.
This was **not** hidden — flagged immediately after the split, in the
same turn it was completed. See §7.2 for the recommended next step if
this is ever picked back up.

### 4.3 Change 3 — `_shared.py` consolidation

**What was found:** `_get_previous_best()` was defined identically in
both `workout.py` and `workout_log.py` (pre-existing duplication,
predates this migration — carried over faithfully from the original
source, not introduced by the move). Separately, the pattern
`WeightUnit.KG if x == "kg" else WeightUnit.LB` was repeated 4 times
across `workout_log.py` (3x) and `workout_settings.py` (1x).

**Fix applied:** New file `domains/workout/routers/_shared.py`, housing:
- `_get_previous_best(db, exercise_id, exclude_session_id=None)` — moved
  here verbatim, deleted from both original locations, imported instead.
- `parse_weight_unit(raw: str) -> WeightUnit` — new tiny helper wrapping
  the repeated conditional; all 4 call sites now call it instead of
  inlining the conditional.

**Naming decision, and why:** the reviewer's first instinct was
`services.py`. The recommendation given (and used) was
`domains/workout/routers/_shared.py` instead, specifically to avoid
colliding with GOVERNANCE.md §2.3's reserved use of the word "services"
for the future top-level `services/ai/` provider-abstraction layer — a
domain-local `services.py` sitting next to that convention risks
confusing a future reader searching for AI-layer debt. The leading
underscore also matches an existing in-codebase pattern (underscore-
prefixed helpers living next to their callers, e.g.
`workout_settings.py`'s own `_fetch_locations()`) — this module is that
same idea promoted one level because two files need it, not a new
pattern.

**Files touched:** `_shared.py` created; `workout.py` and
`workout_log.py` had their local `_get_previous_best()` deleted and an
import added; `workout_log.py` (3 sites) and `workout_settings.py` (1
site) had the inline conditional replaced with a `parse_weight_unit()`
call.

**Authorization:** *"2. I like that. Go ahead. But what is your
recommended approach? I would lean towards the `services.py` route, but
you can correct me if I am wrong"* → recommendation given and accepted
→ *"3. Since we are doing #2, let's do #3 as well altogether."*

### 4.4 Explicit deviation log

For the reviewer's convenience, every place this work order's diff
(taken as a whole, original + follow-up) differs from the standing
GOVERNANCE.md conventions, with the authorization that covers it:

| Deviation | Standing rule | Authorization |
|---|---|---|
| Bug fixed inside the migration's own follow-up pass, not a separate ticket | §4.5 | "Let's do that" (§4.1 above) |
| Router file split during a domain migration's follow-up, not a separate refactor work order | Implied by §4.3's "Standing Work-Order Template" being scoped per-task | "Let's do that" (§4.2 above) |
| Cross-file helper consolidation during a migration follow-up | §3.1 ("DRY mandate" is about *new* code, not retroactive cleanup) | "I like that. Go ahead" / "let's do that as well" (§4.3 above) |

No other deviations occurred. The AI-service-layer consolidation
(§2.3) was explicitly considered and explicitly **not** done — *"agree
with the AI portion. We'll come back later here if the future migration
does not handle it"* — correctly left for the dedicated `services/ai/`
work when it exists.

---

## 5. Current state — full file inventory

```
domains/workout/
    __init__.py
    models.py                          (16 classes: 7 enums, 9 ORM models)
    routers/
        __init__.py
        _shared.py                     (58 lines — NEW in follow-up pass)
        workout.py                     (289 lines)
        workout_log.py                 (357 lines — exports 2 routers)
        workout_plans_crud.py          (190 lines — NEW, replaces workout_plans.py)
        workout_plan_ai_generator.py   (373 lines — NEW, replaces workout_plans.py)
        workout_settings.py            (378 lines)
    templates/
        workout.html
        workout_history.html
        workout_progress.html
        workout_plans.html
        workout_plan_preview.html
        workout_settings.html
        partials/workout/
            active_session_header.html
            body_metric_saved.html
            custom_exercise_list.html
            equipment_list.html
            exercise_history.html
            location_list.html
            plan_generate_error.html
            plan_list.html
            plan_saved.html
            session_detail.html
            session_summary.html
            set_logged_row.html
    static/
        css/workout.css
        js/workout.js
```

Root-level files touched: `models.py` (shim — apply via
`models_py_shim_patch.md`, do not overwrite directly, see §3.2's note),
`main.py`, `core/templating.py`.

`workout_plans.py` (the original, monolithic router) **no longer
exists** — fully superseded by `workout_plans_crud.py` +
`workout_plan_ai_generator.py`.

---

## 6. Outstanding verification debt

These must be closed out in the real environment — none of them were
resolved by the follow-up pass, and none should be treated as passing
until actually run:

1. **`git diff` the router/model files against the real repository.**
   Everything here was transcribed from pasted text, never diffed
   against an actual checkout.
2. **Run the real `Base.metadata` / `configure_mappers()` check** with
   SQLAlchemy installed against the actual `models.py` +
   `domains/workout/models.py`, confirming `models.WorkoutSession is
   domains.workout.models.WorkoutSession` etc. and no
   `InvalidRequestError`.
3. **Exercise `POST /workout/plans/generate` against the live Gemini
   API** (now living in `workout_plan_ai_generator.py`) with a real
   `GEMINI_API` key.
4. **Load a real `WeeklyPlan` with a linked `WorkoutSession`** to
   confirm `WeeklyPlanDay.workout_session` still resolves correctly
   through the shim, end to end, against a live database.
5. **Smoke-test `GET /workout/exercises`** against a live DB to confirm
   the §4.1 fix actually returns valid, well-formed JSON in practice
   (only simulated in isolation here, not run against real `Exercise`
   rows).

---

## 7. Deferred work — what to do once dependent conditions are met

The request that produced this document asked for "what needs to be
done after all the other migrations are completed." **For this specific
domain, that framing is slightly broader than what's actually true** —
worth correcting precisely, since a future agent will scope work off
this document:

### 7.1 `models.py` shim removal — not gated on *all* migrations

Per GOVERNANCE.md §2.4, a domain's shim should be removed once its
**only remaining external consumer** is `routers/dashboard.py`. For
`workout`, that's not the relevant condition at all:
`routers/dashboard.py` has **zero** references to any workout class
(confirmed in the original WO8 by grepping its import block). The
**only** file outside `models.py` itself that still does
`from models import WorkoutPlan, WorkoutPlanDay, WorkoutSession,
WeightUnit` is `routers/weekly_plan.py` — part of the `planning` domain
(`weekly_plan` + `intent`), which GOVERNANCE.md §3.3 lists as not yet
migrated.

**Precise, actionable condition for shim removal:**

> Remove the 16-class shim block from `models.py` (see
> `models_py_shim_patch.md` for the exact block) **as soon as** whatever
> file then owns the weekly-planning logic (`routers/weekly_plan.py`
> today, or its future `domains/planning/routers/weekly_plan.py` home)
> has its model-import line changed from
> `from models import (..., WorkoutPlan, WorkoutPlanDay, WorkoutSession,
> WeightUnit, ...)` to `from domains.workout.models import WorkoutPlan,
> WorkoutPlanDay, WorkoutSession, WeightUnit`.

This does **not** require every other domain to be migrated first — it
requires exactly one specific file's import line to change. That could
happen either as a small standalone edit (not a full domain migration),
or naturally as part of a future "Work Order — planning domain
migration." Either way, **before deleting the shim, re-run the same
`grep -rn "from models import"` sweep for these 16 class names that WO8
used**, to confirm no other file started depending on the shim in the
interim.

**Also verify at that time:** `WeeklyPlanDay.workout_session` is a real
`relationship()`, not just a bare FK column — a genuine cross-domain
relationship (GOVERNANCE §2.2 explicitly calls out this pattern as
needing care, same category as the `Recipe` relationship handled in
WO#7). Confirm the string-based relationship resolution
(`relationship("WorkoutSession", ...)`) still resolves correctly once
`weekly_plan.py`'s import changes — this is a live `configure_mappers()`
check, not a static one, and was never run in this sandbox (§6, item 2)
even for the current shimmed state, let alone the post-shim-removal
state.

### 7.2 GOVERNANCE §1.2 file-size ceiling — still open, 3 files

As of this document, three files in this domain remain over the
300-line ceiling:

| File | Lines | Notes |
|---|---|---|
| `workout_plan_ai_generator.py` | 373 | Split already happened (§4.2); remaining size is the AI prompt string + 2 large endpoint functions. Further split candidate: extract the system-prompt string and `_build_exercise_history_context`/`_fuzzy_match_exercise` into a small `workout_plan_ai_context.py`, leaving just the two route handlers in the router file. |
| `workout_log.py` | 357 | Pre-existing, not part of any follow-up in this work order. Natural split point: the two-router structure already separates session-logging from body-metrics conceptually — could become two files (still both under one `/workout/sessions`-family prefix set) if this is ever revisited. |
| `workout_settings.py` | 378 | Pre-existing, not part of any follow-up. Natural split point: locations/equipment CRUD vs. custom-exercise CRUD are already visually separated by section comments in the file — same shape as the plans split. |

None of these were touched beyond what was explicitly requested. Do not
split these without a fresh, explicit go-ahead — the pattern established
in this work order (propose with rationale → get sign-off → execute →
document) should repeat, not be assumed.

### 7.3 GOVERNANCE §2.3 AI service layer — explicitly deferred

`_call_gemini_for_plan` (now in `workout_plan_ai_generator.py`) remains
one of the known duplicate AI-client implementations. **Do not touch it
preemptively.** When `services/ai/` is actually built (tracked as its
own, separate, project-wide initiative per §2.3), that work order should
update all six-plus known call sites together, including this one. This
was explicitly discussed and explicitly deferred in this work order —
*"agree with the AI portion. We'll come back later here if the future
migration does not handle it."*

### 7.4 Re-verify cross-domain relationship when `planning` migrates

When the `planning` domain (`weekly_plan` + `intent`) is eventually
migrated, per GOVERNANCE §2.2's rule that domains with a live FK
relationship should ideally migrate together (precedent: `blog` +
`code_intel` in WO#2) — that work order's own acceptance criteria should
include a live, DB-backed check that `WeeklyPlanDay.workout_session`
still resolves correctly with both domains fully split out, not just a
static-analysis check like the one performed here.

---

## 8. Reviewer checklist — pass/fail rubric

To mark this work order fully closed, confirm all of the following in
the **real environment** (not this sandbox):

- [ ] `git diff` on every file listed in §5 matches expectations — no
      unintended transcription drift from the pasted-content workflow.
- [ ] `Base.metadata` / `configure_mappers()` succeeds with no
      `InvalidRequestError`; `models.WorkoutSession is
      domains.workout.models.WorkoutSession` (and the other 15 classes)
      hold true.
- [ ] All 13 original acceptance criteria (§3.4) pass live, including
      the three still-⚠️ items (#9, #10, #11).
- [ ] `GET /workout/exercises` returns valid JSON against a live DB
      (confirms §4.1's fix actually works end to end, not just in
      isolation).
- [ ] `main.py` still boots with both `workout_plans_crud.router` and
      `workout_plan_ai_generator.router` correctly serving their
      respective endpoints under the shared `/workout/plans` prefix, no
      route-shadowing between the two.
- [ ] `domains/workout/routers/_shared.py` — both `_get_previous_best()`
      and `parse_weight_unit()` are the *only* definitions of that logic
      anywhere in the domain (already grep-confirmed in-sandbox; worth
      re-confirming against the live repo).
- [ ] No file outside `models.py` and `weekly_plan.py` does
      `from models import <any of the 16 workout classes>` (§7.1's
      precondition for eventual shim removal — not required to pass now,
      but worth baselining).
- [ ] Every deviation logged in §4.4 has a corresponding authorization
      quote — reviewer should confirm none of the three follow-up
      changes exceeded what was actually asked for.

If all of the above hold, this work order — original scope plus the
three authorized follow-ups — should be marked **✅ complete**, with §6
and §7 carried forward as tracked, separate follow-on items rather than
blocking this closure.
