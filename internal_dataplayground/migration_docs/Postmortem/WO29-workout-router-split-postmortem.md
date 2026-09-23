# Work Order #29 — Postmortem: Workout Domain Router Split

**Status:** ✅ Complete — Parts A, B, and C all executed. All acceptance
criteria passing (see §8).
**Work order source:** `migration_docs/Work Orders/work_order_29_workout_router_split.md`
**Domain touched:** `domains/workout/routers/` (+ one necessary `main.py` edit)
**Type:** Location-and-organization refactor only. No endpoint path, HTTP
method, request/response shape, template name, or schema changed anywhere
in this work order.

---

## 1. Purpose of This Document

This is the canonical record of what WO#29 actually did, as distinct from
what the work order *authorized up front*. The distinction matters because
Part C of WO#29 was explicitly **not** pre-authorized — the work order
text required stopping and getting explicit owner sign-off before touching
`workout_log.py`, because doing so meant reversing a HARD BOUNDARY set by
an earlier work order (WO#8). A reviewer checking this migration needs to
be able to see, without re-reading the whole chat transcript, exactly
where that authorization boundary was and how it was crossed.

Section 10 is a forward-looking checklist for whichever agent picks up
cleanup work after the rest of the router-split backlog (the "Track A,
Track B, Track C, Track D" work referenced in WO#29's own closing note)
is done. It is written to be actionable on its own — read it, check each
item, don't assume anything from earlier sections carries over silently.

---

## 2. What Was Authorized Up Front (Original Work Order Scope)

The work order's own **ROLE** was: *"senior refactoring engineer splitting
oversized router files by responsibility... no behavior change, no schema
change."* Its **HARD BOUNDARIES** pre-authorized exactly two splits and
explicitly withheld authorization for a third:

| Part | File(s) in scope | Pre-authorized to split? |
|---|---|---|
| A | `workout_settings.py` → + `workout_locations.py` | ✅ Yes, outright |
| B | `workout_plan_ai_generator.py` → + `workout_plan_prompts.py` | ✅ Yes, outright |
| C | `workout_log.py` → + (new file) | ❌ **No** — read-only investigation only, "stop and ask for explicit sign-off" |

Part C was structured differently in the work order on purpose: WO#8's own
HARD BOUNDARIES had said `workout_log.py`'s two-router-in-one-file
structure (`router` + `body_metrics_router`) "must not be merged or
further split" during that migration. WO#29 Part C's only sanctioned
action, absent further instruction, was to confirm the line count and
report — not to act.

---

## 3. Execution — Part A (`workout_settings.py`)

**Pre-split line count (substitute method — see §9):** 378 lines.

Moved the Locations & Equipment CRUD block into a new file
`domains/workout/routers/workout_locations.py`:
`_fetch_locations`, `_location_list_ctx`, `create_location`,
`update_location`, `delete_location`, `set_default_location`,
`add_equipment`, `update_equipment`, `delete_equipment`.

Kept in `workout_settings.py`: the settings page handler
(`workout_settings`) and Custom Exercises CRUD (`create_custom_exercise`,
`search_exercises`).

**Post-split line counts:** `workout_settings.py` = 155, `workout_locations.py` = 264.

**Necessary addition beyond the literal STEPS** (flagged at the time,
repeating here for the record): `workout_settings()`'s template context
still needs the active-locations list. Rather than duplicating the query,
`workout_settings.py` now does:

```python
from domains.workout.routers.workout_locations import _fetch_locations
```

This is a private (underscore-prefixed) helper imported across a router
file boundary — unusual, but it was the only way to avoid duplicating the
query while keeping `GET /workout/settings`'s response shape identical.
Both routers still share the `/workout` prefix and are registered
separately in `main.py` — the same two-routers-sharing-a-prefix pattern
already used elsewhere in this codebase (`workout_log.py`'s original
`router`/`body_metrics_router` split, and
`workout_plans_crud.py`/`workout_plan_ai_generator.py`'s shared
`/workout/plans` prefix).

Per Part A Step 4, the `search_exercises()` docstring documenting two
already-fixed historical bugs (`NameError` on `rows`, enum
JSON-serialization) was left untrimmed as historical record.

Also dropped, as a side effect of the split (not a separate fix — these
imports became genuinely unused once their only call sites moved out):
`Equipment`, `WorkoutLocation`, `WeightUnit`, `selectinload`,
`parse_weight_unit` from `workout_settings.py`'s import block.

---

## 4. Execution — Part B (`workout_plan_ai_generator.py`)

**Pre-split line count (substitute method):** 375 lines.

Moved `_build_exercise_history_context()` and the inline system-prompt /
prompt-template string construction into a new, non-router file
`domains/workout/routers/workout_plan_prompts.py`, exposed as:
- `PLAN_SYSTEM_INSTRUCTION` (module-level constant, was the inline `system = """..."""` string)
- `build_plan_prompt(target_days, goal, location, equipment_context, exercise_history, additional_notes, catalog_text)` (function, was the inline `prompt = f"""..."""` construction)

Kept in `workout_plan_ai_generator.py`: `generate_plan`,
`save_generated_plan`, `_fuzzy_match_exercise`, and the thin
`_call_gemini_for_plan` wrapper. **`_call_gemini_for_plan`'s routing
through `services.ai.call_gemini_json()` (WO#13) was explicitly not
touched**, per HARD BOUNDARIES.

**Post-split line counts:** `workout_plan_ai_generator.py` = 281,
`workout_plan_prompts.py` = 166.

**Verification performed beyond a visual diff:** the extracted
`PLAN_SYSTEM_INSTRUCTION` string and `build_plan_prompt(...)`'s output
were both checked programmatically against the original inline
construction (see §6) — confirmed byte-identical, including the
`location=None` and `additional_notes=""` branches.

Dropped as a side effect of the split: `from datetime import datetime`
and `text` (from `sqlalchemy`) — both were only used inside
`_build_exercise_history_context()`, which moved.

---

## 5. Part C — Not Pre-Authorized, Then Explicitly Authorized Mid-Engagement

This is the section a reviewer should check most carefully.

### 5.1 What happened during the initial WO#29 run

Per the work order's explicit instruction, `workout_log.py` was **not**
split during the initial pass. Instead:

1. Confirmed the real line count: 357 lines (substitute method).
2. Confirmed the file still exports two `APIRouter` instances from one
   module (`router` → `/workout/sessions/*`, `body_metrics_router` →
   `/workout/body-metrics/*`), and that the only clean split available —
   separating `body_metrics_router` into its own file — would directly
   reverse WO#8's HARD BOUNDARY against merging or further splitting that
   structure.
3. **Stopped.** Reported this situation and explicitly requested owner
   sign-off, without making any file changes to `workout_log.py`.

At the end of the initial run, `workout_log.py` was unchanged (357 lines,
still exporting both routers), and the report to the reviewer/owner ended
with an open request rather than a completed acceptance-criteria row for
Part C.

### 5.2 The authorization

In a follow-up message, the project owner replied:

> "sure, let's separate them. And provide the files"

This is being recorded verbatim because it is the entire authorization
trail for reversing WO#8's precedent. There was no other sign-off
mechanism (no ticket, no separate approval doc) — the decision was made
and communicated in-conversation, in direct response to the explicit
stop-and-ask step Part C required. A reviewer treating this migration as
"done per spec" should treat this message as the artifact that turned
Part C from "held pending sign-off" into "authorized" — it is the fact
that makes Part C's execution compliant with WO#29's own HARD BOUNDARIES,
not a deviation from them.

### 5.3 Execution, after sign-off

Split `workout_log.py` into:
- `workout_log.py` (kept) — `router` (`/workout/sessions/*`):
  `start_session`, `log_set`, `end_session`, `session_detail`,
  `delete_set`.
- `workout_body_metrics.py` (new) — `body_metrics_router`
  (`/workout/body-metrics/*`): `log_body_metric`, `get_body_metrics`.

Both routers are still registered as **separate**
`app.include_router()` calls in `main.py`, exactly as WO#29 Part C Step 4
specified for the sign-off-obtained case.

**Post-split line counts:** `workout_log.py` = 266, `workout_body_metrics.py` = 120.

Import cleanup performed as part of the split (verified by grepping each
name's usage count in the original file before moving anything — see
§6): `Form`, `desc`, `and_`, `WeightUnit`, `WorkoutPlanDay`, `Optional`
were all already-unused imports in the *original, unsplit*
`workout_log.py` (each appeared exactly once — only in its own import
line). None were carried into either resulting file. This is pre-existing
dead code discovered during the split, not introduced by it — noted here
rather than silently dropped, per the "don't fix, report" handling for
pre-existing issues, except that dropping an unused import has no
behavior effect and was treated the same way the Part A/B unused-import
cleanup was (a natural side effect of relocation, not a separate fix
needing its own ticket).

`parse_weight_unit` and `Decimal` were needed in **both** resulting files
(each is called from a handler in `router` and a handler in
`body_metrics_router`) and are imported in both.

### 5.4 Net effect on the WO#8 precedent

WO#8's rule ("this two-router-in-one-file structure must not be merged or
further split") is now **superseded for `workout_log.py` specifically**,
by explicit owner sign-off obtained through the exact mechanism WO#29
built for that purpose. It is **not** superseded as a general rule — any
other file in the codebase following the same two-router-one-file pattern
(if any exist) still requires its own sign-off before being split; this
postmortem does not establish a blanket precedent. See §10.7.

---

## 6. Verification Performed

- **Line counts:** `scripts/check_router_line_limits.py` was not available
  in this engagement's source material. Substituted `wc -l` on every
  touched/created file. This is flagged explicitly in §9 as a substitution,
  not treated as equivalent to running the real lint script.
- **Syntax:** `python3 -m py_compile` on all six router files;
  `ast.parse()` on `main.py` (couldn't `py_compile` it — FastAPI isn't
  installed in this container). All clean.
- **Endpoint completeness:** for every split, grepped both the original
  file and both resulting files for route decorators
  (`@router.<verb>(...)` / `@body_metrics_router.<verb>(...)`) and
  function defs, and confirmed the resulting set matches the original
  set exactly — no endpoint dropped, duplicated, or re-pathed.
- **Byte-identical prompt reconstruction (Part B):** wrote a standalone
  script that reconstructs the original inline `system`/`prompt` string
  logic and diffs it against `PLAN_SYSTEM_INSTRUCTION` /
  `build_plan_prompt(...)`'s actual output, across multiple input
  combinations (with/without `location`, with/without
  `additional_notes`). All matched exactly. (`sqlalchemy` isn't installed
  in this container either — stubbed the two names `workout_plan_prompts.py`
  imports from it, `text` and `AsyncSession`, to make the import
  succeed for this test only; this stub has no bearing on the shipped
  file's correctness, which imports the real package at runtime.)
- **Unused-import detection (Part C):** grepped whole-word occurrence
  counts of each imported name across the original `workout_log.py`
  before splitting, to distinguish "used once → unused" from "used
  elsewhere" rather than relying on memory/visual scan.
- **Cross-file dependency check:** confirmed `workout.py` and
  `workout_plans_crud.py` were never written to the working directory and
  their `main.py` registration lines are byte-for-byte unchanged — see the
  chat exchange immediately preceding this document for that specific
  confirmation.

**Not verified** (no environment available to do so — see §9):
actual `pytest`/runtime execution, a live FastAPI app boot, a real
database round-trip, or Jinja2 template rendering against the moved
context. See §9 and §10.5 for what a real CI run still needs to confirm.

---

## 7. Final File Inventory

| File | Status | Lines | Registered in `main.py`? |
|---|---|---|---|
| `domains/workout/routers/workout_settings.py` | Edited (378→155) | 155 | Yes, unchanged registration |
| `domains/workout/routers/workout_locations.py` | **Created** | 264 | Yes, **newly added** |
| `domains/workout/routers/workout_plan_ai_generator.py` | Edited (375→281) | 281 | Yes, unchanged registration |
| `domains/workout/routers/workout_plan_prompts.py` | **Created** (not a router — nothing to register) | 166 | N/A |
| `domains/workout/routers/workout_log.py` | Edited (357→266) | 266 | Yes, unchanged registration |
| `domains/workout/routers/workout_body_metrics.py` | **Created** | 120 | Yes, **newly added** (replaces old `workout_log.body_metrics_router` reference) |
| `main.py` | Edited | 151 | — |
| `domains/workout/routers/workout.py` | **Not touched** | — | Yes, unchanged |
| `domains/workout/routers/workout_plans_crud.py` | **Not touched** | — | Yes, unchanged |
| `domains/workout/routers/_shared.py` | **Not touched** | — | N/A (helper module, not a router) |
| `domains/workout/models.py` | **Not touched** | — | N/A |

All six router-owning files are now under the 300-line ceiling.

---

## 8. Acceptance Criteria — Final Status

**Part A**
- [✅] Both resulting files under 300 lines (155, 264).
- [✅] Every endpoint reachable at its original path — verified by route-decorator diff.
- [✅] `GET /workout/exercises` pre-existing bug — confirmed **already fixed** in the code (not just documented as fixed); criterion updated accordingly rather than assuming the historical bug report was still accurate.

**Part B**
- [✅] Both resulting files under 300 lines (281, 166).
- [✅] `POST /workout/plans/generate` and `POST /workout/plans/{id}/save` unchanged in path/method/response shape.
- [✅] `_call_gemini_for_plan`'s WO#13 routing untouched.
- [✅] Prompt/system-string content verified byte-identical pre/post-split.

**Part C**
- [✅] Held pending sign-off during the initial run, as required.
- [✅] Sign-off obtained (§5.2) before any file was touched.
- [✅] Both resulting files under 300 lines (266, 120), both routers still registered separately in `main.py`.

**Cross-cutting**
- [✅] Every workout endpoint (session start/log-set/end/detail/delete-set, body-metrics log/get, locations/equipment CRUD, custom-exercise create/search, plan generate/save) reachable at its original path with identical behavior, per the verification in §6.

---

## 9. Known Caveats — Read Before Treating This as Fully CI-Verified

1. **Line-count method is a substitute.** `scripts/check_router_line_limits.py`
   was not available in this engagement. All "under 300 lines" claims in
   this document are `wc -l` counts, not the real lint script's output. If
   the real script counts differently (e.g. excludes blank lines,
   docstrings, or comments), **re-run it for real** before treating any
   line-count acceptance criterion as authoritative.
2. **No live execution.** Nothing in this work order was run against an
   actual FastAPI app, database, or the Jinja2 templates the handlers
   return. `py_compile`/`ast.parse` catch syntax errors, not import-time
   or run-time errors (e.g. a circular import between
   `workout_settings.py` and `workout_locations.py` would not be caught
   by `py_compile` run file-by-file the way it was here). A real
   `python -c "import main"` (or equivalent app-boot smoke test) has not
   been performed.
3. **No dependency install available.** `fastapi` and `sqlalchemy` were
   not installed in the working container (no network egress). Verification
   that depended on their real behavior (§6) used stubs for the modules'
   *names*, not their real implementations — sufficient to prove the
   string-construction logic is preserved, not sufficient to prove the
   files actually import cleanly end-to-end in the real project
   environment.

---

## 10. Follow-Up Required — Once the Rest of the Router-Split Backlog Is Complete

WO#29's own closing note says Parts A and B "can run fully in parallel
with every other Track C work order, Track A, Track B, and Track D" —
meaning this is one of several similar router-split work orders expected
to land across other domains. The items below should be checked **after**
that broader backlog (not just WO#29) is done, since several of them only
make sense to evaluate once the full picture of what moved where is
known. This section is written so an agent can work through it as a
checklist without needing this postmortem's earlier sections as context.

### 10.1 `models.py` — confirm scope, don't assume work is needed here

This work order was a **router split only**. Nothing in Parts A, B, or C
touched `domains/workout/models.py`, and nothing about a router-file
split has any mechanical reason to require a models.py change — routers
import from models, not the reverse, and no model class, relationship, or
enum was added, removed, or renamed.

That said, per `migration_docs/GOVERNANCE.md` §2.4 ("Legacy Import
Shims"), there is a **separate, unrelated** class of models.py cleanup
that this repo has historically tracked: once a domain's only remaining
external consumer of its models is `routers/dashboard.py` (the one
sanctioned cross-domain reader), any legacy shim re-exporting that
domain's classes from a root `models.py` should be deleted and
`dashboard.py` updated to import directly from `domains.<name>.models`.

**Action for the next agent:** do not assume this applies to `workout` —
per GOVERNANCE.md §2.4's own "Status: historical/closed" note, WO#20
already removed every remaining domain shim from root `models.py`, and
WO#22 removed root `models.py` entirely once nothing still depended on
it. **If a root `models.py` file exists in the actual repo when you read
this, that contradicts GOVERNANCE.md's own closed status and should be
treated as a documentation/reality mismatch worth flagging, not as
"WO#29 must have missed something."** Concretely:
- Check whether `models.py` (root level, not `domains/workout/models.py`)
  exists at all. If it doesn't, §2.4 is accurate and there's nothing to
  do for `workout` here.
- If it does exist and still contains a workout-related shim, that's a
  gap between GOVERNANCE.md's claimed status and the actual repo — file
  it as its own cleanup item, separate from any router-split work order,
  per GOVERNANCE.md §4.5 ("Bugs found during migration are not migration
  work").
- Separately, check `routers/dashboard.py` for any import of
  `domains.workout.models` and confirm it imports directly from there
  (not through a shim) — this file was never in scope for WO#29 and was
  not read or modified during this work order.

### 10.2 `migration_docs/GOVERNANCE.md` itself is now stale in two places

Discovered during this work order, not fixed (out of scope — flagging per
§4.5's "bugs/staleness found during migration get their own ticket"
principle):

- **§1.2** lists `workout_plans.py` as one of the files that "exceeded it
  \[300 lines] at time of writing." That file no longer exists under that
  name (it was already split into `workout_plans_crud.py` +
  `workout_plan_ai_generator.py` before WO#29 began, per that file's own
  module docstring). §1.2 was written before that split and was never
  updated. WO#29 adds three more now-resolved oversized files
  (`workout_settings.py`, `workout_plan_ai_generator.py`,
  `workout_log.py`) to the same kind of staleness. Recommend a single
  cleanup pass across §1.2 once the full router-split backlog lands,
  rather than editing it once per work order.
- **§3.3** ("Migration Debt Tracker") lists `workout` among domains "not
  yet moved into the `domains/` structure." This directly contradicts the
  existence of `domains/workout/models.py`, `domains/workout/routers/`,
  `domains/workout/templates/`, and `domains/workout/static/` — i.e., the
  `workout` domain migration (WO#8) is long complete. §3.3 was not
  updated when WO#8 shipped. This is a documentation bug, not a code bug,
  but it will actively mislead an agent who trusts §3.3 over the actual
  filesystem. **Recommend fixing §3.3 before or alongside whatever work
  order next touches GOVERNANCE.md** — an agent should trust the
  filesystem over this section until it's corrected.

### 10.3 `scripts/check_router_line_limits.py`

Not available during WO#29 — all line-count verification in this
document used `wc -l` as a substitute (§9.1). Two follow-up needs:
- Run the real script against every file this work order touched, to
  confirm the `wc -l` substitute numbers agree with it.
- If the script maintains a hardcoded file list (rather than globbing
  `domains/*/routers/*.py`), it needs `workout_locations.py`,
  `workout_plan_prompts.py`, and `workout_body_metrics.py` added — a
  globbing script would pick these up automatically and needs no change.

### 10.4 `main.py` merge coordination across parallel work orders

WO#29 edited `main.py` in three places: the `domains.workout.routers`
import line, and two `app.include_router(...)` insertions (one for
`workout_locations.router`, one replacing
`workout_log.body_metrics_router` with
`workout_body_metrics.body_metrics_router`). If other domains'
router-split work orders (the "Track A/B/C/D" backlog referenced in
WO#29's closing note) are landing in parallel, each one likely needs a
similar `main.py` edit for its own domain. **These edits will conflict at
the text level if applied out of order or without rebasing** — this is a
real merge-coordination risk, not just a theoretical one, since
`main.py`'s import block and include-router block are both single
contiguous regions every domain's split touches. Recommend serializing
`main.py` edits across parallel work orders (one lands, next rebases)
rather than assuming they'll merge cleanly.

### 10.5 Test suite

No test suite was provided as source material for this engagement, so
none was checked or updated. **If a real test suite exists in the actual
repo**, grep it for any import of:
- `workout_settings._fetch_locations` / `workout_settings._location_list_ctx` → now in `workout_locations`
- `workout_settings.create_location` / `update_location` / `delete_location` / `set_default_location` / `add_equipment` / `update_equipment` / `delete_equipment` → now in `workout_locations`
- `workout_plan_ai_generator._build_exercise_history_context` → now in `workout_plan_prompts`
- any test asserting on the literal inline `system`/`prompt` string construction in `workout_plan_ai_generator.py` → now built via `workout_plan_prompts.PLAN_SYSTEM_INSTRUCTION` / `build_plan_prompt(...)`
- `workout_log.body_metrics_router` / `workout_log.log_body_metric` / `workout_log.get_body_metrics` → now in `workout_body_metrics`

Update these import paths before assuming the test suite still passes.

### 10.6 Cross-domain / dashboard impact

`routers/dashboard.py` is the one sanctioned cross-domain reader
(GOVERNANCE.md §2.2) and was never read during WO#29. If it imports
anything from `domains.workout.routers.*` directly (as opposed to
`domains.workout.models`, which would be the expected and sanctioned
pattern), that import path needs the same updates listed in §10.5. Router
files are not supposed to be imported cross-domain per GOVERNANCE.md
§2.2's spirit (routers are mounted via `main.py`, not imported for their
logic elsewhere), so this is expected to be a non-issue — but it was not
verified, since `dashboard.py`'s contents were not part of this
engagement's source material.

### 10.7 The WO#8 precedent reversal is scoped to `workout_log.py` only

Document this explicitly wherever the project tracks precedent decisions
(GOVERNANCE.md would be the natural home, e.g. a new §2.7 or an addendum
to §1.2): **the sign-off obtained in §5.2 authorizes splitting
`workout_log.py`'s two-router-in-one-file structure specifically. It does
not establish a general rule that two-router-in-one-file modules are now
freely splittable.** If any other domain has an analogous structure
(check for other modules exporting more than one `APIRouter` instance
before assuming this is unique to workout), splitting it still requires
its own explicit sign-off, obtained the same way — stop, report the
situation, wait for an explicit answer. Don't cite WO#29 Part C as
blanket precedent.

### 10.8 Repo-wide router audit, once all Track A/B/C/D work is done

Once every parallel router-split work order in this backlog has landed,
run one consolidated pass (not per-domain) to confirm:
- No router file across any domain still exceeds 300 lines.
- Naming conventions from GOVERNANCE.md §1.1 were followed consistently
  (`snake_case.py`, named after the sub-feature it owns) — several
  independently-run work orders picking their own new-file names creates
  drift risk.
- `main.py`'s import and include-router blocks are still internally
  consistent (no duplicate registrations, no stale references to
  now-removed router modules) after all the individual edits from §10.4
  have landed.
