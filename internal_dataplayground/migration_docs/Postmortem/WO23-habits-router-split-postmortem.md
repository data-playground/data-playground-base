# Work Order #23 — Postmortem: Habits Router Split

**Work order:** `migration_docs/Work Orders/work_order_23_habits_router_split.md`
**Suggested location of this file:** `migration_docs/Work Orders/work_order_23_habits_router_split_postmortem.md`
**Date:** 2026-09-19
**Executor:** Claude, in a chat session. **Not** a repo-attached coding agent.
It had no access to the repo, git, a real database, the real templates, or
`scripts/check_router_line_limits.py`. It worked from the pasted source of
eight files: `habits.py`, `domains/habits/models.py`, `core/base_model.py`,
`core/templating.py`, `main.py`, WO#23, `GOVERNANCE.md`, and
`00_MASTER_INDEX.md`. Every "verified" claim below is qualified by that
limit — see §3.5 and §9.

**Who reads what**
- **Reviewer deciding pass/fail on WO#23:** §1, §2, §3, §4, §8.
- **Agent doing the post-migration cleanup:** §5, §6, §7 (the requirements), §9.

---

## 1. Executive summary

WO#23 asked for a pure location-and-organization refactor: split the oversized
`domains/habits/routers/habits.py` (~520–550 lines by hand count of the pasted
source; the lint script was not run) into three files, with no behavior
change. That was done:

| File | Role | Lines (split-only) | Lines (final) |
|---|---|---|---|
| `habits.py` | Daily check-in routes (5) | 264 | 270 |
| `habits_settings.py` (new) | Settings / CRUD routes (6) | 211 | 214 |
| `_shared.py` (new) | 7 non-route helpers | 141 | 141 |

After the split was delivered, the project owner agreed five follow-up changes
in the same conversation. They are **not** part of the original WO and are
recorded separately in §4 so the reviewer can evaluate the migration and the
amendments independently:

1. Delete unused imports (`Optional`, `text`).
2. "Done" count uses only active habits (behavior change).
3. Remove the ignored `_get_week_dates(start_on_sunday)` parameter.
4. `create_habit` places new habits at the bottom via `max(sort_order) + 1`
   (behavior change).
5. Remove the obsolete migration NOTE from the `HabitLog` docstring in
   `models.py` (**specified as a diff only — not applied by the executor**).

The split also surfaced one constraint the WO text did not anticipate: a
**cross-router registration-order dependency** (§3.7). It affects `main.py`
and is directly relevant to WO#24–30.

**Recommended reviewer verdict:** pass the split (§3) as ✅ with ⚠️ on the two
functional-equivalence criteria, because they were verified in a substitute
environment. Evaluate §4 as owner-authorized amendments, not as part of the
"pure relocation" pass criteria. See §8 for the checklist.

---

## 2. Baseline — what WO#23 required

**Scope:** `habits.py` (split); new `_shared.py`; new `habits_settings.py`;
`main.py`; `core/templating.py` "only if a new router needs template access".

**Hard boundaries:**
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- Only touch `habits.py`, files created under `domains/habits/routers/`, and
  the registration lines in `main.py`.
- Stop and report if the file is already comfortably under 300 lines.

**Steps:** (1) run the lint script, confirm >300 lines; (2) move
`_get_grace_period`, `_get_logged_dates_for_habit`, `_calculate_streak`,
`_build_habit_view`, `_get_today_logged_ids`, `_sort_habits_for_display`,
`_get_week_dates` into `_shared.py`; (3) keep the five check-in routes in
`habits.py` and move the six settings/CRUD routes to `habits_settings.py`;
(4) both import from `_shared.py`; (5) register the new router in `main.py`.

**Acceptance criteria:** both files < 300 lines; every route enumerated with
zero path collisions; check-in routes work identically; settings/CRUD routes
work identically; `Base.metadata`/mapper identity unaffected.

---

## 3. What was executed for the original WO (the split)

### 3.1 Files

**Created**
- `domains/habits/routers/_shared.py`
- `domains/habits/routers/habits_settings.py`

**Edited**
- `domains/habits/routers/habits.py` — trimmed to the five check-in routes.
- `main.py` — **specified, not delivered as a file** (two lines; see below).

**Not edited:** `core/templating.py`, any model, any template, any static file.

**Moved (git sense):** none. The code was **retyped from the pasted source**,
not `git mv`'d. Consequences: `habits.py` keeps its path and history;
`_shared.py` and `habits_settings.py` are new files with no blame lineage back
to `habits.py`.

**`main.py` edit to apply**
```diff
-from domains.habits.routers import habits # WO1
+from domains.habits.routers import habits, habits_settings # WO1, WO#23
 ...
 app.include_router(habits.router)
+app.include_router(habits_settings.router)  # WO#23 — must stay AFTER habits.router
```
The order is load-bearing (§3.7).

### 3.2 Where every function went

| Function | Kind | Destination |
|---|---|---|
| `_get_grace_period` | async DB helper | `_shared.py` |
| `_get_logged_dates_for_habit` | async DB helper | `_shared.py` |
| `_calculate_streak` | pure | `_shared.py` |
| `_build_habit_view` | async DB helper | `_shared.py` |
| `_get_today_logged_ids` | async DB helper | `_shared.py` |
| `_sort_habits_for_display` | pure | `_shared.py` |
| `_get_week_dates` | pure | `_shared.py` |
| `habits_page`, `log_habit`, `unlog_habit`, `habit_progress`, `habit_heatmap` | routes | `habits.py` |
| `habits_settings`, `create_habit`, `reorder_habits`, `update_habit`, `deactivate_habit`, `update_grace_period` | routes | `habits_settings.py` |

Note: the WO describes the helpers as "pure-calculation helpers". Only 3 of the
7 are pure; the other 4 perform DB access. This did not change what was moved,
only the accuracy of the description.

`habits_settings.py` uses only one shared helper (`_get_grace_period`).

### 3.3 Route table (all 11, both files)

| Method | Path | Handler | File |
|---|---|---|---|
| GET | `/habits` | `habits_page` | habits.py |
| POST | `/habits/log` | `log_habit` | habits.py |
| DELETE | `/habits/log` | `unlog_habit` | habits.py |
| GET | `/habits/progress` | `habit_progress` | habits.py |
| GET | `/habits/heatmap/{habit_id}` | `habit_heatmap` | habits.py |
| GET | `/habits/settings` | `habits_settings` | habits_settings.py |
| POST | `/habits/new` | `create_habit` | habits_settings.py |
| PATCH | `/habits/reorder` | `reorder_habits` | habits_settings.py |
| PATCH | `/habits/{habit_id}` | `update_habit` | habits_settings.py |
| DELETE | `/habits/{habit_id}` | `deactivate_habit` | habits_settings.py |
| PATCH | `/habits/settings/grace-period` | `update_grace_period` | habits_settings.py |

Result: 11 routes, the set matches the original exactly, zero duplicate
`(method, path)` pairs. Both routers use `APIRouter(prefix="/habits",
tags=["Habits"])`. The original definition order was preserved inside
`habits_settings.py`.

### 3.4 Differences from a literal move (reviewer: expect these in the diff)

- `HabitSettings` import dropped from `habits.py` (the move made it unused).
- Module docstrings rewritten: `habits.py` now lists its own five endpoints
  (the original docstring omitted `GET /habits/progress`); the "Streak
  algorithm" and "Sort order" prose moved into `_shared.py`'s docstring; both
  route files carry a ROUTE-ORDER CONSTRAINT note.
- A section divider was added above `habit_progress` (it had none).
- Trailing whitespace on blank lines inside `habit_progress` was removed.
- `core/templating.py` was **not** edited even though the WO lists it in scope
  ("it will" need it). Routers import the shared `templates` instance, and
  `domains/habits/templates` was already in the `ChoiceLoader`.
- Use `git diff --color-moved=dimmed-zebra` to confirm no other drift; the code
  was retyped, not moved by git.

### 3.5 Verification performed, and its limits

**Substitute environment (per GOVERNANCE §4.3 — the closest achievable check):**
- FastAPI 0.115.6 / Starlette 0.41.3 (pinned — see §6), SQLAlchemy 2.0.54,
  in-memory SQLite via aiosqlite with a static pool.
- The **real** three router files, unmodified.
- A **stub** `models.py` with the same tables, columns, and the
  `uq_habit_logs_habit_date` constraint. One harness-only deviation:
  `HabitLog.id` uses an Integer variant on SQLite (BigInteger doesn't
  autoincrement there).
- **Stub** templates that print their context values. The real templates were
  never seen.
- `py_compile` and `pyflakes` on the three files.

**Results (split-only state):**
- All 11 routes returned expected responses: create ×2, log, idempotent
  re-log, progress (1/2 → 0/2 after unlog), page, heatmap (365 entries, last
  entry today with count 1), unlog, settings page, reorder, update, grace
  period, deactivate, and a 404 for a missing habit.
- **Negative control:** with `habits_settings.router` registered before
  `habits.router`, `DELETE /habits/log` returned **422** (`int_parsing` on
  `habit_id`). Correct order → 200.
- pyflakes: only the two pre-existing unused imports (later removed, §4).

**Not verified:** anything involving the real DB, the real templates
(`habits.html`, `habits_settings.html`, `partials/habit_card.html`,
`habit_progress.html`, `habit_settings_list.html`, `habit_settings_row.html`),
HTMX behavior in a browser, the real lint script, or a real `main.py` boot.
Template context keys were preserved exactly from the original handlers, so
template breakage is unlikely but was not confirmed.

### 3.6 Acceptance criteria (split-only state)

| Criterion | Result | Note |
|---|---|---|
| Both files < 300 lines | ✅ | 264 and 211 (final: 270 and 214). Real lint script not run — rerun it. |
| Every route enumerated, zero path collisions | ⚠️ | Zero *exact* collisions ✅, but there is a *shadowing* dependency (§3.7) the WO text did not anticipate. Safe only with the right include order. |
| Check-in routes work identically | ⚠️ | Verified in the substitute harness, not the real app. |
| Settings/CRUD routes work identically | ⚠️ | Same. |
| `Base.metadata` / mapper identity unaffected | ✅ | No model touched by the split; `_shared.py` imports the same three classes from `domains.habits.models`. |

### 3.7 New constraint discovered: cross-router registration order

WO#23 says the paths are "safe since none of these paths collide". That holds
for exact paths but misses **shadowing by the parametric `/{habit_id}`
routes**:

- `DELETE /habits/log` (habits.py) vs `DELETE /habits/{habit_id}`
  (habits_settings.py): both match `/habits/log`.
- `PATCH /habits/reorder` vs `PATCH /habits/{habit_id}`: both match
  `/habits/reorder`. This one is inside a single file and safe as long as
  `reorder` stays defined above `{habit_id}`.
- Starlette picks the first matching route and does **not** fall through when
  the int conversion later fails, so the loser doesn't get a second chance.
- Before the split, definition order inside one file guaranteed correctness.
  After it, correctness depends on `main.py` including `habits.router` before
  `habits_settings.router`.
- `PATCH /habits/settings/grace-period` is two segments deep and cannot be
  matched by `/{habit_id}`.

The constraint is documented in both files' docstrings. There is no automated
guard yet (§7.6).

---

## 4. Amendments agreed after the initial migration

These were requested by the project owner in the same conversation on
2026-09-19, after the split was delivered. They are **outside the original
WO's scope**. Under GOVERNANCE §4.5/§4.6 they should be committed
**separately** from the split, so the split remains reviewable as a pure
relocation and each fix is independently revertable.

Boundary check: items 1–4 touch only `habits.py`, `_shared.py`, and
`habits_settings.py` — the same files WO#23 was allowed to touch — but items 2
and 4 change behavior, which WO#23's "no behavior change" rule would
otherwise forbid. Item 5 touches `models.py`, which is outside WO#23's file
boundary entirely.

| # | Amendment | Files | Behavior change? | Delivered as | Status |
|---|---|---|---|---|---|
| 1 | Delete unused imports | habits.py | No | file | ✅ done, pyflakes clean |
| 2 | "Done" count = active habits only | habits.py | **Yes** | file | ✅ done, tested |
| 3 | Remove ignored `_get_week_dates` parameter | _shared.py | No (no caller passed it) | file | ✅ done |
| 4 | New habit goes to the bottom (`max + 1`) | habits_settings.py | **Yes** | file | ✅ done, tested |
| 5 | Remove obsolete NOTE from `HabitLog` docstring | models.py | No (comment only) | diff only | ⏳ **not applied** |

### 4.1 Unused imports
- **Removed:** `from typing import Optional`; `text` from
  `from sqlalchemy import select, delete, text, func`.
- **Left in place:** `log = logging.getLogger(__name__)` and `import logging`.
  This is a variable, not an import, and nothing uses it. Owner has not
  decided; see §6.

### 4.2 "Done" count uses only active habits
**Problem (reproduced on the delivered split before the fix):** with habits
A, B, C active, logging all three today and then deactivating A made
`GET /habits/progress` report 3/3 when only 2 of the 3 active habits were done,
because logs belonging to deactivated habits were counted.

**Fix in `habit_progress`:**
```python
done_result = await db.execute(
    select(func.count(HabitLog.id))
    .join(Habit, Habit.id == HabitLog.habit_id)
    .where(HabitLog.logged_date == today)
    .where(Habit.is_active == True)
)
```
**Fix in `habits_page`:**
```python
# was: completed_today = len(today_logged_ids)
completed_today = sum(1 for v in habit_views if v["today_logged"])
```
`_get_today_logged_ids` is unchanged, so `today_logged` on each card is
unaffected. **Verified:** the same scenario now shows 2/3 on both the partial
and the page.

### 4.3 `_get_week_dates` parameter
`def _get_week_dates(start_on_sunday: bool = True)` → `def _get_week_dates()`.
The parameter was never read and the only caller passes nothing. Removed rather
than implemented because the docstring and week grid are Sunday-first; a
Monday-first week should come back as a real setting, not a dead boolean.
Grep confirmed no remaining `start_on_sunday` references in the three files.
(Other files were not searched — **the cleanup agent should grep the whole
repo**, §7.5.)

### 4.4 New habit placed at the bottom
**Problem (reproduced):** `sort_order = (count of active habits) + 1` collides
once anything is deactivated. A/B/C = 1/2/3; deactivate A; add D → D got 3,
tying with C.

**Fix (`create_habit`):**
```python
max_result = await db.execute(
    select(func.coalesce(func.max(Habit.sort_order), 0))
)
next_sort_order = max_result.scalar_one() + 1
...
sort_order=next_sort_order,
```
and `from sqlalchemy import select, func` in `habits_settings.py`.

**Design decisions:**
- The max spans **all** habits including inactive ones, so a later
  reactivation can never tie with a newer habit.
- The first habit still gets `sort_order` 1 (empty table → `coalesce` → 0 + 1),
  matching the old behavior.
- It no longer loads every active row just to count them.
- **Verified:** the same scenario now gives D = 4.
- **Not repaired:** existing tied `sort_order` values already in a database.
  Saving the order once from the settings page rewrites them.

### 4.5 `HabitLog` docstring (owner: "this was done — remove the docstring")
The owner confirmed the unique-constraint migration has been applied, so the
warning in the docstring is obsolete. Interpretation applied: remove only the
**NOTE** paragraph and keep the rest of the docstring (it still explains the
constraint and that `log_habit()` relies on it for idempotency). If the owner
meant the whole docstring, that is a one-line follow-up.

```diff
     assumption living only in this comment. The router's INSERT path
     (`log_habit()` in domains/habits/routers/habits.py) relies on this
     constraint firing an IntegrityError on a duplicate same-day log, which
     it then treats as idempotent (already logged = success).
-
-    NOTE: prior to this declaration, this constraint was assumed to exist
-    "at the DB level" without being represented in the ORM model at all —
-    i.e. `Base.metadata.create_all()` (used by tests / fresh dev DBs) never
-    actually created it, silently allowing duplicate rows in any DB built
-    from the models rather than from the real migration history. If your
-    production DB was provisioned via Alembic and does NOT already have a
-    matching unique index, a migration adding
-    `uq_habit_logs_habit_date` needs to be generated and applied — this
-    model change alone does not touch a live database.
     """
```
**Status: not applied.** The executor never had the real `models.py`. Someone
with repo access must apply it (or fold it into §7).

### 4.6 How a reviewer separates the split from the amendments
The delivered router files are the **final** state (split + items 1–4).
The pure split state is: the delivered files minus these four hunks — the two
removed import lines (habits.py), the `habits_page`/`habit_progress` count
changes (habits.py), the `_get_week_dates` signature (_shared.py), and the
`create_habit` sort-order block plus its `func` import (habits_settings.py).
Line counts differ by +6 (habits.py), +3 (habits_settings.py), 0 (_shared.py).
Preferred: commit the split first, then commit items 1–4 as their own commit,
and review each diff on its own.

---

## 5. Final state after WO#23 + amendments

**Routers** (`domains/habits/routers/`)
- `_shared.py`, 141 lines: 7 helpers, no `@router` decorators, imports `Habit`,
  `HabitLog`, `HabitSettings` from `domains.habits.models`.
- `habits.py`, 270 lines: 5 routes; imports `_build_habit_view`,
  `_get_grace_period`, `_get_logged_dates_for_habit`, `_get_today_logged_ids`,
  `_get_week_dates`, `_sort_habits_for_display` from `_shared`.
- `habits_settings.py`, 214 lines: 6 routes; imports only `_get_grace_period`
  from `_shared`.

**Dependency facts the cleanup agent can rely on:**
- Neither router imports from a root `models` module. Both import
  `domains.habits.models` directly.
- No router reads or writes `Base` directly.
- `core/templating.py` and `core/base_model.py` were not touched.

**Route behavior differences vs. the original, in total:** only the two
intentional ones in §4.2 (progress/page "done" count) and §4.4 (new-habit
`sort_order`). Everything else is unchanged.

---

## 6. Disposition of every issue raised during the work

| Issue | Disposition |
|---|---|
| Unused `Optional`, `text` imports | Fixed (§4.1) |
| Unused `log` logger / `import logging` | **Open — owner decision.** Left in place; delete if wanted. |
| "Done" count included deactivated habits | Fixed (§4.2) |
| `_get_week_dates` ignored its parameter | Fixed (§4.3) |
| `sort_order` collisions on create | Fixed for new habits (§4.4); existing ties unrepaired |
| Unique-constraint migration for `HabitLog` | Owner says already applied — closed; docstring NOTE removal pending (§4.5) |
| Cross-router route-order dependency | Documented in docstrings; **no automated guard** (§7.6) |
| Hardening via `{habit_id:int}` | Not done — changes matching semantics; a candidate ticket (§7.6) |
| Starlette ≥ 1.x drops the legacy `TemplateResponse("x.html", {"request": …})` call style used across the codebase | **Open — verify the lockfile.** The harness needed `starlette==0.41.3`. |
| WO text calls the helpers "pure-calculation" | Cosmetic inaccuracy (§3.2) |
| Pydantic schemas in `domains/habits/models.py` (`HabitCreate`, `HabitUpdate`, `HabitResponse`, `HabitLogResponse`) are unused by the router | Pre-existing, documented in that file. Out of scope; no action. |

---

## 7. Requirements after all other migrations are complete

This section is for the agent that performs the post-migration cleanup. It
adds **habits-specific** requirements on top of whatever WO#20/WO#22 already
define; where they conflict, WO#22 Task 1's end-state for root `models.py`
takes precedence. Per GOVERNANCE §4.5/§4.6, do this as its own work order and
its own commits — not bundled with any router split.

### 7.0 Preconditions and sequencing
- Run **after** WO#22 (root `models.py` end-state, DAG headers,
  `configure_mappers()` check) and after the other Track C splits (WO#24–30)
  land, so `main.py` and the docs are edited once against their final shape.
- **Merge-conflict warning:** the Master Index calls Track C "parallel-safe —
  no shared files", but WO#23 lists `main.py` in its own scope, and the
  other router splits very likely add registration lines to it too. Expect
  textual conflicts in the `app.include_router(...)` block. Rebase
  sequentially and re-check include order after each (§7.6).

### 7.1 Step 0 — establish the real state of root `models.py` first
The sources the executor saw **disagree**, and the executor never saw root
`models.py`, `database.py`, or `routers/dashboard.py`:

| Source | What it says |
|---|---|
| `GOVERNANCE.md` §2.4 | Status "historical/closed": WO#20 removed every domain shim; WO#22 removed root `models.py` itself; mapper registration now lives in an import block in `database.py`. |
| `00_MASTER_INDEX.md` | WO#22 is "Drafted, not executed"; the `models.py` end-state is still described as pending. |
| `core/base_model.py` docstring | Says the top-level `models.py` "still has its own `Base = ...` re-export for backward compatibility". |
| `domains/habits/models.py` header | Says the code was "moved verbatim from models.py". |
| `main.py` (pasted) | Imports nothing from a root `models` module. Only `from database import init_db`. |

`main.py` also already registers `nba`, `soccer`, and `medium` routers and
templating already lists their template dirs, while the Master Index marks
WO#33–35 as not started — further evidence the Index is stale. **Read the
actual files; do not trust the Index or GOVERNANCE for current state.**

Run and record:
```bash
ls models.py 2>&1
grep -rnE "^(from|import) models\b|from models import" --include=*.py .
grep -rnE "\b(Habit|HabitLog|HabitSettings|HabitCreate|HabitUpdate|HabitResponse|HabitLogResponse)\b" \
  --include=*.py . | grep -v "^./domains/habits/"
```
Then branch:

- **Case A — root `models.py` is gone (GOVERNANCE is right).** No shim work.
  Go to §7.3 and §7.4 for the stale comments.
- **Case B — it still exists (Index is right).** Do §7.2, and the comment
  fixes.

### 7.2 Case B: remove the habits references from root `models.py`
1. Find every remaining consumer of the habits symbols via the grep above.
   `routers/dashboard.py` is the one sanctioned cross-domain reader
   (GOVERNANCE §2.2) — it must import from `domains.habits.models` directly.
   Any other consumer is a violation; report it, don't silently fix it.
2. Update each consumer to `from domains.habits.models import ...`.
3. Delete the habits re-export line(s) (`Habit`, `HabitLog`, `HabitSettings`
   and whichever Pydantic schemas it re-exports) and any habits-related
   `Base = ...` re-export, following WO#22 Task 1's end-state.
4. Confirm `database.py` still imports `domains.habits.models` (directly or
   through the registration block). Before removing the shim, that import was
   what guaranteed mapper registration. **Do not delete the shim until this
   is confirmed.**
5. Do not touch other domains' lines.

### 7.3 Stale comments that reference root `models.py`
Comment-only edits. Do not change any code in these files.

- **`core/base_model.py`:** the docstring's sentence that root `models.py`
  "still has its own `Base = ...` re-export for backward compatibility" is
  false in Case A and becomes false after §7.2 in Case B. Rewrite the
  docstring to state only what is durable: this is the single shared
  `DeclarativeBase`; every domain's `models.py` must inherit from this exact
  object; `Base.metadata` is the one registry Alembic and `create_all()` use.
  Keep the warning that importing a different `Base` silently splits the
  registry — that rule remains true.
- **`domains/habits/models.py` header:** "Moved verbatim from models.py as part
  of the domains/habits pilot migration" is historical and accurate. Optional:
  reword to "originally lived in the top-level models.py (removed)". Owner's
  call; not required.

### 7.4 Apply the pending `HabitLog` docstring edit
Apply the diff in §4.5 to `domains/habits/models.py`. Change nothing else in
that file. Confirm `git diff` shows a comment-only change and that
`UniqueConstraint("habit_id", "logged_date", name="uq_habit_logs_habit_date")`
in `__table_args__` is untouched.

### 7.5 Repo-wide greps for leftovers
```bash
grep -rn "start_on_sunday" .
grep -rn "_get_week_dates" .
grep -rn "routers.habits import" .            # any external importer of moved helpers
grep -rn "from domains.habits.routers.habits import" .
```
Any hit outside `domains/habits/routers/` that imports a helper from
`habits.py` is broken by the move. Repoint it to `_shared` or report it.
Also grep `airflow/` and `tests/` for habit-router imports: DAGs must not
import routers at all (GOVERNANCE §2.2).

### 7.6 Route-order guard, and the lesson for the other splits
The order dependency in §3.7 is currently protected only by comments.

- **Add a regression test** (candidate; the snippet below was not run against
  the real app):
  ```python
  # tests/test_habits_route_order.py
  from main import app

  def _idx(method, path):
      for i, r in enumerate(app.routes):
          if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
              return i
      raise AssertionError(f"{method} {path} not registered")

  def test_delete_log_precedes_delete_habit_id():
      assert _idx("DELETE", "/habits/log") < _idx("DELETE", "/habits/{habit_id}")

  def test_reorder_precedes_update_habit():
      assert _idx("PATCH", "/habits/reorder") < _idx("PATCH", "/habits/{habit_id}")
  ```
- **Check WO#24–30 for the same hazard.** Any split that separates
  literal-path routes from a parametric `/{id}` route into different files is
  exposed to it. `main.py` already carries manual ordering comments for
  recipes, media and finance for the same reason.
- **Propose a GOVERNANCE amendment (§6 process):** "When splitting a router,
  enumerate exact-path collisions **and** parametric shadowing across the
  resulting files; record the required `include_router` order in `main.py`."
  Also fold it into the standing work-order template (§4.3).
- **Optional ticket:** declare `{habit_id:int}` so non-numeric segments never
  match. It changes matching semantics, so it needs its own verification.

### 7.7 Documentation updates
- **`00_MASTER_INDEX.md`:** change WO#23 from "📝 Drafted, not executed" to
  executed-and-reviewed once the reviewer signs off, in both the Track C table
  and the status summary. Record that amendments §4.1–§4.5 were applied.
- **`GOVERNANCE.md`:** add the amendment from §7.6; nothing else from this WO
  requires a rule change.
- **Line-limit lint:** re-run `scripts/check_router_line_limits.py` across the
  whole repo and confirm `habits.py` and `habits_settings.py` are absent from
  the offender list. GOVERNANCE §1.2 wants this as a CI step; if it isn't wired
  in, note it rather than assume.

### 7.8 Explicitly out of scope for the cleanup
- Deleting the unused `log` logger — needs an owner decision first.
- Repairing already-tied `sort_order` rows in existing data.
- Using the four unused Pydantic schemas or deleting them.
- Any change to templates, static files, or other domains.
- Generating an Alembic migration for `uq_habit_logs_habit_date` — the owner
  confirmed it is already applied. If a migration check flags a missing
  constraint, **report it and stop**; do not generate one.

### 7.9 Acceptance criteria for the cleanup
- [ ] Root `models.py` state established and recorded (Case A or B), with the
      grep output attached.
- [ ] Case B only: no habits symbol remains in root `models.py`; every consumer
      imports from `domains.habits.models`; `database.py` still registers the
      habits models.
- [ ] `core/base_model.py` docstring no longer claims a root `models.py`
      re-export exists (or the claim is confirmed true).
- [ ] The §4.5 docstring diff is applied; `git diff` on `models.py` is
      comment-only.
- [ ] Mapper and registry check passes:
  ```bash
  python - <<'EOF'
  import database                      # runs the mapper-registration import block
  from sqlalchemy.orm import configure_mappers
  from core.base_model import Base
  configure_mappers()
  assert {"habits", "habit_logs", "habit_settings"} <= set(Base.metadata.tables)
  assert "uq_habit_logs_habit_date" in {c.name for c in Base.metadata.tables["habit_logs"].constraints}
  print("ok")
  EOF
  ```
  *(not run by the executor)*
- [ ] Route-order regression test exists and passes (§7.6).
- [ ] Repo-wide greps in §7.5 come back clean.
- [ ] Lint script no longer lists the habits routers.
- [ ] Master Index (and GOVERNANCE if amended) updated.

### 7.10 Rollback
`git checkout` on every file touched. The cleanup is comment edits, import
repointing, and shim-line deletion; none of it changes the database.

---

## 8. Reviewer checklist (GOVERNANCE §4.4 order)

1. **Hard boundaries respected?**
   - Split: `habits.py`, two new files in `domains/habits/routers/`, and
     `main.py` registration only. ✅ (Confirm `main.py` diff is exactly the
     two lines in §3.1.)
   - `core/templating.py` not edited — acceptable; WO scoped it "only if
     needed".
   - Amendments 1–4 stay within the same three files; 2 and 4 are
     owner-authorized behavior changes. Amendment 5 touches `models.py`,
     outside WO#23's boundary; owner-authorized.
2. **Are ❌/⚠️ items outside the executor's control?** Yes. The ⚠️ items are
   the substitute-environment limits in §3.5, not incomplete work.
3. **Do the Notes surface tickets?** Candidate tickets: `{habit_id:int}`
   hardening; unused `log` logger; route-order test; Starlette pin check;
   GOVERNANCE amendment (§7.6).
4. **Functional criteria** — confirm against the **real** app before merge:
   - Run the real lint script; both files < 300 lines.
   - `git diff --color-moved=dimmed-zebra` on the split commit shows only the
     differences in §3.4.
   - Exercise all 11 routes in the real app (real DB, real templates, HTMX).
   - Confirm `main.py` includes `habits.router` before `habits_settings.router`.
   - Amendment behavior: a logged-then-deactivated habit no longer inflates
     the "X of Y done" count; a new habit lands below every existing habit.

**Pass condition:** the split commit satisfies §2's criteria with §3.6's ⚠️
resolved by real-environment checks, and each amendment commit matches its
description in §4. Do not fail the migration for §4's behavior changes — they
were agreed after the fact and are recorded here as amendments, not
regressions.

---

## 9. Limitations of this postmortem

- The executor had no repo access; all file contents came from what was
  pasted. Nothing here was verified against the real repo, DB, templates, or
  a running app.
- Line counts for the **original** `habits.py` are a hand count of the
  pasted text (≈520–550), not the lint script's number. The final counts (141
  / 270 / 214) come from `wc -l` on the delivered files.
- The `main.py` and `models.py` edits were delivered as diffs, never applied
  or executed.
- The §7.6 test and the §7.9 registry check are untested sketches.
- The conflicts in §7.1 (root `models.py` state, WO#22 status, Index
  staleness) are unresolved; the cleanup agent must resolve them from the real
  files.
- Nothing in this document was checked against WO#24–30; §7.6's warning about
  them is an inference from the WO#23 text and the current `main.py`.
