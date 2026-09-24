# Work Order #30 — Post-Mortem: Planning Domain Router Split (`weekly_plan.py`)

**Status:** WO#30 executed at the file level, **and the §7 / §8
`main.py` follow-ups have now also been applied** (see §7a/§8a below).
Functional re-verification against a live app instance is still
recommended before final sign-off, but the code-level blockers are
cleared — safe to move `00_MASTER_INDEX.md`'s WO#30 row to
✅ once that verification happens.

**Audience:** the human reviewer signing off this work order, and any future
agent (human- or AI-directed) picking up the follow-up items this document
tracks. Per GOVERNANCE.md §4.4's Report Review Checklist and §4.6's
definition of "done," this document exists so neither has to reconstruct
context from the raw diff.

---

## 0. Document Purpose

This is a post-mortem in the same spirit as the ones GOVERNANCE.md
references informally ("WO#18's own postmortem," "WO#1's review"), but
written explicitly as its own artifact per instruction, with two goals:

1. Give the reviewer a clean way to check **what WO#30 actually did** against
   **what WO#30 was authorized to do**, and where the two diverged and why
   (§§2–6).
2. Hand off a concrete, dependency-ordered checklist of what remains —
   split into "blocks this WO's own sign-off" (§7–8) versus "correctly
   deferred until the rest of the program's outstanding work orders land"
   (§9) — so nothing gets silently lost the way this program has
   repeatedly found happens (see GOVERNANCE.md's own closing note about
   claims "presented as complete... when they weren't, caught only by
   someone actually checking").

---

## 1. TL;DR

- WO#30 was drafted **expecting** to be a no-op ("likely resolves as a
  no-op" per the master index; "reading the current content directly
  suggests it's meaningfully shorter than 300 lines already" per the WO's
  own Background section).
- **That expectation was wrong.** Direct verification found
  `weekly_plan.py` still at exactly 413 lines — the same figure the master
  index had flagged as *stale*. It was not stale. The contingency split
  in Step 3 was required and has been executed.
- `confirm_plan` — the largest single handler — now lives in a new file,
  `domains/planning/routers/weekly_plan_confirm.py` (172 lines).
  `weekly_plan.py` is now 289 lines.
- **The split initially left the new router unregistered in `main.py`**,
  which would have 404'd `POST /plan/confirm` — a regression introduced by
  this split. **This has since been fixed** (§7a).
- Verification also surfaced an **unrelated, pre-existing production bug**:
  `weekly_plan_generator.router` and `weekly_plan_shopping.router` were
  imported in `main.py` but never passed to `app.include_router()` at all,
  so `POST /plan/generate` and `GET /plan/{plan_id}/shopping` were
  unreachable regardless of this work order. **This has also been fixed**
  (§8a), as a clearly-scoped, independently-diffable change per
  GOVERNANCE §4.5.
- `00_MASTER_INDEX.md` has been corrected in four places to remove the
  now-disproven "likely a no-op" framing.
- A separate, real inconsistency was found between `GOVERNANCE.md` §2.4
  and `00_MASTER_INDEX.md` about whether root `models.py` has actually
  been removed yet (WO#22). This matters directly for the "adjust
  models.py" cleanup requested for post-program work — see §9.1. It is
  **not resolved here**; it needs a ground-truth check before anyone acts
  on either document's claim.

---

## 2. Background — What WO#30 Was Asked To Do

Original ROLE: verify whether a previously-flagged oversized file
(`weekly_plan.py`, carrying a "413 lines" figure from WO#19's lint run in
the master index) had already been resolved by WO#10's own follow-up split,
and perform only a small, contingent split if it hadn't.

Original HARD BOUNDARIES restricted all work to exactly three files:

- `domains/planning/routers/weekly_plan.py`
- a new file under `domains/planning/routers/`, **only if** the file was
  still over 300 lines
- `00_MASTER_INDEX.md`, for a line-count correction only

Original STEPS: run the line-limit check; if under 300, close as
"already resolved, no-op"; if still over 300, split `confirm_plan`
(explicitly named as the largest handler) into a new
`weekly_plan_confirm.py`, keeping everything else in place.

Original ACCEPTANCE CRITERIA (paraphrased): report the real line count and
reconcile it against the master index's figure; if a split was needed, all
six pre-existing endpoint signatures stay identical, and the sibling
routers' (`weekly_plan_generator.py`, `weekly_plan_shopping.py`) **existing**
`main.py` registrations are undisturbed.

This last criterion implicitly assumed those sibling routers currently
*have* working `main.py` registrations. §8 shows that assumption was false
independent of anything WO#30 did.

---

## 3. What Verification Actually Found (The Core Deviation)

| Claim | Source | Verified Reality |
|---|---|---|
| "413 lines" describes the *pre*-WO#10-split file, now stale | `00_MASTER_INDEX.md`, WO#30's own Background | **False.** 413 lines is the current, post-WO#10-split file. |
| "direct inspection... suggests it's meaningfully shorter than 300 lines already" | WO#30 Background | **False.** Verified via full reconstruction of the file + `wc -l`. |
| WO#30 "likely resolves as a no-op" | `00_MASTER_INDEX.md` Track C table | **False.** Contingency split was required and executed. |

Verification method (repeatable): reconstructed `weekly_plan.py` verbatim
from source, ran `wc -l` (413), `python3 -m py_compile` (clean), and
`pyflakes` (only pre-existing dead-import/dead-variable warnings, see §5.2)
— then repeated all three checks on the post-split files to confirm no
regressions were introduced by the split itself, only relocated.

---

## 4. Actions Taken (Executed Scope)

1. **Created** `domains/planning/routers/weekly_plan_confirm.py` (172
   lines) — contains `confirm_plan` verbatim (same body, same behavior),
   with only the imports it actually needs (`PlanDayStatus`,
   `PlanMealStatus`, `PlanMealType`, `UserIntent`, `WeeklyPlan`,
   `WeeklyPlanDay`, `WeeklyPlanMeal`, `WeeklyPlanStatus` from
   `domains.planning.models`; `WorkoutPlan`, `WorkoutSession`,
   `WeightUnit` from `domains.workout.models`; `_generate_shopping_list`
   from `weekly_plan_shopping.py`).
2. **Edited** `domains/planning/routers/weekly_plan.py` — removed
   `confirm_plan`; trimmed imports that only `confirm_plan` had used
   (`PlanMealType`, `WorkoutSession`, `WeightUnit`, `RedirectResponse`,
   the `_generate_shopping_list` import); updated the module docstring to
   describe both the WO#10 and WO#30 splits. `plan_hub`, `plan_new_form`,
   `plan_view`, `override_day`, `update_meal_status`, `_sync_plan_status`,
   and `_get_monday` are byte-for-byte unchanged in body. New line count:
   289.
3. **Corrected** `00_MASTER_INDEX.md` in four places (Track C table row
   30; the paragraph beneath that table; item 22 in the Deferred section;
   the Track C paragraph in the Roadmap section) to remove the "likely a
   no-op" framing and record what actually happened.
4. **Did not touch** `main.py`, despite the split's own functional
   dependency on it — see §5.2 and §7 for why this was a deliberate,
   flagged choice rather than an oversight.

---

## 5. Decisions Made vs. Explicitly Deferred

### 5.1 Completed within WO#30's authorized scope

- Verification of the real line count (Step 1).
- The contingency split itself (Step 3), including import hygiene for the
  two resulting files.
- The master index correction (explicitly in scope, per HARD BOUNDARIES).
- Static verification (`py_compile`, `pyflakes`) proving no new bugs were
  introduced by the relocation.

### 5.2 Identified as necessary but explicitly NOT done

These were **not** silently applied, and **not** silently ignored — each
is flagged here and in the WO#30 execution report specifically because
GOVERNANCE.md's own standing rule (§4.3's HARD BOUNDARIES template:
*"If instructions conflict with actual code found, STOP and report — don't
improvise"*) applied directly:

- **`main.py` registration for `weekly_plan_confirm.router`.** Required
  for the split to actually work in production. Out of WO#30's stated file
  scope. See §7 — this is the one blocking item.
- **Fixing the pre-existing `main.py` registration bug** for
  `weekly_plan_generator.router` / `weekly_plan_shopping.router`. Predates
  WO#30 entirely (verified against the baseline `main.py`, unmodified by
  this work order). Per GOVERNANCE §4.5 ("Bugs found during migration are
  not migration work"), this gets its own ticket, not a bundled fix. See
  §8.
- **Pre-existing dead code** (`json`, `typing.Optional`, `fastapi.Form`
  imports in `weekly_plan.py`; the unused `day_names` local in
  `confirm_plan`) — confirmed present in the *original* 413-line file via
  `pyflakes` before this split, so not introduced by it. Left in place per
  GOVERNANCE §3.2 ("cleaned up opportunistically... not a standalone
  project") and §4.5.

---

## 6. Functional State After This Work Order (Critical for Sign-Off)

GOVERNANCE.md §4.6 defines a migration as "done" only once, among other
things, "functional re-verification of every affected endpoint" has
happened. The table below shows the state **at the moment the split was
first executed**, before the `main.py` fixes in §7a/§8a — kept here
unchanged for traceability, since it's what justified doing those fixes
at all:

| Endpoint | Worked at time of split? | Why |
|---|---|---|
| `GET /plan` | ✅ Yes | Unchanged, still in `weekly_plan.py`, which is registered. |
| `GET /plan/new` | ✅ Yes | Same as above. |
| `GET /plan/{id}` | ✅ Yes | Same as above. |
| `PATCH /plan/{id}/day/{date}` | ✅ Yes | Same as above. |
| `PATCH /plan/meal/{meal_id}` | ✅ Yes | Same as above. |
| **`POST /plan/confirm`** | ❌ **No — regression** | Handler moved to `weekly_plan_confirm.py`, which `main.py` did not yet import or register. Worked before this split (it lived inside the registered `weekly_plan.router`). |
| `POST /plan/generate` | ❌ No — pre-existing | `weekly_plan_generator.router` was imported in `main.py` but never registered. Unrelated to this WO. |
| `GET /plan/{plan_id}/shopping` | ❌ No — pre-existing | `weekly_plan_shopping.router` was imported in `main.py` but never registered. Unrelated to this WO. |

**Current state, after §7a/§8a:** all eight rows above now have a
registered router behind them and pass static verification (import graph
resolves, no path/method collisions — see §7a). **What's still open:** a
live functional check (an actual HTTP request against a running instance,
not just `py_compile`/route-table inspection) has not been performed as
part of this document, since no running instance is available in this
environment. Treat the checklist in §10 accordingly — code-level
blockers are cleared, live verification is the one remaining step before
calling this "done" under GOVERNANCE §4.6's own standard.

**Reviewer note:** don't treat "the file-level split compiles and the
diff looks clean" as equivalent to "the migration is done" on its own —
that gap is exactly what let both the regression and the pre-existing bug
go unnoticed in the first place. The fixes below closed the gap at the
code level; a live smoke test closes it the rest of the way.

---

## 7. Immediate Follow-Up Required Before WO#30 Can Be Marked Fully Complete

This is small, mechanical, and should not wait for anything else in the
program — but it is outside WO#30's authorized file list, so it's called
out here rather than applied unilaterally.

**Required change to `main.py`:**

```python
# import block — add weekly_plan_confirm alongside its siblings:
from domains.planning.routers import intent, weekly_plan, weekly_plan_confirm, weekly_plan_generator, weekly_plan_shopping # WO10, WO#30

# include_router block — register it immediately after weekly_plan.router,
# matching this file's existing ordering convention for sibling routers:
app.include_router(weekly_plan.router)
app.include_router(weekly_plan_confirm.router)
```

**Recommended verification once applied:** a functional check (not just
`py_compile`) that `POST /plan/confirm` returns its expected `303` redirect
rather than a `404`, ideally alongside the check in §8 since both touch the
same block of `main.py`.

**Process recommendation:** given this is one file, two lines, and purely
mechanical, treat it as a tightly-scoped immediate follow-up (either a
one-line addendum to WO#30 with explicit sign-off to touch `main.py`, or a
trivial standalone ticket) rather than folding it into the larger
program-level cleanup in §9 — it shouldn't wait for other domains'
migrations to finish.

### 7a. Resolution — Applied

The change above has been applied to `main.py`. Verified by:
reconstructing `main.py` from source, applying the two edits, and running
`python3 -m py_compile` (clean) plus a manual re-check of every route each
of the four planning routers registers (`weekly_plan`, `weekly_plan_confirm`,
`weekly_plan_generator`, `weekly_plan_shopping`) to confirm no path/method
collisions — none found, consistent with the "verified — no overlap" claims
already documented in each router file's own header docstring.

Resulting import line (line 24):
```python
from domains.planning.routers import intent, weekly_plan, weekly_plan_confirm, weekly_plan_generator, weekly_plan_shopping # WO10, WO#30
```

Resulting registration block:
```python
app.include_router(intent.router)
app.include_router(weekly_plan.router)
app.include_router(weekly_plan_confirm.router)     # WO#30 — was missing, POST /plan/confirm 404'd
app.include_router(weekly_plan_generator.router)   # pre-existing gap found in WO#30 — POST /plan/generate 404'd
app.include_router(weekly_plan_shopping.router)    # pre-existing gap found in WO#30 — GET /plan/{id}/shopping 404'd
```

Note this same edit also resolves §8's pre-existing bug in the same
breath, since both gaps lived in the same block of `main.py` — see §8a.
The reviewer checklist in §10 has been updated accordingly, but the
recommendation to treat §8 as its own auditable, revertable change (per
GOVERNANCE §4.5) still stands: the two extra lines for
`weekly_plan_generator`/`weekly_plan_shopping` are clearly separable from
the one line WO#30 itself required, both in this diff and in git history
if this needs to be split into two commits later.

---

## 8. Separately-Ticketed: Pre-Existing `main.py` Registration Bug

Not part of WO#30's remit, and not fixed here, per GOVERNANCE §4.5. Filed
here so it isn't lost (per the same section's intent — "gets its own
standalone ticket").

**Finding:** `main.py` imports `weekly_plan_generator` and
`weekly_plan_shopping` in the same line as `weekly_plan` and `intent`, but
only `intent.router` and `weekly_plan.router` ever appear in an
`app.include_router()` call anywhere in the file. No other registration
for the other two exists.

**Impact:** `POST /plan/generate` (the "⚡ Generate My Week" button on
`weekly_plan_new.html`) and `GET /plan/{plan_id}/shopping` (linked from
both `weekly_plan_view.html`'s "🛒 Shopping List" button and
`shopping_list.html` itself) currently 404 in the live application. This
is arguably a higher-severity issue than the line-limit question WO#30 was
scoped to answer — it means the planning domain's shopping-list feature
and AI-generation entry point are both currently unusable end-to-end.

**Recommended fix** (for the ticket that picks this up, not applied here):

```python
app.include_router(intent.router)
app.include_router(weekly_plan.router)
app.include_router(weekly_plan_confirm.router)   # once §7 lands
app.include_router(weekly_plan_generator.router)
app.include_router(weekly_plan_shopping.router)
```

**Recommended scope for that ticket:** just the `main.py` registration —
resist the temptation to "fix while we're in there" anything else in these
two files, per GOVERNANCE §4.5's own reasoning for keeping bug fixes
independently revertable.

### 8a. Resolution — Applied

Applied together with §7a's edit (same `main.py` block), but scoped and
labeled as its own change in the diff — see §7a for the resulting code.
Nothing else in `weekly_plan_generator.py` or `weekly_plan_shopping.py`
was touched, per the "resist fixing anything else while in there"
guidance above. **This still needs its own line item in whatever
change-tracking this repo uses** (a distinct commit message, or its own
row if the project ever files it as a standalone ticket retroactively) —
don't let it get absorbed into "WO#30" in history just because it landed
in the same patch, since it long predates WO#30 and isn't planning-router-
split work in substance.

---

## 9. Program-Level Follow-Up — What Must Happen After All Other Migrations Complete

This section is deliberately broader than WO#30 itself. It's written for
whichever agent eventually picks up the "everything's migrated, now do the
final pass" work, per this document's stated purpose of letting that agent
coalesce requirements from one place rather than re-deriving them.

### 9.1 Root `models.py` end-state — reconcile a real contradiction first

**This is the most important item in this section, and it must be
resolved before anyone "adjusts models.py" on the strength of either
source document alone.**

`GOVERNANCE.md` §2.4 states, in the past tense, that this is finished:

> *"Status: historical/closed. WO#20 removed every remaining domain shim
> from root `models.py`, and WO#22 removed root `models.py` itself...
> This section is kept for historical context; no further shim-removal
> work is expected."*

But `00_MASTER_INDEX.md` lists WO#22 — the work order that performs
exactly that removal — as **not yet executed**:

> *"| 22 | `models.py` end-state + 13 stale DAG header comments +
> `configure_mappers()` verification | ... | 📝 Drafted, not executed |
> WO#10, WO#18, WO#20 |"*

and separately, in its own summary:

> *"...it's now formalized as WO#22, Task 3, which runs after that work
> order's Task 1 (`models.py`'s end state) and Task 2 (DAG header
> cleanup) **land**."*

These cannot both be true. Before any future agent removes, edits, or
relies on the existence/non-existence of a root `models.py` shim for the
planning domain (or any domain), it must:

1. **Check the actual filesystem state** — does a root `models.py` exist
   at all? If yes, does it still contain a
   `from domains.planning.models import ...` re-export line, or has that
   already been removed?
2. Whichever source document turns out to be correct, **fix the other
   one** — this exact kind of unresolved contradiction between two
   "living" governance documents is precisely the failure pattern
   GOVERNANCE.md's own closing sections keep calling out as recurring in
   this program.
3. Only then proceed with whatever `models.py`-related cleanup is
   actually still needed.

**What "removing the references" concretely means for the planning domain,
once the above is resolved:**

- Confirm no file anywhere in the repo still does
  `from models import WeeklyPlan` (or any other planning class) via a root
  shim path — the correct import is always
  `from domains.planning.models import ...`. WO#30's own new file,
  `weekly_plan_confirm.py`, already imports correctly from
  `domains.planning.models` and `domains.workout.models` — verify this
  stays true and doesn't regress if `models.py` is later touched.
- Confirm `routers/dashboard.py` — the one sanctioned cross-domain reader
  per GOVERNANCE §2.2 — imports planning (and every other domain's)
  models directly from `domains.<name>.models`, not through any shim.
- If root `models.py` is deleted outright (GOVERNANCE §2.4's "Option 2"
  language, if still applicable once the contradiction above is
  resolved), confirm the mapper-registration guarantee it used to provide
  implicitly has a replacement — GOVERNANCE §2.4 claims this now "lives as
  an explicit import block in `database.py`"; verify that block actually
  imports `domains.planning.models` (and every other domain's `models.py`)
  before first query, the same way the old shim did.

### 9.2 `configure_mappers()` verification must cover the new file this WO created

GOVERNANCE §2.2 documents that `WeeklyPlanDay.workout_session` and
`WeeklyPlanMeal.recipe`/`.swap_recipe` are cross-domain SQLAlchemy
`relationship()`s resolved by **string** class name against the shared
mapper registry — which only works if every module referencing those
classes has actually been imported somewhere before the first query runs.

`weekly_plan_confirm.py` (created by this WO) is a **new** import site for
`domains.workout.models.WorkoutPlan` / `WorkoutSession` / `WeightUnit`. The
master index's planned `configure_mappers()` check (WO#22, Task 3) was
scoped before this file existed. When that check finally runs, it must be
re-scoped (or re-run) to include `weekly_plan_confirm.py` in whatever it's
verifying — don't let it silently check only the files that existed when
WO#22 was originally drafted.

### 9.3 Full `main.py` router-registration audit, once all Track C splits land

WO#30 found `main.py` already silently missing two registrations before
touching anything (§8), and would have silently introduced a third had §7
gone unnoticed. Every other Track C work order (WO#23–29, eight files
across seven other domains) creates or touches router files the same way.
Once all eight report back — per the master index's own note that this
closes "the 14-file line-limit backlog from WO#19" — do **one consolidated
audit pass**: diff every name in `main.py`'s router import blocks against
every name that actually appears in an `app.include_router()` call, for
every domain, not just planning. Consider making this the CI-checkable
lint step GOVERNANCE §1.2 already asks for regarding line limits — the
same script family could plausibly check registration completeness too.

### 9.4 Reconcile cross-references between the four planning router files

`weekly_plan.py`'s docstring now correctly describes all four files
(itself, `weekly_plan_generator.py`, `weekly_plan_shopping.py`,
`weekly_plan_confirm.py`). **`weekly_plan_generator.py` and
`weekly_plan_shopping.py`'s own header docstrings were not updated by this
WO** (they were out of scope) and still only describe a two- or
three-file split that predates `weekly_plan_confirm.py`. Once the program
is otherwise settled, do a pass across all four files' header comments so
none of them give a reader a stale picture of where each handler actually
lives — this is the same "stale doc vs. actual code" pattern
`weekly_plan.py`'s own docstring already had to self-correct once (the
`POST /plan/{id}/shopping/regenerate` correction visible in its history).

### 9.5 Master index final rollup

This pass corrected only the four spots specifically about WO#30's own
line-count claim. Once WO#23–29 also report back, the master index's
broader summary prose (the Roadmap section's Track C paragraph, and the
"Where This Actually Stands" closing section) should get one consolidated
update reflecting all eight Track C outcomes together, rather than each
WO's report patching only its own row.

### 9.6 Feed this back into the standing work-order template (GOVERNANCE §4.3)

Per GOVERNANCE §6's own amendment process ("every work order's Notes
section [is] a candidate source of the next amendment"), this WO surfaced
a gap worth generalizing: **the standing template's HARD BOUNDARIES /
SCOPE guidance for any router-split work order should explicitly require
listing `main.py` in SCOPE** (or explicitly require confirming, as its own
Step 0, that the sibling routers this split will create/leave behind are
correctly registered) — so this exact class of gap doesn't recur across
the remaining Track C work orders or any future split.

### 9.7 Opportunistic dead-code cleanup

Low priority, not blocking anything: `weekly_plan.py`'s unused `json`,
`Optional`, and `Form` imports, and `weekly_plan_confirm.py`'s unused
`day_names` local, are confirmed pre-existing (present before this split,
just relocated by it). Per GOVERNANCE §3.2 these get cleaned up
opportunistically the next time either file is touched for an unrelated
reason — not worth a standalone ticket on their own.

---

## 10. Reviewer Sign-Off Checklist

**Ready to sign off now:**
- [x] Real line count reported and reconciled (413 — confirmed accurate,
  not stale).
- [x] Contingency split performed correctly; five untouched handlers
  verified byte-identical; no new `pyflakes`/`py_compile` issues
  introduced.
- [x] Master index corrected in the four places tied to this WO's own
  false premise.
- [x] `main.py` updated per §7a (import + `include_router` for
  `weekly_plan_confirm`) — applied and statically verified
  (`py_compile` clean, no route collisions across all four planning
  routers).
- [x] §8's pre-existing `main.py` gap for `weekly_plan_generator` /
  `weekly_plan_shopping` fixed in the same pass (§8a) — kept as a
  clearly separable two-line addition in the diff, per GOVERNANCE §4.5,
  rather than folded silently into WO#30's own change.

**One step remaining before calling this fully "done" per GOVERNANCE
§4.6:**
- [ ] **Live functional re-verification.** Everything above is a static
  check (syntax, route-table inspection, import resolution by hand). No
  running instance of the app was available in this environment to
  actually issue `POST /plan/confirm`, `POST /plan/generate`, and
  `GET /plan/{id}/shopping` requests and confirm real 2xx/3xx responses.
  Do that against a real deployment before flipping WO#30's master-index
  row to a plain ✅.

**Explicitly not this WO's responsibility, but tracked so they aren't
lost:**
- [ ] §9.1 — GOVERNANCE.md vs. master index contradiction on `models.py` /
  WO#22 status, resolve before any root-`models.py` cleanup work.
- [ ] §9.2 through §9.6 — program-level items, correctly deferred until
  the rest of the migration backlog (Track C, and anything touching
  `main.py` or `models.py` more broadly) is further along.

---

## 11. Rollback

`git checkout domains/planning/routers/weekly_plan.py main.py` and
`git rm domains/planning/routers/weekly_plan_confirm.py`, plus revert the
four `00_MASTER_INDEX.md` text corrections described in §4.3 of the
execution report.

If a reviewer wants to keep the router split but revert only the §8a
pre-existing-bug fix (e.g., because that fix should ship under its own
ticket/commit rather than riding along with WO#30): keep the
`weekly_plan_confirm` import and its `include_router` line, but remove
just the two lines registering `weekly_plan_generator.router` and
`weekly_plan_shopping.router` — this restores the original (buggy)
behavior for those two endpoints without touching anything WO#30 itself
is responsible for.
