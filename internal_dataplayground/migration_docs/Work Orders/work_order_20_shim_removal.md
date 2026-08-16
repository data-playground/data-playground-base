# Work Order #20 — Shim Removal Cleanup Pass

*Addresses GOVERNANCE.md §2.4. This work order is explicitly designed to
be safe to run at ANY point — whether zero, some, or all of WO#2–10 have
been executed. For each domain, if its migration hasn't happened yet, skip
it and note "not applicable yet," don't treat that as a failure. Re-run
this same work order again later as more domains get migrated — it's
idempotent by design.*

---

## ROLE
You are a senior refactoring engineer performing a mechanical cleanup pass.
For each domain's shim in root `models.py`, you're checking one fact
(`is dashboard.py still the only consumer?`) and acting on it if true. Do
not attempt to fix, improve, or restructure anything beyond removing a
shim and updating exactly one import line in `dashboard.py` per domain
where applicable.

## HARD BOUNDARIES
- Only touch: the shim blocks in root `models.py`, and
  `routers/dashboard.py`'s import statement(s). Nothing else.
- **For any domain not yet migrated (its classes still live directly in
  root `models.py`, no shim exists for it), skip it entirely** — there's
  nothing to remove. Do not attempt to migrate a domain as part of this
  work order; that's what WO#2–10 are for.
- **For any domain whose shim exists but which has a consumer OTHER than
  `dashboard.py`** (per the "Not in scope, referenced only..." sections of
  each domain's original work order — e.g. `weekly_plan.py`'s references
  into `recipes`/`workout`, or `journal.py`'s reference into `planning`),
  do NOT remove that domain's shim, even if `dashboard.py` doesn't use it.
  The shim exists to serve whichever consumer needs it, not just
  `dashboard.py` — check each domain's original work order's "Not in
  scope" section (or the actual current codebase, if those cross-domain
  references have themselves already been updated to point directly at
  `domains.<name>.models` per a later work order like WO#10's Step 7/8)
  before concluding a shim is safe to remove.
- If a domain's shim is removed, `dashboard.py`'s import for that domain
  must be updated in the same change to point directly at
  `domains.<name>.models` — never leave `dashboard.py` importing from a
  shim that no longer exists.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration, do not fix it —
report it under Notes per the standard rule used throughout this series.

## WORKING METHOD
Process domains in this order, since it matches their migration order and
makes cross-domain dependency checking easier to reason about: `habits`,
`blog`, `code_intel`, `jobs`, `explorer`, `finance`, `journal`, `recipes`,
`workout`, `media`, `planning`. For each domain, do the check-and-act
sequence in Step 1 below before moving to the next — don't batch all the
"checking" first and all the "acting" second, since later domains' checks
may depend on earlier domains' current shim status.

## OUTPUT FORMAT
1. **Files created** — should be none
2. **Files moved** — should be none
3. **Files edited** (path — description of exactly which shim(s) were
   removed and which `dashboard.py` import lines were updated)
4. **Per-domain results table** — for each of the eleven domains, one row:
   `Domain | Migrated? | Shim exists? | Other consumers besides dashboard? | Action taken`
5. **Notes**

## ROLLBACK
`git checkout` on `models.py` and `routers/dashboard.py`.

---

## SCOPE

**Files to edit:**
- `models.py`
- `routers/dashboard.py`

**Reference only (to determine current state — locate wherever each
currently exists, root `routers/` or `domains/*/routers/`):**
- `domains/blog/routers/blog.py`, `domains/code_intel/routers/*.py` (or
  root equivalents) — to check WO#2's "Not in scope" consumers
- `domains/journal/routers/journal.py` (or root equivalent) — to check
  WO#6's local `WeeklyPlanDay`/`WeeklyPlan`/`WeeklyPlanStatus` import and
  whether it's since been updated (per WO#10 Step 8) to point at
  `domains.planning.models` directly
- `domains/planning/routers/*.py` (or root equivalents) — to check WO#10's
  own cross-domain imports into `recipes`/`workout` and whether they were
  updated per WO#10 Step 7

---

## STEPS

**For each of the eleven domains, in the order given in WORKING METHOD:**

1. **Check migration status.** Does a `domains/<name>/models.py` file
   exist? If not, this domain hasn't been migrated — record "not
   applicable yet" in the results table and move to the next domain.

2. **Check shim existence.** Does root `models.py` still contain a
   re-export shim for this domain (the `# TODO: remove after all
   cross-references are updated` block)? If the shim was already removed
   by some prior action, record that and move on.

3. **Check for non-dashboard consumers.** Search the codebase for any
   `from models import <ClassName>` referencing this domain's classes,
   outside of `models.py` itself and `routers/dashboard.py`. Cross-check
   against the specific known cases flagged in each domain's original work
   order (WO#6's journal→planning reference, WO#10's planning→recipes/
   workout references) to confirm whether those have since been updated to
   import directly from the real domain path (in which case they no longer
   count as a "consumer of the shim") or still point at root `models`.

4. **Act.** If migrated, shim exists, and `dashboard.py` is confirmed to
   be the only remaining consumer (or there are zero consumers at all,
   per the `media` domain's situation noted in WO#9):
   - Update `dashboard.py`'s import for this domain's classes to `from
     domains.<name>.models import ...`.
   - Delete this domain's shim block from root `models.py`.
   - If `dashboard.py` doesn't reference this domain at all (the `media`
     case), simply delete the shim — no `dashboard.py` edit needed for
     that domain.

   If any non-dashboard consumer still points at the shim, leave the shim
   in place and record why in the results table.

---

## ACCEPTANCE CRITERIA

- [ ] Every domain's row in the results table is filled in with an
  accurate, verified status (not assumed) — "not applicable yet," "shim
  retained (reason: ...)," or "shim removed, dashboard.py updated"
- [ ] For every domain where a shim was removed: `routers/dashboard.py`
  still imports and uses that domain's classes correctly, with **zero**
  functional change to any dashboard card's rendered output — spot-check
  `/dashboard` renders identically before and after this work order's
  changes
- [ ] For every domain where a shim was removed: root `models.py` no
  longer contains a `# TODO: remove after all cross-references are
  updated` comment for that domain
- [ ] For every domain where a shim was retained: the specific
  non-dashboard consumer justifying its retention is named explicitly in
  the results table, not just asserted generically
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  confirm removing a shim doesn't change table registration in any way —
  same table count, no `InvalidRequestError`, before and after
- [ ] This work order's own idempotency: running it a second time
  immediately after (with no other changes in between) should find
  nothing further to do and produce a results table identical in
  substance (all "already removed" or "not applicable" rows) — confirm
  this by actually re-running it once, briefly, as a sanity check

---

## Closing Note

Once every domain shows "shim removed" (or "not applicable — no
consumers"), root `models.py` should contain only imports and possibly a
handful of genuinely top-level constants, if any remain. At that point,
per WO#10's own closing note, do a final read-through of `models.py` to
confirm nothing was accidentally left behind — this can be folded into the
same pass as this work order's final domain, or done as one more small
follow-up, your call based on how much is left to review by then.

**Separately flagged, not part of this work order:** while producing the
domain work orders in this series, `workout_plans.py` was moved as a
single file in WO#8 without being split, even though GOVERNANCE.md §1.2
uses `workout_plans.py → workout_plans_crud.py +
workout_plan_ai_generator.py` as its own worked example of the 300-line
rule — the same treatment `weekly_plan.py` actually received in WO#10.
This is an inconsistency worth resolving: either split `workout_plans.py`
the same way in a small follow-up work order, or confirm it's actually
under the 300-line threshold and the GOVERNANCE.md example was aspirational
rather than descriptive (run `scripts/check_router_line_limits.py` from
WO#19 to settle this with data rather than guessing). `media_recommend.py`
and `ci_readme.py` were also flagged in the original Phase 1 audit as
likely over the limit — the same lint script resolves all three questions
at once. Worth doing as a short, standalone follow-up once WO#19 has
actually run and reported real numbers.
