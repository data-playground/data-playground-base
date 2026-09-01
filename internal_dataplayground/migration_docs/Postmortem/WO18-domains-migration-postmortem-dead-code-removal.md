# Work Order #19 — Post-Mortem & Post-Migration Requirements

**Status:** WO#19 complete. This document is the handoff record for anyone
(human or agent) picking up work on this codebase after WO#19 — especially
whoever drafts and executes **Work Order #20** (shim removal) once
**Work Orders #2–10** (the per-domain router migration series) have run.

**Purpose of this document:** Part 1 is the factual record of what WO#19
did and verified — this is what a reviewer checks against to confirm WO#19
itself was done correctly. Part 2 and Part 3 are forward-looking: they
define what "the migration is done" means, so that once WO#2–10 execute,
there is an unambiguous checklist to validate against rather than a
judgment call.

---

## PART 1 — POST-MORTEM: WHAT WO#19 ACTUALLY DID

### 1.1 Scope as given

WO#19 bundled two independent, low-risk cleanup tasks:
- **Task 1:** Delete a fully commented-out legacy `_cerebras()` function
  from `airflow/agents/blog_agents.py`.
- **Task 2:** Create a standalone lint script,
  `scripts/check_router_line_limits.py`, enforcing GOVERNANCE.md §1.2's
  300-line router ceiling.

Neither task depended on WO#2–10 having run. Both were explicitly scoped
to touch only the files named above — no other file was in scope, and
none other was modified.

### 1.2 Task 1 — Dead code removal — what was done

- Located the single fully-commented-out prior implementation of
  `_cerebras()` in `airflow/agents/blog_agents.py`. It began at
  `# def _cerebras(` and ended at the closing `    # )` of its final
  `raise RuntimeError(...)`, immediately followed by 3 blank lines and then
  the `# ── CEREBRAS MODEL IDs ──` section header.
- Checked explicitly for a **second, distinct** dead fragment (the work
  order flagged a possible `INTER_REQUEST_DELAY_SEC`-style fragment as a
  thing to verify rather than assume). Found none — the only other
  `INTER_REQUEST`-related symbol in the file is `_CEREBRAS_INTER_REQUEST_SLEEP`,
  which is **live, uncommented code** (not part of any dead block), so it
  was correctly left untouched.
- Deleted exactly the dead block plus its 3 trailing blank lines, and
  replaced the gap with 2 blank lines — matching this file's own existing
  convention of 2 blank lines before a `# ── SECTION ──` header (verified
  against multiple other section headers in the same file before making
  this call).
- Confirmed the live `_cerebras()` function (the one with the retry loop
  and `Cerebras(...).with_raw_response` client) was not touched.

**Verification performed (not just claimed):**
- `grep` for `^# def _cerebras` → no matches (block fully gone).
- `python3 -m py_compile` on the file → syntax OK.
- `git diff` → single contiguous hunk, **120 deletions, 0 insertions**,
  nothing else in the file touched.
- Diffed the full list of `def`/`class` lines before vs. after → identical
  names, in identical order (only line numbers shifted, as expected from a
  pure deletion earlier in the file).
- Counted remaining `_cerebras` occurrences (8) and manually accounted for
  every one: `_cerebras_key` def + its 1 call site, `def _cerebras(` itself,
  1 log string referencing it, and 4 call sites in
  `agent_code_narrator`, `agent_refiner`, `agent_code_commenter`,
  `agent_code_improver`. No orphaned duplicate logic remains.

**Outcome:** ✅ Complete, fully verified.

### 1.3 Task 2 — Router line-limit lint script — what was done

- Created `scripts/check_router_line_limits.py` per the work order's
  contract: `find_router_files()` globs `routers/*.py` (root, for
  not-yet-migrated domains) and `domains/*/routers/*.py` (for migrated
  domains), `main()` walks the result, excludes `__init__.py`, counts
  lines, and reports/exits non-zero on any file over 300 lines.
- Did **not** wire it into any CI config (none exists in this codebase to
  wire into — correctly treated as out of scope per the work order's hard
  boundary).
- Ran the script against the actual repository — did not fix, edit, or
  editorialize on any violation found.

**Verification performed:**
- Ran the script against the live repo (35 router files, excluding
  `__init__.py`): found **14 files over 300 lines**, exit code `1`. Full
  list reproduced in §2.1 below.
- Manually cross-checked the "35 files" count by hand-counting every
  `.py` file under `routers/` and every `domains/*/routers/*.py` file
  (excluding `__init__.py`) — matched exactly (2 at root + 33 across
  migrated domains).
- Stress-tested the two edge cases the work order specifically called out
  as required — since WO#19 could run before *or* after any of WO#2–10:
  - **Pre-migration shape** (only `routers/`, no `domains/` directory at
    all): script found the file, reported correctly, exited 0 when clean.
  - **Fully-migrated shape** (only `domains/*/routers/`, no root
    `routers/` directory at all): same — found the file, no crash, exited
    0 when clean.
  - Both were run in isolated throwaway directories, not against the real
    repo, purely to validate the script's directory-existence handling
    doesn't assume either location exists.

**Outcome:** ✅ Complete, fully verified. The 14 violations are
**pre-existing debt, intentionally left unfixed** per the work order's
explicit instruction not to editorialize or fix — see §2.1 for why this
matters going forward.

### 1.4 Explicit non-scope (things WO#19 correctly did NOT do)

- Did not fix any of the 14 over-limit router files.
- Did not modify `main.py`, any template, any model file, or any router
  file other than the single named target in Task 1.
- Did not wire the lint script into CI.
- Did not touch `_CEREBRAS_INTER_REQUEST_SLEEP`, even though it appears to
  be a live-but-possibly-unused constant, because it was not part of the
  commented-out block and removing live code was outside Task 1's
  boundary. **Flagged, not fixed** — see §3.4.

---

## PART 2 — WHAT WO#19 ESTABLISHES GOING FORWARD

### 2.1 Baseline debt list (as of WO#19, before any of WO#2–10 ran)

This is the **before-migration baseline**. Re-running
`python scripts/check_router_line_limits.py` after each of WO#2–10 should
be treated as a required check — the reviewer should verify no *new*
domain migration introduces a fresh violation without an accompanying
split, and should track how this list changes over time:

```
domains/habits/routers/habits.py: 551 lines (over by 251)
domains/code_intel/routers/ci_readme.py: 448 lines (over by 148)
domains/journal/routers/journal.py: 446 lines (over by 146)
domains/recipes/routers/recipe_extract.py: 440 lines (over by 140)
domains/blog/routers/blog.py: 419 lines (over by 119)
domains/media/routers/media_recommend.py: 416 lines (over by 116)
domains/planning/routers/weekly_plan.py: 413 lines (over by 113)
domains/code_intel/routers/ci_files.py: 382 lines (over by 82)
domains/workout/routers/workout_settings.py: 378 lines (over by 78)
domains/workout/routers/workout_plan_ai_generator.py: 375 lines (over by 75)
domains/recipes/routers/recipes.py: 371 lines (over by 71)
domains/workout/routers/workout_log.py: 357 lines (over by 57)
domains/media/routers/media.py: 355 lines (over by 55)
domains/code_intel/routers/ci_projects.py: 328 lines (over by 28)
```

Note that most of these are **already-migrated** domains (habits, code_intel,
journal, recipes, blog, media, planning, workout) — i.e. GOVERNANCE.md §1.2
splits are a separate, ongoing backlog independent of WO#2–10's router
*relocation* work. Don't conflate "migrated to `domains/`" with "under the
300-line limit" — they're orthogonal. `workout_plans_crud.py` and
`workout_plan_ai_generator.py` are the one visible example in this repo of
a domain that already went through both: relocation *and* a §1.2 split
(explicitly authorized as a WO#8 follow-up — see §3.3).

### 2.2 The lint script is now a standing tool, not a one-off

Any future work order that splits a router file, or moves a router file
into `domains/`, should re-run
`python scripts/check_router_line_limits.py` as part of its own
verification step, the same way WO#19 did. It requires no arguments and
no CI wiring to be useful — it's runnable ad hoc from the repo root.

---

## PART 3 — POST-MIGRATION REQUIREMENTS (After WO#2–10 Complete)

**This section is a planning framework, not a finished checklist.**
Everything below should be treated as the *shape* of the work — the agent
drafting/executing WO#20 must verify every concrete claim (which files
still exist, which imports still point where) against the actual repo
state at that time, not against this document's assumptions. Where I
don't have direct evidence from files I've seen, I've said so explicitly
rather than guessing. Do not skip the verification step just because this
document describes the expected pattern.

### 3.1 Why this cleanup phase exists

The domain migration (WO#2–10) is being done incrementally, one domain at
a time. Incremental migration of this kind typically leaves two categories
of temporary scaffolding behind, and both categories exist **specifically
so that mid-migration, not-yet-updated code keeps working**:

1. **Shim modules** — a thin file left at the *old* location that
   re-exports/re-imports from the *new* `domains/<domain>/...` location,
   so any caller that still imports the old path doesn't break.
2. **Duplicated or stale references in cross-cutting files** — files like
   `routers/dashboard.py` that read from *every* domain and therefore have
   to import from wherever each domain's models/routers currently live.
   Mid-migration, that means `dashboard.py` (and anything else
   cross-domain) may have a mix of old-path and new-path imports depending
   on which domains have migrated so far.

Neither category is meant to be permanent. Once every domain a given file
touches has fully migrated, the shims for that domain should be deletable,
and cross-cutting files' imports for that domain should point exclusively
at the new location.

**This is explicitly what GOVERNANCE.md §2.4 covers** (referenced by name
in the original WO#19 planning notes as "the shim-removal pass"), and it
is why WO#20 is gated on WO#2–10 — you cannot safely remove a domain's
shim until you've confirmed nothing still consumes it, which is only
knowable once that domain's migration work order has actually run and
reported what it did.

### 3.2 Known example already in evidence: `routers/dashboard.py`

At the time of WO#19, `routers/dashboard.py` (still at the root, not
migrated into any single domain — reasonably so, since it aggregates
*every* domain and doesn't belong to one) already imports directly from
several domains' new-location models:

```python
from domains.jobs.models import Job, ApplicationLog, ApplicationStatus, StagingJob, StagingJobStatus
from domains.finance.models import Transaction
from domains.blog.models import BlogIdea, BlogIdeaStatus
from domains.habits.models import Habit, HabitLog, HabitSettings
from domains.journal.models import JournalEntry, WeeklySynthesis
```

This tells us two things worth carrying forward:

- **Model definitions for these five domains already live under
  `domains/<domain>/models.py`**, independent of whether that domain's
  *routers* have finished moving into `domains/<domain>/routers/`. Router
  migration and model migration are not the same milestone — verify both
  separately per domain, don't assume one implies the other.
- `dashboard.py` is a legitimate permanent resident of the root
  `routers/` directory (it's cross-domain by nature), **not** itself a
  shim or migration leftover. Don't flag it for removal — flag its
  *import list* for review once all five (and any other) domains it reads
  from have fully migrated, to confirm none of those imports are still
  quietly pointing at an old shim rather than the real model.

`routers/_helpers.py` is in the same category: a genuinely shared,
cross-domain utility (`html_error()`), not a per-domain shim. Leave it at
root unless a future work order explicitly decides otherwise.

### 3.3 What "explicitly authorized follow-up" looks like in this codebase

Several files already carry docstring/comment evidence of a pattern this
project uses: a migration work order does the mechanical move, and a
**separate, explicitly authorized follow-up** does further cleanup once
the move is settled. This is useful precedent for how WO#20 (and any
models.py cleanup) should be framed and authorized:

- `domains/workout/routers/_shared.py` — created as "an
  explicitly-authorized follow-up to Work Order #8 (not part of the
  original migration diff)" to consolidate two duplicated helper
  functions (`_get_previous_best()`, `parse_weight_unit()`) that had been
  copy-pasted across `workout.py` and `workout_log.py`.
- `domains/workout/routers/workout_plans_crud.py` /
  `workout_plan_ai_generator.py` — the WO#8 follow-up that split a
  523-line `workout_plans.py` in two, specifically citing GOVERNANCE.md
  §1.2's 300-line ceiling as the reason.
- `domains/workout/routers/workout_settings.py` — a bugfix in
  `search_exercises()` (undefined-variable + non-serializable-enum bug)
  explicitly marked as "authorized by project owner... not part of the
  original migration diff."
- `domains/planning/routers/weekly_plan_generator.py` /
  `weekly_plan_shopping.py` — WO#10 follow-up splits of `weekly_plan.py`
  for the same §1.2 reason, with an explicit note in
  `weekly_plan_generator.py` that `airflow/agents/weekly_agents.py` itself
  was "explicitly out of scope for this migration" and imported
  unchanged.
- `workout_plan_ai_generator.py`'s `_call_gemini_for_plan()` — refers to
  itself as one of "six known duplicate AI-client implementations tracked
  in GOVERNANCE.md §2.3," migrated to a shared AI Service Layer under a
  work order (WO#13) *separate from* the router-domain migration series.

**Implication for WO#20 and models.py cleanup:** these should follow the
same pattern — a scoped, explicitly authorized follow-up, referencing
which specific GOVERNANCE.md section justifies it, with a clear statement
of what's in scope and what's deliberately left alone. Do not fold
models.py cleanup into a router-migration work order's diff; keep it
separate and explicit, the same way this codebase already does for
similar cleanup.

### 3.4 The models.py question, specifically

You (Pedro) asked for this section to explicitly cover adjusting
`models.py` and removing stale references once all migrations are done.
Being precise about what I can and can't confirm right now:

- I have **not** seen a root-level `models.py` file in anything reviewed
  during WO#19 — every model import I've observed already points at
  `domains/<domain>/models.py` (jobs, finance, blog, habits, journal, and
  by extension the same pattern almost certainly holds for media, recipes,
  planning, workout, code_intel, explorer, given their routers already
  import from `domains.<domain>.models` throughout).
- That means either (a) a root `models.py` doesn't exist anymore and this
  item is already resolved, (b) it exists but nothing in the files WO#19
  touched happens to import from it, or (c) it exists and is referenced
  by files WO#19 never opened (e.g. `main.py`, Airflow DAG files outside
  `airflow/agents/`, or Alembic migration scripts). **WO#20 must check
  which of these is true before doing anything** — this document is not
  evidence either way, just an honest account of what wasn't observed.
- The generalized, safe-to-state version of this requirement, independent
  of whether a literal `models.py` turns out to exist:

  **Post-migration models/shim audit checklist (run once WO#2–10 are all
  reported complete):**

  1. `grep -rn` the whole repo (routers, Airflow DAGs/agents, services,
     Alembic migrations, `main.py`) for any import that does **not**
     match the `domains.<domain>.models` / `domains.<domain>.routers`
     pattern but clearly refers to model or router code — that's your
     candidate list of stale references.
  2. For each candidate, confirm whether it points at (a) a genuine
     shared/cross-domain module (keep it — see §3.2's `dashboard.py` /
     `_helpers.py` precedent), or (b) a leftover shim/old-path module for
     a domain that has now fully migrated (candidate for deletion).
  3. For every shim file being considered for deletion, confirm via the
     same `grep` sweep that **nothing else in the repo still imports it**
     — not just `dashboard.py` (the file named in the original shim-removal
     note), but genuinely everything, including Airflow DAG files and any
     test suite if one exists.
  4. Only after that confirmation, delete the shim and update
     GOVERNANCE.md (or wherever "pending migration" is tracked) to mark
     that domain's model/router migration as fully complete, not just
     "routers relocated."
  5. Re-run `scripts/check_router_line_limits.py` after every deletion
     round — shim removal shouldn't change line counts of real files, but
     it's a cheap sanity check that the script still finds the same real
     router set it found before (i.e. nothing was accidentally deleted
     that shouldn't have been).

### 3.5 Definition of "migration successful" — for the reviewer

For a reviewer to sign off that the full domain-migration effort (WO#2–10
+ the WO#20 cleanup pass) is genuinely complete, all of the following
should hold, checked **per domain** and then in aggregate:

**Per domain (jobs, finance, blog, habits, journal, media, planning,
recipes, workout, code_intel, explorer — adjust list if the actual domain
set differs):**
- [ ] All of that domain's router files live under
      `domains/<domain>/routers/`; none remain at root `routers/`.
- [ ] All of that domain's model definitions live under
      `domains/<domain>/models.py`; no duplicate/shadow definitions
      remain anywhere else.
- [ ] `main.py`'s router registration imports that domain's router(s)
      exclusively from `domains/<domain>/routers/`.
- [ ] No shim file remains for that domain, **or**, if one is
      intentionally kept (e.g. genuinely still consumed by something
      outside this migration's control), it is explicitly documented as
      a deliberate exception with a reason — not silently left behind.
- [ ] `grep` across the repo turns up zero imports of that domain's old
      pre-migration path.

**In aggregate:**
- [ ] `python scripts/check_router_line_limits.py` is re-run; any
      remaining violations are either resolved or explicitly tracked as
      accepted debt with a named follow-up work order (matching the
      §1.2-split precedent already established for `workout_plans.py` and
      `weekly_plan.py`).
- [ ] `routers/dashboard.py` and `routers/_helpers.py` (or whatever
      genuinely-cross-domain files remain at root) have every import
      verified against the final `domains/` locations — not left over
      from mid-migration.
- [ ] GOVERNANCE.md's migration-status notes (§2.4 and any "pending"
      markers elsewhere) are updated to reflect actual completion, not
      just "routers moved."
- [ ] A brief closing report exists (in the same ✅/❌/⚠️ format this
      project already uses) enumerating exactly which domains were
      confirmed clean and which, if any, were left with a documented
      exception.

---

## Appendix — Quick reference for the next agent

- Lint script: `python scripts/check_router_line_limits.py` (exit 0 =
  clean, exit 1 = violations printed to stdout). No arguments, no CI
  dependency, safe to run from repo root at any point in the migration.
- Baseline violation count at WO#19 close: **14 files**, list in §2.1.
- Do not draft or execute WO#20 until WO#2–10 have each individually
  reported completion — this was true before WO#19 and remains true; WO#19
  did not change that gating condition.
- When WO#20 is drafted, it should explicitly cite this document's §3
  as its planning input, and should replace every "verify" instruction
  above with an actual verified answer before proceeding to deletions.
