# Work Order #22 — `models.py` End State, Stale DAG Header Cleanup, and `configure_mappers()` Verification

*Three independent tasks bundled together (same pattern WO#19 used), because
none is large enough alone to justify its own document and none depends on
the others except that Task 3 should run after Tasks 1–2 land. Keep the
three diffs separable in your report.*

---

## ROLE
You are a senior refactoring engineer doing final cleanup on an
already-completed migration program, plus a purely mechanical cosmetic pass
on DAG header comments, plus a one-time verification check. None of the
three tasks is a feature change.

## HARD BOUNDARIES (all tasks)
- Only touch the files listed in each task's own SCOPE.
- Task 3 (the `configure_mappers()` check) should run against the state
  produced by Tasks 1 and 2 — do it last.
- **Task 1 is the one place this work order diverges from what
  `00_MASTER_INDEX.md` literally recommends.** The master index (summarizing
  the WO#10/WO#20 postmortems) recommends "Option 2 — reduce `models.py` to
  a clean ~15-line import-registry." Reading the actual current file (Step 1
  below) shows that recommendation, taken literally, doesn't accomplish
  anything: a registry file that nothing imports doesn't guarantee mapper
  registration — it's just inert text. Task 1's Step 2 resolves this
  properly rather than mechanically implementing the literal "Option 2"
  text. Flag this in your report as a *correction of prior reasoning*, not
  a scope deviation.
- Task 2 is cosmetic only — do not touch any DAG's `sys.path`, `dag_id`,
  schedule, or logic. One-line comment edits only.
- Task 3 is a verification step to run and report on, not something to
  "fix" preemptively — if it raises, report the exact error and open a
  separate follow-up ticket rather than guessing at a fix here.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
Standard rule: don't fix, reproduce against the pre-change baseline to
confirm it isn't something you introduced, report under Notes, mark the
related criterion ⚠️ rather than ❌.

## WORKING METHOD
Execute Task 1, verify it, then Task 2, verify it, then Task 3 last against
the combined result. Don't defer all verification to the end.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results (✅/❌/⚠️ + reason for non-✅), grouped by task
5. Notes

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.

---

## Task 1 — `models.py` End State

**Current state (confirmed by direct read of the file):** every domain's
re-export shim was removed by WO#20. `routers/dashboard.py` already imports
from each domain's own `models.py` directly (`from domains.jobs.models
import ...`, `from domains.finance.models import Transaction`, etc.).
`main.py` imports every domain's *router* module directly and never imports
root `models.py` at all. What's left in root `models.py` is a block of now-
unused imports (`datetime`, `enum`, `math`, `Decimal`, `Optional`, the
SQLAlchemy column/type imports, `Mapped`/`mapped_column`/`relationship`,
`BaseModel`) plus a comment explaining the WO#20 pass and pointing at this
exact follow-up.

### SCOPE
- `models.py`
- `database.py`
- `migration_docs/GOVERNANCE.md` (§2.4 only — optional, see Step 4)

### STEPS

1. **Grep the whole repo** for `from models import` and `import models`
   (bare — not `from domains.X import models`, not `from core.base_model
   import Base`, not references inside `migration_docs/` or code comments).
   Record the exact command and its output in your report, even if empty.

2. **Branch on the grep result:**

   - **If any real consumer is found** (a live import, not documentation):
     shrink `models.py` to a minimal, explicit registry (imports of every
     domain's `models` module for side effects only), since something
     genuinely depends on importing this module:
     ```python
     # models.py
     """
     Deliberately minimal. Every domain's ORM classes live in
     domains/<name>/models.py — this module's only remaining job is to
     guarantee every domain's models module is imported, so SQLAlchemy's
     mapper registry sees every mapped class before the first query runs.
     """
     from domains.habits import models as _habits_models          # noqa: F401
     from domains.blog import models as _blog_models               # noqa: F401
     from domains.code_intel import models as _code_intel_models   # noqa: F401
     from domains.jobs import models as _jobs_models                # noqa: F401
     from domains.finance import models as _finance_models          # noqa: F401
     from domains.journal import models as _journal_models          # noqa: F401
     from domains.recipes import models as _recipes_models          # noqa: F401
     from domains.workout import models as _workout_models          # noqa: F401
     from domains.media import models as _media_models              # noqa: F401
     from domains.planning import models as _planning_models        # noqa: F401
     ```
     (`explorer` has no `models.py` — omit it, per its own migration's
     explicit "no ORM classes at all" note.)

   - **If zero consumers are found** (expected — this is what direct reading
     of `main.py`/`dashboard.py`/every router suggests): a leftover registry
     file nothing imports doesn't guarantee anything. Instead:
     1. Delete `models.py` from the repo root.
     2. In `database.py`, inside `init_db()` (or immediately above it at
        module scope — your call, state which and why), add the same
        ten-line import block shown above, so the mapper-registration
        guarantee is explicit and centralized in the one place guaranteed
        to run before the app serves its first request, rather than being
        an implicit side effect of `main.py` happening to import every
        domain's router.
     3. Add a one-line comment above that block: `# Guarantees every
        domain's ORM classes are registered with SQLAlchemy before the
        first query — see migration_docs/GOVERNANCE.md §2.4 for history.`

3. Either way, do not add or remove anything else from `database.py` — this
   task touches only the import-registration guarantee.

4. **Optional, only if convenient:** `GOVERNANCE.md` §2.4 ("Legacy Import
   Shims") currently describes shims as ongoing scaffolding to be removed
   "once a domain's only remaining external consumer is
   `routers/dashboard.py`." Since WO#20 already removed every shim and this
   task settles the file's final shape, add one closing sentence to §2.4
   noting it's now historical/closed, with a pointer to this work order. If
   this feels like scope creep given your time budget, skip it and note it
   in Notes instead — it isn't required for this task's acceptance
   criteria.

### ACCEPTANCE CRITERIA
- [ ] Grep command and output reproduced verbatim in the report.
- [ ] Whichever branch was taken, confirmed and stated explicitly.
- [ ] If deleted: `models.py` no longer exists at repo root; the explicit
  registry block exists in `database.py` and is confirmed to run during
  `init_db()`.
- [ ] If kept: `models.py` contains only the docstring and the ten import
  lines — no leftover dead imports, no leftover WO#20 comment.
- [ ] The `configure_mappers()` check in Task 3 passes against the
  post-change state — that's the real proof either branch worked.

---

## Task 2 — Stale DAG Header-Comment Paths

**Finding (from direct read of all 15 current DAG files):** WO#18 moved
every DAG file into a domain subfolder under `airflow/dags/`, but did not
(and, per its own HARD BOUNDARIES, was explicitly forbidden from) touch file
*content* — including each file's own header comment, which still names its
old flat-layout path. `00_MASTER_INDEX.md` tracked this as "12 stale
header-comment paths (cosmetic)"; reading the files directly finds **13** —
the tracked count predates `life_os_staging_promoter.py` being added to the
DAG set, and that file inherited the same stale-header pattern via
copy-paste. Two files need **no** change: `life_os_idea_expander.py` never
had a header path comment at all, and
`life_os_refresh_streaming_availability.py` already has the correct path.

### SCOPE — files to edit (one header-comment line each)

| File | Current | Corrected |
|---|---|---|
| `airflow/dags/blog/life_os_blog_creator.py` | `# airflow/dags/life_os_blog_creator.py` | `# airflow/dags/blog/life_os_blog_creator.py` |
| `airflow/dags/blog/life_os_blog_finalizer.py` | `# airflow/dags/life_os_blog_finalizer.py` | `# airflow/dags/blog/life_os_blog_finalizer.py` |
| `airflow/dags/blog/life_os_blog_scout.py` | `# airflow/dags/life_os_blog_scout.py` (appears twice near the top) | `# airflow/dags/blog/life_os_blog_scout.py` (both occurrences) |
| `airflow/dags/code_intel/life_os_readme_writer.py` | `# airflow/dags/life_os_readme_writer.py` | `# airflow/dags/code_intel/life_os_readme_writer.py` |
| `airflow/dags/code_intel/life_os_code_narrate.py` | `# airflow/dags/life_os_code_narrate.py` | `# airflow/dags/code_intel/life_os_code_narrate.py` |
| `airflow/dags/code_intel/life_os_code_comment.py` | `# airflow/dags/life_os_code_comment.py` | `# airflow/dags/code_intel/life_os_code_comment.py` |
| `airflow/dags/code_intel/life_os_code_improve.py` | `# airflow/dags/life_os_code_improve.py` | `# airflow/dags/code_intel/life_os_code_improve.py` |
| `airflow/dags/jobs/life_os_job_scout.py` | `# airflow/dags/life_os_job_scout.py` | `# airflow/dags/jobs/life_os_job_scout.py` |
| `airflow/dags/jobs/life_os_job_scout_ats.py` | `# airflow/dags/life_os_job_scout_ats.py` | `# airflow/dags/jobs/life_os_job_scout_ats.py` |
| `airflow/dags/jobs/life_os_daily_digest.py` | `# airflow/dags/life_os_daily_digest.py` | `# airflow/dags/jobs/life_os_daily_digest.py` |
| `airflow/dags/jobs/life_os_staging_promoter.py` | `# airflow/dags/life_os_staging_promoter.py` | `# airflow/dags/jobs/life_os_staging_promoter.py` *(the 13th — not in the master index's tracked count)* |
| `airflow/dags/journal/life_os_weekly_synthesis.py` | `# airflow/dags/life_os_weekly_synthesis.py` | `# airflow/dags/journal/life_os_weekly_synthesis.py` |
| `airflow/dags/media/life_os_generate_embeddings.py` | `# airflow/dags/life_os_generate_embeddings.py` | `# airflow/dags/media/life_os_generate_embeddings.py` |

### STEPS
1. Edit exactly the one header-comment line in each of the 13 files above.
2. `life_os_blog_scout.py`'s duplicate docstring-with-header structure
   (the module docstring appears to be pasted twice near the top) — fix
   both occurrences of the stale path, but do **not** otherwise clean up
   the duplication itself; note it under Notes as a separate, trivial
   cleanup candidate.
3. Confirm (don't edit) `life_os_idea_expander.py` has no header comment to
   fix, and `life_os_refresh_streaming_availability.py`'s header is already
   correct.

### ACCEPTANCE CRITERIA
- [ ] All 13 listed files' header comments now match their actual current
  path.
- [ ] `git diff` on each of the 13 shows exactly one changed comment line
  (two, for `life_os_blog_scout.py`) — no `sys.path`, `dag_id`, schedule, or
  logic changes anywhere.
- [ ] `life_os_idea_expander.py` and
  `life_os_refresh_streaming_availability.py` confirmed untouched.

---

## Task 3 — `configure_mappers()` Verification

*Per `00_MASTER_INDEX.md`'s "Where This Actually Stands" section, this is
the single most-repeated "still not verified" item across the whole
20-work-order series. Run it now, against the state Tasks 1–2 produced.*

### SCOPE
Verification only — no files edited in this task.

### STEPS
1. In the real application environment (dependencies installed, `MARIA_DB`
   etc. available, matching how `uvicorn main:app` would run), execute:
   ```python
   import sqlalchemy.orm as orm
   import main
   orm.configure_mappers()
   ```
2. Report the exact result verbatim — either "raised nothing" or the full
   traceback if it raised.
3. If it raises: do **not** attempt a fix in this ticket. File it separately
   with the traceback attached, and mark this task's criterion ⚠️, not ❌.

### ACCEPTANCE CRITERIA
- [ ] `configure_mappers()` result reported verbatim.
- [ ] If it raised nothing: this closes the mapper-registration
  verification gap flagged repeatedly since WO#2.
- [ ] If it raised something: a follow-up ticket exists with the full
  traceback, and this task's own criterion is marked ⚠️, not ❌.

---

## For the next work order (not part of this one)
Once this lands, record the Task 1 decision (delete-and-centralize vs.
shrink-and-keep) in `00_MASTER_INDEX.md`, since it currently still describes
"Option 2" as the open recommendation. Track B is then fully closed.
