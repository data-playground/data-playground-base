# Work Order #18 — DAG Reorganization (Revised Approach)

*This work order revises the plan GOVERNANCE.md §2.5 originally assumed.
That section predicted DAG relocation would need a coordinated
`docker-compose.yml` volume-mount change and carried real risk ("fails
silently rather than loudly"). Closer inspection changes that assessment —
see the DESIGN DECISION section below before reading further. If this
decision is accepted, GOVERNANCE.md §2.5 should be amended to match once
this work order is reviewed (see the closing note).*

---

## DESIGN DECISION — Read Before Starting

**Original plan (GOVERNANCE.md §2.5):** move DAG files into
`domains/<name>/dags/`, requiring a `docker-compose.yml` volume-mount
change so Airflow's containers can still find them.

**Revised plan (this work order):** organize DAG files into **subfolders
within the existing `airflow/dags/` directory** — e.g.
`airflow/dags/blog/life_os_blog_creator.py` — instead of moving them under
`domains/`.

**Why this is lower-risk and requires zero infrastructure changes:**
1. Airflow's default DAG discovery **recursively scans** the configured
   `dags_folder` for `.py` files containing DAG objects — subfolder depth
   doesn't matter. Since `docker-compose.yml` already mounts
   `./airflow/dags:/opt/airflow/dags`, organizing into subfolders *within*
   that already-mounted path requires **no volume-mount change at all**.
2. Every DAG file starts with `sys.path.insert(0, '/opt/airflow/project')`
   and `sys.path.insert(0, '/opt/airflow/project/airflow')` — **absolute
   container paths, not paths relative to the DAG file's own location.**
   Moving a file into a subfolder does not change these lines or break any
   import inside the file.
3. Every DAG's `dag_id` is an explicit string literal in its `DAG(...)`
   constructor (e.g. `dag_id="life_os_blog_creator"`) — **not derived from
   file path.** Airflow's scheduler, UI, run history, and the
   `services/airflow_service.py` trigger helper (which calls
   `/dags/{dag_id}/dagRuns` by ID, never by file path) are all keyed on
   this string, which does not change. Existing DAG run history stays
   intact.
4. Net result: this becomes a **pure file relocation with zero code
   changes required inside any DAG file**, and zero `docker-compose.yml`
   changes. This is a meaningfully different (and much safer) risk profile
   than what GOVERNANCE.md §2.5 anticipated.

This still achieves the original *organizational* goal (DAGs grouped by
domain, easy to find, easy to hand a scoped subfolder to an AI session) —
it just achieves it without touching infrastructure.

---

## ROLE
You are a senior refactoring engineer performing a pure file-relocation
task with an unusually low risk profile — confirmed by the design decision
above. Do not second-guess the "no code changes needed inside DAG files"
conclusion by making changes anyway "to be safe" — verify it's true (per
the acceptance criteria) rather than acting as if it might not be.

## HARD BOUNDARIES
- Only read/edit/move files explicitly listed in SCOPE below.
- **Do not edit `docker-compose.yml`.** Per the design decision above, no
  change to it is needed. If, during verification, you find a reason a
  compose change actually IS needed after all, stop and report — do not
  make the change yourself, since that would mean the design decision
  above was wrong and needs PO review before proceeding.
- **Do not edit any DAG file's internal `sys.path.insert(...)` lines, its
  `dag_id`, or any other logic.** This is a directory-move only. If a
  DAG's content needs to change for any reason you discover during this
  work, stop and report rather than making the change.
- Do not move any file under `airflow/agents/` — this work order is
  scoped to `airflow/dags/` only. Whether/how to reorganize the agent
  modules is a separate, not-yet-scoped question (see closing note).
- Do not rename any DAG file. Only its directory location changes.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline to confirm
   it is not a regression you introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue.

## WORKING METHOD
Execute steps in order. Verify incrementally, not only at the end. Since
this environment likely doesn't have a running Airflow instance to
schedule against, acceptance criteria that need live DAG parsing/scheduling
should be verified via static checks (Python import/syntax validation of
each moved file from its new location, confirming `sys.path` logic still
resolves) — mark those ⚠️ with an explanation rather than skipping them.

## OUTPUT FORMAT
1. **Files created**
2. **Files moved** (old path → new path)
3. **Files edited** (should be empty/none for this work order — see HARD
   BOUNDARIES; if anything appears here, explain why it was necessary
   despite the "zero code changes" expectation)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Notes**

## ROLLBACK
`git mv` (or equivalent) every file back to its original path listed in
"Files moved" above. Since no code content changes, this is a clean,
simple revert.

---

## SCOPE

**Files to move (grouped by target subfolder):**

`airflow/dags/blog/`:
- `life_os_blog_creator.py`
- `life_os_blog_finalizer.py`
- `life_os_blog_scout.py`
- `life_os_idea_expander.py`

`airflow/dags/code_intel/`:
- `life_os_readme_writer.py`
- `life_os_code_narrate.py`
- `life_os_code_comment.py`
- `life_os_code_improve.py`

`airflow/dags/jobs/`:
- `life_os_job_scout.py`
- `life_os_job_scout_ats.py`
- `life_os_daily_digest.py`

`airflow/dags/journal/`:
- `life_os_weekly_synthesis.py`

`airflow/dags/media/`:
- `life_os_generate_embeddings.py`

**Not in scope:**
- `docker-compose.yml` (confirmed unnecessary — see DESIGN DECISION)
- `airflow/agents/*.py` (all agent modules — separate question)
- `airflow/dag_db.py` (unaffected — this work order doesn't touch it)

---

## STEPS

1. **Create the five subfolders** listed above under `airflow/dags/`.

2. **Move each DAG file into its target subfolder**, using a file move
   (`git mv` or equivalent) that preserves git history rather than a
   delete+recreate.

3. **Do not modify file contents.** Confirm after moving that each file's
   `sys.path.insert(0, '/opt/airflow/project')` /
   `sys.path.insert(0, '/opt/airflow/project/airflow')` lines are
   unchanged — these should not need to change per the design decision,
   but confirm rather than assume.

4. **Check for any file that references another DAG file by relative
   path** (as opposed to by `dag_id` string, which is location-independent
   and fine). Based on review of all thirteen DAG files, none currently do
   this — each DAG is self-contained and any cross-DAG coordination goes
   through Airflow's `trigger_airflow()` service call by `dag_id`, not by
   direct file reference. Confirm this holds for every file as you move
   it; if you find an exception, stop and report rather than silently
   patching it.

---

## ACCEPTANCE CRITERIA

- [ ] All thirteen DAG files exist at their new subfolder locations and no
  longer at their original flat location in `airflow/dags/`
- [ ] Every moved file's `sys.path.insert(...)` lines are byte-identical
  to their pre-move content — confirm via diff, not just visual inspection
- [ ] Every moved file's `dag_id` string is byte-identical to its pre-move
  content
- [ ] Each moved file still parses as valid Python and its top-level
  imports resolve when run with `PYTHONPATH` set to include
  `/opt/airflow/project` and `/opt/airflow/project/airflow` (or the
  local-environment equivalent of those paths) — this is a static
  verification substituting for live Airflow DAG parsing, which isn't
  available in this environment; mark ⚠️ with this explanation if a full
  live Airflow scheduler check isn't possible
- [ ] `git diff docker-compose.yml` shows **zero changes** — confirms the
  design decision's "no infrastructure change needed" claim held
- [ ] `git log --follow` on at least one moved file shows its history was
  preserved through the move (confirms `git mv` or equivalent was used,
  not delete+recreate)
- [ ] `services/airflow_service.py` and every router that calls
  `trigger_airflow(dag_id, ...)` (e.g. `ci_readme.py`, `blog.py`,
  `ci_files.py`) required **zero** changes — confirm by reviewing that
  none of them reference DAG file paths, only `dag_id` strings, which are
  unaffected by this move

---

## Closing Note — GOVERNANCE.md Amendment

If this work order's acceptance criteria all pass, GOVERNANCE.md §2.5
should be updated to reflect the revised, lower-risk approach documented
here, replacing its current "requires coordinated docker-compose.yml
change... fails silently" framing. This is exactly the kind of correction
GOVERNANCE.md §6 (Amendment Process) anticipates — a work order surfacing
a better understanding than the original planning assumed.

**Not addressed by this work order, still open:**
- Whether `airflow/agents/*.py` should get similar domain-subfolder
  treatment (e.g. `airflow/agents/blog/`) is a separate question — those
  files are imported by both DAGs and, in some cases (like
  `recipe_agents.py`), FastAPI routers directly, so any reorganization
  there needs its own risk analysis rather than assuming the same
  low-risk profile this DAG-only move enjoys (DAG discovery's
  location-independence doesn't automatically apply to how Python module
  imports resolve for regular, non-DAG files).
