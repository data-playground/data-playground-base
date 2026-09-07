# Work Order #21 — Apply the GOVERNANCE.md §2.5 Amendment (DAG Relocation Risk Reframing)

*Documentation-only. WO#18's own postmortem pre-committed to this update
("If this work order's acceptance criteria all pass, GOVERNANCE.md §2.5
should be updated...") and it has been sitting unapplied since. Nothing here
touches code.*

---

## ROLE
You are a technical writer / governance editor making GOVERNANCE.md match
reality. This is not a code-changing task — resist the urge to "also fix"
anything else you notice in the document while you're in there.

## HARD BOUNDARIES
- Only edit `migration_docs/GOVERNANCE.md`, and only §2.5 within it.
- Do not renumber any other section or edit any cross-reference outside
  §2.5.
- The replacement text must state facts that are actually true of the
  current repo (verified in Step 1) — not aspirational language.

## HANDLING PRE-EXISTING ISSUES DISCOVERED DURING VERIFICATION
If you find something unrelated that looks wrong elsewhere in the document
while doing this: don't fix it, report it under Notes, and leave it alone.

## WORKING METHOD
Verify the facts (Step 1) before writing the new text — don't draft first
and check second.

## OUTPUT FORMAT
1. Files created — none expected
2. Files moved — none
3. Files edited (path — description)
4. Acceptance criteria results (✅/❌/⚠️ + reason for non-✅)
5. Notes

## ROLLBACK
`git checkout` on `migration_docs/GOVERNANCE.md`.

---

## SCOPE
- `migration_docs/GOVERNANCE.md` (§2.5 only)

## STEPS

1. **Confirm the facts before writing anything.** Read the current DAG
   layout — all 15 DAG files should already live under
   `airflow/dags/<domain>/` (`blog/`, `code_intel/`, `jobs/`, `journal/`,
   `media/`), not flat under `airflow/dags/`. Confirm `docker-compose.yml`
   still mounts the whole `./airflow/dags:/opt/airflow/dags` tree (no
   per-subfolder mount was added). If either of these isn't true in the
   environment you're running this against, stop and report — the
   amendment text below assumes both hold.

2. **Replace the current §2.5 body** ("2.5 Why DAGs Haven't Moved Yet")
   with:

   ```markdown
   ### 2.5 DAG Organization — Resolved (see WO#18)

   Earlier versions of this document assumed relocating `airflow/dags/*.py`
   into domain-scoped subfolders would require a coordinated
   `docker-compose.yml` volume-mount change and carried real risk of DAGs
   failing to schedule silently. That assessment has been superseded by
   actual execution.

   **What's actually true, confirmed by WO#18:**
   - Airflow's DAG discovery recursively scans the configured `dags_folder`
     for `.py` files containing DAG objects — subfolder depth doesn't matter.
   - `docker-compose.yml` already mounts `./airflow/dags:/opt/airflow/dags`
     as a whole tree, so organizing into subfolders *within* that
     already-mounted path requires **no volume-mount change**.
   - Every DAG file resolves its own imports via absolute container paths
     (`sys.path.insert(0, '/opt/airflow/project')`, `sys.path.insert(0,
     '/opt/airflow/project/airflow')`), not paths relative to the DAG
     file's own location — moving the file doesn't touch these.
   - Every DAG's `dag_id` is an explicit string literal, not derived from
     file path — the scheduler, UI, run history, and
     `services/airflow_service.py`'s `trigger_airflow(dag_id, ...)` helper
     are all keyed on this string and are unaffected by relocation.

   **Current state:** all DAG files live under `airflow/dags/<domain>/`
   (`blog/`, `code_intel/`, `jobs/`, `journal/`, `media/`) rather than flat
   under `airflow/dags/`. This was a pure file relocation — zero code
   changes inside any DAG file, zero `docker-compose.yml` changes. See
   WO#18 and its postmortem for the full verification (byte-identical
   file diffs, `git log --follow` history preservation).

   **What this does NOT resolve:** moving the *agent modules* under
   `airflow/agents/*.py` into a similar structure is a different, higher-risk
   question. Several of those files (`recipe_agents.py`, `weekly_agents.py`,
   `blog_agents.py`) are imported directly by FastAPI routers as well as (or
   instead of) DAGs — unlike DAG files, which nothing else imports — so the
   "pure relocation, zero code change" property does not automatically
   transfer. See the Track D scoping document
   (`track_D_agent_reorg_scoping_analysis.md`) before attempting that move.

   The original `domains/*/dags/` restructuring this section used to
   describe is no longer the plan — organizing within `airflow/dags/`
   achieved the actual goal (discoverability, being able to hand a scoped
   subfolder to an agent) without the domain-folder move's added risk.
   ```

3. Confirm no other section still references the old "requires coordinated
   docker-compose.yml change" framing (search the whole document for
   "docker-compose" and "fails silently" — neither should appear anywhere
   else describing this as an open risk).

## ACCEPTANCE CRITERIA
- [ ] `git diff` on `GOVERNANCE.md` is isolated to §2.5.
- [ ] No remaining claim anywhere in `GOVERNANCE.md` that DAG relocation
  requires a `docker-compose.yml` change.
- [ ] New §2.5 text accurately reflects the actual current DAG layout
  (5 domain subfolders, 15 files).
- [ ] New §2.5 explicitly states that the `airflow/agents/*.py` question is
  separate and unresolved, with a pointer to the Track D document.

## For the next work order (not part of this one)
WO#22 handles the remaining Track B loose ends (`models.py` end-state, stale
DAG header comments, and the `configure_mappers()` verification). It's
independent of this one — no shared files, safe to run in either order or in
parallel.
