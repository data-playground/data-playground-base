# Work Order #31 — `airflow/agents/` Tier 1 Reorganization (DAG-Only Modules)

*Drafted from `track_D_agent_reorg_scoping_analysis.md`'s consumer map.
Scoped to only the tier that analysis classified as DAG-only — Tier 2
(router-only, misplaced) and Tier 3 (dual-consumed — `blog_agents.py`) are
explicitly out of scope and need their own separate scoping work; see that
document's Part 3 for why. Read the analysis document before running this
one if you haven't already — this work order assumes its findings.*

---

## ROLE
You are a senior refactoring engineer relocating a small, well-understood set
of DAG-only helper modules into domain subfolders, mirroring WO#18's DAG
layout. Unlike WO#18, this requires editing one import line in each
consuming DAG — budget for that, don't assume it's a pure move.

## HARD BOUNDARIES
- Scope is **Tier 1 only**: `job_agents.py`, `job_ats_agents.py`,
  `job_resume_context.py`, `job_dedup.py`, `job_scout_health.py`,
  `media_agents.py`. Do **not** touch `recipe_agents.py`, `weekly_agents.py`,
  `blog_agents.py`, or `email_client.py` — see the Track D analysis
  document for why each is excluded.
- Do not change any function name, signature, or logic inside any moved
  file — content is byte-identical except the file's own location.
- Every consuming DAG's import line must be updated in the **same commit**
  as the file move — there is no safe incremental path for this class of
  change (unlike WO#18's pure directory move).

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.

## WORKING METHOD
Move the files first, then update every consuming DAG's import in the same
pass, then grep-verify nothing was missed. Don't declare this done after
just the file move.

## OUTPUT FORMAT
1. Files created
2. Files moved (old path → new path)
3. Files edited (path — description)
4. Acceptance criteria results
5. Notes

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.

---

## SCOPE

**Files to move:**
- `airflow/agents/job_agents.py` → `airflow/agents/jobs/job_agents.py`
- `airflow/agents/job_ats_agents.py` → `airflow/agents/jobs/job_ats_agents.py`
- `airflow/agents/job_resume_context.py` → `airflow/agents/jobs/job_resume_context.py`
- `airflow/agents/job_dedup.py` → `airflow/agents/jobs/job_dedup.py`
- `airflow/agents/job_scout_health.py` → `airflow/agents/jobs/job_scout_health.py`
- `airflow/agents/media_agents.py` → `airflow/agents/media/media_agents.py`

**Files to edit (import line only, per the consumer map in the Track D analysis):**
- `airflow/dags/jobs/life_os_job_scout.py`
- `airflow/dags/jobs/life_os_job_scout_ats.py`
- `airflow/dags/jobs/life_os_staging_promoter.py`
- `airflow/dags/media/life_os_refresh_streaming_availability.py`

**Not in scope:** `email_client.py` (stays at `airflow/agents/` top level,
per the Track D analysis's reasoning — confirm this decision still holds,
or flag explicitly if it should move too).

## STEPS
1. Create `airflow/agents/jobs/` and `airflow/agents/media/` subfolders.
2. Move each file per the SCOPE table above, using `git mv` to preserve
   history (matching WO#18's own precedent).
3. Update each consuming DAG's import line from `from agents.<name> import
   ...` to `from agents.jobs.<name> import ...` or `from agents.media.<name>
   import ...` as appropriate. Do not touch anything else in these DAG
   files — no `sys.path` changes are needed (the `sys.path.insert(0,
   '/opt/airflow/project/airflow')` line already puts `airflow/agents/` on
   the path; a subfolder under it is still importable the same way, just
   with the extra path segment in the import statement itself).
4. Confirm no other file (router, service, test) imports any of the six
   moved modules under their old path — grep the whole repo for
   `agents.job_agents`, `agents.job_ats_agents`, `agents.job_resume_context`,
   `agents.job_dedup`, `agents.job_scout_health`, `agents.media_agents`
   (with and without the `airflow.` prefix) and confirm every remaining hit
   is inside `airflow/dags/jobs/` or `airflow/dags/media/`.

## ACCEPTANCE CRITERIA
- [ ] All six files exist only at their new path; `git log --follow` on each
  confirms history preservation.
- [ ] Each of the four consuming DAGs' import line is updated and nothing
  else in those files changed (diff should be exactly one line per file,
  or two for DAGs importing more than one of the six).
- [ ] `life_os_job_scout` and `life_os_job_scout_ats` DAGs still parse and
  their tasks still resolve `search_linkedin_jobs`, `score_job_batch`,
  `fetch_all_watched_companies`, etc. correctly (static import-resolution
  check acceptable if a live Airflow scheduler isn't available — mark ⚠️
  with that explanation).
- [ ] `life_os_staging_promoter` still resolves `get_full_job_posting`,
  `extract_linkedin_job_id`, `RESUME_MARKDOWN`, `KEY_STRENGTHS_TO_WEIGHT`.
- [ ] `life_os_refresh_streaming_availability` still resolves
  `get_tmdb_watch_providers`.
- [ ] Grep confirms zero remaining references to any of the six modules'
  old import path anywhere in the repo.
- [ ] `email_client.py` confirmed untouched and still at
  `airflow/agents/email_client.py`.

## For the next work order (not part of this one)
Tier 2 (`recipe_agents.py`, `weekly_agents.py`) and Tier 3
(`blog_agents.py`) both need their own separate scoping work — Tier 2
because the right destination probably isn't inside `airflow/agents/` at
all, and Tier 3 because it can't be moved incrementally. Neither should be
attempted as a follow-on to this work order without a fresh, dedicated
risk pass.
