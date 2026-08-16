# Work Order #3 — Domain Migration: `jobs`

*This domain has no live cross-domain foreign keys (unlike WO#2's blog/code_intel
pair) — its only external dependency is `routers/dashboard.py` reading `Job`,
`ApplicationLog`, `ApplicationStatus`, `StagingJob`, `StagingJobStatus` for
summary cards, which is the same "shim keeps it working" pattern already
proven in WO#1 and WO#2. This should be a comparatively clean, single-domain
move.*

---

## ROLE
You are a senior refactoring engineer performing a structural code migration.
Your job is NOT to improve, optimize, or modernize the code you move — only
to relocate it correctly and verify it still behaves identically. Resist the
urge to "clean up while you're in there." Flag improvement opportunities as
a NOTES section at the end instead of acting on them.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below. If you believe you
  need to touch a file outside this list to complete the task, STOP and
  report why — do not proceed and do not guess.
- No schema changes. No renamed tables, columns, or endpoints. No behavior
  changes. This is a location refactor only.
- If a step's instructions conflict with what you find in the actual code
  (e.g. a file/class isn't where the work order says it is), stop and report
  the discrepancy rather than improvising a fix.
- You may create empty `__init__.py` package markers as needed to support
  new import paths — this is expected scaffolding, not a scope expansion,
  and does not need to be flagged as a deviation (a one-line mention in the
  report's "Files created" section is enough).
- **`services/ats_slug_service.py` is explicitly OUT OF SCOPE and must NOT be
  moved.** It's job-domain-specific in practice but lives in the shared
  `services/` layer alongside other external-API wrappers (GitHub, TMDB,
  etc.) — consistent with how WO#2 kept `github_service.py` in place. Leave
  it where it is; `job_config.py` continues importing it from its current
  path.
- Do NOT move any DAG files (`life_os_job_scout.py`, `life_os_job_scout_ats.py`,
  `life_os_daily_digest.py`) or any files under `airflow/agents/` related to
  jobs (`job_agents.py`, `job_ats_agents.py`, `job_resume_context.py`,
  `job_dedup.py`, `job_scout_health.py`). None of these are imported by the
  FastAPI routers in scope — they're DAG-only and follow the DAG/FastAPI
  boundary rule in `CONTRIBUTING.md`. DAG relocation is a separate, later
  work order requiring a coordinated `docker-compose.yml` change.
- Do NOT move `routers/dashboard.py`. It is a cross-domain consumer, not part
  of the `jobs` domain, and stays at the top level per the "dashboard is
  special-cased, reads-only, cross-domain" note in the project's Phase 3 plan.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline (the code
   as it existed before your changes) to confirm it is not a regression you
   introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket:
   which endpoint/file, the exact error, and root cause if you found one.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue —
   explain the distinction in the one-line reason.

## WORKING METHOD
Execute steps in the order listed. After each step that changes running
behavior (not pure file moves), pause and self-verify against the relevant
acceptance criteria before continuing to the next step. Do not defer all
verification to the end.

If an acceptance criterion requires a resource not listed in SCOPE (e.g. a
config file, external service, or database not provided), do not skip the
criterion silently. Perform the closest verification achievable with what
you have, state explicitly what the substitute check was and why, and mark
the result ⚠️ rather than ✅ or leaving it blank.

## OUTPUT FORMAT
End with a report in exactly this structure:
1. **Files created** (list — include any scaffolding `__init__.py` files here)
2. **Files moved** (old path → new path)
3. **Files edited** (path — one-line description of the change; if a change
   goes beyond what a step literally asked for but was necessary to make
   that step work, say so explicitly and explain why)
4. **Acceptance criteria results** (checklist, ✅/❌/⚠️ per item, with a
   one-line reason for any ❌ or ⚠️ — including substitute-verification
   explanations per the rule above)
5. **Notes / things I noticed but did not act on** (optional — improvement
   ideas, pre-existing bugs found per the rule above, inconsistencies, risks
   spotted while working, explicitly out of scope for this task)

## ROLLBACK
This work order operates on files already tracked in git. If acceptance
criteria fail and cannot be quickly fixed, the safe rollback is `git
checkout` on every file listed in "Files created / moved / edited" above —
do not attempt a partial manual revert.

---

## SCOPE

**Models to extract from `models.py`** (all currently grouped under the
`# --- JOBS MODULE ---` comment):
`ApplicationStatus`, `Job`, `ApplicationLog`, `JobSearchKeyword`,
`WatchedCompany`, `JobScoutRunLog`, `JobResponse`, `ApplicationLogCreate`,
`ApplicationLogResponse`, `StagingJobStatus`, `StagingJob`,
`StagingJobCreate`, `StagingJobResponse`

**Routers:**
- `routers/jobs.py`
- `routers/ats.py`
- `routers/staging.py`
- `routers/job_config.py`

**Templates:**
- `templates/jobs.html`
- `templates/job_config.html`
- `templates/partials/staging_row.html`
- `templates/partials/staging_queue.html`
- `templates/partials/job_detail.html`
- `templates/partials/ats_buttons.html`
- `templates/partials/job_watched_panel.html`
- `templates/partials/job_keyword_list.html`
- `templates/partials/job_candidate_list.html`
- `templates/partials/job_slug_guess.html`

**Static:**
- `static/css/jobs.css`
- `static/css/jobs_enhancements.css`
- `static/js/jobs_enhancements.js`

**Shared infra referenced but not moved (read-only for import-path
confirmation, do not edit):**
- `services/ats_slug_service.py`
- `routers/_helpers.py` (provides `html_error`, used by `staging.py` and
  `job_config.py`)

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:**
- `routers/dashboard.py` (imports `Job`, `ApplicationLog`,
  `ApplicationStatus`, `StagingJob`, `StagingJobStatus` from `models` —
  must keep working via shim, same pattern as `habits`, `blog`, and
  `code_intel` in prior work orders)

---

## STEPS

1. **Create `domains/jobs/models.py`.** Move `ApplicationStatus`, `Job`,
   `ApplicationLog`, `JobSearchKeyword`, `WatchedCompany`, `JobScoutRunLog`,
   `JobResponse`, `ApplicationLogCreate`, `ApplicationLogResponse`,
   `StagingJobStatus`, `StagingJob`, `StagingJobCreate`, `StagingJobResponse`
   there verbatim, preserving their current relative order. Import `Base`
   from `core.base_model`.

2. **Note on internal relationships.** `Job.application_logs` and
   `ApplicationLog.job` reference each other via string class names
   (`relationship("ApplicationLog", ...)` / `relationship("Job", ...)`).
   Since both classes are moving into the *same* new file together, this
   requires no special handling — it's a same-module reference, unlike the
   cross-module case in WO#2. Just confirm both classes end up in
   `domains/jobs/models.py` and the strings still resolve.

3. **In `models.py`:** delete the twelve moved class bodies and replace with
   a re-export shim: `from domains.jobs.models import ApplicationStatus,
   Job, ApplicationLog, JobSearchKeyword, WatchedCompany, JobScoutRunLog,
   JobResponse, ApplicationLogCreate, ApplicationLogResponse,
   StagingJobStatus, StagingJob, StagingJobCreate, StagingJobResponse`. Tag
   it `# TODO: remove after all cross-references are updated`, consistent
   with the shims already in place from WO#1 and WO#2.

4. **Move routers:**
   - `routers/jobs.py` → `domains/jobs/routers/jobs.py`
   - `routers/ats.py` → `domains/jobs/routers/ats.py`
   - `routers/staging.py` → `domains/jobs/routers/staging.py`
   - `routers/job_config.py` → `domains/jobs/routers/job_config.py`

   Update each file's model imports to pull from `domains.jobs.models`
   instead of `models`. Update each file's `templates =
   Jinja2Templates(directory="templates")` to `from core.templating import
   templates`, matching the pattern from WO#1/WO#2. Leave
   `from routers._helpers import html_error` and
   `from services.ats_slug_service import guess_ats_slugs` unchanged —
   neither of those files is moving.

5. **Move templates**, preserving the `partials/` subfolder structure, into
   `domains/jobs/templates/` per the SCOPE list above.

6. **Move static assets** into `domains/jobs/static/css/` and
   `domains/jobs/static/js/` respectively. Update the references inside
   `jobs.html`:
   - `<link rel="stylesheet" href="/static/css/jobs.css">` →
     `/static/jobs/css/jobs.css`
   - `<link rel="stylesheet" href="/static/css/jobs_enhancements.css">` →
     `/static/jobs/css/jobs_enhancements.css`
   - `<script src="/static/js/jobs_enhancements.js"></script>` →
     `/static/jobs/js/jobs_enhancements.js`

   Confirm `job_config.html` has no equivalent asset references to update
   (it currently relies on `base.css` primitives only, with no
   `extra_css`/`extra_js` block of its own) — if you find one during the
   move that isn't accounted for here, update it the same way and note the
   discrepancy in your report.

7. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/jobs/templates/` as an additional search root, alongside the
   roots already added in WO#1 (`domains/habits/templates/`) and WO#2
   (`domains/blog/templates/`, `domains/code_intel/templates/`).

8. **In `main.py`:**
   - Update the four router imports/includes to their new paths (`from
     domains.jobs.routers import jobs, ats, staging, job_config`).
   - Add the new static mount: `app.mount("/static/jobs",
     StaticFiles(directory="domains/jobs/static"), name="jobs_static")`.
     Register it **before** the general `/static` mount, per the ordering
     lesson from WO#1 (Starlette matches `Mount` routes in registration
     order; the general mount would otherwise silently swallow the more
     specific one).

---

## ACCEPTANCE CRITERIA

- [ ] `GET /jobs` renders identically to before the move — jobs table,
  staging panel, filter bar, detail drawer markup all present
- [ ] `GET /jobs/config` renders identically — search keywords list,
  watchlist panel, candidate list, scrape health panel
- [ ] `POST /jobs/stage` (queue a job link) still returns the
  `staging_row.html` fragment correctly and `GET /jobs/stage` still returns
  `staging_queue.html` correctly
- [ ] `GET /jobs/detail/{job_id}` renders `partials/job_detail.html`
  correctly for an existing job
- [ ] `POST /ats/log` still returns the updated `ats_buttons.html` fragment
  with the correct status highlighted
- [ ] `POST /jobs/config/keywords` and its toggle/delete endpoints still
  return `job_keyword_list.html` correctly
- [ ] `POST /jobs/config/watched` and its update/delete endpoints still
  return `job_watched_panel.html` correctly, including the nested
  `job_candidate_list.html` include
- [ ] `POST /jobs/config/watched/detect` still returns `job_slug_guess.html`
  correctly (network calls inside `ats_slug_service.guess_ats_slugs` may be
  mocked/stubbed — mark ⚠️ with explanation if live network probing to
  Greenhouse/Lever isn't available in this environment, same as the GitHub
  caveat in WO#2)
- [ ] `Base.metadata` table-identity check (same method as WO#1/WO#2): same
  table count before/after, `models.Job is domains.jobs.models.Job`,
  `models.StagingJob is domains.jobs.models.StagingJob`, no
  `InvalidRequestError` on mapper configuration
- [ ] `routers/dashboard.py`'s existing `Job` / `ApplicationLog` /
  `ApplicationStatus` / `StagingJob` / `StagingJobStatus` imports (via
  `from models import ...`) still resolve, and `/dashboard`'s job-related
  cards (top unapplied jobs, fit-score histogram, ATS pipeline funnel,
  staging queue summary) all render with correct data
- [ ] `grep -r "from models import"` for each of the 13 moved class names
  across the repo returns only the shim's own lines in `models.py` plus
  `routers/dashboard.py`'s import — nothing else should need updating
- [ ] `airflow/agents/job_agents.py`, `job_ats_agents.py`,
  `job_resume_context.py`, `job_dedup.py`, `job_scout_health.py`, and all
  three job-related DAG files are untouched (`git diff` shows no changes to
  any of them)

---

## For the next work order (not part of this one)

Per the priority list, **Work Order #4 = `explorer`** should be next — it's
the smallest remaining domain (one router, one template, one CSS file, no
partials, no DAGs, no cross-domain model dependencies at all since it reads
the DB schema dynamically rather than importing ORM classes). It should be
the fastest of the four real-domain moves and a good confidence check before
tackling the larger remaining domains (finance, workout, media, recipes,
planning) in whatever batch order is convenient.
