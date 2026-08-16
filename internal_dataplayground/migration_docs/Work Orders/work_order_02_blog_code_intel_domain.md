## Work Order #2 — Domain Migration: `blog` + `code_intel`

*(Moved together because of a live cross-domain foreign key: `BlogIdea.code_file_id` / `code_project_id` → `code_files.id` / `code_projects.id`, with `back_populates` relationships on both sides. Splitting them across two separate work orders would create a window where one domain's models reference a not-yet-relocated class.)*

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
- **`airflow/agents/blog_agents.py` is explicitly OUT OF SCOPE and must NOT 
  be moved or split**, even though it contains functions belonging to both 
  domains (blog: `agent_researcher`, `agent_ghostwriter`, `agent_refiner`, 
  `agent_editor`, `agent_idea_expander`; code_intel: `agent_readme_writer`, 
  `agent_code_narrator`, `agent_code_commenter`, `agent_code_improver`). It 
  also contains shared provider-call helpers (`_gemini_flash`, `_cerebras`, 
  key fetchers, etc.) used by both groups. Splitting this file correctly 
  requires the unified AI service layer planned separately (do not attempt a 
  parallel/duplicate version of that work here). Leave this file exactly 
  where it is; both domains continue importing from 
  `airflow.agents.blog_agents` exactly as they do today.
- `services/github_service.py` and `services/airflow_service.py` are shared 
  infrastructure and stay at their current paths. Do not move them.
- Do NOT move any DAG files (`life_os_blog_*.py`, `life_os_code_*.py`, 
  `life_os_readme_writer.py`, `life_os_idea_expander.py`). DAG relocation 
  requires a coordinated `docker-compose.yml` volume-mount change and is 
  handled in a separate, later work order.
- `templates/desktop.ini` is a stray artifact referencing 
  `code_intelligence.html`. Ignore it — do not move it, do not delete it, do 
  not investigate it further. Note its existence in your report if you 
  notice it, nothing more.

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
the result ⚠️ rather than ✅ or leaving it blank. (Note: GitHub-dependent 
endpoints in code_intel — file pull/push — cannot be verified against the 
real GitHub API in this environment; verify route registration and template 
rendering paths instead, and mark those specific checks ⚠️ with that 
explanation.)

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

**Blog domain — models to extract from `models.py`:**
`BlogProjectType`, `BlogIdeaStatus`, `DIFFICULTY_LEVELS`, `BlogIdea`, 
`BlogIdeaCreate`, `BlogIdeaResponse`

**Code Intel domain — models to extract from `models.py`:**
`ReadmeStatus`, `FolderReadmeStatus`, `CommentedStatus`, `ImprovementStatus`, 
`CodeProject`, `CodeFile`, `FolderReadme`, `CodeProjectCreate`, 
`CodeProjectResponse`, `CodeFileResponse`, `FolderReadmeResponse`, 
`FolderReadmeCreate`

**Routers:**
- `routers/blog.py`
- `routers/ci_projects.py`
- `routers/ci_files.py`
- `routers/ci_readme.py`

**Templates — blog:**
- `templates/blog.html`
- `templates/blog_article.html`
- `templates/partials/blog_card.html`
- `templates/partials/blog_detail.html`

**Templates — code_intel:**
- `templates/code_intelligence.html`
- `templates/partials/project_list.html`
- `templates/partials/project_detail.html`
- `templates/partials/project_agent_panel.html`
- `templates/partials/code_file_detail.html`

**Static:**
- `static/css/blog.css`
- *(code_intel has no dedicated CSS file — its styles are inline in 
  `code_intelligence.html`'s `extra_css` block. No static file to move for 
  this domain.)*

**Shared infra referenced but not moved (read-only for import-path 
confirmation, do not edit):**
- `services/github_service.py`
- `services/airflow_service.py`
- `airflow/agents/blog_agents.py`

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:**
- `routers/dashboard.py` (imports `BlogIdea`, `BlogIdeaStatus` from `models` 
  — must keep working via shim, same pattern as habits)

---

## STEPS

1. **Create `domains/blog/models.py`.** Move `BlogProjectType`, 
   `BlogIdeaStatus`, `DIFFICULTY_LEVELS`, `BlogIdea`, `BlogIdeaCreate`, 
   `BlogIdeaResponse` there verbatim. Import `Base` from `core.base_model`.

2. **Create `domains/code_intel/models.py`.** Move `ReadmeStatus`, 
   `FolderReadmeStatus`, `CommentedStatus`, `ImprovementStatus`, 
   `CodeProject`, `CodeFile`, `FolderReadme`, and their Pydantic schemas 
   there verbatim. Import `Base` from `core.base_model`.

3. **Preserve the cross-domain relationship.** `BlogIdea.code_file` / 
   `code_project` and `CodeFile.blog_ideas` / `CodeProject.blog_ideas` use 
   SQLAlchemy `relationship("CodeFile", ...)` / `relationship("BlogIdea", 
   ...)` with **string** class names, not direct imports — this means they 
   resolve via the shared mapper registry at query time, not at import time. 
   Do not add cross-imports between `domains/blog/models.py` and 
   `domains/code_intel/models.py` to "fix" this — none are needed. The only 
   requirement is that both modules get imported by something before the 
   first query runs, which is satisfied by Step 4's re-export shim (both 
   will load whenever anything does `import models`, exactly like the 
   `habits` migration).

4. **In `models.py`:** delete the moved class bodies (from both domains) 
   and replace with re-export shims: `from domains.blog.models import 
   BlogProjectType, BlogIdeaStatus, DIFFICULTY_LEVELS, BlogIdea, 
   BlogIdeaCreate, BlogIdeaResponse` and `from domains.code_intel.models 
   import ReadmeStatus, FolderReadmeStatus, CommentedStatus, 
   ImprovementStatus, CodeProject, CodeFile, FolderReadme, ...` (full 
   Pydantic list). Tag both with `# TODO: remove after all cross-references 
   are updated`, consistent with the `habits` shim already in place.

5. **Move routers:**
   - `routers/blog.py` → `domains/blog/routers/blog.py`
   - `routers/ci_projects.py` → `domains/code_intel/routers/ci_projects.py`
   - `routers/ci_files.py` → `domains/code_intel/routers/ci_files.py`
   - `routers/ci_readme.py` → `domains/code_intel/routers/ci_readme.py`
   
   Update each file's model imports to pull from the new domain module 
   paths instead of `models`. Update each file's `templates = 
   Jinja2Templates(directory="templates")` to `from core.templating import 
   templates`, matching the pattern established in Work Order #1. Leave all 
   `from services...` and `from airflow.agents.blog_agents import ...` 
   imports unchanged — those files are not moving.

6. **Move templates**, preserving the `partials/` subfolder structure, into 
   `domains/blog/templates/` and `domains/code_intel/templates/` 
   respectively, per the SCOPE lists above.

7. **Move `static/css/blog.css`** to `domains/blog/static/css/blog.css`. 
   Update the `<link rel="stylesheet" href="/static/css/blog.css">` 
   reference inside `blog.html` to `/static/blog/css/blog.css`.

8. **Update `core/templating.py`'s `ChoiceLoader`** to add 
   `domains/blog/templates/` and `domains/code_intel/templates/` as 
   additional search roots, alongside the existing `templates/` and 
   `domains/habits/templates/` from Work Order #1.

9. **In `main.py`:**
   - Update the four router imports/includes to their new paths (`from 
     domains.blog.routers import blog`, `from domains.code_intel.routers 
     import ci_projects, ci_files, ci_readme`).
   - **Preserve the existing include order** — the current code comments 
     that `ci_files` and `ci_readme` must be included *before* `ci_projects` 
     for path-matching specificity. Keep that relative ordering after the 
     import path change.
   - Add the new static mount `app.mount("/static/blog", 
     StaticFiles(directory="domains/blog/static"), name="blog_static")`. 
     Register it **before** the general `/static` mount, per the ordering 
     lesson from Work Order #1 (Starlette matches `Mount` routes in 
     registration order; the general mount would otherwise swallow the more 
     specific one).

---

## ACCEPTANCE CRITERIA

- [ ] `GET /blog` renders identically to before the move — kanban board, all 
  four columns, mobile tab strip
- [ ] `GET /code-intel` renders identically — project list panel, empty-state 
  message if no projects
- [ ] Creating a blog idea via BYOI (`POST /blog/ideas`) still returns the 
  `blog_card.html` fragment correctly
- [ ] Opening a blog idea's detail drawer (`GET /blog/ideas/{id}`) renders 
  `blog_detail.html` correctly, **including its code-project and code-file 
  dropdowns** — this specifically exercises the cross-domain read (`select 
  CodeFile...`, `select CodeProject...` inside `routers/blog.py`'s 
  `idea_detail` handler) and confirms the relationship/import setup from 
  Step 3 didn't break anything
- [ ] Creating a code project (`POST /code-intel/projects`) and opening its 
  detail (`GET /code-intel/projects/{id}/detail`) both render correctly
- [ ] `Base.metadata` table-identity check (same method as Work Order #1): 
  same table count before/after, `models.BlogIdea is domains.blog.models.BlogIdea`, 
  `models.CodeProject is domains.code_intel.models.CodeProject`, no 
  `InvalidRequestError` on mapper configuration — confirms no duplicate 
  registration from the shim + relationship-string combination
- [ ] `routers/dashboard.py`'s existing `BlogIdea`/`BlogIdeaStatus` import 
  and usage (via `from models import ...`) still resolves and its blog 
  pipeline counts render correctly on `/dashboard`
- [ ] `grep -r "from models import.*BlogIdea"` and `grep -r "from models 
  import.*CodeProject\|CodeFile\|FolderReadme"` across the repo return only 
  the shim's own lines in `models.py` plus `routers/dashboard.py`'s blog 
  import — nothing else should need updating
- [ ] GitHub-dependent routes (`POST /code-intel/files/{id}/pull`, `POST 
  .../push-comments`, `POST /code-intel/projects/{id}/sync`) — verify route 
  registration and that the handler reaches the `github_service` call 
  correctly (mocked/stubbed network layer is fine); mark ⚠️ with 
  explanation since live GitHub API calls aren't available in this 
  environment
- [ ] `airflow/agents/blog_agents.py` is untouched (`git diff` shows no 
  changes to this file) and both domains' routers still successfully import 
  from it at their original module path

---

Once this one lands, I'd suggest **Work Order #3 = `jobs` domain** (per your priority list), which is a cleaner single-domain move like `habits` was — no cross-domain FK complications there, just the jobs/ats/staging/job_config router quartet plus its two DAGs. That'll also be the first time we hit the DAG-relocation question directly, so I'd want to draft that one carefully once we've seen how #2 goes.