# Blog & Code Intelligence Domain-Folder Migration — Post-Mortem & Forward Requirements

**Status:** Complete and live in production.
**Commits (chronological):** `dfde5c7` (pre-migration baseline) → `65ec85a` (domain-folder move) → `02ea990` (bugfix: `improve_file`) → `9ead840` (Option B: folder-README persistence).
**Audience:** Whoever (human or agent) performs the next domain-folder migration, and whoever eventually does the final repo-wide cleanup once all domains have moved. Read Part 2 in full before starting either of those.

---

## PART 1 — POST-MORTEM

### 1. Summary

This work order migrated the **Blog** and **Code Intelligence** modules out of the flat `routers/` / `templates/` / `models.py` layout and into `domains/blog/` and `domains/code_intel/`, following the precedent already set by an earlier, separate migration of the **Habits** module (`domains/habits/`). This was a pure relocation — no schema changes, no behavior changes — with one exception: two pre-existing, unrelated bugs were discovered by code inspection during the work and fixed separately, by request, after the relocation itself was verified complete. Everything shipped in the same production deployment; no Alembic migration or `docker compose down` was required (bind-mounted source + `uvicorn --reload`).

### 2. Scope & Objective

Move, without altering behavior:
- **Models:** `BlogProjectType`, `BlogIdeaStatus`, `DIFFICULTY_LEVELS`, `BlogIdea`, `BlogIdeaCreate`, `BlogIdeaResponse` → `domains/blog/models.py`. `ReadmeStatus`, `FolderReadmeStatus`, `CommentedStatus`, `ImprovementStatus`, `CodeProject`, `CodeFile`, `FolderReadme` + their Pydantic schemas → `domains/code_intel/models.py`.
- **Routers:** `routers/blog.py`, `routers/ci_projects.py`, `routers/ci_files.py`, `routers/ci_readme.py` → `domains/{blog,code_intel}/routers/`.
- **Templates:** all blog and code-intel templates (including `partials/`) → `domains/{blog,code_intel}/templates/`.
- **Static:** `static/css/blog.css` → `domains/blog/static/css/blog.css`.
- **Config:** `models.py` (re-export shims, same pattern as the Habits migration), `main.py` (import paths + new static mount, **include-order preserved**), `core/templating.py` (`ChoiceLoader` extended).

Explicitly out of scope and untouched: `airflow/agents/blog_agents.py` (shared by both domains; splitting it needs a planned "unified AI service layer" that didn't exist yet — see §9), `services/github_service.py`, `services/airflow_service.py`, all Airflow DAG files (relocating those needs a coordinated `docker-compose.yml` volume-mount change, deferred separately).

### 3. What shipped (exact diffstat, `dfde5c7` → final)

```
 .gitignore                                          |   2 +
 core/templating.py                                  |   2 +
 domains/blog/models.py                              | 178 ++++++++ (new)
 domains/blog/routers/__init__.py                    |   0 (new, empty)
 {routers => domains/blog/routers}/blog.py            |   9 +-
 {static => domains/blog/static}/css/blog.css         |   0 (pure rename)
 {templates => domains/blog/templates}/blog.html      |   2 +-
 .../blog/templates}/blog_article.html               |   0 (pure rename)
 .../blog/templates}/partials/blog_card.html         |   0 (pure rename)
 .../blog/templates}/partials/blog_detail.html       |   0 (pure rename)
 domains/code_intel/models.py                        | 309 ++++++++++++++ (new)
 domains/code_intel/routers/__init__.py               |   0 (new, empty)
 {routers => domains/code_intel/routers}/ci_files.py  |  37 +-
 {routers => domains/code_intel/routers}/ci_projects.py |  88 +-
 {routers => domains/code_intel/routers}/ci_readme.py |  195 +-
 .../code_intel/templates}/code_intelligence.html    |   0 (pure rename)
 .../templates}/partials/code_file_detail.html       |   0 (pure rename)
 .../templates}/partials/project_agent_panel.html    |  15 +-
 .../templates}/partials/project_detail.html         |  67 +-
 .../templates}/partials/project_list.html           |   0 (pure rename)
 main.py                                              |   6 +-
 models.py                                            | 456 +------ (shim swap)
 22 files changed, 876 insertions(+), 490 deletions(-)
```

`ci_files.py`, `ci_projects.py`, and `ci_readme.py` carry more line changes than a pure move would — that's the two bugfixes below layered on top, not migration noise.

### 4. Issues discovered during migration (root cause analysis)

Neither of these was introduced by the migration. Both were pre-existing, and both were found only because the migration required reading every line of the moved files closely — no test suite existed to catch them beforehand (see §9, "What could be improved").

#### 4.1 Folder-scoped README persistence was completely non-functional

**Symptom:** `PATCH /code-intel/projects/{id}/folder-readme` appeared to succeed (`200 OK`) but the content vanished immediately; `GET /code-intel/projects/{id}/folder-readme` crashed with an unhandled `AttributeError` on every single call.

**Root cause:** `ci_readme.py` read and wrote `project.folder_readme_md`, `project.folder_readme_path`, `project.folder_readme_generated_at` directly on `CodeProject` ORM instances — but `CodeProject` never declared those as mapped columns. Setting an unmapped attribute on a SQLAlchemy declarative instance is a plain Python operation with no persistence effect; `db.commit()` succeeds and silently does nothing. Reading it back on a freshly-queried instance raises `AttributeError` (not a friendly `None`) since the attribute was never actually assigned in that object's lifetime. A separate, fully-built table (`FolderReadme`, one row per `(project_id, folder_path)`, with its own status lifecycle) already existed in the schema and was never wired up to these two endpoints — most likely the model layer was built ahead of the router layer and the connection was simply never made. `ci_projects.py`'s `project_status` endpoint already used a defensive `getattr(project, "folder_readme_generated_at", None)` — a strong signal the original author suspected this attribute might not exist, but the sibling code path wasn't fixed to match.

**Fix (commit `9ead840`, "Option B" — chosen over the cheaper "just add the missing columns to `CodeProject`" alternative):** `save_folder_readme` / `get_folder_readme` now upsert/query the real `FolderReadme` table by `(project_id, folder_path)`. `folder_path` became a required query parameter on the GET endpoint, since a project can now hold **multiple** persisted folder READMEs simultaneously (the cheaper column-based fix would have kept the single-scalar-slot limitation, meaning generating a README for folder B would have silently overwritten folder A's). `project_detail`, `sync_files_from_github`, `generate_readme`, `save_readme_edits`, `push_readme`, and `trigger_readme_dag` were all updated to pass the most-recently-generated `FolderReadme` into their templates so a saved folder README survives a page/panel reload. `project_status` gained an optional `folder_path` query param to look up the correct row instead of `getattr`-ing a phantom attribute.

**A second, related bug found and fixed in the same pass:** `startPolling()` (client-side JS, `project_detail.html`) unconditionally reset `_pollingForFolderReadme = false` on every call — meaning `startPollingForFolderReadme()` setting it `true` was immediately clobbered one line later, so the "Folder README ready ✓" toast could never fire. Fixed by threading a `forFolderReadme` parameter through instead of a bare reset, since the function had to be touched anyway to add `folder_path` to the polling URL.

**Verification:** a dedicated end-to-end test (13 checks) proved: independent folders don't overwrite each other (the exact scenario the cheaper fix would have failed), the required-param 422 behavior, template pre-population surviving a simulated reload, `project_status` correctly scoped by `folder_path`, and `save_as_project` still correctly copying into `CodeProject.readme_md`. An isolated Node.js repro separately proved the polling-flag fix.

#### 4.2 `agent_code_improver` return-type mismatch crashed the Improve action

**Symptom:** `POST /code-intel/files/{id}/improve` would fail at `db.commit()` time.

**Root cause:** `agent_code_improver()` (in `airflow/agents/blog_agents.py`, out of scope, untouched) returns `(content, remaining_tokens)` — a 2-tuple — despite its own `-> str` type annotation, because it doesn't unpack the underlying `_cerebras()` helper's return value the way its sibling functions (`agent_code_narrator`, `agent_code_commenter`) do. `ci_files.py`'s `improve_file` assigned that raw tuple straight to `code_file.improvement_notes`, a `Text` column — no DBAPI driver can bind a tuple as a scalar parameter, so this raised at commit time, uncaught by the surrounding `try/except` (which only wrapped the agent call itself).

**Fix (commit `02ea990`):** one-line change in `ci_files.py`: `notes, _remaining_tokens = agent_code_improver(...)`. Entirely contained inside the in-scope router file; `blog_agents.py` was correctly left untouched per the out-of-scope boundary.

**Verification:** a targeted repro (mocked the agent's tuple return, hit the real endpoint, confirmed the persisted value is a plain string) plus a full regression re-run.

### 5. Verification methodology

No test suite existed in the repository at the start of this work. The following was built as a throwaway verification harness (never committed to the repo) and is a strong starting point for the permanent pytest suite planned for after all migrations complete (see §9):

- **In-memory SQLite** standing in for MariaDB (`sqlite+aiosqlite:///:memory:`), with `database.py`'s `get_db`/`init_db` swapped for a version pointed at it. Schema created fresh per test run via `Base.metadata.create_all()` — proves the ORM layer is internally consistent without needing a live DB server.
- A **real FastAPI app** built from only the routers under test (no need to boot the entire multi-domain `main.py`, which pulls in dozens of sibling modules irrelevant to any one migration), exercised with `httpx.AsyncClient` + `ASGITransport` — real request/response cycles, real Jinja2 rendering, real SQLAlchemy queries.
- **Minimal stand-ins** (clearly marked, never treated as authoritative) for the handful of pre-existing dependencies outside any given migration's scope: a stub `domains/habits/models.py` (just the 4-5 names actually imported elsewhere), and stub `templates/base.html` / `templates/dashboard.html` (just enough block structure for `{% extends %}` to resolve).
- **Before/after comparison**: the harness was run against a git-tagged pre-migration baseline commit first, to get a known-good snapshot, then re-run identically post-migration. Any check that passed before and fails after is a real regression; any check that fails both before and after is a pre-existing bug (handled per §4, not silently fixed).
- **Patch-application verification**: the final unified diff was applied to a byte-exact fresh checkout of the true pre-migration files and diffed against the actual final state — zero discrepancies — before being handed off, to guarantee the diff handed to the person doing the real deployment was complete and exact.

### 6. Deployment

- **Docker:** `web` bind-mounts the whole project root (`.:/app`) and runs `uvicorn --reload`; new/moved files under `domains/` are visible to the container immediately, and the reload watcher picks up `.py` changes automatically (Jinja's `auto_reload=True` handles `.html` changes without even needing that). `docker compose restart web` was recommended anyway as a clean single checkpoint, not because reload wouldn't work. **No `docker compose down`, no rebuild.**
- **Alembic:** not applicable. No column, table, or type changed at any point — confirmed via a `Base.metadata` table-identity check (same 18 tables, same `models.X is domains.Y.models.X` identity, `configure_mappers()` clean) run before and after. The Option B fix specifically added zero schema — `FolderReadme` already existed; only the *router code* that should have been reading/writing it was fixed.
- **Airflow containers:** untouched, not restarted — no DAG files moved, `blog_agents.py` (which the DAGs presumably import) untouched.

### 7. What went well

- Reconstructing a git baseline first (rather than editing "in place" with no history) made every subsequent claim ("blog_agents.py is untouched," "the patch applies byte-exact," "same table count before/after") independently, mechanically verifiable rather than asserted.
- Treating "Options A vs B" as a real decision with a concrete blast-radius estimate (files touched, lines changed, migration-or-not) before writing code avoided committing to the cheaper fix and having to redo it once its limitation (silent cross-folder overwrite) became apparent.
- Catching the `_pollingForFolderReadme` bug as a byproduct of *having to touch that exact function anyway* for an unrelated reason, rather than going looking for extra bugs to fix — kept the diff focused while still not leaving an adjacent, freshly-relevant bug in place.

### 8. What could be improved / lessons learned

- **No test suite existed.** Both real bugs in §4 were found by manual code reading, not by anything automated. This is explicitly being addressed next (owner: repo maintainer, timing: after all remaining domain migrations — see §9).
- **`models.py` accreted duplicate imports and multiple "append these classes to the bottom" comment blocks over time** (`datetime`, `Decimal`, `Optional`, `BaseModel` are each imported 3+ times), a sign the file was built by repeatedly pasting in new modules rather than integrating them. Worth a light consolidation pass whenever it's next opened for a domain extraction.
- **Unmapped-attribute bugs (§4.1) are easy to introduce and invisible until read carefully** — SQLAlchemy raises no error at write time, which is exactly what let this ship silently. When touching any router that does `some_orm_obj.some_attr = value`, it's worth a 5-second sanity check that `some_attr` is an actual `mapped_column` on that class.

### 9. Deferred / backlog items (not migration-blocking; logged for continuity)

These came up in review but were explicitly not acted on this round — listed here so they aren't lost, not as instructions to act on them now:

- `ImprovementStatus.APPLIED` and `ImprovementStatus.PUSHED` are defined but never set anywhere in code — likely an unfinished "apply suggested fix" / "push improved code to GitHub" feature. Owner's call: keep the enum values (GitHub-push capability for code improvements may still be built), don't prune.
- `CodeProject.folder_readme_coverage` (aggregates `FolderReadme` status counts) is fully implemented but rendered by no current template — the start of an unbuilt "folder README coverage dashboard." Noted as a candidate future project; reinforces that Option B (§4.1) was the right call since it already produces the exact data that dashboard would need per-folder.
- Redundant imports in `models.py` (see §8) — owner is addressing `models.py` directly and separately.
- Formal pytest suite — owner's stated plan is to build this **after** all remaining domain migrations land; §5 above is meant to seed that effort.
- AI-provider-layer consolidation for `blog_agents.py` (see Part 2, §9 below) — owner confirmed this is a definite to-do, to avoid the same capability (e.g. Cerebras/Gemini calling code) existing in more than one place as more domains migrate.

---

## PART 2 — REQUIREMENTS FOR FINAL CLEANUP (execute only once ALL domain-folder migrations are complete)

This section is written for whoever — human or agent — does the *last* migration and the subsequent cleanup pass. Do not execute any of this mid-way through the migration program; every item below assumes **every** domain has already been extracted out of the flat `routers/` / `templates/` / `models.py` layout. Executing it early will break things that haven't moved yet.

### 1. How to know you're actually ready

Run this before touching anything in this section:

```bash
grep -rn "from models import" --include="*.py" .
```

**Target end-state:** this returns nothing except `models.py`'s own internal `from core.base_model import Base` line (or nothing at all, if `models.py` is deleted per §2 below). If it returns anything importing a *class name* (not just `Base`) from `models`, that consuming file's domain hasn't fully migrated yet, or hasn't been updated to import from the new domain module directly — **do not proceed** until it's empty.

At the time of writing (end of the Blog/Code Intelligence migration), this command returns:
```
routers/dashboard.py:18:from models import (
models.py:8:  (Base, in a comment)
models.py:343: (comment, inside the Blog shim docstring)
models.py:360: (comment, inside the Code Intel shim docstring)
models.py:381: (comment, inside the Habits shim docstring)
```
i.e. exactly one real remaining consumer (`routers/dashboard.py`) plus the shims' own comments. Confirm this list has shrunk to zero real consumers (dashboard.py included) before proceeding.

### 2. `models.py` final cleanup

Once every class has a domain home, `models.py` has two possible end states — **pick one deliberately, don't leave it half-migrated**:

- **Option 1 — delete `models.py` entirely.** Cleanest, but see the critical risk in §3 below before doing this: something else must guarantee every domain's model module gets imported before the first database query, or cross-domain string-based relationships will fail unpredictably (only on whichever request path happens to be first to touch them).
- **Option 2 (recommended) — reduce `models.py` to a pure registry, no class definitions, no shims:**
  ```python
  # models.py — model registry.
  # Every ORM model lives in its own domain's models.py. This file's only
  # job is to guarantee every domain module gets imported at least once
  # before the first query, so SQLAlchemy's mapper registry has every
  # class available for string-based relationship() resolution.
  # Import this module (or call ensure_all_models_registered()) once,
  # early, from database.py's init_db().
  from domains.blog import models as _blog_models          # noqa: F401
  from domains.code_intel import models as _code_intel_models  # noqa: F401
  from domains.habits import models as _habits_models      # noqa: F401
  # ... one line per remaining domain, added as each one migrates
  ```
  This turns an implicit dependency (whatever happens to import `models.py` triggers registration) into an explicit, auditable one, and is a much smaller diff than deleting the file and chasing every remaining import site.

Either way: **`routers/dashboard.py` must be updated** as part of this step — it's the one confirmed real consumer as of this migration. Change:
```python
from models import (
    Job, ApplicationLog, ApplicationStatus,
    StagingJob, StagingJobStatus,
    Transaction, BlogIdea, BlogIdeaStatus,
    Habit, HabitLog, HabitSettings,
    JournalEntry, WeeklySynthesis,
)
```
to direct per-domain imports, e.g.:
```python
from domains.jobs.models import Job, ApplicationLog, ApplicationStatus         # once Jobs migrates
from domains.staging.models import StagingJob, StagingJobStatus               # once Staging migrates
from domains.finance.models import Transaction                                # once Finance migrates
from domains.blog.models import BlogIdea, BlogIdeaStatus
from domains.habits.models import Habit, HabitLog, HabitSettings
from domains.journal.models import JournalEntry, WeeklySynthesis              # once Journal migrates
```
(domain module names above are inferred, not confirmed — see §8.)

### 3. Critical risk: SQLAlchemy string-based relationship resolution

**Read this before deleting or reducing `models.py`.** `BlogIdea.code_file` / `BlogIdea.code_project` and `CodeFile.blog_ideas` / `CodeProject.blog_ideas` use `relationship("CodeFile", ...)` / `relationship("BlogIdea", ...)` — **string** class names, resolved lazily against SQLAlchemy's shared mapper registry the first time any query touches that relationship, not at import time. This only works if *both* `domains.blog.models` and `domains.code_intel.models` have been imported by *something* before that first query — currently guaranteed transitively by `models.py`'s shim imports.

As more domains migrate, **this problem compounds** — grep the full model layer for every other cross-domain string relationship before finalizing the registry, not just the Blog/CodeIntel one this migration already handles. From the modules already visible in the pre-migration `models.py` (not yet extracted at the time of writing), at least these additional cross-domain relationships exist and will need the same care:
- `WeeklyPlanDay.workout_session` → `relationship("WorkoutSession", ...)` (Weekly Planning ↔ Workout)
- `WeeklyPlanMeal.recipe` / `.swap_recipe` → `relationship("Recipe", ...)` (Weekly Planning ↔ Recipes)
- `WeeklyPlanDay.journal_entry_id` → (Weekly Planning ↔ Journal; confirm whether a `relationship()` exists or it's FK-only)

**Before finalizing whichever registry approach you pick in §2**, run:
```bash
grep -rn 'relationship(' domains/*/models.py
```
and confirm every string argument names a class whose module is included in the registry (§2, Option 2) or is otherwise guaranteed to be imported. If you pick Option 1 (delete `models.py` outright), you must add this same guarantee somewhere else (e.g., inside `database.py`'s `init_db()`), or these relationships will intermittently raise `sqlalchemy.exc.InvalidRequestError: ... failed to locate a name` depending on which endpoint happens to be hit first in a given process — a genuinely nasty, hard-to-reproduce failure mode if this step is skipped.

### 4. `core/templating.py` final cleanup

Once every domain has its own `templates/` directory, decide whether the top-level `templates/` `FileSystemLoader` entry stays. It almost certainly should — it's the natural home for genuinely shared/core templates that don't belong to any one domain: `base.html`, `dashboard.html`, `404.html`, `500.html`. Don't remove it; just confirm nothing *domain-specific* still lives there once every domain's own loader entry has been added. Final `ChoiceLoader` should look like:
```python
templates.env.loader = ChoiceLoader([
    FileSystemLoader("templates"),               # shared/core only: base.html, dashboard.html, 404.html, 500.html
    FileSystemLoader("domains/habits/templates"),
    FileSystemLoader("domains/blog/templates"),
    FileSystemLoader("domains/code_intel/templates"),
    # ... one line per remaining domain
])
```
Also worth resolving at this point: `templates/desktop.ini` is a stray Windows Explorer artifact that's been explicitly punted on twice now (noted, not investigated, in both prior migrations referencing `code_intelligence.html`). Once no domain-specific templates remain at the top level, decide whether to finally delete it or determine why it's tracked in git at all.

### 5. `main.py` final cleanup

By the end, `main.py`'s router imports should be entirely `from domains.X.routers import Y` — zero `from routers import ...` lines except for whatever stays genuinely cross-cutting (see §6). Preserve whatever **relative include-order constraints** exist for path-matching specificity — this migration's own precedent (`ci_files`/`ci_readme` registered before `ci_projects`, because of route path overlap) is exactly the kind of thing to check for in each remaining domain before reordering anything. Also confirm each domain's static mount (if it has one) is registered **before** the general `/static` mount, per the same Starlette route-matching-order rule already established for `/static/habits` and `/static/blog`.

### 6. Fate of `routers/`, `templates/`, `static/`

- `routers/dashboard.py` almost certainly stays at the top level (or becomes its own thin non-domain "core" module) rather than moving into a domain folder — it has no models of its own and structurally aggregates every other domain. Don't force it into a domain just for consistency.
- `routers/explorer.py` — present in `main.py`'s imports but its purpose was never described in anything shared with this migration. Investigate what it actually does before assuming which domain (if any) it belongs to.
- `static/` (top-level) likely keeps whatever is genuinely global (shared CSS/JS used by `base.html` etc.) even after every domain-specific asset has moved to its own `domains/X/static/`.

### 7. Inferred remaining domain boundaries — verify before use

**Important caveat:** the model-layer boundaries below are known with confidence, because the original `models.py` content (Jobs/Finance/Recipe/Workout/Media/Weekly-Planning/Journal sections) was fully read at the start of this engagement. The **router and template contents for these domains were never provided or read** in this engagement — only their `main.py` import names are known. Treat the domain groupings below as a starting hypothesis to confirm against the actual router files, not a pre-validated plan:

| Candidate domain | Model classes (confirmed from `models.py`) | Router modules (names only, from `main.py`) |
|---|---|---|
| Jobs / ATS | `Job`, `ApplicationLog`, `ApplicationStatus`, `JobSearchKeyword`, `WatchedCompany`, `JobScoutRunLog` + schemas | `ats.py`, `jobs.py`, `job_config.py` |
| Staging | `StagingJob`, `StagingJobStatus` + schemas | `staging.py` (possibly merged into Jobs — staging feeds directly into `linkedin_jobs`) |
| Finance | `AccountType`, `Category`, `Account`, `Transaction` + schemas | `finance_summary.py`, `finance_ledger.py`, `finance_upload.py`, `finance_settings.py` |
| Journal | `JournalEntry`, `WeeklySynthesis` | `journal.py` |
| Recipes | `Ingredient`, `RecipeTag`, `Recipe`, `RecipeIngredient`, `PantryItem` + related enums/schemas | `recipe_extract.py`, `recipe_discovery.py`, `pantry.py`, `recipes.py` |
| Workout | `WorkoutLocation`, `Equipment`, `Exercise`, `WorkoutPlan`, `WorkoutPlanDay`, `WorkoutPlanExercise`, `WorkoutSession`, `WorkoutSet`, `BodyMetric` + enums | `workout.py`, `workout_log.py`, `workout_plans.py`, `workout_settings.py` |
| Media | `StreamingService`, `MediaItem`, `UserMedia`, `TVSeasonProgress`, `MediaRecommendation` + enums | `media.py`, `media_search.py`, `media_recommend.py`, `media_settings.py` |
| Weekly Planning | `UserIntent`, `WeeklyPlan`, `WeeklyPlanDay`, `WeeklyPlanMeal`, `ShoppingList` + enums | `intent.py`, `weekly_plan.py` |
| *(unknown)* | *(none identified)* | `explorer.py` — investigate before assigning |

Given the cross-domain relationships already noted in §3 (Weekly Planning ↔ Workout, Weekly Planning ↔ Recipes), **migrating Weekly Planning before Workout and Recipes are done will re-create exactly the same string-relationship risk this migration handled for Blog ↔ Code Intelligence** — plan the migration order with that dependency in mind, or be prepared to handle it the same way (no cross-imports between the two domains' `models.py` files; only the shared registry needs both).

### 8. Related but separate initiatives — sequencing recommendation

- **AI provider-layer consolidation.** `blog_agents.py`'s own docstring already earmarks a "unified AI service layer" as planned; `airflow/agents/gemini_client.py` (`call_gemini_text`/`call_gemini_json`) is the first piece of it, already used by `job_agents.py` but deliberately not yet backported into `blog_agents.py`. Recommended sequencing: extract the remaining shared provider helpers (`_gemini_flash`, `_cerebras`, key fetchers) into that shared service **first**, then split `blog_agents.py`'s domain-specific agent functions into `domains/blog/agents.py` / `domains/code_intel/agents.py` (and equivalently for whichever other domains have agent functions mixed into shared files) — doing the provider consolidation before the domain split avoids touching the same file twice for two different reasons.
- **Formal pytest suite.** See Part 1 §5 for the harness pattern already proven to work (in-memory SQLite, stub only what's genuinely out-of-domain-scope, real FastAPI + httpx request cycles, before/after comparison against a tagged baseline). This is a natural fit to formalize once all domains have landed and there's a complete, final router/model layout to write fixtures against — writing it mid-migration risks having to rewrite fixtures every time another domain moves.
- **Docstring / documentation consistency pass.** Docstrings have been added incrementally to whatever was touched in this and future migrations (Google-style: `Args`/`Returns`/`Raises`, matching the convention `blog_agents.py`'s own Code Commenter agent already prescribes elsewhere in the codebase). Incremental coverage will have gaps by the time all migrations finish — a dedicated audit pass across the full `domains/` tree is worth scheduling once the structure is final, rather than repeatedly partially covering files that get touched again later.

### 9. Final verification checklist (run after the last domain migration + this cleanup)

- [ ] `grep -rn "from models import" --include="*.py" .` returns nothing (or only the registry file's own `Base` import, if `models.py` was kept per §2 Option 2).
- [ ] `grep -rn 'relationship(' domains/*/models.py` — every string-named class's module is covered by the registry mechanism chosen in §2/§3.
- [ ] `Base.metadata` table-identity check: same total table count before/after, `models.X is domains.Y.models.X` (or no `models.py` at all) for every migrated class, `sqlalchemy.orm.configure_mappers()` raises no `InvalidRequestError`.
- [ ] Every remaining router's `Jinja2Templates(...)` instantiation has been replaced with `from core.templating import templates`.
- [ ] Every domain's static mount (if any) is registered before the general `/static` mount in `main.py`.
- [ ] Full regression suite (formal pytest suite, once built per §8, or an equivalent harness per Part 1 §5) passes against the final structure.
- [ ] `docker compose restart web` (bind-mount + `--reload` means no rebuild/`down` needed, consistent with every migration so far) — re-confirm this assumption still holds if the `Dockerfile`/`docker-compose.yml` volume configuration changes for any reason during the remaining migrations.