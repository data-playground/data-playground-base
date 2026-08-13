# Jobs Domain Migration & Follow-On Work — Post-Mortem & Forward Requirements

**Status:** Complete and verified via automated harness. Not yet deployed/observed in
production at the time of writing — see "Outstanding manual actions" below before
considering this fully live.
**Scope of this document:** (1) a post-mortem of the `jobs` domain-folder migration
(Work Order #3) and everything built on top of it in the same engagement — bug fixes,
a new DAG, a schema change, a navigation fix, and a full rework of how the Jobs page
loads data; (2) a checklist for whoever migrates the next domain; (3) a **final
cleanup checklist** to execute only once every domain has been migrated, following
the same deferred-batching principle established in the habits and blog/code_intel
postmortems; (4) an explicit list of decisions that were proposed and **rejected**
during this engagement, so they aren't silently re-proposed without context.

This document assumes no prior conversation context, per the convention set by the
habits and blog/code_intel postmortems. If you're an agent picking up work on the
`jobs` domain, or performing the final cross-domain cleanup, read this whole document
first — specifically Section 5 (lessons learned), Section 9 (outstanding manual
actions), and Part 2 before touching any files.

---

## PART 1 — POST-MORTEM

## 1. Summary

This engagement started as a standard domain-folder migration (Work Order #3),
following the exact precedent set by the habits (WO#1) and blog/code_intel (WO#2)
migrations: move `jobs`-owned models, routers, templates, and static assets out of
the flat `routers/` / `templates/` / `models.py` layout into `domains/jobs/`, leave a
re-export shim in `models.py`, update `main.py` and `core/templating.py`. That part
shipped cleanly and was verified with a real FastAPI + in-memory SQLite harness.

Unlike the prior two migrations, this engagement continued well past the migration
itself, in the same conversation, across seven distinct phases:

1. The domain-folder migration proper (WO#3).
2. A round of bug fixes and small polish items, several of them **pre-existing bugs**
   found during the migration (not introduced by it) — most notably a case-mismatch
   bug that silently broke the ATS status button highlight on every request.
3. A new capability: a "Staging Promoter" Airflow DAG that lets manually-queued job
   links (pasted into the "Add Job Link" panel) get scraped, scored, and promoted
   into `linkedin_jobs` without waiting for a LinkedIn search to find them — either on
   a daily schedule or on-demand via a new "⚡ Process Now" button.
4. A `UniqueConstraint` added to `Job` to guard against duplicate postings, motivated
   directly by phase 3 adding a *third* code path that inserts into `linkedin_jobs`.
5. Two real, pre-existing navigation bugs found and fixed in the shared sidebar
   (`templates/partials/sidebar.html`) — not files owned by the `jobs` domain, but
   directly caused by how the `jobs` domain's routers set `active_module`.
6. A full rework of how `/jobs` loads data — the single largest piece of this
   engagement. The original query had no `LIMIT` at all and rendered every row in
   `linkedin_jobs` twice (once for desktop, once for mobile) on every page load. At
   ~2,300 rows this was hanging the page for over a minute. Fixed by moving all
   filtering into SQL, adding a hard per-request cap (300 rows) enforced via keyset
   (cursor) pagination, and converting every filter control from "hide already-loaded
   rows with CSS" to "fetch a new page from the server."
7. A `search_date`-refresh fix across all three job-ingestion DAGs, so a posting that
   stays open for months doesn't have its `search_date` permanently frozen at
   whenever it was first discovered. A more elaborate "archive old jobs" design (new
   columns, a new enum status, a new retention DAG) was proposed, iterated on, and
   then **explicitly rejected by the project owner** as unnecessary complexity — see
   Section 10, "Considered and Rejected," before proposing anything similar again.

## 2. Scope & Objective

**In scope, and delivered:**
- Everything WO#3's own SCOPE section specified (see that document, preserved in
  full in the conversation this postmortem summarizes) — 13 model classes, 4 routers,
  10 templates, 3 static assets, plus the `models.py` / `main.py` /
  `core/templating.py` edits.
- Bug fixes surfaced during and after the migration (Section 4).
- A new Airflow DAG and one new FastAPI endpoint + template (Section 4, phase 3).
- A schema addition (`UniqueConstraint` + two `Index` declarations on `Job`) —
  declared in the ORM, **not yet applied to the live database** (Section 9).
- A sidebar bug fix in a **shared, non-jobs-owned file**
  (`templates/partials/sidebar.html`) — flagged explicitly since it's outside the
  domain's own folder and future migrations should know this file has already been
  touched once for jobs-specific reasons.
- A full query/rendering rework of `GET /jobs`, plus a new `GET /jobs/rows` endpoint.
- A `search_date`-refresh behavior change across three DAG files.

**Explicitly out of scope, untouched, confirmed via diff review at each phase:**
- `services/ats_slug_service.py` — stays in the shared `services/` layer, same
  reasoning as WO#2 keeping `github_service.py` in place.
- `airflow/agents/job_resume_context.py`, `job_dedup.py`, `job_scout_health.py`,
  `gemini_client.py` — untouched except where explicitly noted (see below).
- `routers/dashboard.py` — the cross-domain reader, deliberately not migrated (see
  Part 2 for what it needs once the *final* cleanup pass happens).
- DAG relocation into `domains/jobs/dags/` — still deliberately deferred per
  GOVERNANCE.md §2.5. All three job-related DAGs (`life_os_job_scout.py`,
  `life_os_job_scout_ats.py`, and the new `life_os_staging_promoter.py`) stay under
  `airflow/dags/`.

**Touched, but not part of the `jobs` domain folder — noted so it isn't missed:**
- `airflow/agents/job_agents.py` — extended (not replaced) with two new functions,
  `get_full_job_posting()` and `extract_linkedin_job_id()`. See Section 4, phase 3,
  for an important caveat about unverified CSS selectors in the former.
- `templates/partials/sidebar.html` — three targeted fixes, all inside the existing
  Jobs-specific block plus one line in the shared top-level nav.

## 3. What shipped (by phase)

### Phase 1 — Domain-folder migration (WO#3)

**New:**
- `domains/jobs/__init__.py`, `domains/jobs/routers/__init__.py`
- `domains/jobs/models.py` — `ApplicationStatus`, `Job`, `ApplicationLog`,
  `JobSearchKeyword`, `WatchedCompany`, `JobScoutRunLog`, `JobResponse`,
  `ApplicationLogCreate`, `ApplicationLogResponse`, `StagingJobStatus`, `StagingJob`,
  `StagingJobCreate`, `StagingJobResponse` — moved verbatim from `models.py`, with
  the file's top-of-file imports trimmed to only what these 13 classes actually use
  (safe in a brand-new file; not a behavior change).

**Moved (`git mv`, then import paths edited):**
- `routers/{jobs,ats,staging,job_config}.py` → `domains/jobs/routers/`
- `templates/jobs.html`, `templates/job_config.html` → `domains/jobs/templates/`
- `templates/partials/{staging_row,staging_queue,job_detail,ats_buttons,
  job_watched_panel,job_keyword_list,job_candidate_list,job_slug_guess}.html` →
  `domains/jobs/templates/partials/`
- `static/css/{jobs,jobs_enhancements}.css`, `static/js/jobs_enhancements.js` →
  `domains/jobs/static/{css,js}/`

**Edited (shared files, targeted diff — not full-file replace):**
- `models.py` — the 13 class bodies replaced with a re-export shim (exact contents
  in Part 2, Section 1).
- `main.py` — router imports repointed to `domains.jobs.routers`; new
  `/static/jobs` mount added **before** the general `/static` mount (Starlette
  registration-order rule, same as every prior domain's static mount).
- `core/templating.py` — `domains/jobs/templates` added to the `ChoiceLoader`.

**Edited (the 4 moved routers):**
- Model imports repointed from `models` to `domains.jobs.models`.
- `Jinja2Templates(directory="templates")` instances replaced with
  `from core.templating import templates`.
- `routers._helpers.html_error` and `services.ats_slug_service.guess_ats_slugs`
  imports left pointing at their original, unmoved locations.

### Phase 2 — Bug fixes and polish

- **`domains/jobs/routers/ats.py`** — `current = status.name.lower()` →
  `current = status.name`. Root cause and verification in Section 4.
- **4 templates** (`jobs.html`, `job_detail.html`, `staging_row.html`,
  `staging_queue.html`) — `rel="noopener noreferrer"` added to every
  `target="_blank"` link (4 total).
- **`domains/jobs/models.py`** — added `Job.latest_status_key` property (returns the
  `ApplicationStatus` member's `.name`, e.g. `"PHONE_SCREEN"`, vs. the pre-existing
  `latest_status` which returns `.value`, e.g. `"Phone Screen"`). Collapses 3 places
  in `jobs.html` that were independently re-deriving the same string via
  `|upper|replace(' ','_')` into one property call — this exact re-derivation is what
  caused the `ats.py` case-mismatch bug in the first place, so this removes the whole
  *class* of bug, not just that one instance.
- **`domains/jobs/routers/jobs.py`** — the `GET /jobs/detail/{job_id}` 404 path now
  uses the shared `html_error()` helper instead of a raw `HTMLResponse`.
- **`domains/jobs/templates/jobs.html`** — the staging queue's `hx-trigger="load,
  every 10s"` poll changed to `"load, every 10s [document.getElementById
  ('staging-body').classList.contains('open')]"` (htmx event-filter syntax) — the
  initial `load` stays unconditional (so the queue is populated before the panel is
  even opened), but the recurring poll only actually fires while the panel is
  visually open.
- **`domains/jobs/templates/jobs.html`** — removed the page-local `#toast` div and
  `showToast()` function; relies on the shared version `base.js` defines, per
  GOVERNANCE.md §3.2. **Not independently verified** that `base.html`/`base.js`
  actually provide this at the point in the DOM lifecycle `jobs.html`'s inline
  `onclick="showToast(...)"` calls need — see Section 9.

### Phase 3 — Staging Promoter DAG

**New:**
- `airflow/dags/life_os_staging_promoter.py` — mirrors `life_os_job_scout.py`'s
  shape (fetch → score → load → log_run) but starts at the detail-fetch step, since a
  staged job already has its URL and never ran a search. Runs daily at 06:00, or
  on-demand. Extracts the numeric LinkedIn job ID from the staged URL so a promoted
  job dedupes correctly against anything the scheduled scout DAG later finds.
- `domains/jobs/templates/partials/staging_process_feedback.html` — feedback
  fragment for the new manual-trigger endpoint.

**Edited:**
- `airflow/agents/job_agents.py` — added `get_full_job_posting(job_link) -> dict`
  (title/company/location/description/salary in one page fetch) and
  `extract_linkedin_job_id(job_link) -> str | None`. **The title/company/location CSS
  selectors in `get_full_job_posting()` were written without the ability to fetch a
  live linkedin.com page and are explicitly flagged in that function's own docstring
  as unverified** — the description/salary selectors reuse the ones already proven
  working in the pre-existing `get_job_details()`. See Section 9, item 3.
- `domains/jobs/routers/staging.py` — new `POST /jobs/stage/process` endpoint,
  triggers the DAG via `services.airflow_service.trigger_airflow()`. Both its
  response paths ("nothing pending" and "Airflow unreachable") deliberately return
  HTTP 200 rather than a 4xx/5xx — see the inline docstring for why (htmx's
  non-2xx swap behavior couldn't be verified without `base.js`).
- `domains/jobs/templates/jobs.html` — added the "⚡ Process Now" button next to
  "+ Queue Job".

### Phase 4 — Duplicate-insert guard

- **`domains/jobs/models.py`** — `Job.__table_args__` gained
  `UniqueConstraint("source", "external_ref", name="uq_linkedin_jobs_source_external_ref")`.
  Directly motivated by phase 3: once there were three independent code paths
  inserting into `linkedin_jobs` (scheduled LinkedIn scout, ATS scout, staging
  promoter) instead of two, a DB-level safety net against re-inserting the exact same
  posting twice from the same source became meaningfully more valuable as a defense
  against races between them. **Declared in the ORM only — see Section 9, item 1, for
  the required manual DB migration and the pre-flight duplicate-check query.**

### Phase 5 — Sidebar navigation bug fixes

**Edited (`templates/partials/sidebar.html` — shared file, not domain-owned):**
- Jobs sub-nav (`{% if active_module == 'jobs' %}`) visibility widened to
  `{% if active_module in ('jobs', 'jobs_settings') %}` — this was the entire cause
  of the submenu being completely absent on `/jobs/config`.
- The "Settings" sub-nav item's highlight condition changed from
  `{% if active_module == 'jobs' %}active{% endif %}` (true whenever the submenu was
  even visible — i.e., permanently "on" on the one page it *could* render) to
  `{% if active_module == 'jobs_settings' %}active{% endif %}`.
- The top-level "Job Tracker" nav item's highlight condition widened from
  `active_module == 'jobs'` to `active_module in ('jobs', 'jobs_settings')`, matching
  the pattern `Habits` already used correctly (`active_module in ('habits',
  'habits_settings')`) — without this, landing on `/jobs/config` left *no* item
  highlighted in the main nav at all. **`Finance` has this same gap and was left
  untouched** — not asked for, flagged as a known parallel issue.
- The "Dashboard" link inside the Jobs sub-nav removed at the project owner's
  request — it either duplicated the cross-domain `/dashboard` page or pointed at a
  Phase-3-planned route that was never built; either way, redundant.
- The "Add Job Link" sub-nav item's `href="#" onclick="openStagingPanel();
  return false;"` — which only worked because the submenu used to be visible
  *exclusively* on `/jobs`, the one page where that inline function actually exists —
  changed to `href="/jobs?open_staging=1"` with an `onclick` guard
  (`typeof openStagingPanel === 'function'`) that calls the function directly when
  already on `/jobs`, and falls through to a real navigation everywhere else. This
  became load-bearing, not just nicer, once the submenu became visible on
  `/jobs/config` too — the old version would have thrown a `ReferenceError` there.

**Edited (`domains/jobs/templates/jobs.html`):**
- Added `?open_staging=1` handling to the page's init block: opens the staging panel
  on arrival, then `history.replaceState`s the URL clean so a refresh doesn't
  re-trigger it.

### Phase 6 — Pagination / performance rework

This is the largest single change in the engagement. Full detail in Section 4.

**New:**
- `domains/jobs/templates/partials/_job_row_desktop.html` — single desktop `<tr>`,
  extracted verbatim from what used to be inline in `jobs.html`.
- `domains/jobs/templates/partials/_job_row_mobile.html` — single mobile
  `.job-card-mobile`, same treatment.
- `domains/jobs/templates/partials/jobs_rows.html` — the AJAX fragment returned by
  the new `GET /jobs/rows` endpoint: both row partials wrapped in `<template>` tags
  (required — browsers strip bare `<tr>` elements parsed outside a table context),
  plus a metadata `<span>` carrying the next cursor and counts.

**Edited:**
- `domains/jobs/models.py` — `Job.__table_args__` gained two more entries:
  `Index("ix_linkedin_jobs_fit_score_id", "fit_score", "ID")` (composite, matches
  the keyset-pagination `ORDER BY`) and `Index("ix_linkedin_jobs_search_date",
  "search_date")`. **Also declared in the ORM only — same outstanding-migration
  caveat as the `UniqueConstraint` above.**
- `domains/jobs/routers/jobs.py` — full rewrite. `GET /jobs` no longer has an
  unbounded query; it now applies the same default filters the old client-side JS
  used to (High Fit / score≥90, hide applied, last 10 days) **in SQL**, capped at
  `PAGE_SIZE = 300` via keyset pagination on `(fit_score DESC, ID DESC)`. New
  `GET /jobs/rows` endpoint accepts the same filters as query params plus an optional
  `cursor`, and always returns "the next `PAGE_SIZE` rows matching these filters,
  after this cursor" — the front end decides whether that's a filter-change (replace)
  or a Load More click (append). Shared helpers `_build_job_query()`,
  `_fetch_job_page()`, `_filters_from_query_params()` are used by both `GET /jobs`
  and `GET /jobs/rows` so the two can never drift apart.
- `domains/jobs/templates/jobs.html` — the desktop table and mobile list markup no
  longer conditionally absent from the DOM (`{% if jobs %}...{% else %}...{% endif
  %}` replaced with an always-rendered `#jobs-list-wrapper`, toggled via an
  `is-empty` CSS class, with `#jobs-empty-state` as an always-present sibling) — this
  was necessary, not cosmetic: AJAX requests need somewhere to insert rows into even
  when the very first page load had zero matches. `applyAllFilters()` (client-side
  hide/show over the full loaded dataset) removed entirely, replaced with
  `fetchJobs({append})` (server request + `<template>` extraction + insert),
  `debouncedFetchJobs()` (350ms debounce for the company text field and score
  slider), and `loadMoreJobs()`. Every filter control (quick-filter pills, score
  slider, company input, date range, status multi-select, hide-applied checkbox,
  clear-filters button) rewired accordingly. Date-range and hide-applied defaults now
  set server-side via Jinja (`value="{{ default_date_from }}"`, `checked`) instead of
  being recomputed by JS on load, so they can never drift from what the SQL query
  actually used.
- `domains/jobs/static/js/jobs_enhancements.js` — the dead
  `window.applyAllFilters = function() {...}` monkey-patch (patching a function that
  no longer exists after the above rewrite) removed, replaced with two functions
  explicitly exposed for `jobs.html` to call: `window._resetKeyboardFocus()` (resets
  the closure-scoped `focusedRowIdx` state after a filter change — this file's
  keyboard-nav state was never reachable directly from `jobs.html`'s own script
  block, an actual bug caught during this rewrite, not shipped) and
  `window._reapplySort()` (re-applies whatever column sort is currently active after
  new rows are inserted, so the header's sort-arrow indicator doesn't silently
  disagree with the actual row order).
- `domains/jobs/static/css/jobs.css` — added `#jobs-list-wrapper.is-empty {
  display:none; }`.

### Phase 7 — `search_date` refresh on rescan

An earlier, more elaborate design (new `Job.is_archived` / `Job.last_scanned_at`
columns, a new `StagingJobStatus.ARCHIVED` member, a new `life_os_jobs_retention.py`
DAG) was proposed, partially built, and then **explicitly rejected** by the project
owner — see Section 10. What actually shipped is much smaller:

**Edited:**
- `domains/jobs/models.py` — no schema change. `Job.search_date`'s docstring updated
  to describe its new meaning (see below); still the exact same column.
- `airflow/dags/life_os_job_scout.py` — `task_search_and_scrape` now splits every
  search result into `new_jobs` (unseen `job_id`) and `rescanned_ids` (already
  present). `new_jobs` proceeds through the pipeline unchanged; `rescanned_ids` gets
  one batched `UPDATE linkedin_jobs SET search_date = %s WHERE job_id IN (...)` and
  stops there — no re-fetch, no re-scoring, just the date.
- `airflow/dags/life_os_job_scout_ats.py` — same split, matched on
  `(source, external_ref)` instead of `job_id` (Greenhouse/Lever postings have no
  numeric `job_id`).
- `airflow/dags/life_os_staging_promoter.py` — the pre-existing "posting already in
  `linkedin_jobs`, skip the insert" branch in `task_load` now also refreshes
  `search_date`, so a manually-pasted duplicate link counts as a "confirmed still
  open today" signal too, consistent with the other two ingestion paths. The list
  previously named `staging_updates` renamed to `db_updates` since it now legitimately
  holds `UPDATE` statements against two different tables (`linkedin_jobs` and
  `staging_jobs`), not just one.

**What `search_date` means now, precisely:** "most recently confirmed to still exist,
by any of the three ingestion paths" — not "first discovered." A job open
continuously for 95 days has its `search_date` refreshed every ~6 hours by the
scheduled scout DAG re-finding it, so it never ages out of anything that filters on
this column (e.g. the cross-source dedup window in both scout DAGs, or the Jobs page
UI's own date-range filter). A job that's actually closed simply stops showing up in
scrapes and its `search_date` just stops moving forward.

## 4. Issues discovered (root cause analysis)

### 4.1 — ATS status highlight silently never applied (pre-existing, confirmed via `git HEAD`)

**Symptom:** clicking any ATS status button (Applied/Phone/Interview/etc.) correctly
persisted the status and correctly swapped in the refreshed button row — but the
just-clicked button never visually looked active. Only a full page reload made it
show correctly.

**Root cause:** `ats.py`'s `create_application_log` computed
`current = status.name.lower()` (e.g. `"applied"`), but `ats_buttons.html` compares
`{% if current == 'APPLIED' %}` (uppercase) to decide which button gets the active
class. The comparison silently never matched. `jobs.html`'s own full-page render
computed the equivalent value differently (`job.latest_status | upper |
replace(' ', '_')`), which is why a reload "fixed" it — that path was never broken.

**Confirmed pre-existing, not introduced by the migration:** `git show
HEAD:routers/ats.py` and `git show HEAD:templates/partials/ats_buttons.html` (i.e.,
the exact pre-migration baseline) both already contained this exact mismatch.

**Fix:** `current = status.name` (drop `.lower()`). **Structural fix on top:** added
`Job.latest_status_key`, collapsing the 3 separate places that were re-deriving this
same string via `|upper|replace(' ','_')` into one property, so this class of bug —
not just this one instance — can't recur.

### 4.2 — `GET /jobs` had no query limit and rendered every row twice (pre-existing, confirmed unrelated to the migration)

**Symptom:** `/jobs` hung for 60+ seconds, reported by the project owner after using
the app normally (not during any migration-adjacent testing).

**Root cause, two compounding factors:**
1. `list_jobs_ui`'s query was `select(Job).order_by(desc(Job.fit_score))` — no
   `.where()`, no `.limit()`. At ~2,300 rows in `linkedin_jobs`, every request
   fetched and rendered all of them.
2. `jobs.html` renders **two full copies** of every row — a desktop `<table>` and a
   separate mobile `.job-card-mobile` list (both always present in the DOM
   simultaneously; CSS, not Jinja, decides which is visible per viewport) — so 2,300
   jobs was actually ~4,600 rendered rows and roughly 32,000 DOM elements (7 ATS
   buttons per row × 2 views × 2,300). On top of that, three separate JS passes ran
   over the full set on every load: `applyStatusTinting()`, `initColumnSort()`
   listener attachment, and `applyAllFilters()` — which existed *only* to hide the
   ~2,100 rows that didn't match the default "High Fit" filter, using CSS
   `display:none`, after already having fully rendered them.

**Confirmed pre-existing:** neither the query nor the dual-rendering structure was
touched by the domain-folder migration itself, and the project owner confirmed the
hang reproduced identically with and without an unrelated query parameter
(`?open_staging=1`) added in an earlier phase, ruling that phase out as the cause.

**Fix:** Section 3, Phase 6, in full. Verified with a seeded 2,313-row dataset:
0.056s response time (down from 60+ seconds observed), 300-row hard cap confirmed
real (an unfiltered "show everything" query correctly reports `total_count=2313`
while the actual page returned is capped at exactly 300), and a full walk across all
8 pages of the unfiltered set confirmed zero overlap and zero gaps — every one of the
2,313 seeded jobs was returned exactly once across the full pagination sequence.

### 4.3 — Sidebar Jobs submenu bugs (pre-existing, confirmed via the file the project owner shared)

Two bugs in the same block of `templates/partials/sidebar.html`, both present in the
exact file the project owner uploaded (not a stale/inferred copy):
1. The submenu's own visibility condition (`active_module == 'jobs'`) meant it could
   never render at all on `/jobs/config` (`active_module == 'jobs_settings'`).
2. The "Settings" item's highlight condition was identical to the submenu's
   visibility condition — meaning on the one page the submenu *could* render
   (`/jobs`), "Settings" was unconditionally highlighted, even though the person was
   looking at the Jobs list, not Settings.

Net effect matched exactly what was reported: Settings shown as active on `/jobs`
(wrong), and the whole submenu simply absent on `/jobs/config` (also wrong). Fixed in
Phase 5, plus one related gap found during verification (the top-level "Job Tracker"
icon also failed to stay highlighted on `/jobs/config`, unlike `Habits`'s equivalent
top-level item, which already handled its own settings state correctly and served as
the fix pattern).

### 4.4 — Self-inflicted bugs caught before shipping (never actually shipped, listed for the verification-methodology lesson)

- While rewriting the filter JS in Phase 6, an early draft referenced
  `focusedRowIdx = -1` directly from `jobs.html`'s own script block — but that
  variable lives inside `jobs_enhancements.js`'s IIFE closure and is not reachable
  from outside it. Caught by re-reading the two files' actual scoping before
  shipping, not by a test failure — no test would have caught this either, since
  non-strict-mode JS would have silently created a stray global instead of throwing.
  Fixed by exposing `window._resetKeyboardFocus()` from `jobs_enhancements.js`
  instead, mirroring the pattern already established by `window._reapplyStatusTinting`.
- The first version of the DAG-logic test harness for Phase 7 used a `dag_db` stub
  that imported back into the currently-executing test script by module name
  (`import test_dags_search_date_refresh as t`) — this created a circular
  self-import that silently re-executed the entire test file a second time from
  scratch with a fresh, unrelated state object, producing a confusing duplicate run
  where the second (spurious) pass's assertions failed even though the first (real)
  pass's had already all passed correctly. Not a bug in the shipped DAG code — a bug
  in the throwaway test harness. Fixed by moving the fake state into its own
  standalone module (`dag_db_state.py`) that neither the test script nor the `dag_db`
  shim it's imported into ever import *back into*.

## 5. Verification methodology

No formal pytest suite exists in this repository yet (same starting point every
prior migration in this project has faced — see blog/code_intel §5, §9). The
following throwaway harnesses were built for this engagement and, per the
established pattern, are a candidate seed for the eventual formal suite rather than
something to reconstruct from scratch each time:

- **In-memory SQLite** (`sqlite+aiosqlite:///:memory:`) standing in for MariaDB, with
  `database.get_db` overridden via FastAPI's `dependency_overrides`. Schema created
  fresh per run via `Base.metadata.create_all()`.
- **Minimal stand-in modules** for every out-of-scope domain `models.py`
  (`domains/habits/models.py`, `domains/blog/models.py`, `domains/code_intel/models.py`)
  — just the class names actually imported by `models.py`'s shims, enough for the
  full import chain and `Base.metadata` identity checks to run. Same methodology
  as blog/code_intel §5.
- **A real FastAPI app** built from only the jobs-domain routers under test, exercised
  via `httpx.AsyncClient` + `ASGITransport` — real request/response cycles, real
  Jinja2 rendering through the real `core.templating` `ChoiceLoader`, real SQLAlchemy
  queries against the in-memory DB.
- **A 16-check regression harness** covering every endpoint in the domain (jobs list,
  job detail, ATS logging, staging queue add/list/process, keyword and watchlist
  CRUD, slug auto-detect with network mocked) — re-run after every phase in this
  engagement, catching zero regressions across phases 2 through 7 despite the
  significant rewrite in phase 6.
- **A seeded-at-scale performance test** (2,313 rows, realistic score/date/remote/
  application-log distribution) specifically for Phase 6 — proved the fix
  quantitatively (0.056s vs. 60+ seconds observed) rather than just qualitatively,
  and proved keyset-pagination correctness (full 8-page walk, zero overlap, zero
  gaps) rather than just "it returns *something* smaller now."
- **A separate, lighter harness** for the three DAG files touched in Phase 7 — real
  `dag_db`/`airflow` package stand-ins (the actual `apache-airflow` package was never
  installed; a minimal `DAG`/`PythonOperator` stand-in sufficed since only the task
  *functions* needed to run, not real DAG scheduling), a hand-rolled in-memory
  `dag_db` implementing only the SQL patterns these three files actually issue.
  Confirmed: a re-scanned job's `search_date` moves forward and a not-rescanned
  sibling's doesn't; a re-scanned job does not continue to scoring/insert a second
  time; the same holds for the ATS DAG's Greenhouse path.
- **Environment-version caveat, worth knowing about for future verification work in
  this same sandbox:** this environment's default `pip install` resolved
  `starlette==1.4.1` / a matching newer `fastapi`, which silently broke the
  `TemplateResponse(name, context)` calling convention this entire codebase uses
  everywhere (`TemplateResponse(request, name, context)` is the newer required
  order). This is **not a bug in the codebase** — pinning `starlette==0.36.3` /
  `fastapi==0.109.2` resolved it immediately and is what every verification pass in
  this engagement actually ran against. Confirmed via a minimal repro against the
  exact pre-migration baseline code, so this is an environment-pinning issue, not
  something introduced by any of the phases above. If a real `requirements.txt`
  exists for this project (never shared with any agent in this engagement), it
  presumably already pins compatible versions — this note is only relevant to
  someone re-running verification harnesses from scratch in a fresh sandbox.

## 6. Deployment

- **`web` (FastAPI):** bind-mounts the project root and runs `uvicorn --reload` per
  every prior migration's postmortem — file changes across all seven phases are
  picked up automatically. `docker compose restart web` recommended as a clean
  checkpoint, not because reload wouldn't work.
- **Airflow:** per GOVERNANCE.md §2.5, `./airflow/dags` (and by the same reasoning,
  presumably `./airflow/agents`) mount directly rather than being baked into the
  image — the scheduler should pick up the new `life_os_staging_promoter.py` and the
  edits to the other two DAGs plus `job_agents.py` on its own periodic rescan, no
  restart strictly required, though forcing an immediate rescan (restart the
  scheduler service) avoids waiting.
- **No rebuild required anywhere.** No new dependency was added in any phase — the
  Phase 3 scraping additions reuse `requests`/`BeautifulSoup4`, already installed for
  the existing `job_agents.py` functions.
- **Database migration REQUIRED, not yet performed** — see Section 9, items 1–2.
  Nothing in the reload/restart flow above substitutes for this. `Base.metadata`
  never alters an existing, populated table; it only creates tables that don't exist
  yet.

## 7. What went well

- Every bug fixed in this engagement (ATS highlight, the performance hang, both
  sidebar bugs) was independently confirmed pre-existing — via `git HEAD` diff for
  the code-level ones, via direct comparison against the project owner's own
  just-uploaded file for the sidebar ones — before being touched. None were
  misattributed to the migration itself.
- The Phase 6 rewrite, the riskiest single change in this engagement, was verified
  quantitatively (measured response time and byte size at realistic scale) rather
  than just structurally (does it return 200) — this is what actually confirmed the
  fix solves the reported problem, not just that it doesn't crash.
- Keyset pagination was chosen over `OFFSET`-based pagination specifically because
  `Job` rows are actively being inserted by three concurrent background DAGs — an
  `OFFSET`-based "page 2" could silently skip or duplicate rows if a new job lands
  between two "Load More" clicks; keyset pagination on a stable sort key cannot.
  This was verified directly (zero overlap across a full multi-page walk), not just
  assumed correct by construction.
- The project owner's own architectural instinct (Section 9's original question:
  "should the filter be applied when talking to the database and make a new
  request?") was exactly the correct direction and matched what got built — worth
  recording as validation that the direction was sound before a large rewrite began,
  not just after.
- When a more elaborate design (archival) was proposed, the project owner's pushback
  ("this adds a whole new level of complexity... I do not believe this layer of
  complexity is necessary") was accepted and acted on immediately — the
  already-partially-built archival scaffolding (new columns, new enum member, drafted
  DAG) was cleanly reverted rather than shipped anyway or left half-built. See
  Section 10.

## 8. What could be improved / lessons learned

### 8.1 — Two schema changes are declared in the ORM but not yet applied to any real database

Both the Phase 4 `UniqueConstraint` and the Phase 6 `Index` declarations follow the
correct process this project has already established (habits postmortem §4.4: verify
against the real DB with `SHOW INDEX FROM ...` before assuming a constraint doesn't
already exist; check for pre-existing duplicate/conflicting data before applying) —
but that process has only been *documented*, not *executed*, since no agent in this
engagement had access to a real database. **This is the single most important
outstanding item — see Section 9.**

### 8.2 — Several "shared infrastructure" assumptions were never independently verified

Three separate points in this engagement made a design decision specifically
*because* something couldn't be verified, rather than verifying it:
- Removing `jobs.html`'s local toast implementation assumed `base.html`/`base.js`
  provide an equivalent `#toast`/`showToast()` at the right point in the page
  lifecycle, per GOVERNANCE.md §3.2's claim — never independently confirmed since
  `base.js`'s content was never shared with any agent in this engagement.
- `POST /jobs/stage/process` deliberately always returns HTTP 200 (even for its
  "Airflow unreachable" error path) specifically to sidestep not knowing whether
  htmx's error-response-swap behavior is customized anywhere in `base.js` — if it
  turns out **not** to be customized (i.e., real htmx defaults apply, which do swap
  4xx/5xx by default in the version pinned via `<script src="...htmx.org@1.9.10">` in
  `base.html`), then several **pre-existing** endpoints elsewhere in this domain
  (`add_keyword`, `add_watched_company`, both of which call `html_error()` with
  409/422 from an `hx-post`-triggered form) may have always displayed their error
  states correctly and this defensive choice was unnecessary caution — but verifying
  this either way needs the real `base.js`, still not available at time of writing.
- `job_agents.get_full_job_posting()`'s title/company/location selectors were written
  from general knowledge of LinkedIn's public job page structure, explicitly because
  `linkedin.com` was not in this environment's outbound-network allowlist at any
  point in this engagement (only `get_job_details()`'s pre-existing description/
  salary selectors, already proven working in production, were reused as-is).

### 8.3 — `dag_db.py`'s real interface was never seen by any agent, across this entire multi-week engagement

Every DAG file in this project (all five now: the two scout DAGs, the digest DAG
referenced in `job_scout_health.py`'s docstring, and the two DAGs touched/added in
this engagement) imports `fetch_all`, `execute`, and `execute_many` from a module
called `dag_db` that has never once been included in anything shared with any agent
working on this codebase. Every SQL pattern used across Phases 3 and 7 (`UPDATE ...
WHERE ... IN (...)`, multi-column `WHERE ... = %s AND ... = %s`, `WHERE ... = %s OR
... = %s`) was inferred to be supported based on call-site conventions already
present in the pre-existing DAG files, and confirmed only against a hand-built stand-
in implementing those same inferred conventions — never against the real module. If
`dag_db.execute_many()`'s real implementation has any restriction on statement shape
not visible from its call sites (e.g., requires all statements in one batch to target
the same table, or doesn't support `IN (...)` clauses with a variable-length
placeholder list the way assumed here), **the Phase 7 rescan-refresh logic in all
three DAGs could fail at runtime in a way no verification performed in this
engagement could have caught.** Strongly recommend sharing `airflow/dag_db.py`'s real
source with whichever agent handles the next domain migration, independent of
whether that domain's own work happens to need it — at minimum to retroactively
verify Phase 7's assumptions.

### 8.4 — `jobs.html` grew large enough that further changes should consider splitting it

Not urgent, but worth flagging given how much of this engagement's Phase 6 work
involved carefully locating and editing specific `<script>` blocks inside a single
large template file. GOVERNANCE.md §1.2 already sets a 300-line hard ceiling for
*routers* — `jobs.html` (template + inline `<script>`) is not a router and isn't
covered by that rule, but has grown large enough (filter panel, staging panel, two
full row-rendering views, an inline `<script>` block handling filtering/pagination/
keyboard-nav-adjacent logic/init) that extracting the `<script>` block into
`domains/jobs/static/js/jobs_filters.js` (separate from `jobs_enhancements.js`, which
already exists and covers a different concern — sort/tint/keyboard-nav) would make
future changes easier to scope and review. Not done in this engagement to keep the
diff for each phase focused on its actual objective, per this project's own
DRY/scoping conventions (GOVERNANCE.md §4.1, §4.6).

## 9. Outstanding manual actions required

None of these are "eventually, once all domains migrate" items — Section 9 items are
things that need to happen regardless of migration sequencing, ideally soon:

1. **Apply the `UniqueConstraint` from Phase 4 to the real database.** Before
   applying, run:
   ```sql
   SELECT source, external_ref, COUNT(*) FROM linkedin_jobs
   WHERE external_ref IS NOT NULL
   GROUP BY source, external_ref HAVING COUNT(*) > 1;
   ```
   If this returns any rows, resolve those duplicates first (the constraint will
   otherwise fail to apply outright). Also run `SHOW INDEX FROM linkedin_jobs` first
   — per the habits postmortem's own established playbook (§4.4), it's possible an
   equivalent constraint already exists under a different name.
2. **Apply the two `Index` declarations from Phase 6** (`ix_linkedin_jobs_fit_score_id`
   on `(fit_score, ID)`, `ix_linkedin_jobs_search_date` on `search_date`) to the real
   database. Building an index on a live, multi-thousand-row table takes a moment and
   can briefly affect write performance depending on MariaDB version/settings — safe
   to run any time, just don't expect it to be instant. `GET /jobs/rows` will still
   *function* without these (the ORM declaration doesn't gate query execution), just
   progressively slower as the table keeps growing — this is the exact class of
   problem Phase 6 was built to fix, so don't let it quietly reappear here.
3. **Spot-check `job_agents.get_full_job_posting()`'s title/company/location
   selectors** against a handful of real LinkedIn job postings before relying on the
   Staging Promoter DAG for anything beyond description/salary. See Section 8.2.
4. **Share `airflow/dag_db.py`'s real source** with whichever agent next touches any
   DAG in this project, and use it to retroactively verify the SQL patterns
   introduced in Phase 7 (Section 8.3) actually work against the real implementation,
   ideally by running `life_os_job_scout.py`, `life_os_job_scout_ats.py`, and
   `life_os_staging_promoter.py` for real (or against a real staging DB) rather than
   only against this engagement's hand-built stand-in.
5. **Confirm `base.html`/`base.js` actually provide the shared `#toast`/`showToast()`**
   GOVERNANCE.md §3.2 describes, at a point in the DOM lifecycle compatible with
   `jobs.html`'s inline `onclick="showToast(...)"` handlers (i.e., loaded and run
   before any user interaction is possible). See Section 8.2.
6. **Confirm (or deliberately set) htmx's error-response-swap behavior** in
   `base.js` — relevant not just to the one endpoint this engagement worked around
   (`POST /jobs/stage/process`) but potentially to pre-existing behavior of
   `add_keyword` and `add_watched_company` too. See Section 8.2.

## 10. Considered and rejected

**Job/staging-job archival, based on age.** Proposed after the Phase 6 performance
fix, as a way to keep `linkedin_jobs` from growing unboundedly forever. The design
that got as far as partial implementation before being reverted:
- New `Job.is_archived` (boolean) and `Job.last_scanned_at` (datetime, distinct from
  `search_date`) columns.
- A new `StagingJobStatus.ARCHIVED` enum member.
- Archival based on `last_scanned_at`, not `search_date`, specifically so a job that
  stays open for months wouldn't be wrongly archived just because it was *discovered*
  90+ days ago.
- Automatic un-archival if a job was re-encountered by any scraper after being
  archived.
- A hard guardrail: never archive anything with `ApplicationLog` history, at any age
  (the FK is `ON DELETE CASCADE`, so even though archiving ≠ deleting, this guardrail
  was treated as non-negotiable regardless).
- A new `life_os_jobs_retention.py` DAG to perform the archival on a schedule.

**Explicitly rejected by the project owner**, in these words: *"let's drop the whole
archival idea... while it might be interesting, it just adds a whole new level of
complexity to the process. The prefilter was magical and made the loading incredible
fast, so I do not believe this layer of complexity is necessary."*

**What shipped instead:** just the `search_date` refresh-on-rescan behavior described
in Section 3, Phase 7 — reusing the pre-existing `search_date` column, no new
columns, no new enum member, no new DAG, no archival/deletion of any kind. The `Job`
table will continue to grow without bound; this is a **known, consciously accepted
tradeoff**, not an oversight — the Phase 6 pagination fix already addresses the
*performance* consequence of that growth (query cost stays bounded by the composite
index + `LIMIT`, regardless of total table size), which was the actual, concrete
problem. Table *size* itself growing forever was judged not worth the added
complexity of a retention system to solve pre-emptively.

**If a future agent or work order proposes reintroducing archival, retention, or
soft-delete for `linkedin_jobs` or `staging_jobs`:** re-read this section first and
raise it as an explicit, standalone proposal with the project owner rather than
assuming it's wanted — it was considered once already and turned down for a specific,
recorded reason.

---

## PART 2 — REQUIREMENTS FOR FINAL CLEANUP (execute only once ALL domain-folder migrations are complete)

This section follows the exact structure and reasoning established in the
blog/code_intel postmortem's own Part 2 — do not execute any of it mid-way through
the migration program; every item below assumes **every** domain has already been
extracted out of the flat `routers/` / `templates/` / `models.py` layout, jobs
included.

### 1. How to know you're actually ready

Same check as every prior postmortem specifies:
```bash
grep -rn "from models import" --include="*.py" .
```
**Target end-state:** nothing except `models.py`'s own internal comments/imports. At
the time of writing (end of the jobs migration and all its follow-on phases), this
command returns:
```
routers/dashboard.py:...:from models import (
models.py:...  (Base, in a comment)
models.py:...  (comment, inside the Jobs shim docstring)
models.py:...  (comment, inside the Blog shim docstring)
models.py:...  (comment, inside the Code Intel shim docstring)
models.py:...  (comment, inside the Habits shim docstring)
```
i.e., exactly one real remaining consumer (`routers/dashboard.py`) plus shim
docstring comments, for **every** domain migrated so far including jobs. Confirm this
list has shrunk to zero real consumers (dashboard.py included) before proceeding —
per §3.3 of GOVERNANCE.md, `finance`, `journal`, `recipes`/`pantry`, `workout`,
`media`, and `planning` (`weekly_plan`/`intent`) still need to migrate before that's
true.

### 2. `models.py` — the jobs shim specifically

The jobs shim currently reads:
```python
"""
JOBS MODULE — moved to domains/jobs/models.py as part of the domain-folder
migration (see domains/jobs/routers/*.py for usage). Re-exported here so
any other file still doing `from models import Job` (etc.) keeps working
unchanged.
"""
# TODO: remove after all cross-references are updated
from domains.jobs.models import (
    ApplicationStatus,
    Job,
    ApplicationLog,
    JobSearchKeyword,
    WatchedCompany,
    JobScoutRunLog,
    JobResponse,
    ApplicationLogCreate,
    ApplicationLogResponse,
    StagingJobStatus,
    StagingJob,
    StagingJobCreate,
    StagingJobResponse,
)
```
Per the batching principle established in the habits postmortem's own final-cleanup
§1 ("Do this for all domains in the same pass, not incrementally, to avoid leaving
the app in a state where some shims are gone and others aren't for no principled
reason") — **do not remove this shim in isolation just because jobs' own migration
and follow-on work is done.** Wait until every domain's shim can be removed in the
same pass, per whichever of the two `models.py` end-states GOVERNANCE.md's final
cleanup section settles on (delete entirely vs. reduce to a pure import-registry —
see the blog/code_intel postmortem Part 2 §2 for the full tradeoff writeup, which
applies identically here; jobs adds no new wrinkle to that decision).

### 3. `routers/dashboard.py` — the one real remaining consumer

Per WO#3's own acceptance criteria, `dashboard.py`'s existing
`from models import (Job, ApplicationLog, ApplicationStatus, StagingJob,
StagingJobStatus, ...)` import was confirmed (at least structurally, via the shim's
exports) to keep resolving throughout this engagement. When the final cleanup pass
happens, update it to import directly:
```python
from domains.jobs.models import Job, ApplicationLog, ApplicationStatus, StagingJob, StagingJobStatus
```
alongside the equivalent direct-import updates for every other domain's classes
`dashboard.py` currently pulls from the shim.

**Note on `dashboard.py`'s actual usage pattern, relevant to Section 4 below:**
`dashboard.py`'s job-related queries (top unapplied jobs, fit-score histogram, ATS
pipeline funnel via `ApplicationLog.status` grouped in SQL, staging queue counts) do
**not** use `Job.latest_status_key` or `Job.latest_status` (the string-derived
properties) anywhere — it groups by `ApplicationLog.status` directly in SQL. This
means the Phase 2 property addition has no bearing on `dashboard.py` and needs no
corresponding update there.

### 4. Cross-domain relationship risk — jobs is one of the *simpler* cases

Per the critical warning in the blog/code_intel postmortem Part 2 §3: any
`relationship()` call using a **string** class name (resolved lazily against
SQLAlchemy's shared mapper registry, not at import time) only works if every module
defining a referenced class has actually been imported by *something* before the
first query touches that relationship.

**Jobs adds no new instance of this risk.** `Job.application_logs` and
`ApplicationLog.job` are both string-based (`relationship("ApplicationLog", ...)` /
`relationship("Job", ...)`), but **both classes now live in the same file**
(`domains/jobs/models.py`) — this is a same-module reference, structurally identical
to any other same-file `relationship()` call, and requires no special registry
handling beyond `domains/jobs/models.py` itself being imported once (which the
`models.py` shim already guarantees, same mechanism as every other domain).

**Still true, and still needs checking at final-cleanup time regardless of jobs
specifically:** whatever registry mechanism gets chosen in Section 2 above must cover
every domain that migrates after jobs too — re-run
```bash
grep -rn 'relationship(' domains/*/models.py
```
one final time across *every* domain's `models.py` before finalizing the registry,
per the same instruction already given in the blog/code_intel postmortem.

### 5. `core/templating.py` and `main.py`

Both already correctly reference the jobs domain as of this engagement —
`domains/jobs/templates` is in the `ChoiceLoader` list, `domains/jobs/routers` is
what `main.py` imports from, and `/static/jobs` is mounted before the general
`/static` mount. **Nothing further needed here specifically for jobs** at final
cleanup time — just confirm these entries survive whatever final-form
`core/templating.py`/`main.py` the last domain's migration converges on, the same
confirmation every other already-migrated domain's entries need.

### 6. Fate of `routers/`, `templates/`, `static/`

Jobs no longer has *anything* remaining in the flat `routers/` folder (all four
files moved), nor in the flat `templates/`/`templates/partials/` folders (all ten
templates moved, plus the three new Phase 6 partials created directly under
`domains/jobs/templates/partials/` from the start), nor in flat `static/css/` or
`static/js/` (all three assets moved). **Jobs does not block deleting these flat
folders once every other domain reaches the same state** — confirm this is still
true (no stray jobs-related file was accidentally left behind or re-created at the
top level during any of the seven phases) before deleting.

### 7. New capabilities from this engagement that final cleanup must not silently break

- **The keyset-pagination cursor format** (`f"{fit_score}_{ID}"`, parsed by splitting
  on the first `_`) is now load-bearing for `jobs.html`'s "Load More" button. Any
  future change to `Job.fit_score`'s type, or `Job.ID`'s type, or any refactor that
  changes what `jobs.py`'s `_parse_cursor()`/cursor-building code assumes about those
  columns, needs to keep this encoding compatible — or update both the encoder in
  `_fetch_job_page()` and the decoder in `_parse_cursor()` together.
- **`GET /jobs` and `GET /jobs/rows` must stay in sync** — they share
  `_build_job_query()`, `_fetch_job_page()`, and `_filters_from_query_params()`
  specifically so this can't drift; any future edit to jobs' filtering logic should
  go through those shared functions, not be added to one endpoint's handler directly.
- **The three DAGs' rescan-refresh logic (Phase 7) depends on `dag_db.execute_many()`
  supporting the SQL shapes described in Section 8.3** — re-verify this specifically
  once `dag_db.py`'s real source is available (Section 9, item 4), independent of
  whether that happens before or after all domains finish migrating.

### 8. Final verification checklist (jobs-specific additions to the standing one)

In addition to every item already in the blog/code_intel postmortem's Part 2 §9
final checklist (which still applies in full and isn't repeated here), for jobs
specifically:
- [ ] `models.Job is domains.jobs.models.Job` (and the other 12 classes) — same
  identity check pattern as every other domain, confirm it still holds after the
  final shim removal changes how `models.py` itself is structured.
- [ ] `UniqueConstraint("source", "external_ref")` and both new `Index` declarations
  from this engagement are present in the real, live database (Section 9, items 1–2)
  — not just declared in the ORM.
- [ ] Full regression pass on `GET /jobs` and `GET /jobs/rows` at whatever the real
  production row count is by the time final cleanup happens (likely well beyond the
  2,313 rows this engagement's performance test used) — confirm the fix still holds
  at real scale, not just the scale it was verified against here.
- [ ] All three job-ingestion DAGs (`life_os_job_scout.py`,
  `life_os_job_scout_ats.py`, `life_os_staging_promoter.py`) run successfully against
  the real `dag_db.py`, with `search_date` refresh behavior spot-checked against at
  least one real re-scanned posting.

---

## 11. Reference — Files touched across this entire engagement (jobs domain, final state)

**New:**
- `domains/jobs/__init__.py`, `domains/jobs/routers/__init__.py`
- `domains/jobs/models.py`
- `domains/jobs/routers/jobs.py`, `ats.py`, `staging.py`, `job_config.py` (moved,
  then `jobs.py` and `staging.py` subsequently rewritten/extended in place)
- `domains/jobs/templates/jobs.html`, `job_config.html` (moved, `jobs.html`
  subsequently edited across every phase)
- `domains/jobs/templates/partials/staging_row.html`, `staging_queue.html`,
  `job_detail.html`, `ats_buttons.html` (edited), `job_watched_panel.html`,
  `job_keyword_list.html`, `job_candidate_list.html`, `job_slug_guess.html` (moved)
- `domains/jobs/templates/partials/_job_row_desktop.html`, `_job_row_mobile.html`,
  `jobs_rows.html`, `staging_process_feedback.html` (all new, Phases 3 and 6)
- `domains/jobs/static/css/jobs.css` (moved, subsequently edited), `jobs_enhancements.css`
  (moved, unedited)
- `domains/jobs/static/js/jobs_enhancements.js` (moved, subsequently edited)
- `airflow/dags/life_os_staging_promoter.py` (new, Phase 3)

**Edited in place (shared files — applied as targeted diffs, per the standing rule):**
- `models.py` — Jobs shim added (Phase 1)
- `main.py` — jobs router imports + `/static/jobs` mount (Phase 1)
- `core/templating.py` — `domains/jobs/templates` added to `ChoiceLoader` (Phase 1)
- `templates/partials/sidebar.html` — 4 targeted fixes inside/around the Jobs block
  (Phase 5)
- `airflow/agents/job_agents.py` — 2 new functions added (Phase 3)
- `airflow/dags/life_os_job_scout.py` — rescan/search_date-refresh logic (Phase 7)
- `airflow/dags/life_os_job_scout_ats.py` — same, ATS variant (Phase 7)

**Untouched, confirmed via diff review, despite being adjacent/related:**
- `routers/dashboard.py`, `services/ats_slug_service.py`, `routers/_helpers.py`,
  `airflow/agents/job_resume_context.py`, `job_dedup.py`, `job_scout_health.py`,
  `gemini_client.py`, `airflow/agents/job_ats_agents.py`

**Endpoints covered by the regression suite used throughout (reuse this list's shape
for verifying any future change to this domain):**
`GET /jobs`, `GET /jobs/rows`, `GET /jobs/detail/{id}`, `POST /jobs/stage`,
`GET /jobs/stage`, `POST /jobs/stage/process`, `POST /ats/log`,
`GET /jobs/config`, `POST /jobs/config/keywords`,
`PATCH /jobs/config/keywords/{id}/toggle`, `DELETE /jobs/config/keywords/{id}`,
`POST /jobs/config/watched`, `PATCH /jobs/config/watched/{id}`,
`DELETE /jobs/config/watched/{id}`, `POST /jobs/config/watched/detect`.
