# Media Domain Migration & Follow-On Work — Post-Mortem & Forward Requirements

**Status:** Migration (WO#9) complete and verified to the extent this
engagement's environment allowed (no live MariaDB, Airflow, or external
API access — see each section's Verification subsection for exactly what
was and wasn't exercised). Three pieces of follow-on work, explicitly
agreed to **after** WO#9 was already delivered, are also complete and
documented separately in Part 2 — see the note at the top of that part
for why the separation matters and how a reviewer should use it.

**This document supersedes and retires**
`migration_docs/Postmortem/wo9_notes_pending_postmortem.md`, which was a
temporary holding note written before this postmortem existed (Claude's
cross-session memory is not enabled for this account, so that file was
the durable substitute at the time). Its content is fully folded into
Part 2 below. Delete the pending-notes file once this document is
accepted.

**Scope of this document:** (1) a post-mortem of the `media` domain-folder
migration itself (Work Order #9), reported separately from (2) a
post-mortem of three items of follow-on work agreed to afterward, in a
later conversation — a pagination/performance fix, a new weekly DAG, and
exploratory (not-shipped-to-the-app) research; and (3) a **thorough,
self-contained final-cleanup requirements section** (Part 3), written so
whoever performs that pass — human or agent — doesn't need to
cross-reference the habits, blog/code_intel, jobs, or explorer postmortems
to get the complete current picture. It reproduces and updates the
standing structure those four documents established, with `media`'s own
specifics filled in.

This document assumes no prior conversation context, per the convention
set by every prior postmortem in this series. If you're an agent picking
up final cleanup, or the next domain migration, read Section 0 immediately
below, then Part 3 in full, before touching any files.

---

## SECTION 0 — How to read this document (important for reviewers)

This project's own governance rules are explicit that migration work and
follow-on work are reviewed separately, and one must not be allowed to
quietly inflate or obscure the other:

> **GOVERNANCE.md §4.5 — Bugs Found During Migration Are Not Migration
> Work.** ...gets its own standalone ticket with its own fix and its own
> verification — never bundled into the migration's diff...
>
> **GOVERNANCE.md §4.6 — What "Done" Means for a Domain Migration.** ...It
> is *not* considered done if "cleanup" happened alongside it (dead code
> removal, style fixes, bug fixes) — those are separate, separately
> reviewable changes by design.

This postmortem is structured around that split, explicitly, because it
was directly requested for this document:

- **Part 1** covers **only** what WO#9 itself — the domain-folder
  relocation, as originally scoped — actually shipped. Judge "did the
  migration succeed" against Part 1 alone. Nothing in Part 1 includes any
  of the follow-on work.
- **Part 2** covers three items of follow-on work, each individually
  scoped, agreed to, and delivered **after** WO#9 was already reported
  complete. None of it was bundled into WO#9's diff. Two of the three
  items touch a file WO#9 also touched (`domains/media/routers/media.py`)
  — Part 2 is explicit, per item, about exactly what changed on top of
  WO#9's delivered version and why.
- **Part 3** is forward-looking only — requirements for whenever every
  domain (not just media) has migrated.

---

# PART 1 — POST-MORTEM: THE MIGRATION (WORK ORDER #9, AS ORIGINALLY SCOPED)

## 1.1 Summary

WO#9 migrated the `media` module — four routers, thirteen templates, one
CSS file, and fifteen model/schema definitions — out of the flat
`routers/` / `templates/` / `models.py` layout and into `domains/media/`,
following the exact precedent of every prior domain migration in this
series (habits, blog+code_intel, jobs, explorer, finance, journal). This
was a pure relocation: no schema changes, no behavior changes, no
endpoint renames.

Unlike every domain migrated before it, `media` has **zero confirmed
external consumers of its own shim** — not even `routers/dashboard.py`,
the one sanctioned cross-domain reader per GOVERNANCE.md §2.2. This was
true going in (per WO#9's own SCOPE section) and was independently
re-confirmed during the migration itself.

## 1.2 Scope & Objective

**In scope, and delivered — matches WO#9's SCOPE section exactly:**

- **Models moved to `domains/media/models.py`:** `PREDEFINED_MOOD_TAGS`
  (module-level constant), `MediaExternalSource`, `MediaType`,
  `UserMediaStatus`, `RecommendationMediaType` (4 enums),
  `StreamingService`, `MediaItem`, `UserMedia`, `TVSeasonProgress`,
  `MediaRecommendation` (5 ORM classes), `MediaItemResponse`,
  `UserMediaCreate`, `UserMediaUpdate`, `UserMediaResponse`,
  `StreamingServiceResponse` (5 Pydantic schemas). 15 definitions total.
- **Routers moved:** `media.py`, `media_search.py`, `media_recommend.py`,
  `media_settings.py` → `domains/media/routers/`.
- **Templates moved:** 5 full-page templates + 8 partials →
  `domains/media/templates/` (partials under their own `partials/`
  subfolder).
- **Static asset moved:** `static/css/media.css` →
  `domains/media/static/css/media.css`.
- **Config files edited:** `models.py` (shim swap), `main.py` (import
  repoint + new static mount), `core/templating.py` (new `ChoiceLoader`
  entry).

**Explicitly out of scope, and confirmed untouched:**
- `services/ml_service_client.py`, `services/tmdb_service.py`,
  `services/openlibrary_service.py` — shared external-API wrappers, same
  tier as `github_service.py` in prior migrations. None import `models`,
  so unlike `recipe_service.py` (WO#7's own scope), they needed zero
  internal edits either.
- `routers/_helpers.py` (provides `html_error`) — shared, not moved.
- The `MEDIA_RECOMMEND_AI` import-time environment-variable toggle in
  `media_recommend.py`, and its `_gemini_explain()` Gemini-calling
  function — one of the six known duplicate AI-client implementations
  tracked in GOVERNANCE.md §2.3. Left completely untouched, same
  treatment as `finance_upload.py`'s Gemini call and `blog_agents.py` in
  every prior migration.

## 1.3 What shipped

**Files created:**
```
domains/media/__init__.py                              (empty, scaffolding)
domains/media/routers/__init__.py                       (empty, scaffolding)
domains/media/models.py
domains/media/routers/media.py
domains/media/routers/media_search.py
domains/media/routers/media_settings.py
domains/media/routers/media_recommend.py
domains/media/templates/media.html
domains/media/templates/media_search.html
domains/media/templates/media_recommend.html
domains/media/templates/media_settings.html
domains/media/templates/media_rec_history.html
domains/media/templates/partials/media_card.html
domains/media/templates/partials/media_detail.html
domains/media/templates/partials/media_drawer_status.html
domains/media/templates/partials/media_rating.html
domains/media/templates/partials/media_seasons.html
domains/media/templates/partials/recommendations.html
domains/media/templates/partials/search_results.html
domains/media/templates/partials/streaming_service_list.html
domains/media/static/css/media.css                      (byte-identical move)
```

**Files edited (shared files — applied as targeted diffs, not full-file
replacement, per the standing rule from the habits postmortem §4.1):**

- **`models.py`** — the 15 media definitions replaced with:
  ```python
  # TODO: remove after all cross-references are updated
  from domains.media.models import (
      PREDEFINED_MOOD_TAGS,
      MediaExternalSource,
      MediaType,
      UserMediaStatus,
      RecommendationMediaType,
      StreamingService,
      MediaItem,
      UserMedia,
      TVSeasonProgress,
      MediaRecommendation,
      MediaItemResponse,
      UserMediaCreate,
      UserMediaUpdate,
      UserMediaResponse,
      StreamingServiceResponse,
  )
  ```
  Structurally identical to every prior domain's shim. **Confirmed via
  AST inspection** that no other line in `models.py` changed, and that
  none of the 15 names remained defined (only imported) anywhere else in
  the file.
- **`main.py`** — router import repointed to
  `from domains.media.routers import media, media_search, media_recommend, media_settings`;
  `app.mount("/static/media", ...)` added immediately before the general
  `/static` mount; the pre-existing include order (`media_search` →
  `media_recommend` → `media_settings` → `media` catch-all last) was
  preserved exactly.
- **`core/templating.py`** — `FileSystemLoader("domains/media/templates")`
  appended to the `ChoiceLoader` list.

**Files moved (old → new):** every router/template/static path listed in
§1.2 above, one-to-one.

## 1.4 Issues discovered during the migration

**No pre-existing functional bugs were found** in the media domain's code
during this migration — same outcome as the explorer domain (WO#4), and
unlike jobs (ATS highlight bug, unbounded-query hang) or blog/code_intel
(non-functional folder-README persistence). The domain's small,
self-contained shape (no cross-domain relationships, no
commit-then-reuse ORM patterns beyond what was already exercised safely)
likely explains the difference, consistent with the same reasoning the
explorer postmortem gave for its own clean result.

**One self-inflicted mistake, caught before delivery, never shipped:** an
early draft of `domains/media/templates/media_rec_history.html` included
an `extra_css` block linking `media.css` — a block that does **not**
exist in the original `templates/media_rec_history.html`. Caught during
this engagement's own verification pass (by checking the original
template's actual block structure rather than assuming every page in the
domain needed the same treatment as the other four), corrected before
the file was delivered. Mentioned here for the same reason the habits and
jobs postmortems document their own self-caught mistakes: it's evidence
the verification step was actually adversarial/thorough, not just
performative.

## 1.5 Verification methodology

**No live MariaDB, Airflow, or external API access was available in this
engagement** — same starting constraint every prior postmortem in this
series has faced, addressed the same way: build the closest achievable
substitute, state explicitly what was substituted and why, mark results
accordingly.

- **Static verification:** `ast.parse()` on every created/edited Python
  file (syntax-only, but catches a real class of mistake).
- **Model-layer verification, actually run against real SQLAlchemy (not
  a description of what it should do):** `domains.media.models` imported
  in isolation and `sqlalchemy.orm.configure_mappers()` executed against
  it — confirmed all 5 tables (`media_items`, `user_media`,
  `tv_season_progress`, `media_recommendations`, `streaming_services`)
  register, and that the same-module string-based relationships
  (`MediaItem.user_media` ↔ `UserMedia.media_item`,
  `UserMedia.season_progress` ↔ `TVSeasonProgress.user_media`) resolve to
  the correct classes.
- **Template-layer verification, also actually run, not just read:**
  every one of the 13 moved templates compiled successfully through a
  `ChoiceLoader` built the same way `core/templating.py` builds its own
  (confirms zero Jinja2 syntax errors were introduced by the move).
  Beyond compilation, 9 of those templates/partials were actually
  **rendered** against mock context objects shaped like the real
  ORM/enum classes (`media_type.icon`, `genre_list`, the half-star rating
  math, nested nested `{% include %}` chains in `media_detail.html`) —
  this is the same class of check that caught real context-mismatch bugs
  in the habits and blog/code_intel migrations (the `view` dict-spread
  bug, the folder-README persistence bug). None found here.
- **Structural shim verification:** confirmed via AST that the shim's
  `ImportFrom` node imports exactly the 15 expected names, no more, no
  fewer, and that none of the 15 class/constant names remain **defined**
  (only imported) anywhere else in `models.py`.
- **Zero-external-consumers verification:** every file made available in
  this engagement (`routers/dashboard.py` in particular — its full
  import block was inspected directly, not assumed) was checked for a
  reference to any of the 15 media names. None found. **Caveat, stated
  the same way every prior postmortem states its own equivalent gap:**
  this could not be checked against files not shared in this engagement
  (`recipes.py`, `workout_plans.py`, `weekly_plan.py`, etc.) — no prior
  work order's SCOPE section lists any of them as a media consumer
  either, which is corroborating evidence, not proof.
- **CSS/static path verification:** confirmed via grep that
  `/static/css/media.css` (old path) appears zero times in the moved
  templates, and that `/static/media/css/media.css` (new path) appears
  in exactly the 4 templates that had a CSS link in the original
  (`media.html`, `media_search.html`, `media_recommend.html`,
  `media_settings.html`) — **not** 5, since `media_rec_history.html`
  never linked its own CSS in the original (see §1.4's self-caught
  mistake above).

**Not verified, and explicitly flagged rather than silently assumed:**
- `GET /media/search/query`, `POST /media/recommend/generate` — route
  code and template rendering confirmed identical to pre-migration; the
  live TMDB, OpenLibrary, ml-service, and Gemini calls behind them could
  not be exercised.
- A full, multi-domain `import models` (i.e., importing the *entire* root
  `models.py`, not just `domains.media.models` in isolation) was not run
  — `domains/jobs/models.py`, `domains/finance/models.py`,
  `domains/blog/models.py`, `domains/code_intel/models.py`,
  `domains/habits/models.py`, and `domains/journal/models.py` were never
  shared in this engagement to stub faithfully. The isolated
  `domains.media.models` check above is a genuine, real verification of
  the media domain's own mapper configuration; it is not a substitute for
  confirming the *whole* `Base.metadata` registry (all domains combined)
  is still internally consistent — that requires either the real files or
  a live environment.

## 1.6 Deployment

Not deployed in this engagement — no Docker/Airflow/MariaDB stack was
available. Per every prior migration's own deployment notes: `web`
bind-mounts the project root and runs `uvicorn --reload`, so once these
files land in the real repository, no rebuild or `docker compose down`
should be required — `docker compose restart web` as a clean checkpoint,
consistent with precedent. **This specific claim (reload picks up the new
`domains/media/` tree automatically) was not independently re-verified in
this engagement** — it's precedent from prior migrations' own deployment
notes, not something newly confirmed here.

## 1.7 What went well

- The zero-consumer discovery held up under actual scrutiny, not just
  assumption — the migration's own verification pass re-confirmed it
  independently (via direct inspection of `dashboard.py`'s import block)
  rather than trusting WO#9's SCOPE section's claim at face value.
- The self-caught `media_rec_history.html` mistake (§1.4) is exactly the
  kind of thing a less adversarial verification pass would have shipped
  — catching it demonstrates the verification step actually compared
  against the original file's structure rather than pattern-matching
  across the domain's other four pages.
- The model-layer and template-layer checks were **executed**, not just
  described — real `configure_mappers()` runs, real Jinja2 compilation
  and rendering against realistic mock data. This is a stronger
  verification bar than a purely static/read-only review would have
  cleared.

## 1.8 What could be improved / lessons learned

- **No live-infrastructure verification was possible at any point in this
  engagement**, for the migration or for any of the follow-on work in
  Part 2. This is the single largest gap across this entire document.
  Whoever deploys this should treat every "static/isolated verification
  passed" statement as a floor, not a ceiling — a real smoke test against
  live MariaDB, and ideally a real weekly run of the new DAG (Part 2.2),
  should happen before either is trusted unattended.
- Per §1.5's caveat: the full multi-domain `Base.metadata` identity check
  established as standard practice since WO#1 could not be run here for
  lack of the other domains' real model files. This isn't a media-specific
  gap — it's a standing limitation of doing domain work in an environment
  that only has some of the repository's files available — but it's worth
  naming plainly rather than letting the isolated single-domain check
  quietly stand in for something broader than it actually is.

## 1.9 Acceptance criteria results (WO#9, as originally scoped)

| # | Criterion | Result | Notes |
|---|---|---|---|
| 1 | `GET /media` renders identically | ✅ | Rendered against mock data through the real `partials/media_card.html` include chain |
| 2 | `GET /media/search`, `GET /media/search/query` render/return identically | ⚠️ | Template + route code confirmed identical; live TMDB/OpenLibrary calls not exercisable |
| 3 | `POST /media/search/add` find-or-create logic | ✅ | Router logic unchanged, confirmed via diff — only import lines changed |
| 4 | `GET /media/recommend` renders identically | ✅ | Rendered against mock context, all liked-count/ML-availability branches present |
| 5 | `POST /media/recommend/generate` reaches ML + Gemini layer | ⚠️ | Pipeline code and `_USE_GEMINI`/`_gemini_explain` untouched; ml-service and Gemini API not reachable |
| 6 | `GET /media/recommend/history` renders correctly | ✅ | Rendered against mock `MediaRecommendation` data |
| 7 | `GET /media/settings` renders identically | ✅ | Rendered against mock `StreamingService` data |
| 8 | `POST /media/settings/subscriptions` bulk-update | ✅ | Router logic unchanged; returns `streaming_service_list.html`, confirmed via render test |
| 9 | Detail/status/rate/notes/seasons/delete endpoints | ✅ | All relevant partials rendered successfully with mock context; router logic unchanged |
| 10 | `Base.metadata` identity check | ⚠️ | `domains.media.models` mapper-configures cleanly **in isolation** (real `configure_mappers()` run); full multi-domain check not run — see §1.5/§1.8 |
| 11 | Zero external consumers confirmed | ✅ | Confirmed against every file available in this engagement; caveat re: unseen files stated in §1.5 |
| 12 | `dashboard.py` required zero changes | ✅ | Confirmed directly by inspecting its actual import list |

**Per GOVERNANCE.md §4.6, WO#9 as originally scoped is considered
complete**: its models, routers, templates, and static assets all live
under `domains/media/`; `main.py` and `core/templating.py` reference the
new paths; a legacy shim exists in root `models.py`; every acceptance
criterion above is ✅ or an explained ⚠️; and — per §1.4 — no unrelated
behavior changed. **Nothing in Part 2 below was part of this
determination.**

---

# PART 2 — POST-MORTEM: FOLLOW-ON WORK (AGREED AND DELIVERED AFTER WO#9)

**Read this preamble before the three items below.** None of this was
part of WO#9's acceptance criteria (Part 1, §1.9) — it was proposed,
discussed, refined, and delivered in a separate, later conversation,
**after** WO#9 had already been reported complete and its own files
delivered. This is intentional, per GOVERNANCE.md §4.5 and §4.6 (quoted
in full in Section 0 above): migration work and improvement work are kept
separately reviewable by design, specifically so a reviewer can assess
"did the relocation succeed" independently of "were these separate
enhancements any good."

Two of the three items below edit a file WO#9 also touched
(`domains/media/routers/media.py`). Each subsection is explicit about
what WO#9 delivered vs. what changed on top of it, and why.

## 2.1 Follow-on item 1 — `GET /media` pagination / SQL-level filtering fix

**What prompted this:** raised as a recommendation when directly asked
"do you have any recommendations for what should be improved here," in
the conversation immediately following WO#9's delivery — not discovered
during WO#9's own acceptance-criteria verification pass. Explicitly
requested by the project owner as a follow-up change ("I like that,
please add make changes to add that").

**Root cause — pre-existing, not introduced by WO#9's relocation:**
`media_board()` (in the router WO#9 delivered, and in the original
pre-migration `routers/media.py` before it) built its query with no
`.limit()`, and applied the streaming-service filter as a second, Python
pass over the *entire* fetched result set:
```python
if service_filter:
    svc = await db.get(StreamingService, service_filter)
    if svc and svc.tmdb_provider_id:
        pid = svc.tmdb_provider_id
        user_media_list = [
            um for um in user_media_list
            if pid in (um.media_item.streaming_available_on or [])
        ]
```
This is structurally the same shape of bug documented in the jobs
postmortem §4.2: `list_jobs_ui`'s pre-fix query had no `.limit()` and
rendered every row twice, causing 60+ second page loads at ~2,300 rows.
Media's library is smaller today — this was caught by pattern-matching
against that documented precedent, before it became symptomatic, not by
a user-reported hang.

**Fix, delivered on top of WO#9's version of `domains/media/routers/media.py`:**
- Added `PAGE_SIZE = 300` and `.limit(PAGE_SIZE)` on the row query.
- Moved the service filter into SQL via
  `func.json_contains(MediaItem.streaming_provider_ids, str(provider_id))`,
  so it's applied *before* the cap truncates results, not after.
- Split the topbar stats (`stats.total`, `.completed`, `.in_progress`,
  `.want_to`) into their own `func.count(...)` query, computed
  independently of `PAGE_SIZE` — so "N tracked" can't silently become
  "N shown" once a library exceeds the cap.
- Dropped the `all_genres` computation from the router. **Confirmed via
  grep, not assumed,** that `media.html` never actually referenced
  `all_genres` in the template — it was being computed, via the same
  per-item Python loop this fix removes, for zero consumers.
- **Deliberately not done:** no "Load More" / keyset-pagination UI (the
  jobs domain's eventual WO#3 Phase 6 pattern). `PAGE_SIZE` is a hard cap
  with no way to see beyond it from the UI today, other than narrowing
  filters. This matches the scope of what was actually requested — worth
  building real pagination later if/when libraries approach the cap, not
  preemptively.

**Verification performed:**
- Syntax (`ast.parse`) on the edited router.
- The `JSON_CONTAINS` + `LIMIT` query **compiled successfully against the
  real MySQL/MariaDB SQLAlchemy dialect** (not just SQLite or an abstract
  dialect) — confirms the SQL SQLAlchemy would actually send is valid
  MariaDB syntax. **Not run against a live MariaDB instance** — no DB was
  available in this engagement.
- `media.html` re-rendered against mock data with the new,
  `all_genres`-free context — confirmed it still renders correctly and
  the stats display is intact.

## 2.2 Follow-on item 2 — new weekly DAG: `life_os_refresh_streaming_availability`

**What prompted this:** raised as a design question by the project owner
("should I have something that... updates on a weekly basis... to keep
the streaming services updated?"), refined once ("we can also query for
completed/abandoned, as long as it does not hurt the API... put a limit
to how many MediaItem IDs it can confirm... if want_to/in_progress don't
reach that limit, add completed then abandoned"), then explicitly
approved for building ("let's build the DAG as you lined up"). This is a
**new capability**, not a bug fix — `streaming_provider_ids` /
`streaming_fetched_at` have only ever been set once, at add-time, in
`media_search.py`'s `add_from_search()`, since before WO#9 existed; this
DAG is the first thing in the project that ever refreshes them.

**Design decisions, and why:**
- **Weekly, not real-time.** TMDB's watch-providers endpoint has no batch
  mode — confirmed via research, not assumed; TMDB has repeatedly and
  explicitly declined this as a feature request. Checking on every page
  load would mean one TMDB call per rendered title on every visit to
  `/media`, adding latency to a page this project has already had one
  real performance incident on (jobs), and risking TMDB rate limits.
- **Priority ordering as a single query, not three.** Implemented as one
  `ORDER BY` (priority tier: `want_to`/`in_progress` → `completed` →
  `abandoned`, then never-fetched-first, then oldest-fetched-first) +
  `LIMIT`, rather than three separate queries with manual
  remaining-budget bookkeeping — simpler, and structurally can't get the
  "spill into the next tier" math wrong.
- **A new `airflow/agents/media_agents.py`, not a `services/` import.**
  DAGs in this codebase never import from `services/` — GOVERNANCE.md
  §2.2 states this predates the domain-folder governance pass and is
  treated as absolute ("All DAG database access goes through
  `airflow/dag_db.py` raw SQL helpers"). `services/tmdb_service.py`
  already has a `get_streaming_providers()`-shaped function, proven
  working in `media_search.py` — but it could not be reused by a DAG
  under this project's own stated rule. **This is flagged explicitly as
  new debt, not silently absorbed:** the codebase now has two independent
  implementations of "call TMDB's watch-providers endpoint" — one in
  `services/tmdb_service.py` (never seen in this engagement — its exact
  parsing logic is unconfirmed), one in the new
  `airflow/agents/media_agents.py` (written from general knowledge of
  TMDB's public API shape, tested against a realistic mocked response,
  but never diffed against the real `services/tmdb_service.py`). See Part
  3, §3.9 for the recommended resolution path.

**Files created:**
- `airflow/agents/media_agents.py` — `get_tmdb_watch_providers(tmdb_id,
  media_type)`, extracts US `flatrate` (subscription) provider IDs only,
  matching `MediaItem`'s own docstring ("available for streaming in the
  US"). Raises on HTTP failure rather than returning `None`, specifically
  so the DAG can distinguish "confirmed nothing streams this" from "the
  request itself failed" and only advance `streaming_fetched_at` on a
  genuine answer.
- `airflow/dags/media/life_os_refresh_streaming_availability.py` — the
  DAG. `schedule_interval="@weekly"`, `REFRESH_INTERVAL_DAYS = 7`,
  `MAX_ITEMS_PER_RUN = 200` (both plain constants at the top of the file,
  easy to retune). Placed under `airflow/dags/media/` — matching the
  target subfolder structure WO#18 (DAG reorganization) already
  establishes for `life_os_generate_embeddings.py`, even though WO#18
  itself has not run. Two tasks: `select_and_refresh` (query candidates,
  call TMDB per item, push results to XCom — a per-item TMDB failure is
  logged and skipped, not raised, so one bad title doesn't fail the whole
  weekly run) → `apply_updates` (writes results back via
  `dag_db.execute()`, one `UPDATE` per row).

**Verification performed — real, executed, not just described:**
- **TMDB response-parsing logic** (`airflow/agents/media_agents.py`)
  unit-tested against a realistic `/watch/providers` JSON payload shape
  (mocking only the HTTP layer via `unittest.mock`): confirmed it
  correctly extracts US `flatrate` provider IDs only (not `rent`/`buy`,
  not non-US regions), returns `None` cleanly for a title with no US
  entry at all, and raises `ValueError` for an invalid `media_type`.
- **DAG task-function logic**, tested against a hand-built `dag_db`
  stand-in — same methodology the jobs domain's own Phase 7 DAG
  verification used (per the jobs postmortem §5, §4.4), including
  keeping the fake in-memory state in its own separate module
  specifically to avoid the circular self-import bug that postmortem's
  §4.4 documents an earlier version of this exact test pattern hitting.
  Confirmed, with real assertions, not just "it ran without erroring":
  - Priority ordering: a never-fetched `want_to` item and a stale
    `want_to` item both rank ahead of an equally-stale `in_progress`
    item within the same tier (never-fetched-first, then oldest-first);
    when the cap is reached mid-tier, the remaining budget correctly
    spills into the `completed` tier next, not `abandoned`.
  - A recently-refreshed item (inside `REFRESH_INTERVAL_DAYS`) and a
    non-TMDB item (an OpenLibrary book) are both correctly excluded from
    selection.
  - A simulated TMDB failure for one candidate is logged and skipped
    without aborting the rest of the run — the other candidates in the
    same batch still get processed and written.
  - A confirmed "nothing streams this right now" answer (`None`) is
    **written** to the row, not mistaken for a failure and silently
    dropped — this is the specific behavior the whole DAG exists to
    enable, so it was tested explicitly rather than assumed to follow
    from the other tests passing.

**Not verified, and explicitly flagged rather than glossed over:**
- **The real `airflow/dag_db.py` interface.** Per the jobs postmortem
  §8.3, this module's real interface has never been shared with any
  agent working on this codebase, across every prior DAG-touching work
  order in this project — this engagement inherits that exact same gap,
  not a new one. `fetch_all()`/`execute()` are called here using the
  conventions already established at other DAGs' call sites (`%s`
  parameterization, one `execute()` per row rather than assuming
  `execute_many()` supports differing values per row in one batch — that
  capability was never confirmed for any DAG in this project, this one
  included).
- Live TMDB API behavior (real rate limits, real auth failure modes).
- Real Airflow DAG registration/scheduling — `apache-airflow` was not
  installed in this engagement; verification used a minimal
  `DAG`/`PythonOperator` stand-in (same approach WO#18's own verification
  used), sufficient to exercise the task functions but not to confirm the
  DAG actually registers and schedules correctly inside a real Airflow
  instance.
- Real-data behavior of the `INNER JOIN` against `user_media` — a
  `MediaItem` with no corresponding `UserMedia` row is possible today
  (`DELETE /media/{id}` only deletes the `UserMedia` row, not the
  `MediaItem`) and is deliberately excluded from refresh by this join;
  this is an intentional design choice (an orphaned catalog item isn't in
  anyone's want-to/in-progress/completed/abandoned bucket, so there's no
  meaningful priority to assign it), but it was never confirmed against
  real orphaned rows.

**Outstanding manual action:** deploy, then watch at least one real
weekly run's logs — specifically the `dag_db` calls and the `INNER JOIN`
behavior noted above — before trusting this unattended. Same shape of
outstanding action every DAG-touching work order in this project has
left behind.

## 2.3 Follow-on item 3 — Libby/NYPL (OverDrive) availability research

**What prompted this:** a direct question ("would there be a way to check
if my account/library would have a digital copy of certain books..."),
answered first as a landscape overview, then — once the project owner
asked for it explicitly — with real documentation and a test script.

**This item ships nothing to the running application.** No router, no
model, no template, no DAG. It is exploratory/reference material only,
delivered as two new files:

- **`docs/libby_overdrive_api_notes.md`** — documents that OverDrive's
  official APIs require library-level (not individual-developer)
  registration, confirmed directly against OverDrive's own developer
  portal; and documents five endpoints of OverDrive's separate, public,
  unauthenticated "thunder" API (the one that actually powers
  `nypl.overdrive.com` and `libbyapp.com`), **sourced from a real, working,
  actively-maintained open-source client** (`ping/odmpy`, GPLv3, 400+
  stars) rather than guessed at. Explicitly documents the one confirmed
  gap in that research — no sourced, confirmed full-text
  search-by-title endpoint was found — and gives a concrete, low-risk way
  to close that gap (inspecting the site's own real network traffic via
  browser DevTools) rather than guessing at a plausible-looking endpoint
  shape and shipping it unverified.
- **`scripts/test_libby_availability.py`** — a standalone CLI implementing
  the five confirmed thunder-API endpoints (library lookup, single-title
  lookup, **bulk multi-title lookup in one call** — unlike TMDB, this API
  does support batching — and per-library availability). Not wired into
  the app. **Could not be run against live NYPL data from this
  environment** — the sandbox this work was performed in has no network
  path to `thunder.api.overdrive.com`; a direct attempt to fetch that
  domain during this engagement was blocked by the environment's own
  network allowlist. Treat this script as **unverified against live data**
  until it's actually run.

**Explicitly recommended against, and why (not merely deferred):**
automating patron authentication or hold placement. This requires storing
library-card credentials in a personal tool and automating a write action
against a commercial platform whose terms restrict this to approved
integrations; community tools that do this exist, but building custom
automation on top of that trades a small convenience for real fragility
(breaks silently whenever Libby's internal API changes) and ToS exposure.
The recommended pattern instead — check availability via the confirmed
read-only endpoints, then hand off to a real Libby deep link so the
actual borrow/hold action happens in the genuine app — is documented in
the notes file and should be treated as this project's actual position on
the question, not just one option among several.

---

# PART 3 — REQUIREMENTS FOR FINAL CLEANUP (execute only once ALL domains are migrated)

This section follows the exact structure and reasoning established in the
blog/code_intel, jobs, and explorer postmortems' own Part 2/Part 3
sections — and per explorer's own precedent, is written to be **thorough
enough to stand on its own**, without requiring the reader to
cross-reference all four prior postmortems to get the complete picture.
**Do not execute any of this mid-way through the migration program** —
every item below assumes every domain has already been extracted out of
the flat `routers/` / `templates/` / `models.py` layout, media included.

**As of this postmortem, that is not yet true.** See §3.7 below for the
current, directly-confirmed state of the migration backlog — it differs
in one respect from what WO#9's own "For the next work order" note
assumed.

## 3.1 How to know you're actually ready

Same standing check every prior postmortem specifies:
```bash
grep -rn "from models import" --include="*.py" .
```
**Target end-state:** nothing except `models.py`'s own internal
comments/imports. As of the end of this postmortem's coverage (WO#9 plus
its follow-on work), this command's expected output is the four prior
postmortems' snapshots **plus one new, unusual line**:
```
routers/dashboard.py:...:from models import (
models.py:...  (Base, in a comment)
models.py:...  (comment, inside the Jobs shim docstring)
models.py:...  (comment, inside the Finance shim docstring)
models.py:...  (comment, inside the Blog shim docstring)
models.py:...  (comment, inside the Code Intel shim docstring)
models.py:...  (comment, inside the Habits shim docstring)
models.py:...  (comment, inside the Journal shim docstring)
models.py:...  (comment, inside the Media shim docstring)   ← NEW (WO#9)
```
i.e., exactly one real remaining consumer (`routers/dashboard.py`) plus
shim docstring comments, for every domain migrated so far — **including
media, even though media's shim has zero actual consumers** (see §3.2
below for why the shim still exists and still shows up here despite
that). Confirm this list has shrunk to zero real consumers
(`dashboard.py` included) before proceeding — per GOVERNANCE.md §3.3,
`recipes`/`pantry` (WO#7), `workout` (WO#8), and `planning` (WO#10) still
need to migrate before that's true (§3.7).

## 3.2 `models.py` final cleanup — full current shim picture, with a precise per-domain usage breakdown

Reproduced here in full, per domain, so this document is self-contained.
**New in this postmortem, not present in any prior one:** an exact
breakdown of which names within each shim `routers/dashboard.py` actually
imports — not just "dashboard.py is a consumer," but *which specific
names*, confirmed by direct inspection of `dashboard.py`'s real import
statement rather than inferred. This matters because two of the seven
shims below turn out to be **fully orphaned already** — not just
"probably safe to remove eventually," but zero actual usage today.

`dashboard.py`'s real, complete import block (confirmed verbatim, not
paraphrased):
```python
from models import (
    Job, ApplicationLog, ApplicationStatus,
    StagingJob, StagingJobStatus,
    Transaction, BlogIdea, BlogIdeaStatus,
    Habit, HabitLog, HabitSettings,
    JournalEntry, WeeklySynthesis,
)
```

| Domain | Shim size | Names dashboard.py actually imports | Fully orphaned? |
|---|---|---|---|
| Jobs | 13 names | `Job`, `ApplicationLog`, `ApplicationStatus`, `StagingJob`, `StagingJobStatus` (5) | No — genuinely used |
| Finance | 9 names | `Transaction` (1) | No — genuinely used |
| Blog | 6 names | `BlogIdea`, `BlogIdeaStatus` (2) | No — genuinely used |
| Code Intel | 12 names | *(none)* | **Yes — 0 names used by dashboard.py today** |
| Habits | 7 names | `Habit`, `HabitLog`, `HabitSettings` (3) | No — genuinely used |
| Journal | 2 names | `JournalEntry`, `WeeklySynthesis` (both) | No — genuinely used |
| **Media (WO#9)** | 15 names | *(none)* | **Yes — 0 names used by dashboard.py today, and 0 known consumers anywhere else (see §1.5)** |

**Important distinction between Code Intel and Media, despite both
showing "fully orphaned" above:** Code Intel's zero-usage-by-dashboard
finding is a **byproduct** of this postmortem's own dashboard.py
inspection — no prior postmortem in this series (including
blog/code_intel's own) broke down dashboard.py's usage per-domain this
precisely, so it's newly surfaced here, not something WO#2's own
engagement checked for. **It has not been independently confirmed that
nothing else in the repo references Code Intel's classes** — treat it as
a strong, well-evidenced lead, not a settled fact, until someone runs the
same "confirm zero consumers anywhere" check WO#9 ran for media
specifically. Media's zero-consumer status, by contrast, **was** the
subject of a direct, dedicated check across every file available in this
engagement (§1.5) — it carries a stronger evidentiary basis.

**Practical implication for whoever runs the shim-removal pass (WO#20):**
media (and very likely code_intel) can be removed with **zero
corresponding `dashboard.py` edit** — a pure deletion, unlike jobs,
finance, blog, habits, and journal, which each require `dashboard.py`'s
import line to be repointed in the same change. Per the batching
principle established in the habits postmortem ("do this for all domains
in the same pass, not incrementally") and restated in WO#20's own HARD
BOUNDARIES, **do not remove media's shim in isolation before WO#20
actually runs** — this section documents that it's a low-risk removal
when that pass happens, not an invitation to do it early.

The eventual `models.py` end-state (per the blog/code_intel postmortem's
own Option 1 vs. Option 2 tradeoff writeup, which applies identically
here — media introduces no new wrinkle to that decision) is either full
deletion or reduction to a pure import-registry. Either way, `media`'s
entry in whichever form that takes is a one-line addition:
```python
from domains.media import models as _media_models          # noqa: F401
```

## 3.3 `routers/dashboard.py` — confirmed unaffected by media, now and at final-cleanup time

Per §3.2's table, `dashboard.py` never imported anything from the media
domain, at any point — WO#9 required zero changes to this file (§1.9,
criterion 12), and **at final-cleanup time, media requires zero
corresponding edit to `dashboard.py`** either, unlike every domain whose
shim `dashboard.py` actually consumes. When the final cleanup pass
updates `dashboard.py`'s import block to pull directly from each
migrated domain (per the blog/code_intel postmortem's own Part 2 §2
example), media simply never appears in that updated block — there is
nothing to add.

## 3.4 Cross-domain SQLAlchemy relationship risk — media is a non-issue here

Per the critical warning first raised in the blog/code_intel postmortem
(Part 2 §3) and re-confirmed by every postmortem since: any
`relationship()` call using a **string** class name only resolves
correctly if every module defining a referenced class has been imported
by *something* before the first query touches that relationship.

**Media adds zero risk here.** None of its five ORM classes
(`StreamingService`, `MediaItem`, `UserMedia`, `TVSeasonProgress`,
`MediaRecommendation`) reference any class outside `domains/media/models.py`
— every `relationship()` in that file is a same-module reference,
structurally identical to the `Job`/`ApplicationLog` case from WO#3, and
requires no special registry handling beyond `domains/media/models.py`
itself being imported once (already guaranteed by the shim in §3.2, the
same mechanism as every other domain).

Still true, and still needs checking at final-cleanup time regardless of
media specifically — re-run, one final time across *every* domain's
`models.py`, before finalizing whichever registry approach §3.2 settles
on:
```bash
grep -rn 'relationship(' domains/*/models.py
```

## 3.5 `core/templating.py` final cleanup

Media's entry (`FileSystemLoader("domains/media/templates")`) is already
correctly appended to the `ChoiceLoader` list as of WO#9. At final
cleanup time, the eventual full list should look like:
```python
templates.env.loader = ChoiceLoader([
    FileSystemLoader("templates"),               # shared/core only
    FileSystemLoader("domains/habits/templates"),
    FileSystemLoader("domains/blog/templates"),
    FileSystemLoader("domains/code_intel/templates"),
    FileSystemLoader("domains/jobs/templates"),
    FileSystemLoader("domains/explorer/templates"),
    FileSystemLoader("domains/finance/templates"),
    FileSystemLoader("domains/journal/templates"),
    FileSystemLoader("domains/media/templates"),
    # ... one line per remaining domain (recipes, workout, planning)
])
```
Nothing further needed for media specifically beyond confirming this line
survives whatever final-form `core/templating.py` the last domain's
migration converges on.

## 3.6 `main.py` final cleanup

Media's router import (`from domains.media.routers import media,
media_search, media_recommend, media_settings`) and static mount
(`/static/media`, registered before the general `/static` mount) are both
already correctly in place as of WO#9. **Confirmed by direct inspection
of the current `main.py`** (not assumed): the pre-existing include order
— `media_search` → `media_recommend` → `media_settings` before the
`media` catch-all — was preserved exactly, matching the comment already
in the file explaining why.

By the end of the full migration program, `main.py`'s router-import block
should be entirely `from domains.X.routers import Y` — as of this
postmortem, it is **not** yet: `recipe_extract`, `recipe_discovery`,
`pantry`, `recipes`, `workout`, `workout_log`, `workout_plans`,
`workout_settings`, `intent`, and `weekly_plan` are all still imported
via `from routers import ...` (§3.7 has the full detail). Nothing here is
media's responsibility to fix — noted so whoever does the final pass
knows exactly what's still outstanding without re-deriving it from
scratch.

## 3.7 Domains still outstanding — confirmed current state, with an explicit discrepancy flagged

**This is a direct finding from this postmortem's own inspection of the
actual `main.py` provided in this engagement, not an assumption carried
over from any prior document.**

WO#9's own "For the next work order" section states: *"Per GOVERNANCE.md
§3.3, Work Order #10 = `planning`... is the last of the real domains...
`weekly_plan.py` alone is on record as depending on:
Recipe/RecipeMealType/Ingredient/PantryItem... from WO#7,
WorkoutPlan/WorkoutPlanDay/WorkoutSession/WeightUnit... from WO#8..."* —
phrasing that assumes, by the time planning (WO#10) is drafted, recipes
(WO#7) and workout (WO#8) will already be done.

**The actual `main.py` inspected during this engagement contradicts that
assumption.** It contains, unchanged from before WO#9:
```python
from routers import recipe_extract, recipe_discovery, pantry, recipes
from routers import workout, workout_log, workout_plans, workout_settings
...
from routers import intent, weekly_plan
```
None of these are `domains.*` imports. **Recipes+pantry (WO#7) and
workout (WO#8) have not actually been executed**, despite WO#9 (media)
having already run. This means the domains actually still outstanding,
confirmed as of this postmortem, are:

- `recipes` + `pantry` (WO#7)
- `workout` (WO#8)
- `planning` (`weekly_plan` + `intent`, WO#10 — genuinely last, since it
  depends on WO#7 and WO#8 both being done first, per its own real
  cross-domain dependencies, which remain accurate regardless of
  execution order)

This isn't presented as a problem to solve — only as a factual correction
to whatever sequencing a prior document implied, so the next agent
doesn't assume recipes/workout are further along than they actually are.
**Confirm this grep-style check yourself against the real, current
`main.py` before relying on it** — this postmortem's finding is only as
current as the file it inspected.

## 3.8 Fate of `routers/`, `templates/`, `static/`

Media contributed **zero** leftover files to the flat top-level
directories — every router, template, and the one static asset were
fully moved, confirmed via the file lists in §1.3. Media does not block
deleting the flat `routers/`, `templates/`, or `static/css/` directories
once every other outstanding domain (§3.7) reaches the same state —
confirm no stray media-related file was accidentally left behind before
deleting, same check every prior domain's postmortem specifies for
itself.

## 3.9 New from the follow-on work (Part 2) — items final cleanup, or some other future work, needs to account for

None of these were introduced by WO#9 itself — they're new as of the
follow-on work in Part 2, and are collected here because they're the kind
of thing a final-cleanup pass (or some other future work order) should
know about rather than rediscover:

1. **Two independent TMDB watch-providers implementations now exist** —
   `services/tmdb_service.py` (router-facing, pre-existing, never seen in
   this engagement) and `airflow/agents/media_agents.py` (DAG-facing, new,
   §2.2). This is a direct, structural consequence of GOVERNANCE.md §2.2's
   DAG/service import boundary, not an oversight — but it's real
   duplication and deserves a deliberate decision at some point: diff the
   two implementations for behavioral drift (region/type extraction logic
   in particular — confirm `services/tmdb_service.py` also only extracts
   US `flatrate`, not `rent`/`buy`, before assuming the two agree), or
   formally document the split as the correct, permanent DAG/router
   boundary. Do not silently "fix" this by having the DAG import the
   service — that would violate GOVERNANCE.md §2.2 as currently written;
   if that rule itself should change, that's a GOVERNANCE.md amendment
   (§6), not a quiet workaround.
2. **`airflow/dag_db.py`'s real interface is still unseen**, same
   standing gap the jobs postmortem §8.3 already flagged — this
   postmortem's new DAG inherits that exact gap rather than introducing a
   new one. Whoever eventually gets the real module's source should use
   it to retroactively verify both the jobs domain's Phase 7 logic *and*
   this DAG's `select_and_refresh`/`apply_updates` tasks.
3. **`life_os_refresh_streaming_availability.py` is already placed under
   `airflow/dags/media/`**, matching WO#18's target reorganized structure
   even though WO#18 itself hasn't run. When/if WO#18 does run, this file
   needs **zero** additional move — worth confirming explicitly in that
   work order's own acceptance criteria rather than assuming.
4. **The Libby/NYPL research (§2.3) is not part of any domain's cleanup
   footprint** — it shipped no application code. It's a fully separate,
   still-open thread (no `services/libby_service.py`, no router, nothing
   wired in) for whoever picks it up next, independent of the domain
   migration program's own sequencing.

## 3.10 Final verification checklist (run after the LAST domain migration + this cleanup)

Standing checklist from every prior postmortem's own closing section,
reproduced and extended with media-specific items:

- [ ] `grep -rn "from models import" --include="*.py" .` returns nothing
  (or only the registry file's own `Base` import, per §3.2 Option 2).
- [ ] `grep -rn 'relationship(' domains/*/models.py` — every string-named
  class's module is covered by the registry mechanism chosen. Media
  contributes zero entries requiring special handling (§3.4).
- [ ] `Base.metadata` table-identity check: same total table count
  before/after the **entire** migration series; `models.MediaItem is
  domains.media.models.MediaItem` (and the other 4 media classes) —
  same identity-check pattern as every other domain, **not yet run
  end-to-end in this engagement** (§1.5/§1.8) — must happen for real
  before this checklist item can be checked off.
- [ ] Every remaining router's `Jinja2Templates(...)` instantiation has
  been replaced with `from core.templating import templates` — media's
  is already done (confirmed, §1.3).
- [ ] Every domain's static mount is registered before the general
  `/static` mount — media's is already done and confirmed correctly
  ordered (§3.6).
- [ ] Full regression suite passes against the final structure, including
  a **live** re-run of every check in §1.5/§1.9 that was only run
  statically/in isolation in this engagement, plus the `life_os_
  refresh_streaming_availability` DAG's first real weekly run (§2.2).
- [ ] **Media-specific:** confirm `docker compose restart web` (or
  equivalent) actually picks up `domains/media/` with no rebuild — stated
  as precedent in §1.6, never independently re-confirmed here.
- [ ] **Media-specific:** confirm the two TMDB watch-providers
  implementations (§3.9, item 1) were either reconciled or deliberately
  documented as separate by design — don't let this silently stay an open
  question past final cleanup.
- [ ] **Media-specific:** confirm §3.7's "recipes/workout not yet
  migrated" finding has been resolved (i.e., WO#7 and WO#8 have actually
  run) before treating the domain migration program as complete.
