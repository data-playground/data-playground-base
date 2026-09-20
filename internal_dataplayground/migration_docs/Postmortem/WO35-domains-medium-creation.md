# WO#35 Post-Mortem — Medium Article Extraction (`domains/medium/`)

**Status as of this document:** Phase 1 and Phase 2 both complete per the
original work order's own definition of done. Four schema migrations exist
(`m3d1um_d0ma1n001` → `m3d1um_d0ma1n004`); **none are confirmed applied
against the real production database** as of this writing — every migration
in this series was authored and syntax-checked in a sandbox with no live
MariaDB connection. That is the single biggest gap between "this document
says done" and "this is actually running in production," and it's why §7
exists.

This document is written for a future agent or reviewer to reconcile
requirements against, not as a narrative to be read once. It assumes no
memory of the conversation that produced it.

---

## 1. Original Work Order

WO#35 opened as a **revisit of an existing implementation**, not a
greenfield build — explicitly different in kind from the sports-domain work
orders (WO#33/#34) that preceded it. Its structure:

- **ROLE:** modernize an existing feature; treat the prior implementation as
  real prior work to understand before changing it, but with real latitude
  to rebuild once understood — not a line-for-line preservation migration.
- **HARD BOUNDARIES:**
  - Phase 1 (assessment) must complete and be reviewed before Phase 2
    (build) starts.
  - Any AI/LLM calls must route through `services/ai/` — not applicable in
    practice, since nothing in the medium domain calls an LLM.
  - If a DAG is part of the design, it must never import `models.py`,
    `database.py`, or any router/service.
  - If folded into `blog` instead of its own domain: preserve the
    Researcher→Ghostwriter→Refiner→Editor pipeline shape unless there's a
    clear reason not to. (Moot — own domain was chosen; see §2.)
- **Phase 1 deliverable:** a report covering what the existing
  implementation actually did, how it integrated with the rest of the app,
  what needed revisiting, a recommendation (own domain vs. fold into blog),
  and open questions — with **no code written** during this phase.
- **Phase 2 acceptance criteria (own-domain branch):** `models.py` imports
  `Base` from `core.base_model`; routers under 300 lines each; templates
  extend `base.html`; `static/` only if needed; if a DAG exists, it lives
  under `airflow/dags/medium/` and only touches the DB through `dag_db.py`;
  the domain is wired into `main.py`, `core/templating.py`, static mount
  ordering, and the sidebar.

## 2. Phase 1 — Assessment & Direction

The actual prior implementation could not be produced — the owner reported
it was deleted by accident, with no code surviving. Phase 1 proceeded as a
reconstruction from the owner's recollection instead of a code trace, which
the work order's own text explicitly allows ("...or a working description
if the code itself isn't readily shareable").

**What the deleted implementation did, per that recollection:** used
Medium's *internal, undocumented* GraphQL endpoint
(`medium.com/_/grout/graphql`) to pull a personalized "recommended
articles" feed, including per-category recommendations, and full article
text. It was never scheduled — run manually while experimenting — and
wrote output to local files, with no integration into the rest of the app
(no DB writes, no router, no DAG).

**Key finding that shaped everything downstream:** that endpoint is not
Medium's public, documented API — it's the same private endpoint Medium's
own web frontend uses internally. Building a daily automated pipeline on
top of it carries real risk: no stability guarantee, and pulling
personalized ("recommended") results requires an authenticated session,
not a clean API token, which is a different and more fragile trust
boundary than everything else in this app's `services/` layer.

**Recommendation made and accepted:** own domain, `domains/medium/`. Blog
exists to help the owner *write and publish* their own content; this domain
exists to help them *read* other people's — different purpose, not a good
fit for folding into blog's pipeline.

**Design that came out of Phase 1 and was explicitly agreed before Phase 2
started — the "two-flank" plan:**
1. **RSS flank** (this repo builds this): Medium's public, documented,
   stable per-profile/publication/topic/custom-domain RSS feeds, for
   sources the owner already knows they want to follow. No auth, no
   personalization, low fragility.
2. **GraphQL flank** (explicitly **not** built by this work — owner's own,
   separate effort): the internal GraphQL endpoint, kept to *manual,
   occasional discovery* of new sources rather than an automated pipeline.
   An automated version of this flank was directly requested at one point
   during the build and **declined** — the specific script shown used
   `curl_cffi`'s browser-TLS-impersonation feature together with Cloudflare
   clearance cookies, which is built specifically to defeat a site's
   anti-bot detection regardless of run cadence (manual vs. scheduled
   doesn't change what the code does). That refusal stands; nothing in
   `domains/medium/` touches the GraphQL endpoint at all, by design.

## 3. Phase 2 — What Was Actually Built

### Domain code (`domains/medium/`)
| File | Purpose |
|---|---|
| `models.py` | `MediumFeedSource` (tracked sources) and `MediumArticle` (ingested articles). See §7.2 for two stale docstrings in this file that need fixing. |
| `rss_ingest.py` | Feed URL construction (`build_feed_url`), fetch/parse (`fetch_feed`, `parse_feed`), source auto-detection from a pasted link (`candidate_sources`, `identify_source`), and thumbnail extraction (`extract_thumbnail`). Deliberately stdlib-only (`xml.etree`, `email.utils`, `re`) — no `feedparser` dependency, matching this codebase's stated "don't add a dependency without a real functional gain" discipline (see `services/ai/README.md`'s SDK Exceptions section, which sets that precedent). 334 lines — not a router, not subject to the 300-line ceiling. |
| `formatting.py` | `relative_time()`, `estimate_read_minutes()` — pure display helpers, split out specifically so they're unit-testable without FastAPI/a DB. |
| `routers/medium_settings.py` (224 lines) | `/medium/settings` — list/add/toggle/delete tracked sources, auto-detect-and-add (Airflow-triggered, see §4), manual-add fallback, "Run Ingest" button. |
| `routers/medium_feed.py` (151 lines) | `/medium` and `/medium/more` — the reading page: four view layouts (Latest/Feed/Card grid/Compact) rendered from one query, source filter, "Show More" pagination. |
| `templates/settings.html`, `templates/articles.html`, `templates/partials/more_articles.html` | Extend `base.html`; use the app's real design tokens (`var(--accent)`, `var(--bg-card)`, etc.) and class conventions, copied from `domains/habits/` and `domains/blog/` once those were available as reference — **not** invented ad hoc (see §5, this was a real bug in an earlier round). |
| `static/css/medium.css` | Domain-scoped styling, linked via `{% block extra_css %}`, mounted at `/static/medium`. |
| `fixtures/`, `tests/` | `sample_feed.xml` fixture (synthetic, not real Medium content) plus test files: `test_rss_ingest.py`, `test_identify_source.py`, `test_extract_thumbnail.py`, `test_formatting.py`. |

### Airflow (`airflow/dags/medium/`)
| File | Purpose |
|---|---|
| `life_os_medium_ingest.py` | Daily scheduled DAG. Reads active sources from `medium_feed_sources` via `dag_db.fetch_all()`, calls `rss_ingest.ingest_sources()`, upserts into `medium_articles` via `dag_db.execute_many()` — one `INSERT ... ON DUPLICATE KEY UPDATE` per article, all in **one transaction per run** (see §6 — this is a known, currently-accepted limitation, not an oversight). |
| `life_os_medium_detect_source.py` | On-demand DAG, triggered from the settings page. Takes a raw pasted URL via `conf`, runs `identify_source()`, inserts directly into `medium_feed_sources` on success. No separate "pending request" table — see §4, this was a deliberate design change from the first instinct. |
| `tests/test_life_os_medium_ingest.py`, `tests/test_life_os_medium_detect_source.py` | Import the **real** DAG files (not reimplementations) with `airflow`/`dag_db` faked via `sys.modules`, to exercise the pure logic (column mapping, upsert statement building, error propagation) without needing a live Airflow/MariaDB. |

### Wiring
- `main.py` — `medium_feed`/`medium_settings` router registration, `/static/medium` mount (positioned before the catch-all `/static` mount, per the existing ordering rule).
- `core/templating.py` — `domains/medium/templates` added to the `ChoiceLoader`.
- Sidebar nav entries for `medium`/`medium_settings` already existed in `sidebar.html` before this work order touched anything; the routers simply weren't passing `active_module` in their template context, so highlighting was silently broken until that was added.

### Schema (`alembic/versions/`)
| Revision | What it did |
|---|---|
| `m3d1um_d0ma1n001` | Created `medium_feed_sources` and `medium_articles`. No FK between them (deliberate — see the model's own denormalization note). No seed data. |
| `m3d1um_d0ma1n002` | Added `medium_articles.raw_item` (`TEXT` at the time) — verbatim `<item>` XML, added as a forward-looking safeguard so a field nobody thought to extract could be backfilled later without re-fetching (Medium's feeds only ever show the ~10-25 most recent items per source). |
| `m3d1um_d0ma1n003` | Added `medium_articles.thumbnail_url` (`String(1000)`, nullable). |
| `m3d1um_d0ma1n004` | Widened `raw_item`, `content_html`, `summary` from `TEXT` (65,535-byte cap) to `MEDIUMTEXT` (16MB) — reactive fix for a real production failure, not preventive. See §5. |

Each migration's `down_revision` carries a "verify via `alembic heads`
before applying" warning, because this chain's head drifted more than once
during the build from unrelated parallel work landing on other domains
(documented in `s0cc3r_l1n3ups001`'s own docstring, which this chain
observed directly). **Do not assume `m3d1um_d0ma1n004` is still the current
head without checking** — see §7.1.

## 4. Decisions That Changed After the Initial Migration

This is the section a reviewer most needs: what shipped after
`m3d1um_d0ma1n001` deviated from the shape implied by that first migration,
and why, on the record.

**1. Source detection went from synchronous-in-request to
Airflow-triggered-async — a real architecture change, not a refactor.**
The first working version of "paste a link, add a source" ran
`identify_source()` inline inside the settings page's POST handler (off the
event loop via `run_in_threadpool`, since it makes blocking HTTP calls).
This worked and was tested. It was then explicitly replaced, at the owner's
request, with the current design: the settings page fires the
`life_os_medium_detect_source` DAG and gets back only "queued" or
"couldn't reach Airflow" — detection success/failure is now visible in
Airflow's own run history, not in the app. The stated reason: wanting
proper retry/tracking without reinventing it in the app. **A new
"pending request" table was proposed as part of this change and rejected**
— the owner's own judgment was that it would duplicate what Airflow's run
history already gives for free, and the final design has no such table.
This is worth a reviewer's attention specifically because it means the
settings page's UX intentionally regressed (no more immediate "found 3
articles, added as publication X" feedback) in exchange for better
failure recovery — that trade was made deliberately, not by accident.

**2. `raw_item` and `thumbnail_url` were not in the original schema.** Both
were added after `m3d1um_d0ma1n001` at the owner's explicit request, as
follow-on hardening/features, not because the first migration was wrong.

**3. The filter UI shipped twice.** The first version (a native
`<select multiple>`, auto-submitting on every change) was explicitly
rejected on sight — too large, and refreshing on every click was
unpleasant. It was rebuilt as a collapsed dropdown with checkboxes and an
explicit "Apply" button, gated behind one page load per filter change
instead of one per checkbox.

## 5. Bugs Found and Fixed During the Build

Kept distinct from §4's deliberate design changes — these are defects,
found either in review or in production, not reconsiderations.

- **Templates were initially placed at the wrong path** (`templates/medium/`
  instead of `domains/medium/templates/`), with router code referencing
  them via a `"medium/"`-prefixed filename. Caught by the owner comparing
  directly against `domains/habits/`. Fixed: templates moved, routers
  switched to bare filenames, and — while in there — routers were also
  switched from each creating its own local `Jinja2Templates(...)` instance
  to importing the shared `templates` object from `core.templating`,
  matching `habits.py`'s actual pattern.
- **CSS was initially built with invented custom properties**
  (`--mp-text`, `--mp-border`, etc.) because `base.css`/`habits.css`
  weren't available yet at that point in the build. Once `habits.css` and
  `blog.css` were provided as reference, the domain's CSS was rebuilt
  against the app's real tokens. A follow-on round of that same rebuild
  also turned up genuinely missing rules (`.settings-section`,
  `.form-group`, `.form-input`, `.action-btn`, `.toggle-active-btn` were
  referenced in templates but never actually defined in the CSS file — an
  override for a base rule had been written without the base rule itself).
- **`build_feed_url()` produced `https://https://...` for a `CUSTOM_DOMAIN`
  source** when a full URL (not a bare domain) was typed into the manual
  "Add Manually" form. Root cause: the function always prepended `https://`
  without checking whether the stored identifier already had a scheme.
  Fixed in two places — defensively in `build_feed_url()` itself (so an
  already-bad stored row self-corrects without manual DB intervention) and
  at input time in the manual-add handler (so it can't happen again for
  new entries).
- **`docker-compose.yml`'s `web` service was missing
  `AIRFLOW_SECRET_KEY`** in its environment block — present for the three
  `airflow-*` services (used there to set Airflow's actual admin password
  and Flask secret key) but never passed through to `web`, where
  `services/airflow_service.py` needs it to authenticate outbound trigger
  calls. This was a **pre-existing bug, not introduced by WO#35** — it
  silently broke every "click a button, fire a DAG" feature in the app
  (blog's Scout/Creator/Finalizer/Idea Expander, code_intel's README
  Writer and Narrate/Comment/Improve DAGs), not just anything in the medium
  domain. It surfaced because WO#35 was the first feature built that
  actually exercised `trigger_airflow()` and got tested end to end. Fixed
  by adding the one missing line; confirmed working by the owner
  afterward (both "Run Scout" on the blog page and the medium detect flow).
  **This is a strong candidate for a GOVERNANCE.md amendment** — see §7.3.
- **`medium_articles.raw_item` (and, latently, `content_html`/`summary`)
  exceeded MySQL's `TEXT` type's 65,535-byte cap** on a real, long article,
  failing the daily ingest DAG with `DataError 1406`. Because every
  article in a run lands in one `execute_many()` transaction, this rolled
  back the **entire day's batch across all 4 active sources**, not just
  the one oversized article. Fixed via `m3d1um_d0ma1n004` (widened to
  `MEDIUMTEXT`, 16MB). The all-or-nothing transaction behavior itself was
  **not** changed — see §6.

## 6. Outstanding / Not Yet Done

- **GraphQL-based recommendation discovery.** By design (§2) — this is the
  owner's own, separate effort. No code for it exists anywhere in
  `domains/medium/`.
- **Recommendation-based article ordering.** The owner has stated intent
  (recorded separately) to build their own recommendation system later and
  add a second selector next to the current Latest/Feed/Card
  grid/Compact view toggle to switch between date-ordering and
  recommendation-ordering. Not started; no code, no schema for it.
- **All-or-nothing ingest transaction.** Flagged explicitly to the owner
  after the `MEDIUMTEXT` fix — one bad row in a daily run currently costs
  the entire batch across every source, not just the offending row. This
  was surfaced as an open question, not resolved either way. Worth a
  decision (and if changed, its own migration-adjacent work, not a schema
  change) before the next time something in a single article breaks the
  whole run.
- **Visual QA on real thumbnails** (real `<img>` at 112×75 with
  `object-fit: cover`) has not been confirmed against actual Medium cover
  images — only structurally verified (correct HTML/CSS, correct
  fallback-to-placeholder behavior).
- **`rss_ingest.py` parsing logic is unit-tested against synthetic/fixture
  data**, not a comprehensive real-world corpus. Real-world issues have
  surfaced incrementally as they were hit in production (the double-scheme
  bug, the oversized-column bug) rather than being caught in advance by the
  test suite — worth treating the test suite as a floor, not a ceiling, on
  confidence.

## 7. What Needs to Happen After All Migrations Are Completed

This is the section this document was specifically requested to include.
"All the other migrations" means: every `alembic` migration currently
sitting unapplied in this repo — not just the medium domain's four, but
whichever others (soccer's lineup tables, and anything that's landed since)
are ahead of or interleaved with them — actually applied to the real
database, with a single confirmed head.

### 7.1 Verification checklist once migrations are applied
1. Run `alembic heads` — confirm a single head, and that it's
   `m3d1um_d0ma1n004` or a later revision that chains through it.
2. Confirm the live schema matches `models.py` exactly: `medium_articles`
   should have `raw_item`, `content_html`, and `summary` as `MEDIUMTEXT`
   (not `TEXT`), and `thumbnail_url` should exist as a nullable
   `VARCHAR(1000)`.
3. Manually re-trigger `life_os_medium_ingest` (the settings page's "Run
   Ingest" button) rather than waiting for the next scheduled run, to
   confirm the `MEDIUMTEXT` widening actually resolves the DataError
   against real data — the 2026-09-18 failure means that day's batch was
   never saved, and Medium's feeds only hold ~10-25 recent items per
   source, so there's real risk of losing content to feed rotation the
   longer this waits.
4. Spot-check `medium_feed_sources` for any `custom_domain` row whose
   `identifier` still contains a scheme (`http://`/`https://`) — a
   leftover from the double-scheme bug (§5). The running code now handles
   this defensively at read time, so it's not urgent, but a clean row is
   better than a defensively-patched one.

### 7.2 Known-stale references in `models.py` — fix regardless of migration timing
These are factually wrong **right now**, independent of anything pending;
they describe an earlier state of the schema that the code around them has
since outgrown:
1. **Module-level docstring** (top of the file): states article storage
   "is intentionally not modeled yet" and that adding it "would be guessing
   at the shape before we've seen real data." `MediumArticle` has existed
   since the very first migration in this series (`m3d1um_d0ma1n001`) —
   this description was accurate before Phase 2 started and has been wrong
   since.
2. **`MediumArticle`'s class docstring**, `guid` field description: refers
   to "the eventual `dag_db.py`" — `dag_db.py` is not eventual, it's a
   real, confirmed file that `life_os_medium_ingest.py` and
   `life_os_medium_detect_source.py` both call today.
3. **`MediumArticle`'s class docstring**, thumbnail note: states "No
   thumbnail column yet" — `thumbnail_url` has existed since
   `m3d1um_d0ma1n003`, directly contradicting the column defined a few
   lines below that same comment.

None of these three require a migration to fix — they're documentation,
not schema — but they should be corrected together with (not before) the
migration-completion pass, so whoever does it can also confirm anything
else in the file has caught up with reality at the same time.

### 7.3 Documentation — candidate GOVERNANCE.md amendment
Per GOVERNANCE.md §6's own amendment process ("this document is updated
whenever a work order surfaces a rule worth generalizing"), the missing
`AIRFLOW_SECRET_KEY` bug (§5) is a strong candidate: a new domain's feature
was the first thing to actually exercise `trigger_airflow()` end to end,
and in doing so surfaced a pre-existing gap that had silently affected
several other domains' Airflow-trigger features. The generalizable rule:
**when a new router dependency needs a secret/env var, verify it's passed
through in every `docker-compose.yml` service that will actually call it
— not just the service that owns the secret's other config values.**
Not applied to GOVERNANCE.md by this document; flagged for the owner to
decide whether to add.

### 7.4 Open architecture decisions carried forward
- Whether to change `_upsert_articles()` from one all-or-nothing
  transaction to per-article isolation (§6) — unresolved, needs an
  explicit decision, not a default.
- Recommendation-based ordering (§6) — waiting on the owner's own
  recommendation-system work; the second view-selector it implies is not
  built and shouldn't be guessed at ahead of that work landing.

## 8. Acceptance Criteria — Mapped Against the Original Work Order

| Criterion (own-domain branch) | Status |
|---|---|
| `models.py` imports `Base` from `core.base_model` | ✅ |
| Routers under 300 lines each | ✅ — `medium_feed.py` 151, `medium_settings.py` 224 |
| Templates extend `base.html` | ✅ |
| `static/` only if needed | ✅ — `static/css/medium.css`, real need (page styling) |
| DAG never imports `models.py`/`database.py`/router/service | ✅ — both DAGs verified against this at write time; `dag_db.py` is the only DB touchpoint |
| Wired into `main.py`, `core/templating.py`, static mount ordering, sidebar | ✅ — sidebar entries pre-existed and needed `active_module` wiring, not new nav markup |
| Any AI/LLM calls route through `services/ai/` | N/A — no AI calls in this domain |
| Basic functional smoke test: Medium input → final output works end to end | ⚠️ **Not yet confirmed** — this is exactly what §7.1's checklist is for. Individual pieces are unit-tested; the full real-feed → DAG → DB → page pipeline has not been confirmed working end to end since the `MEDIUMTEXT` fix landed. |

Every ✅ above was true at the time each relevant file was written and
reviewed in this conversation, based on static inspection and the test
suites described in §3 — none of it was confirmed against a live
Airflow/MariaDB stack, since no such stack was ever reachable during this
build. The ⚠️ is the one item a reviewer should treat as blocking sign-off,
not the ✅s.
