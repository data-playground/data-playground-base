# WO#34 — Soccer Data Domain: Postmortem & Handoff

**Written:** 2026-10-02 · **Status addendum:** 2026-10-06 (owner-reported, see §0)
**State of the work order:** functionally delivered and in use, **not yet closed** (see §0).
**Audience:**
1. The **reviewing agent**, who decides whether WO#34 passes. Read §0, §1–§3 (what was asked vs. what changed), §6 (criteria and verification) and §7 (known issues).
2. The **implementing agent** who takes the follow-up work. Read §7, §8 (improvements) and §9 (post-migration cleanup).

**How to read the claims in this document.** Every statement is one of: *verified* (checked against a file, a query result, or the session record, and says so), *owner-reported* (Pedro ran it and told us), or **UNVERIFIED** (an inference — labelled). Nothing is "tested" unless the text says who tested it. The implementing session had no live database, Airflow, or network access, so every functional result below came from the owner's environment, not from the author's.

---

## 0. Status at a glance

| Area | State | Notes |
|---|---|---|
| Domain code (routers, templates, helpers, agent, DAG) | Delivered; owner-run | Shipped in `soccer_domain_wo34.zip` (18 files in 12 folders). Iterated against real DAG runs; defects found and fixed (§5). |
| Original acceptance criteria | Met with caveats | See §6.1. Two ⚠️ remain (WO#33 alignment, formal smoke test). |
| Migrations | 4 written (`s0cc3r_*`) | First three appear applied; **`s0cc3r_st4ts001` is not applied** — `/soccer` currently returns 500 (`Unknown column 'soccer_matches.home_penalty_score'`). |
| Alembic chain | **Open defect** | Owner ran `alembic upgrade head` and got `Cycle is detected in revisions (m3d1um_d0ma1n004, s0cc3r_t34m1ds001)`. The copies of these files in the delivered ZIP form a straight line, so the repo's copies differ. Cause **not determined** (K2). |
| Data | Clean slate decided | Owner chose to wipe all soccer data (keep schema) and re-pull. Wipe SQL was provided; **execution not confirmed**. |
| Owner-requested improvements | 3 logged, none built | §8: shootout goals in the tally (I1), shots definition (I2), formation placement templates (I3). |
| Post-migration cleanup | Not started | §9 — to be run only after every other in-flight migration has landed. |

### 0.1 Addendum — owner-reported updates (2026-10-06)

These were reported after the table above was written. They are **owner-reported, not re-verified by the author**; confirm with the commands in §6.2.

- **Migrations/chain (K2, K3):** the owner's database is stamped at head `expl0r3r_d0ma1ns001` (a later, explorer-domain revision this document never saw) with the soccer migrations applied **through `s0cc3r_st4ts001`**. K2 and K3 therefore appear resolved for the database; K2 stays open only as a **repo-hygiene** item (confirm the on-disk `down_revision` values match what is applied). Read every "intended single head `s0cc3r_st4ts001`" below as "`s0cc3r_st4ts001` is an ancestor of the single head".
- **I1:** the owner has confirmed that `Period == 11` on a `Goals` row denotes a **penalty-shootout kick**, and that the shootout result lives in `HomeTeamPenaltyScore`/`AwayTeamPenaltyScore`. The discriminator in I1 is no longer a hypothesis; what remains is confirming the counts on the real payload.
- **Working order:** the owner is taking the follow-ups in the order **K1 → I1 → K2 → §9 cleanup**.

**One thing to know before the next DAG run:** `ingest_match_details` currently fetches every scheduled fixture in the next 60 days, every day (K1). On the first run after the clean slate this is the largest single source of FIFA API calls. It is a one-line fix and should be decided before triggering the DAG.

---

## 1. The Work Order as issued

*(Source: the WO#34 text pasted at the start of the session. The copy under `migration_docs/` in the repo, if any, is authoritative.)*

### 1.1 Role, boundaries, method

- **Role:** full-stack engineer building a new domain from scratch, with latitude on schema/endpoints/templates, but bound by existing conventions (domain-folder structure, DAG/FastAPI import boundary, shared AI service layer).
- **Hard boundaries:**
  - New code lives under `domains/soccer/` (models, routers, templates, static — only what is used).
  - If ingestion is an Airflow DAG: it never imports `models.py`, `database.py`, or any router/service; DB access only through `airflow/dag_db.py` raw-SQL helpers.
  - Any LLM call goes through `services/ai/`.
  - Router files stay under GOVERNANCE §1.2's 300-line ceiling; split early by competition or concern.
  - Step 1 must complete, in full, before any code is written.
- **Method:** Step 1, wait for answers, then build with judgment; surface only hard-to-reverse decisions.
- **Output format:** (1) materials requested/received, (2) files created, (3) files edited, (4) design decisions, (5) acceptance results, (6) notes/open questions.

### 1.2 Step 1 — questions and the owner's answers (decisions of record)

| # | WO question | Owner's answer |
|---|---|---|
| 1 | Data source / access / existing code | Provided `fifa_data_pull.py`. It loads to BigQuery; **this build must load to the Life OS MariaDB instead**. "You can change anything else as needed." |
| 2 | Competition scope | Competitions are reachable by ID; examples in the script's lines 55–70. **World Cup (ID 17) is "a great test".** |
| 3 | Ingestion pattern | **DAG-scheduled, daily pull.** |
| 4 | Granularity | "I like all your examples" plus what the script already pulls. **Ideally store the API's raw data so many endpoints can reuse the same pattern.** |
| 5 | Dashboard/sidebar in scope? | **Future fast-follow** (not in this pass). |

### 1.3 Step 2 — build instructions (summary)

`domains/soccer/models.py` (Base from `core.base_model`); routers split by competition/concern; templates extending `base.html`; static only if needed; match NBA (WO#33) conventions where sensible; DAG(s) under `airflow/dags/soccer/` using `dag_db.py` only; wire up `main.py`, `core/templating.py`, any static mount (before `/static`), sidebar/dashboard if in scope.

### 1.4 Acceptance criteria as issued

1. Step 1 happened before code was written.
2. All new code lives under `domains/soccer/`.
3. If a DAG exists: it never imports `models.py`, `database.py`, or any router/service.
4. Any AI/LLM calls route through `services/ai/`.
5. Router(s) registered in `main.py`; templates path added; static mount ordered correctly if applicable.
6. Every router file under 300 lines.
7. Design decisions documented, including whether/how they aligned with WO#33's NBA conventions.
8. Basic functional smoke test passes for the competitions scoped into this pass.

---

## 2. What was delivered against the original WO (baseline)

This is the first delivery, as reported at the time — i.e. the state **before** the owner began running it and before any change in §3.

### 2.1 Files created

| File | Purpose at first delivery |
|---|---|
| `domains/soccer/__init__.py`, `routers/__init__.py` | Package markers |
| `domains/soccer/models.py` | `SoccerCompetition`, `SoccerMatch` (normalized, from the calendar endpoint only), `SoccerRawPayload` (verbatim FIFA JSON) |
| `domains/soccer/routers/soccer.py` | `/soccer` fixtures list + `/soccer/match/{id}` (108 lines) |
| `domains/soccer/templates/soccer.html`, `soccer_match_detail.html` | Browse pages from existing `base.css` primitives; the detail page rendered match details/events as **formatted raw JSON** |
| `airflow/agents/soccer_agents.py` | FIFA HTTP client ported from the owner's script (BigQuery code dropped) |
| `airflow/dags/soccer/life_os_soccer_ingest.py` | Daily DAG: seed watch list → ingest fixtures → fetch match detail/events (raw only) |

### 2.2 Files edited
Instructions only (neither file was on disk in the author's environment): `main.py` (register router) and `core/templating.py` (add `domains/soccer/templates` to the `ChoiceLoader`).

### 2.3 Design decisions made at first delivery

- **Raw-first storage.** Every FIFA response stored verbatim in `soccer_raw_payloads` (endpoint + competition + optional match id → JSON), so adding an endpoint needs no migration. Normalized tables were deliberately limited to what browsing needed.
- **FIFA IDs as strings** (the Euros ID is alphanumeric).
- **Watch list seeded in the DAG** from the competitions named in the WO's own interests list (World Cup, Champions League, Premier League, Brasileirão, Libertadores) — *the owner had not explicitly confirmed this set*.
- **"Field-name honesty."** The owner's script never showed a sample response body, so field names were guessed defensively (`.get()` everywhere) with the raw JSON always preserved. *(This guess was wrong in several places — see §5.)*
- **Fetch details once** (`details_fetched_at` set once, mirroring `streaming_fetched_at` in `media_agents.py`).
- **Single router file** (108 lines) since v1 was browse-only.

### 2.4 Acceptance criteria — result claimed at first delivery

| # | Criterion | Claimed | Caveat stated then |
|---|---|---|---|
| 1 | Step 1 before code | ✅ | — |
| 2 | Code under `domains/soccer/` | ✅ | (DAG and agent live under `airflow/` per Step 2.4 — see §6.1) |
| 3 | DAG import boundary | ✅ | — |
| 4 | AI via `services/ai/` | ✅ | No AI calls in this domain |
| 5 | Registered / templating / static | ☐ | Written as instructions, not applied |
| 6 | Router < 300 lines | ✅ | 108 lines |
| 7 | Design decisions documented | ✅ | WO#33 not available to align with |
| 8 | Smoke test | ⚠️ | Not run: no DB/Airflow/network in the author's environment |

### 2.5 Gaps declared at first delivery
No Alembic migration (no `alembic/` folder had been shared); match details/events raw-only; no WO#33 to align with; FIFA API is unauthenticated and unofficial.

---

## 3. Changes agreed after the initial migration

**Definition used here.** The "initial migration" is `s0cc3r_d0ma1n001` — the first migration written for this domain (it did not exist at first delivery; see §2.5). Everything below happened after the owner began running the code. Each change is classed:

- **A — Scope addition** requested by the owner (the original WO did not cover it).
- **B — Defect fix** (the first delivery was wrong or incomplete).
- **C — Design reversal** (a decision in §2.3 was deliberately replaced).
- **D — Proposed and declined/deferred** (explicitly *not* part of this WO).

### 3.1 Change log

| # | Change | Class | Where | Migration |
|---|---|---|---|---|
| C1 | **Settings page** (`/soccer/settings`): ingest window (past/future days), watch-list activate/deactivate/remove, live FIFA competition search (cached 6 h) with "+ Add". Removal blocked when a competition has ingested matches (deactivate instead). | A | `routers/soccer_settings.py`, `templates/soccer_settings.html`, `templates/partials/*`; `soccer_settings` table | `s0cc3r_d0ma1n001` |
| C2 | **Backfill date moved into the UI** (per competition, set at add time, inline-editable). DAG's hard-coded `_SEED_COMPETITIONS` **removed** — the migration is now the only seeder. "Needs backfill" changed from "competition has zero matches" to "`backfill_from_date` is earlier than the earliest match on file", re-checked every run. | A + C | DAG `_needs_backfill()`, settings UI | none beyond `d0ma1n001` (column ships there — reviewer: confirm with `SHOW CREATE TABLE soccer_competitions`) |
| C3 | **Fixtures list filters**: competition, status, since (default yesterday), until (default tomorrow); filters preserve each other; "Show full history" uses explicit sentinel dates; ascending by kickoff. | A | `routers/soccer.py`, `soccer.html` | — |
| C4 | **Sidebar highlight**: settings page passes `active_module="soccer_settings"` (was `"soccer"`). Sidebar entries themselves are outside the delivered file set. | B | `soccer_settings.py` | — |
| C5 | **MatchStatus map replaced** with the owner's authoritative table (`0` finished, `1` scheduled, `3` finished/penalties, `7` postponed, `9` finished/forfeit; unknown → `"unknown"`). First map was a guess and mislabeled matches, which also caused premature `details_fetched_at` locks. A surgical SQL cleanup repaired the locked rows. | B | `_MATCH_STATUS_MAP` | — |
| C6 | **Parser rewritten against real payloads**: calendar uses `Home`/`Away` (not `HomeTeam`/`AwayTeam`); names are locale lists (`_extract_localized_text`); `_scalar_or_none`; **per-match try/except** so one bad match no longer aborts a batch. | B | `soccer_agents.py`, DAG | — |
| C7 | **Match detail page rebuilt** from a raw-JSON dump into a rendered page: score header with crests, scorers/assists, formation + coach, pitch diagrams with goal/assist/card/sub badges, GK styling, "(C)" captain marker, substitution lists, possession bar, empty state. Layout chosen by the owner after three mockup directions. *(The first delivery had flagged this as a "real follow-up work order"; it was folded into WO#34 by agreement.)* | A | `soccer_match_detail.html`, `lineup_helpers.py`, router | — |
| C8 | **Normalized detail tables** `soccer_match_lineups`, `soccer_goals`, `soccer_bookings`, `soccer_substitutions`, `soccer_coaches`; new `soccer_matches` columns `home_formation`, `away_formation`, `possession_home`, `possession_away`, `attendance`; `parse_match_lineup_data()`; DAG task 3 `parse_finished_match_details`. | A | models, agent, DAG | `s0cc3r_l1n3ups001` |
| C9 | **Team IDs** `fifa_home_team_id` / `fifa_away_team_id` on `soccer_matches`, needed to build crest image URLs. | A | models, parser, `_upsert_match` | `s0cc3r_t34m1ds001` |
| C10 | **Detail-fetch policy — reversed twice.** (1) First: fetch once, lock forever. (2) Then: locked only when the match is *currently* confirmed finished; non-finished matches refetched within ±1 day of kickoff. (3) **Final (owner direction):** default to the **daily ingestion window**; everything in the window is refetched and **fully overwritten** every run; outside the window a match is fetched/parsed only if it has never been. | C | DAG tasks 2 & 3 | — |
| C11 | **Raw-payload pruning**: new task 4 `prune_raw_payloads` keeps the 3 most recent snapshots per (endpoint, competition, match) using `ROW_NUMBER()`. Owner asked for this after seeing `soccer_raw_payloads` grow with every reprocess. | A | DAG | — |
| C12 | **Event-derived stats** (shots, corners, fouls, offsides per side, counted from `/timelines` Type codes 12/16/18/15) and **penalty-shootout score** columns (`home_penalty_score`, `away_penalty_score` from `/live`). | A | `parse_match_event_stats()`, models, template | `s0cc3r_st4ts001` |
| C13 | **Formation row splitting**: use the `Tactics` string (e.g. `3-4-2-1`) to split FIFA's undifferentiated `Position: 2` group into the right number of rows. | B | `lineup_helpers.py` | — |
| C14 | **`alembic/env.py` fix** `from models import Base` → `from core.base_model import Base` (root `models.py` was removed in WO#22). **Pre-existing bug outside WO scope**; fixed so migrations could run. Per GOVERNANCE §4.5 it should have been its own ticket. | B (out of scope) | `alembic/env.py` | — |
| C15 | **Clean-slate decision**: wipe all data in all soccer tables (schema kept) and re-pull, because several policy/parser changes left existing rows inconsistent. | Operational | — | — |

### 3.2 Proposed and declined / deferred (D) — explicitly **not** in this WO

| Item | Disposition |
|---|---|
| Manual "Run backfill" button per competition, "backfill for all" toggle, and a separate on-demand backfill DAG | **Declined by owner** ("we have a fair strategy of the backfill here already… no need to add more complexity"). Backfill remains: edit `backfill_from_date` and let the daily run close the gap. |
| Dashboard / sidebar integration (`routers/dashboard.py`) | Deferred (owner: future fast-follow). Not built. |
| AI-generated match recap | Placeholder only; not built. |
| Team kit colours / jersey-shaped player tokens | Deferred (needs colour data). |
| Standings/tables endpoints | Not built (the raw-first design makes this cheap later). |
| xG, big chances, passes, duels, saves counts | Not available as confirmed FIFA fields; early mockup numbers came from a reference screenshot of another site, used for layout only. |
| Shots **on target** | Not built (see I2 — now believed derivable). |

### 3.3 Scope statement for the reviewer
**WO#34 as delivered = the original WO (§1) + changes C1–C13.** C14 is an out-of-scope fix that should be ticketed separately; C15 is an operational step; §3.2 items are not part of this WO and must not be held against it.

---

## 4. Final state of the system

### 4.1 Data flow

```
FIFA public API (no key)
  calendar ──► ingest_fixtures ──► soccer_matches (upsert) + soccer_raw_payloads('matches')
  /live, /timelines ──► ingest_match_details ──► soccer_raw_payloads('match_details','match_events')
  soccer_raw_payloads ──► parse_finished_match_details ──► lineups/goals/bookings/subs/coaches
                                                          + scalar columns on soccer_matches
  prune_raw_payloads ──► keeps newest 3 snapshots per (endpoint, competition, match)
FastAPI: /soccer, /soccer/match/{id}, /soccer/settings  (read from the normalized tables)
```

### 4.2 Tables (9)

| Table | Role |
|---|---|
| `soccer_competitions` | Watch list (FIFA competition id, name, active flag, `backfill_from_date`). Seeded with 5 competitions by `s0cc3r_d0ma1n001`; edited via `/soccer/settings`. |
| `soccer_settings` | Single row: `window_past_days` / `window_future_days` (defaults 3 / 60). |
| `soccer_matches` | One row per match; identity = (competition, season, stage, match) FIFA ids. Scores/status from the calendar; formation/possession/attendance/penalty scores/shots/corners/fouls/offsides filled by task 3. |
| `soccer_raw_payloads` | Verbatim FIFA responses (`endpoint`, ids, `payload`, `fetched_at`). |
| `soccer_match_lineups`, `soccer_goals`, `soccer_bookings`, `soccer_substitutions`, `soccer_coaches` | Parsed per-match detail; fully replaced on each reprocess. |

### 4.3 DAG `life_os_soccer_ingest` (`@daily`)

| Task | Does | API calls? | Scope rule |
|---|---|---|---|
| `ingest_fixtures` | Upsert fixtures per active competition; window from `soccer_settings`; widens to `backfill_from_date` when a gap exists | Yes (calendar, paginated) | Rolling window or backfill gap |
| `ingest_match_details` | Fetch `/live` + `/timelines`, store raw | Yes (2/match) | kickoff in window **or** no `match_details` raw row yet — **see K1** |
| `parse_finished_match_details` | Parse latest raw payloads; `DELETE`+`INSERT` the five detail tables in one transaction per match; update scalar columns | No | kickoff in window **or** no lineup rows yet |
| `prune_raw_payloads` | Retention (`RAW_PAYLOAD_RETENTION_COUNT = 3`) | No | all rows |

Design intent: a *fresh* fetch is the only cure for a stale snapshot, so fetching stays bounded to the window; re-parsing is free, so it is not.

### 4.4 Web layer
`/soccer` (list, filters), `/soccer/match/{id}` (detail), `/soccer/settings` (+ HTMX partials for watch list and FIFA search). Router sizes (verified `wc -l`): `soccer.py` 206, `soccer_settings.py` **261** (past GOVERNANCE §1.2's 250-line "split soon" mark). `lineup_helpers.py` holds pitch-token math and `build_crest_url()`.

### 4.5 Migration chain

Soccer migrations, as they exist in the delivered ZIP:

| Revision | down_revision (in ZIP) | Adds | Applied? |
|---|---|---|---|
| `s0cc3r_d0ma1n001` | `j0b_sc0ut_upgr4de001` | 4 base tables; seeds 5 competitions and default settings | Yes (owner-run) |
| `s0cc3r_l1n3ups001` | `m3d1um_d0ma1n002` | 5 detail tables; formation/possession/attendance columns | Yes — **UNVERIFIED**, inferred from the owner querying `soccer_goals` |
| `s0cc3r_t34m1ds001` | `m3d1um_d0ma1n003` | `fifa_home_team_id`, `fifa_away_team_id` | Yes — **UNVERIFIED**, inferred: in the 500 error's SELECT list these columns precede the first unknown column |
| `s0cc3r_st4ts001` | `m3d1um_d0ma1n004` | penalty scores; shots/corners/fouls/offsides (10 columns) | **No** (500 error) |

Chain as last understood (other domains' revisions are not owned by this WO):

```
j0b_sc0ut_upgr4de001 → s0cc3r_d0ma1n001 → nb4_d0ma1n001 → [m3d1um_d0ma1n001, not seen by the author]
  → m3d1um_d0ma1n002 → s0cc3r_l1n3ups001 → m3d1um_d0ma1n003 → s0cc3r_t34m1ds001
  → m3d1um_d0ma1n004 → s0cc3r_st4ts001   (intended single head)
```

**This chain drifted four times** while soccer migrations sat unapplied, because NBA and Medium migrations landed in parallel. Each soccer migration's `down_revision` was rebased by hand. See K2 for the open cycle error.

### 4.6 Manual wiring that lives outside the ZIP
`main.py` (import and `include_router` for **both** `soccer` and `soccer_settings`), `core/templating.py` (`FileSystemLoader("domains/soccer/templates")`), `templates/partials/sidebar.html` (Soccer entry + Settings sub-nav, `active_module` values `soccer` / `soccer_settings`). Owner-reported working at the time of the last screenshots; re-verify in §6.2.

---

## 5. Incidents and lessons

| Incident | Root cause | Resolution | Lesson |
|---|---|---|---|
| Parser read nothing/`NULL` teams | Field names guessed (`HomeTeam`/`AwayTeam` vs real `Home`/`Away`; names are locale lists) | Parser rewritten against real captured payloads | Defensive `.get()` hides a wrong field name — it doesn't catch it. Validate against a real payload first. |
| Wrong match statuses; matches locked on empty snapshots | Guessed status-code map; `details_fetched_at` set eagerly | Owner's authoritative map; lock only when *currently* finished; targeted SQL repair | Enums need authoritative sourcing; locks must be state-verified at write time. |
| One malformed match aborted a whole run | No per-item isolation | Per-match try/except | Batch ingestion needs per-record error isolation. |
| Alembic head drift (×4) and later a reported cycle | Parallel domain work landing between sessions | Manual rebases; cycle unresolved | Treat `alembic heads` as the first command of any migration session. See §9.1. |
| Alembic could not run | `env.py` imported the deleted root `models.py` | One-line import fix | Pre-existing; should have been its own ticket (GOVERNANCE §4.5). |
| **Stale/incomplete snapshots frozen forever** — seen as a `match_events` payload going `[]`, `[]`, populated across three pulls, and a match whose goal rows were captured before its shootout finished | "Fetch once, lock when finished" | Window-based full overwrite (C10) | Never trust a single snapshot as final; keep refreshing until it ages out of the window. |
| **Misdiagnosis of the LDU–Palmeiras goals** | Two successive guesses (extra time; then an unfinished shootout contradicted by status code `0`) were stated as fact before verification. Compounding it, the payload pasted as "raw payload data for issue 1 (LDU/Palmeiras game)" was actually a **Premier League fixture involving Crystal Palace** (`IdCompetition 2000000000`), not the Libertadores match. | Resolved only when the owner, who watched the match, confirmed the extra goal rows were **penalty-shootout kicks** | State inferences as inferences. Note also that status code `0` ("Completed") was observed on a shootout-decided match, so status `3` cannot be relied on to detect shootouts — use the penalty-score fields. The real LDU–Palmeiras payload had **not** been reviewed at that point. |
| `/soccer` returns 500 after deploy | New `models.py` deployed (web container bind-mounts the repo) before `alembic upgrade head` added its columns | Apply the migration | Deploy order: migrate first, then ship models that need the columns. |

---

## 6. Reviewer guide

### 6.1 Acceptance criteria — final assessment

| # | Criterion | Final | Basis |
|---|---|---|---|
| 1 | Step 1 before code | ✅ | Session record: Step 1 acknowledged and answered before any file was written. |
| 2 | Code under `domains/soccer/` | ⚠️ read with Step 2.4 | Code lives in four places by the WO's own instruction: `domains/soccer/` (models, routers, templates, `lineup_helpers.py`), `airflow/dags/soccer/` (Step 2.4), `airflow/agents/soccer_agents.py` (follows `job_ats_agents.py` precedent), and `alembic/versions/` (4 files). |
| 3 | DAG import boundary | ✅ verified | DAG imports only `dag_db` and `agents.soccer_agents` plus stdlib/Airflow. `soccer_agents.py` has no ORM/router/service imports. (The FastAPI-side `soccer_settings.py` importing `airflow.agents.soccer_agents.fetch_competitions` has precedent in `recipe_service.py`.) |
| 4 | AI via `services/ai/` | ✅ | No AI calls in this domain. |
| 5 | Registration/templating/static | ✅ owner-reported | `/soccer/settings` loaded after registration. No static mount needed (no static assets). Re-verify per §6.2. |
| 6 | Routers < 300 lines | ✅ verified | 206 and 261. Note 261 is past the 250 "split soon" mark. |
| 7 | Design decisions / WO#33 alignment | ⚠️ | Decisions documented here and in §2.3/§3. **WO#33 alignment was never performed**: NBA's design wasn't available, and an NBA migration (`nb4_d0ma1n001`) now exists in the repo. A diff pass on ID format, table naming and season/date handling is still owed. |
| 8 | Smoke test | ⚠️ | Owner-run evidence: pages render, World Cup test ingested, detail data displayed, defects found and fixed. No formal smoke-test record exists, and the app is currently in a broken state until `s0cc3r_st4ts001` is applied. Run §6.2 to close this. |

### 6.2 Verification checklist (run in order)

Run from `internal_dataplayground/` on the host; DB commands inside the container.

1. **Chain.** `docker compose exec web alembic heads` → exactly one head (owner reports `expl0r3r_d0ma1ns001`); `s0cc3r_st4ts001` must be an ancestor of it. `alembic history | head -15` shows the §4.5 order (allowing for the unseen `m3d1um_d0ma1n001`). If a cycle or multiple heads appear, stop and resolve K2 first.
2. **Schema.** After `alembic upgrade head`: `SHOW COLUMNS FROM soccer_matches` includes `home_penalty_score`, `away_penalty_score`, `shots_*`, `corners_*`, `fouls_*`, `offsides_*`, `fifa_home_team_id`, `fifa_away_team_id`. `/soccer` loads (no 500).
3. **Import boundary.** `grep -nE "^(from|import) " airflow/dags/soccer/*.py airflow/agents/soccer_agents.py` shows no `models`, `database`, `routers`, `services`.
4. **Router size.** `wc -l domains/soccer/routers/*.py` all < 300.
5. **Wiring.** `grep -n "soccer" main.py core/templating.py` shows both routers included and the templates loader entry.
6. **First DAG run** (after the clean slate, with backfill dates narrowed — see K1/§7): all four tasks succeed. Then
   ```sql
   SELECT 'matches', COUNT(*) FROM soccer_matches
   UNION ALL SELECT 'raw', COUNT(*) FROM soccer_raw_payloads
   UNION ALL SELECT 'lineups', COUNT(*) FROM soccer_match_lineups
   UNION ALL SELECT 'goals', COUNT(*) FROM soccer_goals;
   ```
   and confirm `soccer_raw_payloads` has ≤ 3 rows per (endpoint, competition, match) after task 4.
7. **UI.** Open a finished match: lineups, formation strings, scorers, subs, possession, and the four count stats render; a match whose events payload is empty shows no fabricated zeros (see K4).
8. **Reference match.** LDU vs Palmeiras — look it up **by team names and date**, not by id (see I1 on the id discrepancy).

---

## 7. Known issues at close

Severity: **H** = fix before the next DAG run or before closing; **M** = fix soon; **L** = backlog.

| ID | Sev | Issue | Suggested action |
|---|---|---|---|
| K1 | **H** | **`ingest_match_details` window is symmetric.** It selects every match with `kickoff_at BETWEEN today−past_days AND today+future_days` (defaults 3 / 60), i.e. every scheduled fixture for the next 60 days — two FIFA calls and two raw rows per fixture **per day**, for payloads that cannot yet contain lineups or events. The earlier design bounded future fetches to ±1 day of kickoff. | Give the details fetch its own short lookahead (e.g. `DETAIL_LOOKAHEAD_DAYS = 1`): `kickoff_at BETWEEN window_start AND today + lookahead`. Apply to task 2 only; leave `ingest_fixtures` on the full window. |
| K2 | **H** | **Alembic cycle error** (`m3d1um_d0ma1n004` ↔ `s0cc3r_t34m1ds001`) in the owner's repo, while the delivered ZIP's copies are linear. Most likely the repo's `s0cc3r_t34m1ds001` (or `m3d1um_d0ma1n003/004`) carries a different `down_revision`. **Cause unconfirmed.** | Run `grep -n "^revision\|^down_revision" alembic/versions/s0cc3r_*.py alembic/versions/m3d1um_*.py alembic/versions/nb4_*.py`; set the three soccer `down_revision` values per §4.5 (use `sed` patterns that skip docstring lines — match `^down_revision: `); confirm one head. |
| K3 | **H** | `s0cc3r_st4ts001` unapplied → `/soccer` 500. | Resolve K2, then `alembic upgrade head`. |
| K4 | M | `parse_match_event_stats()` returns **zeros** (not `NULL`) for an empty `/timelines` payload, so a match with no timeline data shows "0 – 0" shots/corners/fouls/offsides. `[]` payloads were observed for days on a real match. | Return `None` for every stat when the event list is empty; template already hides stats that are `None`. |
| K5 | M | Shootout goals are counted as goals in scorer lines and pitch badges (I1). | See I1. |
| K6 | M | "Shots" undercounts (I2). | See I2. |
| K7 | M | Crest URL `…/picture/teams-sq-4/{IdTeam}` is **UNVERIFIED**; owner screenshots showed fallback initials (consistent with either IDs not yet populated at the time or the image failing). | Open one crest URL in a browser; if it fails, find the real format or proxy images through the backend. |
| K8 | M | `soccer_settings.py` is 261 lines (past the 250 "split soon" mark). | Split (e.g. watch-list CRUD vs. FIFA search) before adding features. |
| K9 | L | Manager shown only for `Role == 0`; that code was confirmed on two matches only. No fallback if a team lists no `Role 0` coach. | Fallback to the first listed coach; verify on more matches. |
| K10 | L | Card badge is binary (`has_card`): a second-yellow/red (`Card: 3`, observed in the Crystal Palace data) renders like a plain yellow. Plain red code not yet observed. | Render by card code; confirm red code on a real example. |
| K11 | L | Task 3's "no lineup rows yet" guard re-parses, every run, any match whose payload legitimately has no lineup (low-coverage competitions, far-future fixtures). Cheap (local) but unbounded. | Add a `details_parsed_at` marker or skip matches with an empty `Players` array. |
| K12 | L | Scores come only from the calendar endpoint; `/live`'s own `HomeTeam.Score`/`AwayTeam.Score` (confirmed present) are unused and could disagree after extra time/penalties. | Decide a single source of truth; compare on a shootout match. |
| K13 | L | No "live" status exists: the in-progress code was never captured, so a live match cannot be labelled. | Capture a payload during a live match. |
| K14 | L | Task `parse_finished_match_details` is a misnomer (it now processes any in-window match). | Rename only with care — Airflow history is keyed on `task_id`. |
| K15 | L | **Model/migration nullability drift** on four timestamp columns: `models.py` lines 83, 195, 337, 370 omit `nullable=False`; `s0cc3r_d0ma1n001` declares them NOT NULL. | Align the models to the migration (see §9.2). |
| K16 | L | **Wrong comment**: `soccer_agents.py` (~line 570) says keeper saves and outfield blocks are both Type 17 and distinguishable only by text. The shared Bournemouth–Brentford payload shows them as **different Type codes** (8 events of Type 57 — "The goalkeeper of Brentford pulls off a save" — and 3 of Type 17 — "…blocks the shot"). | Correct the comment when I2 is done (§9.8). |
| K17 | L | No automated tests. | Add pure-function tests with fixtures from real payloads (see §9.12). |

---

## 8. Improvement backlog

### I1 — Penalty-shootout goals must not appear in the goal tally *(owner request)*

**Problem.** The owner confirmed (having watched the match) that extra goal rows on LDU vs Palmeiras are penalty-shootout kicks. The penalty **score** is correct, but the individual kicks currently count as goals in (a) the scorer lines under each team in the header, (b) the ⚽ badges/counts on pitch tokens, and (c) anything else derived from `soccer_goals`.

**Evidence.**
- During the investigation `soccer_goals` rows for `match_id = 3059` included row `id 8110` (home, `92'`, `period: 11`) and `id 8113` (away, `91'`); the analysis identified two `period: 11` rows. Other goals on that match carried `period: 3` or `5`.
- **Id discrepancy:** the owner now cites **match id 3097** for LDU vs Palmeiras; the investigation used **3059**. Surrogate ids may differ after the clean-slate reload. Resolve the match by team names + date; don't hard-code either.
- Only one `period: 11` row per side was stored. A completed shootout normally has more kicks, which fits the owner's suspicion that the snapshot was taken mid-shootout. **Re-fetch under the new policy before drawing conclusions about counts.**
- The raw LDU–Palmeiras payload has **never been inspected in full** (the one pasted was the Crystal Palace fixture above). The `period: 11 = shootout` link was first inferred from two rows; the owner has since **confirmed** it (§0.1). Counts on the real payload are still unchecked.

**Required behaviour.**
1. Header scorer lines list only regulation and extra-time goals.
2. Pitch badges and assist counts ignore shootout kicks.
3. The shootout appears only as the `(x – y pens)` line from `home_penalty_score` / `away_penalty_score`.
4. Per side, the number of listed goals equals the displayed score (calendar `HomeTeamScore`/`AwayTeamScore`).

**Remaining verification.** The owner has confirmed `Period == 11` marks a shootout kick (§0.1). On the real payload for this match and at least one other shootout match, check that `HomeTeamPenaltyScore`/`AwayTeamPenaltyScore` equal the number of period-11 goals, and whether goal `Type` values (observed `1` and `2`; meaning unknown) separate in-play penalties or own goals. **Do not treat `Type` as meaningful until confirmed.**

**Implementation options.** (A) Skip period-11 goals in `parse_match_lineup_data()` via a named, commented constant (`_SHOOTOUT_PERIOD_CODE`). Simplest; raw payloads keep the originals. (B) Store them with an `is_shootout` flag (needs a migration) and filter at render — choose only if shootout kicks should ever be displayed. **Recommendation: A**, unless the owner wants a shootout timeline.

**Acceptance.** For LDU vs Palmeiras after reload: per-side listed goals match the regulation/extra-time score, the pens line matches FIFA's penalty score, and no shootout kick contributes a badge. Add a parser unit test using the real payload.

### I2 — Shots count does not include goals or goal preventions *(owner request)*

**Problem.** `shots_home/away` is `COUNT(Type 12 "Attempt at Goal")` per team. Goals and keeper saves/blocks are *different event types*, so the figure appears low.

**Evidence from the real Bournemouth–Brentford timeline** (shared during the session; verified by counting the raw payload):
- Goals are `Type 0` (not part of Type 12).
- Keeper saves are `Type 57`, attributed to the **defending** team — 8 such events, all Brentford, sample text "The goalkeeper of Brentford pulls off a save."
- Outfield blocks are `Type 17` (3 events, e.g. "JANELT (Brentford) blocks the shot.").
- The reference figures used for the early mockups (from another site; layout reference only) were **shots on goal 10 vs 2** and **saves 0 vs 8**. They are consistent with: Bournemouth on target = 8 Brentford saves + 2 goals = **10**; Brentford on target = 0 Bournemouth saves + 2 goals = **2**. One match, one external source — **suggestive, not proof.**

**To resolve before building.**
1. Does each `Type 57` event point (qualifier `233`, "OppositeRelatedEventId") at a `Type 12` event already in the payload? If **yes**, saved shots are already inside Type 12 and only goals are missing from "total shots"; if **no**, saves must be added too. Check this first — it decides double-counting.
2. Same question for `Type 17` blocks.
3. Define the stats explicitly:
   - **Shots on target** = goals (Type 0, excluding own goals and shootout kicks) + opposing keeper saves (Type 57).
   - **Total shots** = per the result of (1)/(2).
4. Decide naming and schema: add `shots_on_target_home/away` (migration) and keep or rename `shots_*`. Label what the UI shows ("Shots" vs "Shots on target").

**Acceptance.** On Bournemouth–Brentford the derived shots on target equal 10 / 2 (matching the reference), on at least one more match the definition holds against an independent source, the definition is documented in the parser docstring, and K16's wrong comment is corrected.

### I3 — Predefined player placements by tactic *(owner request)*

**Problem.** `build_pitch_tokens()` spaces every row evenly across the pitch width, so tokens bundle and labels collide (e.g. five players in one row). The row-splitting fix (C13) restores the *number* of rows but not realistic *spread*.

**Constraints, from real data.**
- FIFA gives only coarse `Position` codes (`0` GK, `1` DEF, `2` mid/wide/wing-back catch-all, `3` FWD, `6` DM) — `LineupX/LineupY`-style coordinates were not present in any observed payload.
- Crystal Palace's 3-4-2-1 had **six** players at `Position: 2` and none at `6`, so which player belongs to the "4" band versus the "2" band **is not in the data**. Any slot assignment is a heuristic (the current code falls back to the API's `Players` array order — also unverified as a convention).

**Proposal.**
1. Add a lookup of slot templates keyed by formation string, each giving per-row `top%` and an explicit `left%` list (not evenly spaced). Cover the formations seen so far — 4-2-3-1, 3-4-2-1, 3-4-1-2, 3-5-2 — plus common ones: 4-3-3, 4-4-2, 4-1-4-1, 5-3-2, 3-4-3. Back-3 and back-5 rows should be narrower/wider than a back-4; wing-back and wide-mid slots should hug the touchlines.
2. Fall back to the current generic split for any formation not in the table.
3. Enforce a minimum horizontal gap between name labels; prefer a short name (the payload carries `ShortName`; the schema stores only the full `player_name`) to avoid overlap.
4. Keep the slot-assignment heuristic isolated in one function so it can be improved if better data appears.

**Acceptance.** For each formation in the table, no two name labels overlap at the page's standard width; wide slots are visibly wider than central ones; unknown formations still render via the fallback; `build_pitch_tokens()` has unit tests per template.

### Additional recommendations (not owner-requested)
- **K1 fix** — shorten the details lookahead (do this first).
- **K4, K9, K10, K11, K12** — see §7.
- **Crest verification** (K7).
- **Reconcile with WO#33** — one diff pass against the NBA domain.

---

## 9. Post-migration cleanup

**Run this section only after every other in-flight migration (NBA, Medium, anything else) has landed and `alembic heads` prints a single head.** Pause the soccer DAG while working. Make each item its own small, reviewable change (GOVERNANCE §4.5).

### 9.1 Alembic chain hygiene
- Confirm the four soccer migrations' `down_revision` values against the final chain (§4.5). **Do not hand-edit migrations that have been applied** — change docstrings only, and only if factual errors remain.
- `s0cc3r_d0ma1n001`'s docstring still describes a DAG seeding step that was later removed (C2). Either leave it and note the discrepancy, or correct the docstring text (no code change).
- Record the final revision ids in GOVERNANCE (§9.9).
- Do **not** squash applied migrations.

### 9.2 `domains/soccer/models.py`
1. **Remove references that only make sense to someone who had this conversation** and replace them with durable pointers (e.g. "see `migration_docs/Work Orders/work_order_34_soccer_domain_postmortem.md` §I2"):
   - line 147 — "mockups, which came from a reference ESPN screenshot…"
   - lines 159–161 — "this conversation initially misdiagnosed… the project owner confirmed…"
   - any other "WO#34 conversation" wording.
2. **Rewrite comments to describe present behaviour**, not history: `details_fetched_at` is informational ("last refreshed"), not a gate; penalty scores hold the shootout result; event stats follow whatever definition I2 settles on.
3. **Fix nullability drift (K15):** add `nullable=False` to `created_at` (line 83), `updated_at` (line 195), `fetched_at` (line 337), `updated_at` (line 370) so models match the applied migration. Align models to the migration, not the reverse.
4. Confirm `from core.base_model import Base`; no cross-domain `models` imports; remove unused imports.
5. Validate with the autogenerate dry run in §9.4 (expect an empty diff).

### 9.3 Other files with the same kind of reference

| File | Lines (at time of writing) | Action |
|---|---|---|
| `domains/soccer/routers/soccer.py` | 22–23 | Replace "WO#34 mockups… ESPN screenshot" with a statement of which stats are real. |
| `domains/soccer/templates/soccer_match_detail.html` | 18, 229 | Line 229 is **user-visible** text ("not the earlier placeholder mockup numbers") — remove it; the "stats appear once parsed" note should say only that. Line 18 is a template comment — rewrite. Also remove any "Future phase: AI-generated recap" placeholder shown to users. |
| `airflow/dags/soccer/life_os_soccer_ingest.py` | 28, 31, 510 | Replace "this session / the project owner" with a postmortem pointer. |
| `airflow/agents/soccer_agents.py` | 37, 241, 551 | Same; keep the *facts* (status map provenance) but drop conversational framing. |
| `alembic/versions/s0cc3r_l1n3ups001_*.py`, `…st4ts001_*.py` | 14, 27 / 24, 26 | Docstring-only edits; allowed on applied migrations. |

Re-run: `grep -rniE "conversation|mockup|ESPN|project owner|the owner|this session|screenshot" domains/soccer airflow/dags/soccer airflow/agents/soccer_agents.py alembic/versions/s0cc3r_*.py` — expect no hits.

### 9.4 Mapper registration and `alembic/env.py`
- `env.py` imports only `core.base_model.Base`; soccer models reach `Base.metadata` only transitively through the routers imported by `main.py`. **If `env.py` doesn't import the domain models, `alembic revision --autogenerate` will see an empty metadata and propose dropping every table. Do not apply such a revision.**
- GOVERNANCE §2.4 says the mapper-registration guarantee lives in an explicit import block in `database.py`; the copy of `database.py` reviewed during this WO had none. Reconcile: either add the import block (or a `core/model_registry.py`) and have `env.py` import it, or correct GOVERNANCE.
- Verify: generate an autogenerate revision into a scratch copy, confirm `upgrade()` is empty, then delete it.

### 9.5 Legacy shim check
This domain was built after root `models.py` was removed (WO#22); **no soccer shim should exist.** Verify: `grep -rn "from models import" --include=*.py .` returns nothing referencing soccer classes. `routers/dashboard.py` does not import soccer (integration is deferred).

### 9.6 DAG file
- Decide the fate of `details_fetched_at` (keep as "last refreshed" or drop via migration). If dropped, remove the `CASE WHEN … 'finished'` clause in `_upsert_match` and update the model.
- Task name `parse_finished_match_details` (K14): rename only with an explicit note about Airflow history.
- Apply the K1 lookahead change and the K11 guard if accepted.
- Re-read the module docstring after I1/I2 land so it describes the final design.

### 9.7 Templates
Remove placeholder/mockup language (§9.3). After I3, re-check label collisions at the standard page width and at the mobile breakpoint in `static/css/mobile.css`.

### 9.8 `airflow/agents/soccer_agents.py`
- Correct the Type 17/57 comment (K16) and, after I2, document the exact event definitions (Types 0, 12, 15, 16, 17, 18, 57) in `parse_match_event_stats()`.
- Add `_SHOOTOUT_PERIOD_CODE` (I1) with its provenance and the status of verification.
- Trim the long "field-name caveat" narrative to a short factual reference (Appendix A).

### 9.9 GOVERNANCE.md amendments
- §3.3 (migration debt tracker): soccer was built greenfield in the domain structure — not migration debt.
- §2.5 (DAG organization): list `airflow/dags/soccer/`.
- Add standing rules that this WO taught: (a) validate parser field names against a real captured payload before writing the parser; (b) status/enum maps need an authoritative source; (c) any "fetch once" lock must be state-verified, and prefer window-based refresh over locks; (d) run `alembic heads` first in any migration session, and migrate before shipping models that need new columns; (e) keep a bounded-history policy for raw-payload tables from day one.

### 9.10 Artifacts not meant for the repo
The early mockup HTML files and `soccer_domain_wo34.zip` are delivery artifacts. Confirm none were committed.

### 9.11 Ticket the out-of-scope fixes separately
C14 (`alembic/env.py` import) belongs in its own ticket/commit.

### 9.12 Tests (recommended)
Pure-function tests with fixtures cut from real payloads: `parse_match_lineup_data()` (incl. shootout handling per I1), `parse_match_event_stats()` (incl. the empty-list case, K4), `_parse_tactics()`/`_split_am_rows()`/`build_pitch_tokens()` (incl. the Crystal Palace 3-4-2-1 case: six `Position: 2` players, no DM), `_extract_localized_text()`, and the status map.

### 9.13 Final regression
Repeat §6.2 end to end, then re-open the LDU–Palmeiras match and a 4-2-3-1 and a 3-4-2-1 match and confirm: header scorers, badges, penalty line, stats, and pitch spread.

---

## Appendix A — FIFA payload reference

**Endpoints** (from `soccer_agents.py`; unauthenticated, browser-style headers)
- Calendar: `https://api.fifa.com/api/v3/calendar/matches` (paginated)
- Competitions list: `https://cxm-api.fifa.com/fifaplusweb/api/sections/matches/…`
- Live detail: `https://api.fifa.com/api/v3/live/football/{competition}/{season}/{stage}/{match}?language=en`
- Timelines: `https://api.fifa.com/api/v3/timelines/{competition}/{season}/{stage}/{match}?language=en`
- Crest (**UNVERIFIED**): `https://api.fifa.com/api/v3/picture/teams-sq-4/{IdTeam}`

**Competition ids seeded:** World Cup `17`; Champions League `2000001032`; Premier League `2000000000`; Brasileirão `2000000078`; Libertadores `2000001035`. (Club World Cup was added later through the settings UI.)

**MatchStatus (owner-authoritative):** `0` Completed, `1` Scheduled, `3` Decided on Penalties, `7` Postponed, `9` Forfeited/Suspended. Stored labels: `finished` (0, 3, 9), `scheduled` (1), `postponed` (7), `unknown` (anything else). **Caveat:** a shootout-decided match was observed with code `0`.

**Calendar fields:** `Home`/`Away` (with `TeamName` locale list, `IdTeam`), `HomeTeamScore`/`AwayTeamScore`, `Stadium.Name` (locale list), `Date`, `MatchStatus`, `IdCompetition`/`IdSeason`/`IdStage`/`IdMatch`.

**`/live` fields:** `HomeTeam`/`AwayTeam` each with `Players[]`, `Goals[]`, `Bookings[]`, `Substitutions[]`, `Coaches[]`, `Tactics` (formation string, counts 10 outfield players), `Score`; `BallPossession.OverallHome/OverallAway`; `Attendance`; `HomeTeamPenaltyScore`/`AwayTeamPenaltyScore`.

**Player fields:** `IdPlayer`, `ShirtNumber`, `Status` (`1` starter, `2` bench), `Captain`, `PlayerName` (locale list), `ShortName`, `Position`, `FieldStatus` (`1` on pitch at the end).
**Position codes:** `0` GK, `1` DEF, `2` mid/wide/wing-back (catch-all), `3` FWD, `6` DM.

**Period codes seen:** `3` first half, `5` second half, `10` match end (match-level and end-marker event), `11` on two goal rows from a shootout match (**working hypothesis: shootout**). Extra-time codes: **not observed — do not assume.**

**Card codes seen:** `1` yellow; `3` second yellow/red (Crystal Palace data). Plain red code: not yet observed.
**Coach `Role`:** `0` head coach (confirmed on two matches only).

**`/timelines` event Types** (all from the Bournemouth–Brentford and other shared payloads):

| Type | Meaning | Used for |
|---|---|---|
| 0 | Goal | I2 |
| 2 | Yellow card | — (bookings come from `/live`) |
| 5 | Substitution | — |
| 7/8/9/10 | Start/End/Pause/Resume time | — |
| 12 | Attempt at Goal | shots |
| 15 | Offside | offsides |
| 16 | Corner | corners |
| 17 | Goal Prevention — outfield block | I2 |
| 18 | Foul | fouls |
| 26 | Match end | — |
| 57 | Goal Prevention — keeper save | I2 |

Qualifier `233` ("OppositeRelatedEventId") links an event to its counterpart; see I2.

## Appendix B — Delivered file inventory (18 files in 12 folders, `soccer_domain_wo34.zip`)

Domain: `domains/soccer/{__init__,models,lineup_helpers}.py`; `routers/{__init__,soccer,soccer_settings}.py`; `templates/{soccer,soccer_match_detail,soccer_settings}.html`; `templates/partials/{watchlist_rows,competition_search_results}.html`.
Airflow: `airflow/agents/soccer_agents.py`; `airflow/dags/soccer/life_os_soccer_ingest.py`.
Alembic: `s0cc3r_d0ma1n001_add_soccer_domain_tables.py`, `s0cc3r_l1n3ups001_add_lineup_tables.py`, `s0cc3r_t34m1ds001_add_team_ids.py`, `s0cc3r_st4ts001_add_stats_and_penalties.py`; plus the `alembic/env.py` fix (C14).
**Not in the ZIP and applied by hand:** `main.py`, `core/templating.py`, `templates/partials/sidebar.html`. This postmortem was written after the ZIP and is not in it.
