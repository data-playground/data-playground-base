# WO#33 — NBA Data Domain — Post-Mortem

**Location in repo:** `migration_docs/WO33_POSTMORTEM.md`
**Work order type:** Greenfield domain build (not a refactor) — see GOVERNANCE.md §4.2
**Recommended disposition:** **PASS WITH CONDITIONS** (§1.3)

**What this document is for**

1. **The reviewer** deciding pass/fail. §2–§5 establish what the Work Order required, what was built first, what changed afterwards, *who agreed to each change*, and how every original acceptance criterion stands at close. §7 separates what was verified live from what was verified only in a sandbox. Appendix A is a command-by-command verification checklist.
2. **The next agent** coalescing requirements. §10 (post-migration cleanup) and Appendices B–C are written to be executed, not just read. Every cleanup item has a trigger, the exact files, a "do not" list, and an acceptance check.

**Attribution markers used throughout**

| Marker | Meaning |
|---|---|
| `[OWNER]` | The project owner explicitly asked for or agreed to it in conversation. |
| `[OWNER-ACCEPTED]` | The agent proposed it; the owner proceeded without objection. Treat as agreed, not as explicitly signed off. |
| `[FORCED]` | Caused by external reality (a blocked host, an API shape), not by a design preference. |
| `[AGENT]` | The agent found/decided it within the latitude the owner delegated ("review the process and make changes as you see fit"). Listed so the reviewer can challenge it. |
| `[PENDING]` | The agent made a judgment call that has **not** been confirmed by the owner. Decision register in §9. |

---

## 1. Executive summary

### 1.1 Outcome

A new `nba` domain exists: 14 MariaDB tables, two pages (`/nba` game log, `/nba/game/{id}` box score + play-by-play), a daily Airflow ingest DAG with a built-in backfill path, and a weekly player-sync DAG that is now non-essential. Game metadata, the traditional box score, play-by-play, arena and attendance all load from a single `core-api.nba.com` request per game.

### 1.2 The headline deviation

The Work Order was built on the assumption — inherited from the owner's original `nba_game_summary.py` — that all data would come from `stats.nba.com`. **That host blocks requests from the deployment host** (silent read timeouts; a long-documented anti-scraping behavior against cloud/datacenter IPs). Everything after the first delivery is, directly or indirectly, a consequence of that:

- Game summary, traditional box score, play-by-play and arena/attendance moved to `core-api.nba.com/cp/api/v1.9/gameDetails` (endpoint found by the owner via browser dev tools).
- Player names moved from the failing weekly `commonallplayers` call to the rosters embedded in every box score.
- The other nine box-score categories (advanced, misc, scoring, usage, four factors, tracking, hustle, matchup, defensive) are **still only available from `stats.nba.com`**. Their tables exist and are wired, but are empty and gated behind an off-by-default flag. This is the single largest piece of the original scope that is not delivered as working data.

### 1.3 Why "pass with conditions"

The original acceptance criteria are met or met-with-explanation (§5). The conditions exist because some evidence is sandbox-only and one agent judgment call needs owner sign-off:

| # | Condition | Blocking? |
|---|---|---|
| 1 | `main.py` registers `domains.nba.routers.nba_games.router`. The file was never in the agent's materials; the owner applied it (a live `/nba` page was observed). The reviewer must confirm the diff exists in the repo. | Yes — verify only |
| 2 | `alembic heads` shows a single head before applying; both NBA revisions then apply cleanly (end state: `nb4_t34ms_null4bl3001`). The second revision has **not** yet been applied in production. | Yes |
| 3 | The owner confirms decision **D-1** (§9): player names now come from box-score rosters and `life_os_nba_player_sync` is deprecated. This was an agent call, not an agreed one. | Yes — sign-off |
| 4 | The October 2025 backfill is re-run past 2025-10-03 and completes. This is the live confirmation of the exhibition-team fix (C-14). Until then acceptance criterion 8 is ⚠️, not ✅. | Yes |
| 5 | The documentation defects listed in §10 P-08 are tracked (they do not affect behavior). | No |

---

## 2. The original Work Order — what was required

*(Summarized faithfully from the WO#33 text; the WO itself is the authority.)*

**Role.** Full-stack engineer building a new domain inside an established codebase. Unlike refactor WOs, latitude on schema, endpoints, templates and implementation — but **not** on the architectural conventions (domain folders, DAG/FastAPI boundary, shared AI layer).

**Hard boundaries**

1. New code lives under `domains/nba/` (only the subfolders actually used).
2. A DAG must never import `models.py`, `database.py`, any router, or any service; DB access only via `airflow/dag_db.py` raw-SQL helpers.
3. Any LLM call goes through `services/ai/`; do not write a seventh provider implementation.
4. Router files stay under 300 lines from the start.
5. **Step 1 (requesting materials) must happen before any code is written.**

**Step 1 — materials to request:** (1) the existing extraction process or the intended data source; (2) ingestion pattern — default DAG→tables→router, confirm vs. on-demand; (3) data granularity; (4) whether dashboard/sidebar integration is in scope; (5) existing naming/format conventions.

**Step 2 — build:** follow the domain shape (`models.py` importing `Base` from `core.base_model`; routers each <300 lines; templates extending `base.html`; static only if needed); new DAG(s) under `airflow/dags/nba/` using `dag_db.py` only; wire into `main.py` and `core/templating.py` (static mount before the general `/static` mount if one exists); sidebar/dashboard only if in scope.

**Acceptance criteria (verbatim intent)**

- [ ] AC-1 Step 1 happened before code; the report shows what was requested and supplied.
- [ ] AC-2 All new code under `domains/nba/` following the established shape; no unused empty subfolders.
- [ ] AC-3 If a DAG exists: confirmed it never imports `models.py`, `database.py`, or any router/service.
- [ ] AC-4 Any AI/LLM calls route through `services/ai/`.
- [ ] AC-5 Router(s) registered in `main.py`; templates path added to `core/templating.py`; any static mount ordered before `/static`.
- [ ] AC-6 Every router file under 300 lines.
- [ ] AC-7 The report documents every non-obvious design decision.
- [ ] AC-8 Basic functional smoke test: pages render and (if DAG-based) the DAG runs and populates data.

**Required report format:** (1) materials requested/received, (2) files created, (3) files edited, (4) design decisions, (5) acceptance results, (6) notes/open questions. §3–§7 below follow this order, extended with the post-delivery change register.

---

## 3. Baseline — Step 1 record and the first delivery

### 3.1 Step 1 record (AC-1 evidence)

The agent acknowledged the WO and requested the five items **before writing any code** (the first response contains only the request). Outcomes:

| Item | Owner response | How it was resolved |
|---|---|---|
| 1. Extraction process | Uploaded `nba_game_summary.py`. Stated: final DB must be the local MariaDB, not BigQuery; the Gemini enrichment step was never turned on and did not work. Delegated: *"review the process and make changes to it as you see fit."* `[OWNER]` | Script reviewed; BigQuery layer replaced; enrichment not ported. |
| 2. Ingestion pattern | Chose **daily Airflow DAG**, replaying the script's "yesterday" default. `[OWNER]` | Daily DAG at 09:00 UTC. |
| 3. Granularity | Chose: game log + traditional box score + play-by-play UI now; ingest all ten box-score variants headless; first-class UI for every category as a **clearly stated next step**. `[OWNER]` | Option 1 built; §10 P-10 carries the next step. |
| 4. Dashboard/sidebar | **Defer to a fast-follow work order.** `[OWNER]` | Not built; §10 P-09. |
| 5. Conventions | **Not answered.** | Assumption: NBA-native conventions (10-char string game IDs, numeric team IDs, `YYYY-YY` season derived from the game ID, ISO dates). Flagged in §9 (D-7) for the soccer cross-read. |

### 3.2 First delivery (baseline) — what was built

**Files (13 + a README, delivered as individual files, later as a zip):**

- `domains/nba/models.py` — 14 tables: `Team`, `Player`, `Game`, `BoxScoreTraditional`, eight further per-player-per-game variants (`Advanced`, `Misc`, `Scoring`, `Usage`, `FourFactors`, `Tracking`, `Hustle`, `Defensive`), `BoxScoreMatchup` (per defender/offensive-player pair), `PlayByPlayEvent`. Shared identity columns via a mixin.
- `domains/nba/seed_data.py` (30 teams), `domains/nba/reference_ddl.sql` (convenience only).
- `domains/nba/routers/nba_games.py` (96 lines): `GET /nba?date=` and `GET /nba/game/{game_id}`.
- `domains/nba/templates/nba_games.html`, `nba_game_detail.html` — extend `base.html`, reuse `base.css` primitives; no `static/` folder.
- `airflow/agents/nba_agents.py` — DAG-facing, **pure** fetch/parse/flatten layer (no DB code), following the `media_agents.py` precedent. Ported the endpoint catalog and the three response-shape parsers (v2/v3/v4) from the owner's script; added a request throttle and retry/backoff.
- `airflow/dags/nba/life_os_nba_ingest.py` — daily; per game: summary + ten box-score variants + matchup + play-by-play, all from `stats.nba.com`; idempotent `INSERT … ON DUPLICATE KEY UPDATE`.
- `airflow/dags/nba/life_os_nba_player_sync.py` — weekly roster snapshot from `commonallplayers`.
- `core/templating.py` — one `ChoiceLoader` line added.

**Baseline design decisions**

1. **Normalized relational schema** instead of BigQuery's nested RECORD columns (MariaDB has no equivalent). Confirmed with the owner as the one hard-to-reverse call. `[OWNER-ACCEPTED]`
2. `nba_teams` is a static 30-row reference table (matchups/box scores only carry bare team IDs).
3. `nba_players` synced weekly rather than derived per game.
4. Stateless functions instead of the script's mutable `NBA` class.
5. Request throttle added (the original had none).
6. AI enrichment dropped; if revived it must use `services/ai/call_gemini_json`. `[OWNER]`
7. All writes idempotent via unique constraints, so any date can be safely re-run.

**Not in the baseline:** no Alembic migration (the agent had no migration history and declined to fabricate a `down_revision`), no `main.py` edit (file not provided).

---

## 4. Change register — everything changed after the first delivery

Grouped by origin. IDs (`C-nn`) are stable references used elsewhere in this document.

### 4A. Changes the owner requested or directed

**C-01 — Alembic migration `nb4_d0ma1n001`** `[OWNER]`
- The owner attached the latest migration (`s0cc3r_d0ma1n001`, soccer) and asked for an Alembic version based on it.
- Chained `down_revision = 's0cc3r_d0ma1n001'`; mirrored that migration's conventions (leet-style revision IDs, seeded static table via `bulk_insert`, "verify `alembic heads` first" docstring, composite unique constraints added after `create_table`).
- Seeds the 30 teams (duplicating `seed_data.py` on purpose, as the soccer migration does, so the migration has no application-code dependency).
- Files: `alembic/versions/nb4_d0ma1n001_add_nba_domain_tables.py`. Live evidence it was applied: the owner's production error cites `jobs.nba_games … nba_games_ibfk_2`, which only exists if the migration ran.

**C-02 — Delivery as a ZIP in project structure** `[OWNER]` — `internal_dataplayground/{domains,airflow,alembic,core}/…`.

**C-03 — Backfill capability** `[OWNER]` (request) / `[OWNER-ACCEPTED]` (design)
- Request: backfill from 2025-10-01, start/end dates defaulting to yesterday, minimal process change; "run manually, or a settings page?"
- Delivered in `life_os_nba_ingest.py`: `dag_run.conf` `{start_date,end_date}` or `{game_date}`; fallback to Airflow's logical date `ds` (so `airflow dags backfill` also works); a direct CLI (`--date`, `--start-date/--end-date`, `--with-advanced-stats`); each date upserts before the next, so a long run keeps partial progress.
- **Settings page: recommended against** (a one-time operation; Airflow's trigger-with-config covers it). The owner proceeded with the CLI. Revisit only if backfills become routine (`services/airflow_service.trigger_airflow` exists for that).

**C-04 — Discovery request format** `[OWNER]` — the owner found the live nba.com request (`gamedate=MM/DD/YYYY`, lowercase) and the agent adopted it. The previous `GameDate=YYYY-MM-DD` had in fact worked (12 games for 2025-10-22, matching the public schedule); the change aligns with the real contract instead of relying on backend leniency.

**C-05 — Source switch for summary/box score/play-by-play** `[OWNER]`
- The owner found `core-api.nba.com/cp/api/v1.9/gameDetails` and supplied full `tabs=all` payloads, and directed: *"try to go the route of the API and see if we can get every boxscore through that."*
- Outcome: **partial.** `tabs=header,summary,pbp,boxscore` yields summary, traditional box score, play-by-play, arena and attendance. NBA's own validation error enumerates the only legal tabs (`all, header, summary, highlights, boxscore, pbp, ced, insights`), and the owner confirmed the site's other box-score tabs call `stats.nba.com`. So the nine other categories cannot be had from this host.
- Implemented as `fetch_game_details_full()` (one request replaces three). Details and field mapping: Appendix B.

**C-06 — Exhibition/preseason opponents must not crash the run** `[OWNER]` — see C-14 for the implementation; the owner's words: *"be aware and account for those instead of having it as an error."*

**C-07 — Deferrals** `[OWNER]`: dashboard/sidebar integration (Step 1 Q4); visual redesign (team colors, logos — "come back later and discuss mockups"). Both carried into §10.

### 4B. Changes forced by external reality

**C-08 — `stats.nba.com` is blocked from the deployment host** `[FORCED]`
- Symptom: read timeouts on every `stats.nba.com` call (box-score summary, then the players list), while `core-api.nba.com` stayed reachable throughout.
- Research found this is a long-standing, widely reported block on cloud/datacenter IPs, not a code bug.
- Options presented to the owner: residential proxy; a paid hosted API (balldontlie's tier with box scores/PBP/advanced stats); slowing requests (judged unlikely to help); headless-browser scraping (judged low-probability). The owner chose none of these and instead found alternative hosts (C-05).
- Mitigation shipped: `_ingest_date(..., fetch_advanced_stats=False)` — the nine categories + matchup are skipped by default so a backfill is not stuck burning minutes per game on calls known to fail. CLI flag `--with-advanced-stats`; conf key `fetch_advanced_stats`.

**C-09 — Empty `modules` on days with no games** `[FORCED]` — `data["modules"][0]` raised `IndexError` on any off-day (the first scheduled run happened in the offseason). Now treated as "no games."

**C-10 — Airflow 3 compatibility** `[FORCED]` — `schedule_interval` → `schedule`.

### 4C. Defects in the baseline, found and fixed by the agent `[AGENT]`

Listed in full in §8 with root causes. Summary of code changes:

**C-11 — `autoincrement=False`** on `nba_teams.id` and `nba_players.person_id` (SQLAlchemy had silently given externally-supplied IDs `AUTO_INCREMENT`). Applied to `models.py` and the migration.

**C-12 — Removed redundant single-column `game_id` indexes** from the per-game tables (each already has a composite `UNIQUE` whose leftmost column is `game_id`). Only `nba_games.game_date` keeps an index.

**C-13 — Placeholder player rows** — box-score/PBP rows FK into `nba_players`, which a *weekly* job populated; a player debuting between syncs would have failed an entire night's ingest. `_ensure_players_exist` inserts `Player #<id>` when missing (a genuine no-op if the row exists, so a real name is never overwritten). Verified in the sandbox: ingest → simulated sync → ingest; the real name survives.

**C-14 — Placeholder team rows; `tricode` nullable; migration `nb4_t34ms_null4bl3001`**
- Live failure: the 2025-10-03 slate raised `IntegrityError 1452` on `nba_games.away_team_id`. Inferred cause (the actual team ID was not seen): a preseason exhibition opponent whose ID was never one of the 30.
- Fix: `_ensure_teams_exist()` inserts `Team #<id>` (no tricode/conference/division) before the games upsert. `Team.tricode` became nullable, via a **new** migration (the first was already applied in production and must not be edited). NULLs do not violate the `UNIQUE` index in MariaDB/InnoDB, so many placeholders coexist while real tricodes stay unique.
- Follow-on bug caught before delivery: the templates did `team.tricode if team else team_id`, so a placeholder (row exists, tricode NULL) would have rendered the literal text "None". Added `Team.display_code` (`tricode or full_name`) and switched both templates to it.

**C-15 — Player names from box-score rosters** `[AGENT]` → **`[PENDING]` D-1**
- `fetch_game_details_full()` now also returns `player_rows`; `_upsert_player_names()` writes real `full_name`/`team_id` into `nba_players` every run (overwriting placeholders). Rosters in the box score include starters, bench, DNPs and inactives, each with a real name.
- This replaced the need for the weekly `commonallplayers` call, which kept timing out even after the request was shrunk to the current season, with a 45 s timeout and longer backoff. `life_os_nba_player_sync.py` was left in place with a docstring marking it non-essential. **The owner asked to "approach the player_sync aspect"; deprecating it was the agent's chosen approach and has not been confirmed.**

**C-16 — CLI run hung / printed nothing** `[AGENT]`
- Two causes were in play. Confirmed: the owner's container was running a stale copy of the file (the deprecation warning still showed `schedule_interval` after the fix was "delivered"). Hypothesized, **not proven**: constructing the `DAG` object at import time on a direct `python3 file.py` run stalled against the live scheduler.
- Fix: `_register_dag()` runs only when the file is imported by Airflow (`if __name__ != "__main__"`), so a direct run builds no DAG object. The guard is harmless but may not have been strictly necessary.

**C-17 — Resilience in the fetch layer** `[AGENT]` — per-endpoint timeouts; retry backoff 10/20/30 s on timeouts; `None`-safe parsing; DNP/team-event rows with no `personId` handled.

### 4D. Not changed (to avoid ambiguity)

- Table count remains 14; the nine headless tables and the matchup table were **kept**.
- UI scope is unchanged (game log, traditional box score, play-by-play).
- No `domains/nba/static/` folder, no sidebar/dashboard edits.
- No AI/LLM code exists.
- No legacy shim was added to root `models.py`, and none must be (GOVERNANCE §2.4 is closed).

---

## 5. Acceptance criteria at close

Legend: ✅ met · ⚠️ met with a caveat / partially evidenced · ❌ not met.

| AC | Status | Evidence and caveats |
|---|---|---|
| AC-1 Step 1 before code | ✅ | §3.1. Q5 (conventions) went unanswered; handled as a stated assumption (D-7). |
| AC-2 Code under `domains/nba/`, domain shape, no empty folders | ✅ | `domains/nba/{models.py, seed_data.py, routers/, templates/}`; `find domains/nba -type d` → only `routers`, `templates`. Ingestion logic is in `airflow/agents/nba_agents.py` and `airflow/dags/nba/` — placement the WO itself sanctions (Step 2.3) and follows the `media_agents.py` precedent. Migrations in `alembic/versions/` per CONTRIBUTING. |
| AC-3 DAG boundary | ✅ | Grep of both DAG files and `nba_agents.py` found no `models`/`database`/router/service/`domains` imports; the DAG files import only `airflow`, `dag_db`, and `agents.nba_agents`. Command in Appendix A. |
| AC-4 AI via `services/ai/` | ✅ (vacuous) / ⚠️ docs | No AI calls exist. **Caveat:** the `nba_agents.py` module docstring refers to an "intentionally-unused hook" in `models.py`; that hook was never built (§10 P-08). |
| AC-5 Registration | ⚠️ | `core/templating.py` edited ✅. No static mount needed ✅. `main.py` registration was **applied by the owner** (evidence: a live `/nba` page with real data) but is **not part of the delivered files** — the reviewer must confirm the diff (Condition 1). |
| AC-6 Router < 300 lines | ✅ | `nba_games.py` = 96 lines. |
| AC-7 Decisions documented | ✅ | This document (§3.2, §4) plus inline docstrings. Known stale docstrings listed in §10 P-08. |
| AC-8 Smoke test | ⚠️ | Live: the `/nba` game log rendered real scores; one real game (2025-10-02: 1 game, 42 roster rows, 634 play-by-play rows) ingested end to end. **Not live:** the detail page with real box score/PBP, the nine headless tables, the fixed exhibition-team path, a scheduler-triggered run on the current code, and the full October backfill. See §7. Condition 4 closes this. |

---

## 6. Final state of the system

### 6.1 Data flow

```
core-api.nba.com  gamecardfeed (gamedate=MM/DD/YYYY)  ──►  game IDs for a date
core-api.nba.com  gameDetails  (tabs=header,summary,pbp,boxscore)
        │   ONE request per game
        ├─► summary  ──► nba_games              (scores, status, records, arena, attendance)
        ├─► boxscore ──► nba_box_score_traditional  (per player, incl. DNP/inactive rows)
        ├─► pbp      ──► nba_play_by_play
        └─► rosters  ──► nba_players            (real names, current team)

stats.nba.com  (BLOCKED — gated off by default via fetch_advanced_stats)
        └─► 8 per-player variants + matchup ──► 9 "headless" tables (EMPTY)

Placeholders written by the ingest DAG:  nba_teams ("Team #id", NULL tricode)
                                         nba_players ("Player #id")
```

Ingest order inside `_ingest_date`: ensure teams → upsert games → upsert player names → ensure players (fallback) → upsert traditional + PBP → (optional) nine variants + matchup.

### 6.2 Table lineage — what is actually populated

| Table | Source | Populated? | Never-populated columns |
|---|---|---|---|
| `nba_teams` | Migration seed (30) + ingest placeholders | Yes | — |
| `nba_players` | Box-score rosters; `Player #id` fallback | `full_name`, `team_id` | `roster_status`, `from_year`, `to_year` (only the failing sync DAG wrote them) |
| `nba_games` | `gameDetails` header/summary/boxscore | Yes | `duration`, `game_label`, `game_sub_label`, `series_text` (no source in these tabs) |
| `nba_box_score_traditional` | `gameDetails` boxscore | Yes | — |
| `nba_play_by_play` | `gameDetails` pbp | Yes | `action_id` (no equivalent field); `shot_value` is derived from `actionType` |
| 8 variant tables + `nba_box_score_matchup` | `stats.nba.com` | **Empty** | all |

### 6.3 Delivered files (15)

```
domains/nba/__init__.py                       domains/nba/routers/__init__.py
domains/nba/models.py                         domains/nba/routers/nba_games.py
domains/nba/seed_data.py                      domains/nba/templates/nba_games.html
domains/nba/reference_ddl.sql                 domains/nba/templates/nba_game_detail.html
airflow/agents/nba_agents.py                  core/templating.py   (edited — one line)
airflow/dags/nba/life_os_nba_ingest.py
airflow/dags/nba/life_os_nba_player_sync.py   (deprecated in place; see D-1)
alembic/versions/nb4_d0ma1n001_add_nba_domain_tables.py
alembic/versions/nb4_t34ms_null4bl3001_make_tricode_nullable.py
```

**Migration chain:** `… → s0cc3r_d0ma1n001 → nb4_d0ma1n001 → nb4_t34ms_null4bl3001`.

### 6.4 Runtime behavior worth knowing

- Schedule: `0 9 * * *` UTC, `catchup=False`; default date = the run's Airflow logical date (the previous day for an on-time run).
- Off-days are a quiet no-op (0 games), not an error.
- Every write is an idempotent upsert; re-running any date or range is safe. Real names/placeholders interact safely (a real name is never overwritten by a placeholder).
- The nine variants are skipped by default; `--with-advanced-stats` re-enables them (they will time out until the block is resolved).

---

## 7. Verification evidence

### 7.1 Observed live (owner's environment)

- The migration was applied (FK name `nba_games_ibfk_2` in the production error).
- Discovery for 2025-10-22 returned 12 games (matches the public schedule).
- 2025-10-02 ingested end to end: 1 game, 42 player rows ensured, 42 traditional rows, 634 play-by-play rows.
- The `/nba` game log rendered the 2025-10-22 slate with real scores (first version of the summary fetch).
- Each failure in §4 was first observed here.

### 7.2 Verified by the agent in a sandbox (real MariaDB 10.11, synthetic payloads shaped like the owner's pasted responses)

- Both migrations: `upgrade`, `downgrade`, `upgrade` (original); `upgrade` (tricode). Column/constraint/seed checks via `SHOW CREATE TABLE`.
- Multiple NULL tricodes coexist; a duplicate real tricode (`LAL`) is still rejected (`ERROR 1062`).
- A synthetic preseason game against an unrecognized team ID ingested cleanly twice (idempotent); the placeholder team was created with NULL tricode; real franchises' tricodes untouched; roster names landed; box-score and PBP rows referencing the unknown team inserted.
- Parsing of the `gameDetails` shape, including: ISO-duration minutes, string scores, `x`/`y` coordinates, DNP/inactive/nameless roster entries, team-level PBP events with no `personId`.
- Both templates rendered with a real ORM `Team` that has a NULL tricode (no "None" leak).
- Placeholder-player behavior across ingest → sync → ingest.

> Why SQLite was not used: the first attempt failed because `ALTER`-style constraints need batch mode there. MariaDB was installed instead — it is the real target.

### 7.3 Not verified (do not claim as passed)

- Anything against the live NBA API from the sandbox (it cannot reach those hosts).
- The `/nba/game/{id}` page with real box-score/PBP data.
- The tricode migration in production.
- A scheduler-triggered (non-CLI) run on the final code, including the `ds` date fallback under Airflow 3.
- The complete October 2025 backfill.
- The nine headless categories (no source).

---

## 8. Defects in the baseline, root causes, and lessons

| ID | Defect | How found | Impact if shipped | Fixed in |
|---|---|---|---|---|
| F-1 | `stats.nba.com` assumed reachable | Production | **Majority of the original design unusable** | C-05, C-08 |
| F-2 | `AUTO_INCREMENT` on external IDs | DDL review | Semantically wrong, latent | C-11 |
| F-3 | Redundant `game_id` indexes | DDL review | Write overhead | C-12 |
| F-4 | Player FK dependent on a weekly job | Real-DB test | A debut between syncs fails the **whole night** | C-13 |
| F-5 | `modules[0]` on empty list | Production (offseason) | Every no-game day fails | C-09 |
| F-6 | Full-history player request | Production | Timeouts | C-17 (insufficient) → C-15 |
| F-7 | Direct-run CLI stalled/printed nothing | Production | Backfill unusable | C-16 |
| F-8 | Team FK assumed only 30 teams | Production (preseason) | Crash on any exhibition game | C-14 |
| F-9 | `tricode NOT NULL` blocked placeholders | Fix design | — | C-14 |
| F-10 | Template rendered "None" for placeholder teams | Pre-delivery review | Cosmetic but visible | C-14 |
| F-11 | Docstring cites a nonexistent enrichment hook | Post-mortem audit | Misleading | **Open** (P-08) |

**Lessons**

1. **Reachability is a design input, not a runtime detail.** A hostname existing in someone's working notebook does not mean it is reachable from the deployment host. Probe every external host from inside the target container *before* designing around it (Appendix C, amendment 5).
2. **Test migrations on the real engine.** DDL compilation passed while two real defects (F-2, F-4) were only visible against MariaDB.
3. **Dimension tables fed on a different schedule from the facts that reference them need a placeholder-on-first-sight path from day one.** This was needed twice (players, teams) before it was generalized.
4. **Verify deployment freshness.** A stale file in the container produced a misleading failure and cost two rounds. Add a one-line check after every file drop (`grep` a known symbol inside the container).
5. **Keep module-level side effects out of DAG files** so they can double as CLIs.
6. **Say what is a hypothesis.** The "contending with the scheduler" explanation in C-16 was never proven; it is recorded as such.

---

## 9. Open items and decision register

| ID | Decision needed | Owner input required? | Default if no decision |
|---|---|---|---|
| **D-1** | Confirm player names now come from box-score rosters and `life_os_nba_player_sync` is deprecated (pause in the Airflow UI, keep or delete the file). | **Yes — Condition 3** | Left in place, non-essential, still failing. |
| **D-2** | `stats.nba.com` strategy for the nine categories: (A) abandon permanently, (B) proxy/alternate egress, (C) find a `core-api`/`www.nba.com` equivalent. | Yes | Status quo: tables exist, empty, flag off. |
| **D-3** | Model the extra fields now available in `gameDetails` (starter/played/DNP reason, per-period line score, `foulsDrawn`, `blocksReceived`, fast-break / paint / second-chance points, team totals, officials) and/or hide DNP rows in the UI. | Yes | Not modeled; DNP/inactive players appear as rows with NULL minutes and zero stats. |
| **D-4** | Namespace ORM class names (`Team`/`Player`/`Game` → `NbaTeam`/…) now vs. when a collision appears (§10 P-03). | Yes | Left generic. |
| **D-5** | Retention policy for placeholder rows (`Team #id`, `Player #id`). | No (low risk) | Keep; monitor counts (§10 P-12). |
| **D-6** | Re-run the October backfill and confirm (Condition 4). | Yes | — |
| **D-7** | Conventions cross-read with the soccer domain (team ID format, season format, date handling) — the WO's "next work order" note. | Yes (review) | NBA-native conventions stand. |

Other known limitations: undocumented endpoints can change without notice; `CORE_API_HEADERS` embeds NBA's public web-client subscription key as a constant (not a secret we own — revisit if NBA rotates it); `duration`/labels/`action_id`/career metadata are structurally NULL (§6.2).

---

## 10. After the rest of the migration program completes — cleanup and hardening

**Trigger.** Run this section once (a) the remaining domain migrations in GOVERNANCE §3.3 and the soccer work order (WO#34) have landed and (b) decision **D-2** has been made. Several items are conditional on D-2; each says so. Do them as separate, separately-revertable changes (GOVERNANCE §4.5/§4.6: cleanup is never bundled into another change).

### P-01 — Reconcile Alembic heads (do this first)

- **Why:** both NBA revisions were written against the head that existed at the time (`s0cc3r_d0ma1n001`). Migrations authored in parallel by other work orders will have created additional heads.
- **Do:** `alembic heads`. One head → nothing to do. More than one → `alembic merge heads -m "merge after nba + <others>"` and apply.
- **Do not:** edit `down_revision` of an already-applied NBA revision, and do not squash `nb4_t34ms_null4bl3001` into `nb4_d0ma1n001` — the first is applied in production, so history must stay append-only.
- **Accept:** `alembic heads` prints exactly one hash; `alembic upgrade head` is a no-op on a fully migrated database; on a fresh database both NBA revisions apply in order.

### P-02 — Mapper registration and model/DB drift check

- **Why:** GOVERNANCE §2.4 says the mapper-registration guarantee lives in an explicit import block in `database.py` (root `models.py` no longer exists). The `database.py` provided with this WO did not show such a block, so it is unverified that `domains.nba.models` is imported anywhere the metadata registry is assembled (the app, and Alembic's `env.py` `target_metadata`).
- **Do:** confirm `import domains.nba.models` (or the repo's equivalent) appears in the registration block and is visible to Alembic's `env.py`. Add it if missing.
- **Accept (strongest check):** `alembic revision --autogenerate -m "nba drift check"` produces an **empty** `upgrade()`/`downgrade()`. Delete the generated file. Any diff means `models.py` and the migrations disagree — resolve before anything else. The expected-equal points: `nba_teams.id`/`nba_players.person_id` not auto-increment; `tricode` nullable + unique; no standalone `game_id` indexes; one `nba_games.game_date` index.

### P-03 — Namespace the ORM class names (decision D-4)

- **Why:** every domain shares one `Base` and one mapper registry. GOVERNANCE §2.2 resolves cross-domain relationships by *string class name*. `Team`, `Player` and `Game` are generic; a second domain defining any of them yields SQLAlchemy's "Multiple classes found for path" at mapper configuration — a failure that can appear in an unrelated domain.
- **Check first:** `grep -rnE "^class (Team|Player|Game)\b" domains/`. If another domain already defines one, this item is urgent.
- **Do:** rename to `NbaTeam`, `NbaPlayer`, `NbaGame`; update the `relationship("Team", …)` strings in `Game`, the imports in `routers/nba_games.py`, and (when built) `routers/dashboard.py`. Templates only use attribute names, so they need no edit. `display_code` stays on `NbaTeam`.
- **Do not:** touch `__tablename__` values — **no migration is required**, and none should be generated.
- **Accept:** app boots; `/nba` and `/nba/game/{id}` render; autogenerate (P-02) is still empty.

### P-04 — `models.py` content cleanup

**a. Stale documentation (no behavior change).**
- `BoxScoreTraditional` docstring says "Source: BS_TRAD (boxscoretraditionalv3)" and `PlayByPlayEvent` says "Source: PBP (playbyplayv3)" — both now come from `core-api.nba.com` `gameDetails`.
- `Game` docstring describes a two-pass discovery + `BS_SUMMARY` flow — now one `gameDetails` call.
- Module docstring still frames the nine variants as "ingested by the DAG on the same schedule"; they are skipped by default.
- `Team` and `Player` docstrings should state the placeholder-row behavior and (for `Player`) that names come from roster data.

**b. Columns that no current source populates — decide per column.**

| Column | Options |
|---|---|
| `nba_games.duration`, `game_label`, `game_sub_label`, `series_text` | Populate from a source if one is found (D-2/C; playoff labels likely matter once the postseason is ingested) **or** drop via migration. Do not leave silently NULL without a comment. |
| `nba_play_by_play.action_id` | No source field exists in the `gameDetails` shape; drop, or keep only if the `stats.nba.com` path is revived. |
| `nba_players.roster_status`, `from_year`, `to_year` | Only meaningful if player sync is revived/replaced (D-1). Otherwise drop. |

Every drop is a migration; add it as a new revision after P-01. Do not drop the nine headless tables unless D-2 resolves to (A).

**c. Optional additions (decision D-3).** Fields confirmed present in the `gameDetails` payload but not modeled: per-player `starter`, `played`, `notPlayingReason`, `foulsDrawn`, `blocksReceived`, `pointsFastBreak`, `pointsInThePaint`, `pointsSecondChance`; per-team totals (a "team totals" row the UI currently has to compute); per-period line score; officials. The `played`/`notPlayingReason` pair solves the DNP-rows-with-zero-stats display problem. Each addition = migration + parser mapping in `fetch_game_details_full` + template change.

**Accept:** `grep -rn "boxscoretraditionalv3\|playbyplayv3\|BS_TRAD\|BS_SUMMARY" domains/` returns nothing; autogenerate (P-02) empty after any column change.

### P-05 — Remove dead code from `airflow/agents/nba_agents.py` (conditional on D-2)

The audit at close found these symbols with no caller outside their own definition: `fetch_game_summary`, `fetch_game_details_header`, `fetch_play_by_play`, `flatten_play_by_play`, `discover_games_for_season`.

**Always safe to remove (superseded by `fetch_game_details_full`, regardless of D-2):**
- functions: `fetch_game_summary`, `fetch_game_details_header`, `fetch_play_by_play`, `flatten_play_by_play`;
- the `BS_SUMMARY`, `PBP` entries in `ENDPOINTS`;
- the `BS_TRAD` entries in `BOX_SCORE_STAT_MAPS` and `_BOX_SCORE_URLS` (the DAG's `BOX_SCORE_TABLES` already excludes it);
- the `GAMES` endpoint and `discover_games_for_season` **unless** a season-level backfill is wanted (it was kept as a possible helper; nothing calls it).

**If D-2 = (A) abandon `stats.nba.com`:** additionally remove `STATS_HEADERS`, the whole v2/v3 machinery (`_filter_data_to_skeleton`, `_filter_results_by_list`, `_player_box_score_skeleton`, `BOX_SCORE_STAT_MAPS`, `_BOX_SCORE_URLS`, `ENDPOINTS` entries other than `GAME_DATA`, `fetch_endpoint`'s v2/v3 branches), `fetch_box_score`, `flatten_player_box_score`, `flatten_matchup_box_score`, `fetch_all_players`; drop the nine tables (migration); remove the `fetch_advanced_stats` flag and `--with-advanced-stats`; delete `life_os_nba_player_sync.py` (and pause/remove the DAG in Airflow).

**If D-2 = (B) or (C):** keep the machinery; remove only the "always safe" list and update the flag's docstring.

**Do not:** remove `_fetch_json`, `_throttle`, `CORE_API_HEADERS`, `discover_games_for_date`, `fetch_game_details_full`, `derive_season_from_game_id`, `_parse_iso_minutes`, `_find_card` — all in the live path.

**Accept:** for each removed symbol `grep -rn "<symbol>" airflow/ domains/` is empty; the DAG file imports cleanly (`python -c "import ast,sys; ast.parse(open('airflow/dags/nba/life_os_nba_ingest.py').read())"`); a single-date CLI run (Appendix A) still succeeds.

### P-06 — DAG-file cleanup

- Remove the `fetch_advanced_stats` plumbing from `_ingest_date`, `task_ingest_games`, and the CLI **only** under D-2 = (A); otherwise keep it and update its docstring.
- Keep `_ensure_teams_exist`, `_ensure_players_exist`, `_upsert_player_names`, `_upsert` — they are live.
- **DRY check (GOVERNANCE §3.1):** compare `_upsert` with the equivalent helpers in the jobs/soccer DAGs. If three or more near-identical copies exist, extract one shared helper (e.g. under `airflow/`), imported by all, as its own change.
- Airflow 3: the imports `from airflow import DAG` / `airflow.operators.python.PythonOperator` still work but are deprecated paths in Airflow 3. This is a program-wide concern; handle it for all DAGs together, not just these.

### P-07 — Remove duplicated seed and reference artifacts

- `domains/nba/reference_ddl.sql` is a convenience file explicitly *not* authoritative; remove it once P-02's drift check passes (otherwise it will rot).
- `domains/nba/seed_data.py` duplicates the migration's seed. Keep only if an idempotent out-of-band reseed is wanted; otherwise remove. Never remove the migration's `SEED_TEAMS`.

### P-08 — Documentation corrections (known defects at close)

1. `nba_agents.py` module docstring, last bullet: "see models.py for the intentionally-unused hook this would attach to" — **no such hook exists** (no `narrative_summary` column was ever added). Delete the clause, or implement the hook if enrichment is revived (via `services/ai/call_gemini_json`, never a raw client).
2. `nba_agents.py` module docstring and ENDPOINT comments still describe `BS_SUMMARY`/`BS_TRAD`/`PBP` as live sources.
3. `life_os_nba_ingest.py` module docstring still describes "box score summary, all ten box score variants, play-by-play" as one flow; update to the three-source story in §6.1.
4. `life_os_nba_player_sync.py` status note (added at close) should be reconciled with the D-1 outcome.
5. The statement "traditional comes from `stats.nba.com`" anywhere in `models.py`/templates/comments.

**Accept:** `grep -rn "intentionally-unused hook" airflow/ domains/` empty; a reader following only the docstrings gets the same picture as §6.1.

### P-09 — Sidebar and dashboard integration (the deferred fast-follow, `[OWNER]`)

- Templates already pass `active_module='nba'`; `templates/partials/sidebar.html` has no NBA entry. Add a nav item and, if wanted, a sub-nav block following the media/blog pattern.
- Dashboard card ("last night in the NBA") goes in `routers/dashboard.py`, the single sanctioned cross-domain reader (GOVERNANCE §2.2): import **directly** from `domains.nba.models` — do **not** create a shim in any root `models.py`.
- **Accept:** sidebar highlights on `/nba` and `/nba/game/*`; the dashboard card renders on an off-day (no games) without error.

### P-10 — First-class UI for the nine categories (the next step the owner required, `[OWNER]`)

- **Precondition:** D-2 must yield a data source. Building pages over empty tables is pointless; sequence this *after* D-2.
- **Do:** one router file per concern (stay under 300 lines — split before 250), tabs on the game-detail page, treat a missing category for a game as "no data for this game" (the owner confirmed not every game has every category), not an error.
- **Accept:** each category page renders for a game that has data and shows an explicit empty state for one that does not.

### P-11 — Visual redesign (deferred, `[OWNER]`)

- Team logos: the payload exposes CDN logo URLs of the form `https://cdn.nba.com/logos/nba/{team_id}/primary/L/100x100/logo.png`; confirm before depending on it. Team colors are **not** in any payload — a static map keyed by `team_id` is needed.
- Placeholder teams (NULL tricode) and `display_code` must have a graceful fallback (no logo, neutral color).
- Settle mockups with the owner first.

### P-12 — Post-backfill data-quality checks and monitoring

Run after the October backfill completes, then schedule as periodic checks (queries in Appendix A, §A.4). Expected-zero checks: final-game score ≠ last play-by-play score; player points sum ≠ team score. Trend-to-zero metrics: `Player #…` placeholders; list of `tricode IS NULL` teams (should equal the known exhibition opponents).

### P-13 — Fixture-based tests and lint (NBA can change shapes silently)

- Save the real `gameDetails` payload the owner pasted (game `0022500003`) as a fixture and test `fetch_game_details_full` against it, asserting the quirks in Appendix B.
- Add CI-checkable lints GOVERNANCE §1.2 already calls for: router line count; the DAG boundary grep (Appendix A, check 4).

### P-14 — Governance document updates

Apply the amendments in Appendix C to GOVERNANCE.md / CONTRIBUTING.md, and add this WO's standing lessons to §6 (Amendment Process).

### P-15 — Cross-domain consistency with the soccer domain (decision D-7)

Compare conventions and record one standard: external IDs (NBA numeric team IDs and 10-char string game IDs vs. FIFA VARCHAR IDs), season representation (NBA derives `YYYY-YY` from the game ID), date/time handling (NBA stores the Eastern calendar date; soccer stores `kickoff_at` as a datetime), and the seed/migration idiom shared by both migrations.

### P-16 — Operational hardening

Alert on DAG failure; verify the 9 am UTC schedule against late games and the `ds` semantics under Airflow 3; confirm retries (`retries=1`, 5-minute delay) are adequate once the backfill is done; track the public subscription-key constant for rotation.

### Suggested order

`P-01 → P-02 → (D-2 decision) → P-05/P-06 → P-04 → P-03 → P-07/P-08 → P-12/P-13 → P-09 → P-10 → P-11 → P-14/P-15/P-16`. P-03 can move earlier if the collision check in it finds a clash.

---

## Appendix A — Reviewer verification checklist

Run from the repo root (adjust paths if the layout differs). Expected results in comments.

### A.1 Static checks

```bash
# 1. DAG boundary (AC-3) — expect NO output
grep -nE "import (models|database)|from (models|database|routers|services|domains)" \
     airflow/dags/nba/*.py airflow/agents/nba_agents.py

# 2. AI layer (AC-4) — expect only docstring/comment hits, no import or call
grep -rnE "services\.ai|genai|call_ai|call_gemini" airflow/dags/nba airflow/agents/nba_agents.py domains/nba

# 3. Router size (AC-6) — expect < 300 (currently 96)
wc -l domains/nba/routers/*.py

# 4. Shape (AC-2) — expect only: routers, templates (no static/)
find domains/nba -type d

# 5. Templating (AC-5)
grep -n "domains/nba/templates" core/templating.py

# 6. Router registered (AC-5, Condition 1) — expect an include of the nba router
grep -n "nba_games" main.py
```

### A.2 Migrations (Condition 2)

```bash
alembic heads                       # one head; before applying: s0cc3r_d0ma1n001 (or a later one — see P-01)
alembic upgrade head
alembic current                     # nb4_t34ms_null4bl3001
mysql <db> -e "SHOW CREATE TABLE nba_teams\G"
#   id        INT NOT NULL   (no AUTO_INCREMENT)
#   tricode   varchar(3) DEFAULT NULL  + UNIQUE KEY
mysql <db> -e "SELECT COUNT(*) FROM nba_teams WHERE tricode IS NOT NULL;"   # 30 on a fresh DB
alembic revision --autogenerate -m "drift check"    # expect empty upgrade()/downgrade(); delete the file
```

### A.3 Runtime

```bash
# Confirm the container has the current file (stale files caused earlier confusion)
docker compose exec airflow-scheduler grep -c "_ensure_teams_exist" /opt/airflow/dags/nba/life_os_nba_ingest.py   # 1

# Single-date smoke
docker compose exec airflow-scheduler python3 -u /opt/airflow/dags/nba/life_os_nba_ingest.py --date 2025-10-22
#   expect: "12 game(s) ingested"

# The failing slate from the incident (Condition 4) — expect it to complete
docker compose exec airflow-scheduler python3 -u /opt/airflow/dags/nba/life_os_nba_ingest.py \
    --start-date 2025-10-01 --end-date 2025-10-31

# Idempotency — run the same date twice; row counts must not change
```

Then open `/nba` (game log) and `/nba/game/<id>` (box score + play-by-play tabs) for one regular-season and one preseason game.

### A.4 Data-quality queries (also P-12)

```sql
-- 12 games on opening-week day (public schedule)
SELECT COUNT(*) FROM nba_games WHERE game_date = '2025-10-22';

-- Final games: last play-by-play score equals final score (expect 0 rows)
SELECT g.game_id, g.home_score, p.score_home, g.away_score, p.score_away
FROM nba_games g
JOIN nba_play_by_play p ON p.game_id = g.game_id
WHERE g.game_status = 3
  AND p.action_number = (SELECT MAX(action_number) FROM nba_play_by_play WHERE game_id = g.game_id)
  AND (g.home_score <> p.score_home OR g.away_score <> p.score_away);

-- Final games: player points sum to team score (expect 0 rows)
SELECT g.game_id,
       g.home_score, SUM(CASE WHEN b.team_id = g.home_team_id THEN b.points END) AS home_sum,
       g.away_score, SUM(CASE WHEN b.team_id = g.away_team_id THEN b.points END) AS away_sum
FROM nba_games g JOIN nba_box_score_traditional b ON b.game_id = g.game_id
WHERE g.game_status = 3
GROUP BY g.game_id, g.home_score, g.away_score
HAVING home_sum <> g.home_score OR away_sum <> g.away_score;

-- Placeholder health
SELECT COUNT(*) FROM nba_players WHERE full_name LIKE 'Player #%';   -- trends toward 0
SELECT id, full_name FROM nba_teams WHERE tricode IS NULL;           -- exhibition opponents only

-- Preseason games (NBA game IDs encode season type in characters 2-3: 01 = preseason)
SELECT game_id, game_date FROM nba_games WHERE game_id LIKE '001%';
```

---

## Appendix B — `gameDetails` mapping and shape quirks

**Request:** `GET https://core-api.nba.com/cp/api/v1.9/gameDetails?gameid=<id>&leagueid=00&platform=web&tabs=header,summary,pbp,boxscore` with `CORE_API_HEADERS`. Legal `tabs` values (from NBA's own validation error): `all, header, summary, highlights, boxscore, pbp, ced, insights`. **Discovery:** `…/feeds/gamecardfeed?gamedate=MM/DD/YYYY&platform=web`; `modules` is legitimately `[]` on no-game days.

**Mapping**

| Target | Source path |
|---|---|
| `nba_games.game_date` | `header.cardData.gameTimeEastern[:10]` |
| `game_status`, `game_status_text`, `period` | `header.cardData.gameStatus / gameStatusText / period` |
| `home_/away_ team_id, score, wins, losses` | `header.cardData.homeTeam / awayTeam` (`teamId`, `score`, `wins`, `losses`) |
| `arena_name / city / state` | `summary.modules[].cards[]` where `cardType == 'gameInfo'` → `cardData.arena` |
| `attendance` | `boxscore.attendance` |
| `nba_box_score_traditional.*` | `boxscore.{homeTeam,awayTeam}.players[].statistics` (`fieldGoalsMade/Attempted/Percentage`, `threePointers…`, `freeThrows…`, `reboundsOffensive/Defensive/Total`, `assists`, `steals`, `blocks`, `turnovers`, `foulsPersonal`, `points`, `plusMinusPoints`); `position` from the player object |
| `nba_players.full_name, team_id` | `boxscore.{homeTeam,awayTeam}.players[].name` and the team's `teamId` |
| `nba_play_by_play.*` | `pbp.actions[]` (`actionNumber`, `period`, `clock`, `teamId`, `teamTricode`, `personId`, `playerName`, `actionType`, `subType`, `description`, `scoreHome/Away`, `pointsTotal`, `shotDistance`, `shotResult`, `isFieldGoal`, `location`, `x`→`shot_x`, `y`→`shot_y`) |

**Quirks (each one would silently corrupt data if missed)**

1. `boxscore.homeTeam/awayTeam.score` is always `0` — use `header.cardData` scores.
2. `boxscore.arena` has empty city/state — use the `gameInfo` card in `summary`.
3. The `gameInfo` card's position among `summary.modules` is not stable — search by `cardType`.
4. `minutes` is an ISO-8601 duration (`PT37M30.00S`), converted to `M:SS`; empty string for DNPs.
5. PBP `scoreHome`/`scoreAway` are **strings**.
6. PBP coordinates are `x`/`y` (not `xLegacy`/`yLegacy`); `shotDistance` is a float (rounded to the integer column).
7. PBP has no `actionId`; `shot_value` is derived from `actionType` (`2pt`→2, `3pt`→3).
8. Team-level PBP events (team rebounds, timeouts) have no `personId`; `played`-0 and INACTIVE roster entries are real rows with names but empty/zero stats.
9. `sellout` is the string `'1'` (not persisted).
10. Not every game has every tab populated — callers must treat empty as "no data."

---

## Appendix C — Proposed governance amendments

1. **GOVERNANCE §2.2 addition — placeholder-on-first-sight.** A dimension table fed by a different schedule (or a static seed) from the fact tables that FK into it needs a placeholder-insert step in the ingest path (`ON DUPLICATE KEY UPDATE <pk>=<pk>`), with the placeholder's non-key columns nullable. Reference implementation: `_ensure_players_exist`, `_ensure_teams_exist` in `life_os_nba_ingest.py`.
2. **CONTRIBUTING addition — externally supplied primary keys** use `autoincrement=False` (SQLAlchemy otherwise assumes auto-increment on a single-column integer PK).
3. **CONTRIBUTING addition — no standalone index on the leftmost column of an existing composite unique constraint.**
4. **GOVERNANCE §2.5 addition — DAG-facing fetch modules stay pure** (HTTP + parsing, no DB code); the DAG file is the only writer. Reference: `nba_agents.py` / `life_os_nba_ingest.py`.
5. **§4.3 Work-Order template — add a "SOURCE REACHABILITY" gate before design:** the owner (or agent via the container) confirms each external host responds from inside the deployment container (`docker compose exec <svc> curl …`). This single check would have avoided most of this WO's rework.
6. **§4.3 Work-Order template — add "DEPLOYMENT VERIFICATION":** after any file drop, `grep` a known new symbol inside the running container.
7. **CONTRIBUTING addition — migrations are tested on the target engine** (MariaDB), with `upgrade → downgrade → upgrade`, not on SQLite or by DDL compilation alone.
8. **CONTRIBUTING addition — DAG files avoid import-time side effects** (construct the `DAG` only when imported by Airflow) so they can double as CLIs.
9. **Post-mortem convention:** `migration_docs/WO<NN>_POSTMORTEM.md`, using the attribution markers defined at the top of this document so a reviewer can separate agreed changes from agent judgment.

---

## Appendix D — Questions a reviewer is likely to ask

- **Why do nine tables exist but hold nothing?** The owner asked for all ten variants to be ingested with UI only for traditional (§3.1 Q3). The source for nine of them is blocked; the tables, parsers and flag are retained so the work resumes the moment access exists (D-2).
- **Why does the DAG code live partly outside `domains/nba/`?** The WO itself places DAGs under `airflow/dags/nba/` and points to the `media_agents.py` pattern for the fetch layer.
- **Why a second migration instead of editing the first?** The first was already applied in production; Alembic history must be append-only.
- **Why not delete `life_os_nba_player_sync.py`?** The owner has not confirmed deprecation (D-1); leaving it, clearly annotated, is the reversible choice.
- **Is the exhibition-team fix proven?** Proven on real MariaDB with a synthetic game of the same shape; **not yet** re-run against the live backfill (Condition 4).
- **Anything that could fail silently?** Yes — NBA changing the undocumented response shape. P-13's fixture tests and P-12's consistency queries exist to catch that.
