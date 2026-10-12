# WO#34 — Soccer Data Domain: Follow-Up Postmortem & Cleanup Handoff

**Written:** 2026-10-09
**Supersedes / extends:** the postmortem dated 2026-10-02 ("WO#34 — Soccer Data Domain: Postmortem & Handoff"), referred to below as **PM-1**. This document is **PM-2**. PM-1 is not repeated; its item numbers (K1–K17, I1–I3, §9.x) are reused so the two documents can be read side by side.
**State of the work order:** all follow-up items the owner asked for in this round are delivered; **two groups of work remain** — owner-side application/verification (§7) and the post-migration cleanup (§10), which must wait until every other in-flight migration has landed.

**Audience**
1. The **reviewing agent**, who decides whether WO#34 passes. Read §0, §3, §4, §6 and §7.
2. The **implementing agent** who performs the final cleanup. Read §8 (open items) and §10 (the cleanup spec) — §10 is written to be executed on its own.

**How to read the claims.** Each statement is one of:
- **[verified]** — checked against a file, a command result or the session record, and says so;
- **[owner-reported]** — Pedro ran it and told us;
- **[UNVERIFIED]** — an inference or a value written from memory;
- **[not run]** — code that compiles/was exercised only on stand-in data, never against the real database, Airflow or FIFA.
The authoring sessions had **no live database, Airflow, network or repository access**. Every functional result below came from the owner's environment, not from the author's.

---

## 0. Status at a glance

| Area | State | Evidence |
|---|---|---|
| PM-1 headline defects (K1, K2, K3) | K1 fixed and deployed; K2 gone; K3 (migration unapplied) resolved | K1/I1 "deployed" [owner-reported]; repo shows one head, no cycle [owner-reported output of `alembic_chain_check.py`]; `alembic current` → `expl0r3r_d0ma1ns001 (head)` [owner-reported] |
| Owner-requested improvements | **I1 done and deployed**; I2, I3 not started | §3.2 |
| PM-1 §9 cleanup | Partially done; the remainder is gated on other migrations (§10) | §3.3 |
| New work added this round | Crest images via FIFA's `PictureUrl`; CSS moved to a domain static folder; "View on FIFA" link; model registry | §4, §5 |
| Database | At head `expl0r3r_d0ma1ns001` **before** the new migration `s0cc3r_cr3sts001` | owner-reported; new migration is **not applied** |
| Application of the last delivery (`soccer_followups.zip`, scripts) | **Not confirmed** | owner has not reported applying it |
| `/soccer` 500 from PM-1 | Expected fixed (all `soccer_matches` columns now exist at head); **not confirmed by the owner** | §3.1 K3 |
| Overall | Functionally complete; **not yet closed** — see §6.3 for the exact sign-off conditions | |

---

## 1. The work order, in brief (unchanged since PM-1)

The full text is in PM-1 §1. Summary of what a reviewer must hold the delivery against:

**Role/boundaries.** Full-stack engineer building a new domain under `domains/soccer/`; the DAG must not import `models.py`, `database.py`, routers or services (DB access only via `airflow/dag_db.py`); any LLM call goes through `services/ai/`; router files stay under 300 lines; Step 1 (questions) before any code.

**Owner's Step 1 answers (decisions of record).** (1) source = the supplied `fifa_data_pull.py`, loading to MariaDB instead of BigQuery; (2) competitions by ID, World Cup (17) as the test; (3) daily DAG pull; (4) all examples in the script plus raw-data storage reusable across endpoints; (5) dashboard/sidebar integration is a **future fast-follow**.

**Acceptance criteria as issued:** (1) Step 1 before code; (2) code under `domains/soccer/`; (3) DAG import boundary; (4) AI via `services/ai/`; (5) router registered, templates path added, static mount correctly ordered if used; (6) routers < 300 lines; (7) design decisions documented incl. WO#33 alignment; (8) functional smoke test.

---

## 2. What PM-1 raised (the baseline for this round)

PM-1 (2026-10-02) recorded the delivered state and handed off three groups of work:

1. **Known issues K1–K17** (§7 of PM-1). The three high-severity ones:
   - **K1** — `ingest_match_details` fetched every scheduled fixture in the next 60 days, every day (two FIFA calls + two raw rows each), because the details query reused the full future window.
   - **K2** — an Alembic *"Cycle is detected in revisions (m3d1um_d0ma1n004, s0cc3r_t34m1ds001)"* error in the owner's repo; cause undetermined.
   - **K3** — `s0cc3r_st4ts001` unapplied, so `/soccer` returned HTTP 500 (`Unknown column 'soccer_matches.home_penalty_score'`).
2. **Owner-requested improvements I1–I3** (§8 of PM-1): I1 shootout kicks must not count as goals; I2 the "shots" definition undercounts; I3 predefined player placements per formation.
3. **Post-migration cleanup §9** (§9.1–§9.13 of PM-1): migration docstrings, `models.py` references, narrative comments, mapper registration/`env.py`, shim check, DAG details, templates, agent comments, GOVERNANCE amendments, artefact hygiene, separate ticket for the `env.py` fix, tests, final regression.

PM-1 also stated its own scope: **WO#34 = original WO + changes C1–C13**, with C14 (`alembic/env.py` import fix) an out-of-scope fix to be ticketed separately and C15 (clean-slate wipe) an operational step.

---

## 3. What was done in this round — item by item

### 3.1 PM-1 known issues (K1–K17)

| ID | PM-1 problem | Disposition now | Detail / evidence |
|---|---|---|---|
| **K1** | Symmetric details window | **Fixed; deployed** [owner-reported]; behaviour in production **not yet observed** | `DETAIL_LOOKAHEAD_DAYS = 1` constant; upper bound is `min(future_days, 1)` days out, taken to **end of that day** (`datetime.max.time()`; the old bare `date` compared as midnight and would have excluded that day's own kickoffs). **Both** branches of the query are capped: the "kickoff in window" branch and the "no `match_details` raw row yet" branch (the latter was the real culprit — unbounded). Matches with `kickoff_at IS NULL` are no longer fetched. `ingest_fixtures` and task 3 untouched. File compiles [verified]; never run against a database [not run]. |
| **K2** | Alembic cycle | **Resolved in the owner's repo**; cause **not identified** | Owner's `alembic_chain_check.py` run: 41 revisions, **no cycle, one head** (`expl0r3r_d0ma1ns001`), soccer stretch in the intended order. The six "FORK" lines the first version of that script printed were **false positives** (all re-converge through merge revisions); the script was corrected to report re-converging forks as `info` only. Likely mechanism (a stale `down_revision` on `s0cc3r_t34m1ds001`) was a hypothesis and is **UNVERIFIED**; the owner's repo had already been fixed by the time it was inspected. |
| **K3** | `/soccer` 500 | Expected fixed | `alembic current` reports head `expl0r3r_d0ma1ns001` [owner-reported] ⇒ `s0cc3r_st4ts001` applied ⇒ the missing columns exist. Page load **not confirmed** by the owner. |
| **K4** | `parse_match_event_stats()` returns zeros for an empty `/timelines` payload | **Open** | Not touched. Suggested fix unchanged (return `None` for all stats when the event list is empty). |
| **K5** | Shootout goals counted as goals | **Fixed (= I1)** | §3.2. |
| **K6** | Shots undercount | **Open (= I2)** | §8. |
| **K7** | Crest URL unverified / failing | **Addressed in code; resolution unverified** | The owner found the cause: the URL was built from `IdTeam`; the payload carries `PictureUrl`. Now stored and used (§4.1). The resolved URL's `{format}`/`{size}` values and the club (non-flag) case are **[UNVERIFIED]**. |
| **K8** | `soccer_settings.py` at 261 lines | **Open** | Not touched (past the 250 "split soon" mark). |
| **K9** | Manager shown only for `Role == 0` | **Open** | A screenshot this round showed a Saudi Arabia coach with `Role: 1`, consistent with PM-1's "1 = assistant" — **one more data point, not a spec**. |
| **K10** | Card badge binary | **Open** | |
| **K11** | Task 3 re-parses matches with no lineup forever | **Open** | |
| **K12** | Score source of truth (calendar vs `/live`) | **Open** | |
| **K13** | No "live" status code known | **Open** | |
| **K14** | Task name `parse_finished_match_details` is a misnomer | **Open** | Rename only with care — Airflow history is keyed on `task_id`. |
| **K15** | Model/migration nullability drift on four timestamp columns | **Fixed in the delivered `models.py`**; application not confirmed | Verified against the migration the owner supplied: `s0cc3r_d0ma1n001` declares `NOT NULL` on exactly `soccer_competitions.created_at`, `soccer_matches.updated_at`, `soccer_raw_payloads.fetched_at`, `soccer_settings.updated_at`. Models aligned to the migration (not the reverse). A by-eye comparison of column names/nullability across the four migrations and `models.py` found no other drift; **an `--autogenerate` comparison has not been run**. |
| **K16** | Wrong comment: saves and blocks both "Type 17" | **Fixed** | Comment now states keeper saves = Type 57 (defending team), outfield blocks = Type 17, goals = Type 0, and that the relationship to Type 12 is unverified. |
| **K17** | No automated tests | **Partial** | Three pure-function tests for the I1 parser change, run as plain Python (pytest absent in the authoring sandbox) [verified pass]. Their fixture is **synthetic**, not a captured payload. Nothing else is tested. |

### 3.2 Owner-requested improvements (I1–I3)

**I1 — penalty-shootout kicks excluded from the goal tally. DONE; deployed [owner-reported].**
- Owner confirmed **`Period == 11` is a penalty shootout** (this removes the "working hypothesis" status PM-1 gave it).
- `soccer_agents.py`: `_SHOOTOUT_PERIOD_CODE = 11` and `_is_shootout_period()` (accepts an int or numeric string); `parse_match_lineup_data()` skips such goals. The shootout result continues to come from `home_penalty_score`/`away_penalty_score`. Raw payloads still hold every kick.
- Implementation = PM-1's **Option A** (skip at parse; no schema change). Header scorer lines, pitch ⚽ badges and assist counts all read `soccer_goals`, so no template/router change was needed.
- Existing matches heal on the next DAG run: `soccer_goals` is rebuilt (`DELETE`+`INSERT`) from the stored raw payload with **no new FIFA call**.
- **Not yet verified on real data.** The reference match (LDU vs Palmeiras) must be checked by team names + date, not by surrogate id (PM-1 cited 3059; the owner later cited 3097 — ids may differ after the clean-slate reload). The real payload for that match has **never been reviewed** by the authors.

**I2 — shots definition. NOT STARTED.** Still blocked on the question PM-1 posed: whether a saved shot (`Type 57`) already appears as a `Type 12` event (via qualifier `233`). See §8.

**I3 — formation placement templates. NOT STARTED.** See §8.

### 3.3 PM-1 §9 cleanup items

| PM-1 § | Item | Disposition | Notes |
|---|---|---|---|
| 9.1 | Migration docstrings | **Decided: leave as-is** (owner's call; the author agreed) | They record what was true when each migration was written; Alembic never reads them. The one stale claim (`s0cc3r_d0ma1n001` says the DAG's `_SEED_COMPETITIONS` also seeds the table — that code is gone) is **left deliberately**. Do **not** edit applied migrations' code. |
| 9.2 | `models.py` references | **Partly done** by script; remainder in §10.2 | See §10. |
| 9.3 | Narrative references in other files | DAG + agents **done**; router/template/helpers by script | Remaining stale text in the agents file is listed in §10.4. |
| 9.4 | Mapper registration / `env.py` | **Done (delivered); application not confirmed** | Confirmed from the supplied `env.py` and `database.py`: `env.py` imports only `core.base_model.Base`; `database.py` has no model-import block ⇒ `target_metadata` is empty ⇒ `--autogenerate` would propose dropping every table. `core/model_registry.py` + one import line in `env.py` fix this. GOVERNANCE §2.4's claim that `database.py` holds the registration is stale and is corrected by script. |
| 9.5 | Legacy shim check | Owner | |
| 9.6 | DAG details | K1 done; `details_fetched_at` **kept** (informational, not dropped — dropping needs a migration); K11/K14 open | |
| 9.7 | Templates | CSS extracted; placeholder text removed | |
| 9.8 | Agents comments | K16 done; shootout constant added; field-name caveat trimmed to a factual section; I2 documentation pending | |
| 9.9 | GOVERNANCE amendments | Delivered as a script (§2.8 ingest rules, `soccer/` DAG folder, amendment-log entry, §2.4 correction); **not applied** by the authors (the document was not available on disk and is too large to retype) | |
| 9.10 | Artefacts not for the repo | Owner to confirm | |
| 9.11 | Separate ticket for the `env.py` import fix (C14) | Owner | |
| 9.12 | Tests | Partial (K17) | |
| 9.13 | Final regression | Owner | |

---

## 4. Changes agreed after the initial migration

"Initial migration" = `s0cc3r_d0ma1n001`, the first migration written for this domain. This section is the **complete** list, merging PM-1's C1–C15 with this round's additions, so a reviewer needs only this document.

Classes: **A** scope addition (owner request); **B** defect fix; **C** design reversal; **D** proposed and declined/deferred; **Op** operational.

### 4.1 Carried over from PM-1 (C1–C15, unchanged)

| # | Change | Class |
|---|---|---|
| C1 | Settings page `/soccer/settings` (ingest window, watch-list toggle/remove, live FIFA competition search cached 6 h, "+ Add"; removal blocked when matches exist) | A |
| C2 | Backfill date moved into the UI; DAG's hard-coded `_SEED_COMPETITIONS` **removed**; `_needs_backfill()` compares `backfill_from_date` to the earliest match on file every run | A + C |
| C3 | Fixtures filters: competition, status, since (default yesterday), until (default tomorrow), "Show full history" sentinel dates | A |
| C4 | Sidebar highlight: settings page passes `active_module="soccer_settings"` | B |
| C5 | MatchStatus map replaced with the owner's authoritative table (`0` finished, `1` scheduled, `3` finished/penalties, `7` postponed, `9` finished/forfeit; else `unknown`) | B |
| C6 | Parser rewritten against real payloads (`Home`/`Away` on the calendar, locale-list names, `_scalar_or_none`, per-match try/except) | B |
| C7 | Match detail page rebuilt from a raw-JSON dump into a rendered page (crests, scorers, pitch diagrams, badges, subs, possession) | A |
| C8 | Normalized detail tables (lineups, goals, bookings, substitutions, coaches) + formation/possession/attendance columns; task 3 | A |
| C9 | Team IDs on `soccer_matches` (`fifa_home_team_id`/`fifa_away_team_id`) — originally for crest URLs; **their purpose is superseded by C22** (they remain, and are still required for event-stat attribution) | A |
| C10 | Detail-fetch policy reversed twice; final: everything in the ingestion window is refetched and fully overwritten every run | C |
| C11 | Raw-payload pruning, 3 newest snapshots per (endpoint, competition, match) | A |
| C12 | Event-derived shots/corners/fouls/offsides; penalty-shootout score columns | A |
| C13 | Formation row splitting from the `Tactics` string | B |
| C14 | `alembic/env.py` import fix (`from models import Base` → `from core.base_model import Base`) — **out of scope; ticket separately** | B (out of scope) |
| C15 | Clean-slate decision: wipe soccer data, keep schema, re-pull — **execution never confirmed** | Op |

### 4.2 Added in this round (C16–C27)

| # | Change | Class | Where | Migration |
|---|---|---|---|---|
| **C16** | **K1 fix.** Details fetch limited to a 1-day lookahead (§3.1 K1). Behavioural consequences: lineups published > 1 day before kickoff are picked up one day before the match; NULL-kickoff matches are skipped | B | DAG | none |
| **C17** | **I1.** Shootout kicks (`Period == 11`, owner-confirmed) excluded from `soccer_goals` | A | `soccer_agents.py` | none |
| **C18** | **K16.** Comment corrected (Type 57 saves / Type 17 blocks / Type 0 goals) | B | `soccer_agents.py` | none |
| **C19** | **§9.1 decision:** migration docstrings are history; left alone | Decision | — | none |
| **C20** | **K15.** `nullable=False` on four timestamp columns to match `s0cc3r_d0ma1n001` | B | `models.py` | none (models only) |
| **C21** | **Model registry.** `core/model_registry.py` (one explicit import per domain) imported from `alembic/env.py`; GOVERNANCE §2.4 stale claim corrected. Candidly low-urgency: it only matters for `alembic revision --autogenerate` or standalone scripts importing a single domain — the running app registers mappers through the routers, and all migrations so far are hand-written. Kept because it turns a possible drop-everything autogenerate into a non-event | A (hygiene) | `core/`, `alembic/env.py` | none |
| **C22** | **Crest images use FIFA's own `PictureUrl`.** Owner found the made-up `IdTeam` URL was the cause of failing crests. `HomeTeam.PictureUrl`/`AwayTeam.PictureUrl` (from `/live`) stored verbatim in `home_team_picture_url`/`away_team_picture_url` (`VARCHAR(255)`, nullable); the literal `{format}`/`{size}` placeholders are filled at render time (`CREST_FORMAT="sq"`, `CREST_SIZE="4"`, **[UNVERIFIED]**). No `PictureUrl` ⇒ coloured-initials fallback (no guessed URL). Observed payload: `…/picture/flags-{format}-{size}/KSA` — for national teams this is a **flag**, not a crest. Columns fill when task 3 parses a match's `/live` data | B | agents, DAG, `models.py`, `lineup_helpers.py`, router | **`s0cc3r_cr3sts001`** (down_revision `expl0r3r_d0ma1ns001`; **unapplied**) |
| **C23** | **CSS to the domain static folder.** The inline `<style>` of `soccer_match_detail.html` moved to `domains/soccer/static/css/soccer.css` (served at `/static/soccer/css/soccer.css`), matching the pattern in `domains/media/static/css/media.css`; `/static/soccer` mount added in `main.py` **before** the general `/static` mount (GOVERNANCE §2.6) | A | template, `main.py`, new CSS | none |
| **C24** | **"View on FIFA ↗" button** in the match page's top bar, built from the four stored FIFA ids. URL template `https://www.fifa.com/en/match-centre/match/{competition}/{season}/{stage}/{match}` is **[UNVERIFIED]** (written from memory; a web search could not confirm it) | A | `lineup_helpers.py`, router, template | none |
| **C25** | `details_fetched_at` **kept** as an informational "last refreshed" column (not dropped) | Decision | — | none |
| **C26** | Cleanup of narrative/"this conversation" wording across DAG, agents, models, router, template (the content of PM-1 §9.2/§9.3) | B | several | none |
| **C27** | Tooling added for the owner's repo: `alembic_chain_check.py`, `check_model_registry.py`, `s9_repo_cleanup.py`, `s94_model_registry.py`, `s_crests_css_link.py`. **These are helpers, not part of the delivery** (see §10.8) | Op | outside the app | none |

### 4.3 Proposed and declined / deferred (D) — **not** part of this WO

| Item | Disposition |
|---|---|
| Manual "Run backfill" button, "backfill for all" toggle, separate backfill DAG | **Declined by the owner** (existing edit-the-date strategy is sufficient) |
| Dashboard integration (`routers/dashboard.py`) | Deferred (future fast-follow). The **sidebar** entry and Settings sub-nav already exist (`sidebar.html`) — see §10.3 for the stale docstring that says otherwise |
| AI match recap; team kit colours; standings endpoints | Not built |
| xG / big chances / passes / duels / saves | Not available as confirmed FIFA fields |
| Shots on target | Not built — see I2 |
| Reading `PictureUrl` from the **calendar** endpoint (so scheduled matches get a crest earlier) | Not done: whether the calendar `Home`/`Away` objects carry it is **unseen** |

### 4.4 Scope statement for the reviewer
**WO#34 as delivered = original WO (§1) + C1–C13 (PM-1) + C16–C26 (this round).** C14 is an out-of-scope fix to be ticketed on its own; C15 is operational; C27 is tooling, not delivery; §4.3 items are not part of this WO and must not be held against it.

---

## 5. Final state of the system

### 5.1 Data flow (unchanged in shape)

```
FIFA public API (no key)
  calendar ──► ingest_fixtures ──► soccer_matches (upsert) + soccer_raw_payloads('matches')
  /live, /timelines ──► ingest_match_details ──► soccer_raw_payloads('match_details','match_events')
      scope: kickoff between window start and TODAY + DETAIL_LOOKAHEAD_DAYS (end of day),
             OR never fetched and kickoff <= that upper bound            [changed by C16]
  soccer_raw_payloads ──► parse_finished_match_details ──► lineups/goals/bookings/subs/coaches
        + scalar columns on soccer_matches (formation, possession, attendance, penalty scores,
          team picture URLs [C22], shots/corners/fouls/offsides)
        goals exclude Period 11 [C17]
  prune_raw_payloads ──► newest 3 snapshots per (endpoint, competition, match)
FastAPI: /soccer, /soccer/match/{id} (+ "View on FIFA" [C24]), /soccer/settings
```

### 5.2 Tables (9, unchanged) and their columns added since PM-1
`soccer_competitions`, `soccer_settings`, `soccer_matches`, `soccer_raw_payloads`, `soccer_match_lineups`, `soccer_goals`, `soccer_bookings`, `soccer_substitutions`, `soccer_coaches`.
New column pair on `soccer_matches` (migration `s0cc3r_cr3sts001`, unapplied): `home_team_picture_url`, `away_team_picture_url`. No new tables. The explorer's one-time table→domain seed (`expl0r3r_d0ma1ns001`, 69 tables) already lists all nine soccer tables; **new tables created later appear as "unassigned"** until assigned on `/explorer/settings`.

### 5.3 Migration chain

Soccer-relevant stretch **as it exists on the owner's disk** [owner-reported via `alembic_chain_check.py`]:

```
… → j0b_sc0ut_upgr4de001 → s0cc3r_d0ma1n001 → nb4_d0ma1n001 → m3d1um_d0ma1n001 → m3d1um_d0ma1n002
  → s0cc3r_l1n3ups001 → m3d1um_d0ma1n003 → s0cc3r_t34m1ds001 → m3d1um_d0ma1n004
  → s0cc3r_st4ts001 → expl0r3r_d0ma1ns001  (DB head)
  → s0cc3r_cr3sts001   (NEW — delivered, not applied)
```
The first four soccer migrations and the explorer migration are unchanged. `s0cc3r_cr3sts001`'s `down_revision` was written against the head at the time; **re-run `alembic heads` before applying** (this chain has drifted repeatedly while migrations waited — see PM-1 §5).

### 5.4 Files touched in this round (all under `internal_dataplayground/`)

| File | Change | Size after (verified) |
|---|---|---|
| `airflow/dags/soccer/life_os_soccer_ingest.py` | C16, C26, writes picture URLs | — |
| `airflow/agents/soccer_agents.py` | C17, C18, C22 (reads `PictureUrl`), C26 | — |
| `domains/soccer/models.py` | C20, C22 columns, C26 | 368 lines |
| `domains/soccer/lineup_helpers.py` | C22 (`build_crest_url` now resolves a template), C24, typo fix | 213 lines |
| `domains/soccer/routers/soccer.py` | passes crest + FIFA URLs; docstring cleanup | 204 lines (< 300) |
| `domains/soccer/templates/soccer_match_detail.html` | stylesheet link, FIFA button, cleanup | 216 lines |
| `domains/soccer/static/css/soccer.css` | **new** (extracted CSS) | 69 lines |
| `main.py` | `/static/soccer` mount before `/static` | — |
| `core/model_registry.py` | **new** | — |
| `alembic/env.py` | one added import | — |
| `alembic/versions/s0cc3r_cr3sts001_add_team_picture_urls.py` | **new** | — |

`soccer_settings.py` (261 lines, K8) and the four earlier soccer migrations were **not** modified.

### 5.5 Process notes (errors made while producing this round, all caught before delivery)
Recorded so the reviewer can judge confidence in the delivery: (1) K1 was initially blocked because the DAG file had not been supplied — no patch was written from the postmortem's description alone; (2) the chain checker's first version over-reported forks as problems (fixed); (3) the crest/CSS script had an idempotency bug (a second `--apply` would have duplicated edits; fixed and re-tested); (4) while rebuilding `lineup_helpers.py` for the zip, the `assist_count` line was mistyped and then restored from the original; (5) comments written by the cleanup point at a postmortem filename that was **invented** (`migration_docs/Work Orders/work_order_34_soccer_domain_postmortem.md`) — see §10.8. Files that were retyped (not on disk) are `models.py`, `lineup_helpers.py`, `soccer.py`, `soccer_match_detail.html`, `main.py`: **diff them against the repo before overwriting**; any difference outside the changes listed in §5.4 means either a transcription error or that the repo has moved on.

---

## 6. Reviewer guide

### 6.1 Acceptance criteria — final assessment

| # | Criterion | Final | Basis |
|---|---|---|---|
| 1 | Step 1 before code | ✅ | PM-1 §6.1. |
| 2 | Code under `domains/soccer/` | ✅ (read with Step 2.4) | Domain code, templates and — new this round — **static CSS** live under `domains/soccer/`; DAG and agent under `airflow/` per Step 2.4 and the `job_ats_agents.py` precedent; migrations under `alembic/versions/`. |
| 3 | DAG import boundary | ✅ | The DAG imports only `dag_db` and `agents.soccer_agents` plus stdlib/Airflow; `soccer_agents.py` imports no ORM/router/service. The new `PictureUrl` and shootout code add no imports. The FastAPI-side `soccer_settings.py` importing `airflow.agents.soccer_agents` has precedent in `recipe_service.py`. **Re-verify with the grep in §6.2 step 5.** |
| 4 | AI via `services/ai/` | ✅ | No AI calls in this domain. |
| 5 | Registration / templating / **static mount order** | ✅ (static now applies) | `/static/soccer` is mounted after `/static/medium` and **before** the general `/static` mount (verified in the rebuilt `main.py`); templates path was already registered. Owner-reported working at PM-1 time; re-verify §6.2 step 8. |
| 6 | Routers < 300 lines | ✅ | `soccer.py` 204, `soccer_settings.py` 261 [verified/PM-1]. `soccer_settings.py` is past GOVERNANCE's 250 "split soon" mark (K8). |
| 7 | Design decisions documented / WO#33 alignment | ⚠️ | Decisions are documented in PM-1 and here. **WO#33 (NBA) alignment was never performed** (NBA's design was unavailable; `nb4_d0ma1n001` now exists). A diff pass on ID format, table naming and season/date handling is still owed. |
| 8 | Functional smoke test | ⚠️ | Owner-run evidence exists (pages rendered, World Cup test ingested, defects found and fixed; DB at head). **No formal smoke-test record**, and nothing in this round has been run against the database. §6.2 closes it. |

### 6.2 Verification checklist (run in order, from `internal_dataplayground/`; DB commands in the web container)

1. **Chain.** `python3 alembic_chain_check.py` → `RESULT: OK`, exactly one head. After applying the new migration the head is `s0cc3r_cr3sts001`.
2. **Apply.** `docker compose exec web alembic upgrade head`; then `alembic current` → `s0cc3r_cr3sts001 (head)`.
3. **Schema.** `SHOW COLUMNS FROM soccer_matches` includes `home_team_picture_url`, `away_team_picture_url`, `home_penalty_score`, `away_penalty_score`, `shots_*`, `corners_*`, `fouls_*`, `offsides_*`, `fifa_home_team_id`, `fifa_away_team_id`.
4. **Registry.** `docker compose exec web python3 check_model_registry.py` → no "in the DB snapshot but NOT on Base.metadata" entries other than any table the owner knowingly leaves unmodelled; then `alembic revision --autogenerate -m scratch` → `upgrade()` empty (ignore `compare_type` noise); **delete the file, do not apply it**.
5. **Boundaries/size.** `grep -nE "^(from|import) " airflow/dags/soccer/*.py airflow/agents/soccer_agents.py` shows no `models`, `database`, `routers`, `services`. `wc -l domains/soccer/routers/*.py` all < 300.
6. **Wiring.** `grep -n "soccer" main.py core/templating.py` shows both routers, the templates loader and the `/static/soccer` mount **above** `/static`.
7. **DAG run.** Trigger `life_os_soccer_ingest`. All four tasks succeed. In the log, the "Fetched details/events for N/M candidate matches" line shows N **far below** the number of scheduled fixtures (K1). `soccer_raw_payloads` holds ≤ 3 rows per (endpoint, competition, match) after task 4.
8. **UI.** Open a finished match: lineups, formations, scorers, subs, possession and the four count stats render; the stylesheet loads from `/static/soccer/css/soccer.css` (browser network tab, HTTP 200); a crest or flag image loads, or the initials fallback shows; **the "View on FIFA" button opens a real match page** (if not, fix `FIFA_MATCH_URL_TMPL`).
9. **I1.** Open LDU vs Palmeiras (found by team names + date). Per side, the listed goals equal that side's score; the `(x – y pens)` line equals FIFA's penalty score; no pitch token carries a shootout ⚽ badge.
10. **Crest values.** Open one resolved `PictureUrl` in a browser. If it fails, change `CREST_FORMAT`/`CREST_SIZE` in `lineup_helpers.py` (these are **[UNVERIFIED]** defaults).

### 6.3 Conditions for passing the work order

The reviewer can pass WO#34 when: (a) §6.2 steps 1–10 are satisfied; (b) the owner confirms the zip/script delivery was applied (or the equivalent edits exist in the repo); (c) the two ⚠️ criteria (7, 8) are accepted as explained, or closed; and (d) the §8 open items are accepted as backlog rather than blockers. **The post-migration cleanup (§10) is not a pass condition for the migration itself** — it is a separate, gated task.

---

## 7. What remains for the owner (not for the implementing agent)

1. Apply the delivered files (`soccer_followups.zip`) **after diffing** the five retyped files (§5.5), or apply the exact-string scripts instead.
2. Run `alembic heads` then `alembic upgrade head` to apply `s0cc3r_cr3sts001`.
3. Run the §6.2 verification and report any failures.
4. Verify and, if needed, correct the three unverified constants in `lineup_helpers.py`: `CREST_FORMAT`, `CREST_SIZE`, `FIFA_MATCH_URL_TMPL`.
5. Run the clean-slate wipe **only after** the upgrade (so the new columns exist) if it is still wanted; PM-1 §0 notes its execution was never confirmed.
6. Ticket C14 separately; confirm PM-1 §9.10 (no mockup HTML or `soccer_domain_wo34.zip` committed).
7. Decide where the postmortem documents live and tell the cleanup agent the final path (§10.8).

---

## 8. Open items and improvement backlog

Severity: **H** fix soon; **M** soon; **L** backlog. (K-numbers as in PM-1.)

| ID | Sev | Item | Note |
|---|---|---|---|
| I2 | M | Shots definition (PM-1 §8-I2) | **First** check whether each Type 57 event points (qualifier 233 "OppositeRelatedEventId") at a Type 12 event already in the payload — it decides double-counting. Then define "shots on target" = goals (Type 0, excluding own goals and **shootout kicks**) + opposing keeper saves (Type 57), add `shots_on_target_*` columns (migration) and label the UI precisely. Reference figures (10 vs 2) are from one match and one external source: suggestive, not proof. |
| I3 | L | Formation placement templates (PM-1 §8-I3) | Unchanged. Position codes remain coarse; slot assignment inside the split rows is a heuristic. |
| K4 | M | Zeros for empty `/timelines` | Return `None`, template already hides `None`. |
| K7 | M | Crest/flag verification | Confirm resolved URLs; decide whether to label national-team images as flags. |
| K8 | M | Split `soccer_settings.py` | Watch-list CRUD vs FIFA search. |
| K9–K14 | L | Coach role fallback; card codes (code `3` seen = second yellow/red; plain red unobserved); task-3 re-parse guard; score source of truth; live status; task rename | See PM-1 §7. |
| N1 | L | Optionally read `PictureUrl` from the calendar endpoint | Only if a captured calendar payload shows the field. |
| N2 | L | WO#33 alignment pass | Criterion 7. |
| N3 | L | Real-payload fixtures for tests | Replace the synthetic I1 fixture with a trimmed real LDU–Palmeiras payload; add tests listed in PM-1 §9.12 (`parse_match_event_stats` incl. empty list, `_split_am_rows`/`build_pitch_tokens` incl. the six-player `Position: 2` Crystal Palace case, `_extract_localized_text`, the status map). |

---

## 9. Lessons (additions to PM-1 §5)

| Incident | Cause | Lesson |
|---|---|---|
| Made-up crest URL failed intermittently | URL constructed from an id instead of using the field FIFA supplies | Prefer a URL the API hands you over one you derive; keep a graceful fallback. Found by the owner reading the raw payload. |
| `PictureUrl` is a **template** | Literal `{format}`/`{size}` placeholders | Store such values verbatim and resolve at render time, so the choice of size/format stays a one-line change. |
| Alembic cycle disappeared without a recorded cause | Fixed in the repo between sessions | Record the fix when it is made; a diagnostic that prints the cycle path and each node's `down_revision` (now `alembic_chain_check.py`) shortens this to minutes. |
| Diagnostic over-reported forks | Treated every fork as an error | A fork is only a defect if it never re-converges. |
| Edit-script not idempotent | "Already applied" test failed when the new text contained the old | Check for the *new* text first. |
| Retyping files for a zip introduced an error | No on-disk copy | Where possible, patch the real file; if a file must be retyped, apply the edit scripts' exact-string checks and diff against the repo. |

---

## 10. Post-migration cleanup — specification for the implementing agent

### 10.0 When and how to run this section

**Precondition (all must hold):**
1. Every other in-flight migration (NBA, Medium, anything else) has landed.
2. `python3 alembic_chain_check.py` prints `RESULT: OK` and **one head**; `alembic current` equals that head.
3. `s0cc3r_cr3sts001` is applied; `soccer_followups.zip` (or equivalent edits) is in the repo.
4. The decisions in §10.1 have been made by the owner.

**Rules:** pause the soccer DAG while editing; make each numbered subsection its own small, reviewable change (GOVERNANCE §4.5); **do not edit the code of any applied migration** (docstring-only edits are unnecessary — see §3.3 9.1); **do not squash migrations**; do not change behaviour (comment/doc edits only, except where a subsection says otherwise); do not fix unrelated bugs inside these changes — ticket them.

**Boundaries:** the scope is `domains/soccer/`, `airflow/dags/soccer/`, `airflow/agents/soccer_agents.py`, `core/model_registry.py`, `alembic/env.py`, `migration_docs/GOVERNANCE.md` and the tooling/test files named below. Touch nothing else.

### 10.1 Decisions needed from the owner first

| # | Decision | Default if unanswered |
|---|---|---|
| D1 | Final path/filename of the postmortem documents (PM-1 and this one) | Use the actual path of this file; do not guess |
| D2 | Keep or drop `details_fetched_at` | **Keep** (a drop needs a migration; benefit is cosmetic) |
| D3 | Whether I2 (shots on target) is built before the cleanup | If yes, do §10.2 M8/M11 **after** I2 so each comment is written once |
| D4 | Where helper scripts live (§10.8) | `migration_docs/tools/` |

### 10.2 `domains/soccer/models.py` — remove stale references and rewrite history as present behaviour

Run first: `grep -nE "WO#34|conversation|mockup|ESPN|owner|_SEED_COMPETITIONS|formatted JSON|follow-up|unverified|No red-card|teams-\{format\}|Qualifier decoding" domains/soccer/models.py`. Items **M1–M12** below are the references that are stale or wrong **in the delivered file**; the cleanup scripts already rewrote the ESPN/"this conversation"/"UPDATED (date)" passages (verify they no longer match the grep).

| ID | Where | Problem | Required result |
|---|---|---|---|
| M1 | Module docstring | Says match detail/events "stay in SoccerRawPayload only, rendered as formatted JSON in the match detail page" and that normalizing them is "a follow-up in the WO#34 report Notes, not done here". **False now** — detail data is normalized into five tables and the page renders from them. Also references "WO#34 Step 1 Q4" and a non-existent "store_raw()" phrasing history | Describe the **three layers as they are**: (1) raw verbatim payloads, (2) normalized match row, (3) normalized per-match detail tables (lineups, goals, bookings, substitutions, coaches) populated by task 3 from the stored raw payloads. Keep the "FIFA ids are strings" paragraph. Replace WO references with a pointer to the postmortem (D1) |
| M2 | `SoccerCompetition` docstring | "See … `_SEED_COMPETITIONS` for what ships watched by default" — **that constant no longer exists** (C2) | "Seeded by migration `s0cc3r_d0ma1n001`; managed on `/soccer/settings`." |
| M3 | `SoccerMatch` docstring | "goes scheduled -> live -> finished" — no "live" status code has ever been confirmed (K13) | State the labels actually produced by `_MATCH_STATUS_MAP`: `scheduled`, `finished`, `postponed`, `unknown` |
| M4 | `fifa_home_team_id` / `fifa_away_team_id` comment | "added specifically to build real crest image URLs" — superseded by C22 | Say they identify teams for event-stat attribution (`parse_match_event_stats`) and for identity; crest/flag images come from the picture-URL columns |
| M5 | `fifa_match_status_code` comment | "flagged there as unverified against a live response" — the map is now authoritative for codes 0, 1, 3, 7, 9 | State that, and that any other code maps to `unknown` |
| M6 | formation/possession block | Verify the script rewrite removed "Populated once…" and the ESPN passage | Present-tense description: parsed from `/live`, refreshed on every in-window reprocess |
| M7 | penalty-score block | Verify the "this conversation" passage is gone | Shootout result; kicks are **not** in `soccer_goals` (`_SHOOTOUT_PERIOD_CODE`) |
| M8 | shots/corners/fouls/offsides comment | Says saves/blocks "would need Qualifier decoding or English-text matching" — **contradicts K16** (saves are Type 57, blocks Type 17) | Rewrite per the outcome of I2 (D3); at minimum remove the false claim |
| M9 | `details_fetched_at` comment | Verify script rewrite; must read "informational last-refreshed-at, not a gate" (D2) | |
| M10 | `backfill_from_date` comment | Verify the "UPDATED (date)" wording is gone | Plain description of the first-ingest/gap-closing behaviour |
| M11 | `SoccerBooking` docstring | "Card code 1 = yellow, confirmed. No red-card example seen yet." — code **`3`** has since been observed (second yellow/red, Crystal Palace data); plain red still unobserved | State both observations, and that the labels beyond `1` are unconfirmed (K10) |
| M12 | `SoccerCoach` docstring | "two data points" for `role_code` | Update to the evidence available when this is written (K9); do not state more than the data supports |
| M13 | Nullability (K15) | The four timestamp columns must carry `nullable=False` | `grep -n "nullable=False, server_default" domains/soccer/models.py` shows `created_at`, `fetched_at` and **two** `updated_at` lines |
| M14 | New picture-URL columns | Comment must say: FIFA's template with literal `{format}`/`{size}`; flag for national teams; NULL until the match's `/live` data is parsed | |
| M15 | Any column added after this round (e.g. I2's `shots_on_target_*`) | Add and document | |
| M16 | All "WO#34 report"-style pointers | Replace with the real postmortem path (D1) | |

**Acceptance for §10.2:** the grep above returns no stale hit; `python3 -c "import domains.soccer.models"` succeeds; the §10.7 autogenerate check shows an empty `upgrade()` for the soccer tables.

### 10.3 `domains/soccer/routers/soccer.py` and templates

- **R1 (stale, found while writing this document):** the module docstring says the router is "deliberately NOT wired into routers/dashboard.py's cross-domain summary **or the sidebar's Modules nav** in this pass". The sidebar **is** wired (`templates/partials/sidebar.html` has the Soccer entry and a Settings sub-nav). Correct to: dashboard not wired (deferred); sidebar wired.
- **R2:** confirm the stats paragraph the scripts inserted lists exactly the stats that are real (score, penalty score, possession, formations, derived shots/corners/fouls/offsides), updated if I2 lands.
- **T1:** `soccer_match_detail.html` — `grep -niE "mockup|placeholder|conversation|ESPN"` returns nothing; no user-visible text refers to earlier mockups; no "Future phase: AI recap" placeholder is shown to users.
- **T2:** the stylesheet is linked, not inline, and `soccer.css` has no Jinja (it is static).
- **T3:** after I3 (if built), re-check label overlap at the standard width and at the mobile breakpoint in `static/css/mobile.css`.

### 10.4 `airflow/agents/soccer_agents.py` and the DAG

**Agents file — stale text that the earlier cleanup did **not** cover:**
- **A1:** the long docstring of `_extract_localized_text()` ("CONFIRMED BUG #1 … BUG #2 … inferred from behavior, not from a captured sample — if it's still wrong, pull one raw match object …") is incident narrative. The calendar's real shape (`Home`/`Away`, name under a `TeamName` locale list) has since been confirmed; reduce to a factual description of the shapes handled, keeping the `str`/`dict`/locale-list branches.
- **A2:** `parse_match_lineup_data()` docstring says it is "intended to be called only once a match is confirmed 'finished'". **Stale** (C10: any in-window match). Fix.
- **A3:** the section comment above the `/live` parser says it is "built against a real, fully-populated, finished match (Bournemouth 2-2 Brentford, 2026-09-12)". Keep the facts (which codes are confirmed), drop the session framing.
- **A4:** after I2 (D3), document the exact definitions for Types 0, 12, 15, 16, 17, 18 and 57 in `parse_match_event_stats()`.
- **A5:** confirm `_SHOOTOUT_PERIOD_CODE` comment states the owner-confirmed provenance and nothing speculative.

**DAG file:**
- **G1:** the "Refresh policy" docstring contains a pointer to the postmortem filename that was **invented** (`migration_docs/Work Orders/work_order_34_soccer_domain_postmortem.md`). Replace with the real path (D1).
- **G2:** `grep -n "_SEED_COMPETITIONS"` must return nothing in executable code; any docstring mention must say seeding moved to the migration.
- **G3:** decide K11 (add a guard so a match whose payload has an empty `Players` array is not re-parsed every run) and K14 (leave the task name; add a docstring note that it processes in-window matches) — **only with the owner's go-ahead**, since either changes DAG behaviour or history.
- **G4:** confirm `details_fetched_at` handling matches D2.

**Acceptance:** `grep -rniE "conversation|mockup|ESPN|project owner|the owner|this session|screenshot" airflow/dags/soccer airflow/agents/soccer_agents.py` returns nothing (legitimate uses of "watched competition" are not matches for these words); both files compile; the three I1 tests still pass.

### 10.5 Mapper registration and `env.py` (PM-1 §9.4 — verify, do not redo)

- `core/model_registry.py` lists one `import domains.<name>.models` per domain. Run `check_model_registry.py` in the web container; for each domain whose `models.py` does not exist (the explorer line is deliberately a comment), remove the import rather than creating an empty module.
- `alembic/env.py` imports the registry **once**.
- `database.py` is **not** where registration lives; GOVERNANCE §2.4 must say so (see §10.6).
- **Acceptance:** `alembic revision --autogenerate -m scratch` yields an empty `upgrade()`; delete the file.

### 10.6 GOVERNANCE.md amendments (PM-1 §9.9)

Apply (the exact-string edits are in `s9_repo_cleanup.py`/`s94_model_registry.py`, or make them by hand): §2.4 correction (registration lives in `core/model_registry.py`); §2.5 note that `nba/`, `medium/` and `soccer/` DAG folders were created directly; **new §2.8 "Ingest & Parser Rules"** — (a) validate parser field names against a real captured payload before writing the parser, (b) enum/status maps need an authoritative source, (c) never lock a record on a single snapshot — prefer window-based refresh-and-overwrite, (d) isolate failures per record in batch ingestion, (e) bound raw-payload tables and bound fetch scope from day one, (f) run `alembic heads` first and migrate before shipping models that need new columns; §3.3 note that soccer was built greenfield (not migration debt); amendment-log entry. **Acceptance:** the document renders, each amendment appears once, and no other section changed.

### 10.7 Alembic hygiene

1. `python3 alembic_chain_check.py` → OK, one head (re-run after every other domain's migration lands).
2. Record the final soccer revision ids in GOVERNANCE (`s0cc3r_d0ma1n001`, `s0cc3r_l1n3ups001`, `s0cc3r_t34m1ds001`, `s0cc3r_st4ts001`, `s0cc3r_cr3sts001`).
3. **Do not** hand-edit applied migrations (§3.3 9.1). If the explorer table seed (`expl0r3r_d0ma1ns001`) is ever re-seeded, new soccer tables created by I2 must be added through `/explorer/settings`, not by editing that migration.
4. Ticket C14 (the original `env.py` import fix) separately if not already done.

### 10.8 Repository hygiene (PM-1 §9.10, §9.11)

- The helper scripts produced during this work are **not part of the application**: `alembic_chain_check.py`, `check_model_registry.py` (keep both under the location chosen in D4 if the owner wants them), and `s9_repo_cleanup.py`, `s94_model_registry.py`, `s_crests_css_link.py` (one-shot edit scripts — delete after the edits are in the repo). None of them should sit at the repository root.
- Confirm no mockup HTML files and no `soccer_domain_wo34.zip` / `soccer_followups.zip` were committed.
- **Postmortem path:** pick the final location for PM-1 and PM-2 (D1) and update every in-code pointer (G1, M16, M1). Search: `grep -rn "work_order_34" .`.
- Tests: place the I1 tests where the repo keeps tests (none was observed; if there is no tests folder, ask before creating one). Replace the synthetic fixture with a trimmed real payload (N3).

### 10.9 Final regression and sign-off

1. Repeat §6.2 end to end.
2. Re-open LDU vs Palmeiras, a 4-2-3-1 match and a 3-4-2-1 match; confirm header scorers, badges, penalty line, stats, crests/flags and pitch spread.
3. Confirm the greps in §10.2–§10.4 are clean and the autogenerate check is empty.
4. Report: files edited (each with the item ids it satisfied), the output of each grep, anything not done and why. Mark any item that could not be verified ⚠️, not ✅.

### 10.10 Things the implementing agent must NOT do

- Drop `details_fetched_at`, rename the task `parse_finished_match_details`, or change any DAG/query behaviour without the owner's explicit go-ahead.
- Edit the code of an applied migration, or rebase a migration's `down_revision` without running `alembic heads` and the chain check first.
- Treat `CREST_FORMAT`, `CREST_SIZE` or `FIFA_MATCH_URL_TMPL` as verified.
- Present any I1 behaviour on real data as verified before the §6.2 step 9 check has been run.
- Bundle pre-existing-bug fixes (K4, K8–K14) into cleanup commits.

---

## Appendix A — FIFA payload notes added this round

- **Team object (`/live` `HomeTeam`/`AwayTeam`, from a screenshot of the owner's payload):** `Score`, `Side` (null), `IdTeam` (string), `PictureUrl` (template, e.g. `https://api.fifa.com/api/v3/picture/flags-{format}-{size}/KSA`), `IdCountry`, `TeamType`, `AgeType`, `Tactics` (e.g. `"4-3-3"`), `TeamName` (locale list), `Abbreviation`, `Coaches[]`.
- **Coach object:** `IdCoach`, `IdCountry`, `PictureUrl` (null in the sample), `Name` and `Alias` (locale lists), `Role` (**`1`** in the sample — an assistant), `SpecialStatus` (null).
- **Period `11`** = penalty shootout (owner-confirmed). Other period codes in PM-1 Appendix A are unchanged; extra-time codes remain unobserved.
- **Still unconfirmed:** valid `{format}`/`{size}` values for the picture endpoints; whether club teams carry a crest-type `PictureUrl`; whether the calendar `Home`/`Away` objects carry `PictureUrl`; the fifa.com match-centre URL shape.

## Appendix B — Delivery artefacts of this round
`soccer_followups.zip` (11 files + `APPLY_README.md`, overlay for `internal_dataplayground/`); `K1.patch`, `I1.patch`, `S9_dag.patch`, `S9_agents.patch` (earlier, partly superseded by the zip); `s9_repo_cleanup.py` (`--apply`, `--nullable`), `s94_model_registry.py`, `s_crests_css_link.py`, `alembic_chain_check.py`, `check_model_registry.py`, `test_soccer_shootout.py`.
