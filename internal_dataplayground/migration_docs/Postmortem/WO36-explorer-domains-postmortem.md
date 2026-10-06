# WO#36 — Explorer Table → Domain Grouping: Post-Mortem

**Status:** Delivered; awaiting reviewer sign-off. Follow-up cleanup (Part D) is **blocked** until all other in-flight migrations are applied.
**Work order number:** assumed `#36` (the next after #35 Medium). Code comments in `main.py` use `WO#36` / `WO36`. Rename if the project's index differs.
**Domain:** `explorer` (`domains/explorer/`)
**Suggested repo location:** `migration_docs/explorer-domains-postmortem.md`
**Document date:** 2026-10-02
**Audience:** (1) the reviewer deciding whether to pass this work; (2) a future coding agent that will execute the Part D follow-up and must derive its requirements from this file alone.

---

## 0. Reviewer quick-read

This work was **delivered in three rounds**, not one. The reviewer must judge it against the *final agreed state*, not against the first delivery. The differences matter:

| Round | What it was | Triggered by |
|---|---|---|
| **Delivery 1** | The initial work order: inline code for every file, pasted into the conversation | Owner's original request |
| **Delivery 2** | Complete files in a ZIP (`explorer_domains.zip`, v1) with a real migration chained to the latest head | Owner's second message: asked for a ZIP, supplied real source for verification, supplied the corrected table list |
| **Delivery 3** | ZIP v2: visual adjustments + stylesheet cache-busting | Owner's third message (screenshots) |

**Final agreed state in one paragraph.** A new table `explorer_table_domains` (table name → free-text domain) is created and seeded by Alembic revision `expl0r3r_d0ma1ns001` (parent `s0cc3r_st4ts001`) with the owner's **final** 69-table assignment plus the new table itself (70 rows, 15 domains). `GET /explorer/schema` returns a `domain` per table. The SQL Explorer sidebar is a collapsible **domain → table → column** tree, collapsed by default, with any table lacking an assignment shown in an **"unassigned"** group (last, yellow). A new page, `/explorer/settings`, assigns tables to domains (inline edit, bulk assign, filters). New tables need no code change: they appear automatically as unassigned.

**One thing the reviewer must know up front:** the owner reported that the visual problems seen after Delivery 2 were caused by a **browser-cached `explorer.css`**, and that "the only change needed was really the settings page." So the sidebar-CSS changes in Delivery 3 are **not confirmed adopted** (see §3.4 and §4). Both variants are acceptable; the reviewer should verify whichever one is deployed.

**Pass criteria** are in §6. In short: AC-1 to AC-9 and AC-11 must be ✅; every ⚠️ is an item that could not be exercised outside the owner's environment and must be acknowledged, not silently accepted.

---

## 1. Problem statement and objective

The Life OS MariaDB database (`jobs` schema) grew to **69 tables across 14 domains**, all in one database. The SQL Explorer's table browser listed them in one flat, alphabetical list, which made it hard to find anything.

**Owner's requirements, as originally stated:**
1. Group the explorer's tables by domain.
2. Default to domains **collapsed** (individual tables hidden).
3. Expanding a domain shows its tables; expanding a table shows its columns (existing behavior).
4. A **settings page** where each table in the database (the set returned by `SHOW TABLES;`) is assigned to a domain.

**Requirements added or confirmed in the second message:**
5. New tables added to the database later must be handled — and must land in an **"unassigned"** category if not assigned.
6. Deliver **all complete files in a ZIP**.
7. Verify model registration against a real domain (`code_intel`) rather than assuming.
8. Chain the migration after the real latest Alembic revision (the soccer stats migration file).
9. Use the owner's **updated** table → domain list (three tables moved; see §2.4).

**Requirements added in the third message (visual):**
10. More padding between the panel edge and the domain icon.
11. Domain header should look clickable (hand cursor).
12. Domain table-count right-aligned, like the tables' row counts.
13. The filter controls on the settings page look wrong in the dark theme.

---

## 2. Part A — The initial work order (Delivery 1)

### 2.1 Design decisions (and why)

| # | Decision | Rationale |
|---|---|---|
| D-1 | Store the mapping in a **new table** `explorer_table_domains(table_name PK, domain)` rather than hard-coding it or deriving it from SQLAlchemy model modules | The owner wants to edit assignments in a UI; not every table has a model (e.g. `alembic_version`, Airflow-owned tables); DAG-written tables are ORM-defined but the explorer must also see anything created by manual DDL |
| D-2 | **No foreign keys**; `domain` is free text (not an enum or a `domains` table) | Tables are created/dropped outside this app. A stale row must be harmless; a missing row must degrade to "unassigned". Free text means a new domain needs no migration |
| D-3 | **Unassigned is the absence of a row**, not a stored value | Makes "new table appears automatically" true by construction — nothing has to run when a table is created |
| D-4 | Table list is read live from `information_schema` on every request | Same reason as D-3. Never hard-code the table list |
| D-5 | Separate router file `explorer_settings.py` | `CONTRIBUTING.md`: "Every new module gets its own router file — no adding endpoints to existing routers" |
| D-6 | HTMX partials under `templates/partials/`; settings page extends `base.html` | `CONTRIBUTING.md` rules |
| D-7 | `HIDDEN_TABLES` moved from `explorer.py` into a new shared module `table_domains.py` | The settings router needs the same list; one router must not import another router |
| D-8 | Domain input normalized server-side to `[a-z0-9_]{1,50}`; blank or "unassigned" clears | Prevents junk domains and makes "clear" a first-class action |
| D-9 | No DAG involvement | Respects the DAG/FastAPI boundary rule; nothing here is written by Airflow |
| D-10 | Collapsed-by-default tree state lives client-side (`openDomains` set) | Not persisted; the owner asked for "default collapsed", not "remember state" |

### 2.2 Files — Delivery 1

**Created**

| File | Purpose |
|---|---|
| `domains/explorer/models.py` | `ExplorerTableDomain` ORM model |
| `domains/explorer/table_domains.py` | `list_tables()`, `get_domain_map()`, `UNASSIGNED`, `HIDDEN_TABLES` |
| `domains/explorer/routers/explorer_settings.py` | `GET /explorer/settings`, `POST /assign`, `POST /bulk-assign` |
| `domains/explorer/templates/explorer_settings.html` | Settings page |
| `domains/explorer/templates/partials/explorer_settings_row.html` | One `<tr>` (returned by `/assign`) |
| `domains/explorer/templates/partials/explorer_settings_rows.html` | All rows (returned by `/bulk-assign`) |
| `domains/explorer/static/css/explorer_settings.css` | Settings page styles |
| Alembic migration | Create + seed the table. **Delivery 1 gave only the body**; the owner was told to run `alembic revision -m "add explorer_table_domains"` and paste it in, because the real revision chain was not yet visible |

**Edited**

| File | Change |
|---|---|
| `domains/explorer/routers/explorer.py` | Removed the local `HIDDEN_TABLES`; `get_schema` now returns `domain` per table (see §2.5 for the extra change bundled here) |
| `domains/explorer/templates/explorer.html` | Domain-grouped tree JS; "Expand all / Collapse all"; topbar link to settings; mobile strip became a domain `<select>` plus chips |
| `domains/explorer/static/css/explorer.css` | Appended domain-group styles |
| `main.py` | Import and `include_router(explorer_settings.router)` |

### 2.3 Behavior as first delivered
- Sidebar: domains collapsed by default; "unassigned" sorted last; searching auto-expands matching domains; matching on table name **or** domain name.
- Settings page: unassigned rows first; text filter + domain filter; row checkboxes with a bulk-assign bar; inline edit saves on `change`.
- Fallback: if `explorer_table_domains` does not exist, `get_domain_map()` returns `{}` and everything shows as unassigned (the explorer keeps working before the migration is applied).

### 2.4 Initial seed and the three disputed assignments

Delivery 1 seeded the **owner's first list**. Three entries in that list differed from where the code defines the model, and Delivery 1 flagged this explicitly:

| Table | Owner's first list | Where the model lives in code |
|---|---|---|
| `application_logs` | `life_os` | `domains/jobs/models.py` |
| `shopping_lists` | `recipes` | `domains/planning/models.py` |
| `weekly_syntheses` | `planning` | `domains/journal/models.py` |

The owner then **adopted the code's grouping** in the updated list (§3.1). This is the single most important data change between Delivery 1 and the final state.

### 2.5 Changes in Delivery 1 that went beyond the literal request (disclosed at the time)

| Change | Why it was made | Revert path |
|---|---|---|
| `get_schema` rewritten from **~2 `information_schema` queries per table (≈138 queries)** to **3 queries total** | Page-load cost; the grouping feature made the schema payload the hot path | Self-contained inside `get_schema`; per GOVERNANCE §4.5 it can be reverted independently of the feature. **Not a bug fix**; it is an optimization bundled with the feature and called out as separable |

### 2.6 Assumptions made in Delivery 1 that were unverified at the time

| Assumption | Status after Delivery 2 |
|---|---|
| HTMX is loaded in `base.html` | ✅ Verified (HTMX 1.9.10, from `templates.zip`) |
| `showToast(message)` takes a single string | ⚠️ Usage in `code_intel` seen to accept `(msg, isError)`; `base.js` itself was not provided |
| `html_error(message, status_code=...)` matches the real signature | ⚠️ **Still unverified** — `routers/_helpers.py` was never provided |
| GOVERNANCE says model registration lives in a `database.py` import block, but the `database.py` provided has none | Delivery 1 **flagged the contradiction as unresolved** and asked the owner to add the import wherever other domains' models are imported (`database.py` or `env.py`). ✅ Resolved in Delivery 2: the real mechanism is router imports (see §3.2). GOVERNANCE §2.4 is the stale document |

---

## 3. Part B — What was agreed to change after the initial work order

### 3.1 Inputs the owner supplied after Delivery 1

| Input | Used for |
|---|---|
| `code_intel.zip` | Confirming how a real domain registers its models and uses HTMX/toasts |
| `templates.zip` (incl. `base.html`, `partials/error_fragment.html`, sidebar) | Confirming HTMX presence, error-fragment markup, no template-name collisions |
| `s0cc3r_st4ts001_add_stats_and_penalties.py` | The current latest Alembic head, to chain the new migration after |
| Updated table → domain list | The authoritative seed (three reassignments from §2.4) |

### 3.2 Questions the owner asked, and the answers that became requirements

| Question | Answer / agreed behavior |
|---|---|
| Does the process account for new tables? | **Yes.** Table list is live from `information_schema`; no code or migration is needed |
| How are new tables classified? | **"unassigned"**, shown last in the tree and first on the settings page |
| Can there be an unassigned category? | **Yes — it is the default state** (D-3). The topbar link also reads `⚙ Domains · N unassigned` when any exist |
| How are models registered? | **By router import.** `code_intel` has no registration block: its routers import `domains.code_intel.models` directly and `main.py` imports the routers. The explorer routers do the same with `domains.explorer.models`. **Not verified:** Alembic's `env.py` (not provided); see Part D, task D4 |

### 3.3 Change log: Delivery 1 → Delivery 2 → Delivery 3

Origin key: **OWNER** = requested/decided by the owner; **VERIF** = forced by verifying against real source or a real database; **AGENT** = agent-initiated and disclosed to the owner.

| ID | Change | Origin | Delta vs. Delivery 1 | Files |
|---|---|---|---|---|
| C-01 | Seed uses the **updated** list: `application_logs`→`jobs`, `shopping_lists`→`planning`, `weekly_syntheses`→`journal` | OWNER | Seed content | migration |
| C-02 | Migration given a real identity: revision `expl0r3r_d0ma1ns001`, `down_revision = 's0cc3r_st4ts001'`; docstring carries the "head may have moved" procedure copied from the soccer migration's own convention | OWNER (supplied head) | Delivery 1 had no revision ids | migration |
| C-03 | Seed generated **programmatically from the owner's list** and asserted to equal the real tables (70 rows) | VERIF | New | migration |
| C-04 | `_apply()` changed from a MySQL-specific `INSERT … ON DUPLICATE KEY UPDATE` to SQLAlchemy `merge()` | VERIF | Dialect-neutral; made the router testable | `explorer_settings.py` |
| C-05 | `get_domain_map()` now `rollback()`s after catching a failure | VERIF | A failed statement left the session unusable for the next query | `table_domains.py` |
| C-06 | `_apply()` returns `bool`; save failure returns a readable error ("has the explorer_table_domains migration been applied?") instead of a bare 500 | AGENT | New | `explorer_settings.py` |
| C-07 | Settings page rebuilds the domain `<select>` and the autocomplete `<datalist>` from the rows after every swap | AGENT | Removed a Delivery 1 caveat (new domains needed a reload) | `explorer_settings.html` |
| C-08 | "Saved." toast on success; server's validation message surfaced as an error toast (via `error_fragment.html`'s `.error-fragment-message`) | AGENT | New | `explorer_settings.html` |
| C-09 | While a search is active, a domain can still be closed by the user (`searchCollapsed`) | AGENT | Delivery 1: header clicks did nothing during a search | `explorer.html` |
| C-10 | Topbar link shows `⚙ Domains · N unassigned` | AGENT (supports requirement 5) | New | `explorer.html` |
| C-11 | Cosmetic: search placeholder "Search tables or domains…"; settings `<title>` no longer repeats the `Life OS —` prefix `base.html` adds; link-buttons get `text-decoration:none`; `check-all` resets after a swap; `tr[hidden]{display:none}` guard | AGENT | New | templates, CSS |
| C-12 | Deliverable became **complete files in a ZIP** incl. a full `main.py` and a README | OWNER | Delivery 1 was inline snippets | all |
| C-13 | "Unassigned" count pill turned yellow | AGENT | Delivery 1 only colored the name | `explorer.css` |
| C-14 | **Settings page, dark theme:** `<select>` popup and `<datalist>` get dark colors (`color-scheme: dark`, dark `option` background); filter input widened; the "unassigned" orange border scoped to table rows so the bulk-bar input is no longer tinted | OWNER (requirement 13) | New | `explorer_settings.css` |
| C-15 | Both stylesheet links gain `?v=2` | AGENT (response to a cached-CSS symptom) | New | `explorer.html`, `explorer_settings.html` |
| C-16 | Sidebar header tweaks: left padding 14→18px; count rendered as plain right-aligned text (no pill, same right edge as row counts); tables indented 32px and columns 50px; `.table-list` specificity prefix; hover brightens domain name | OWNER (requirements 10–12) | Visual | `explorer.css` |

### 3.4 The visual-feedback round, root cause, and what was actually adopted

The owner sent two screenshots after Delivery 2 and asked for requirements 10–13.

**Findings**
- Requirements 10, 11 and 12 (icon padding, hand cursor, right-aligned count) were **already implemented** in the CSS shipped in Delivery 1/2 (`padding:7px 14px`, `cursor:pointer`, flex row with the count pushed right). The screenshot showed unstyled domain headers: large names, the count touching the name, the icon against the panel edge.
- The agent loaded the shipped `explorer.css` in a test browser and confirmed the headers computed as `display:flex`, `cursor:pointer`, `padding-left:14px`, `font-size:10px`, `text-transform:uppercase`. The file was correct; the owner's browser was not using it.
- **Owner confirmed: the styles were cached.**
- Requirement 13 (dark-theme settings filter) was a **genuine defect**: the browser painted the `<select>` popup with light defaults, giving pale text on a white list.

**Resolution**
- Genuine fix: C-14 (settings CSS).
- Preventive: C-15 (`?v=2` cache-busting).
- C-16 was applied in ZIP v2 anyway to honor the literal requests. **The owner then stated that only the settings-page change was needed.**

**Therefore, for review:**

| Change set | Status |
|---|---|
| C-01 – C-13 | Part of the delivered baseline (Delivery 2 content) |
| C-14 (settings CSS) | **Needed and adopted** — the owner's stated final change |
| C-15 (`?v=2`) | Delivered; adoption **not confirmed** for `explorer.html` (the settings link is bundled with the settings change) |
| C-16 (sidebar header look) | Delivered in ZIP v2; **not confirmed adopted — the owner said it was unnecessary.** The agent told the owner they could keep their existing `explorer.css`. The reviewer must check which `explorer.css` is deployed and accept either (§4) |

### 3.5 Deliberately **not** changed
- DAGs, `airflow/`, `dag_db.py`.
- Any other domain's models, routers, templates or static files.
- `core/templating.py` (explorer's template directory was already registered) and the `/static/explorer` mount in `main.py` (already present).
- `base.html`, `base.css`, `base.js`, sidebar partials.
- `docker-compose.yml`, secrets/environment wiring.
- The SQL execution path in `explorer.py` (`/query`, `_validate_sql`, `BLOCKED_PATTERN`, `ROW_CAP`) — unchanged.

---

## 4. Final state — the baseline the reviewer should check

### 4.1 Files

| File | New / Edited | Status |
|---|---|---|
| `alembic/versions/expl0r3r_d0ma1ns001_add_explorer_table_domains.py` | New | Final. Parent `s0cc3r_st4ts001` (**re-verify with `alembic heads` before applying**) |
| `domains/explorer/models.py` | New | Final |
| `domains/explorer/table_domains.py` | New | Final |
| `domains/explorer/routers/explorer_settings.py` | New | Final (140 lines) |
| `domains/explorer/routers/explorer.py` | Edited | Final (210 lines) |
| `domains/explorer/templates/explorer_settings.html` | New | Final |
| `domains/explorer/templates/partials/explorer_settings_row.html` | New | Final |
| `domains/explorer/templates/partials/explorer_settings_rows.html` | New | Final |
| `domains/explorer/static/css/explorer_settings.css` | New | **Final — includes C-14 (the owner's adopted change)** |
| `domains/explorer/templates/explorer.html` | Edited | Final. Contains C-09/C-10 and `?v=2` |
| `domains/explorer/static/css/explorer.css` | Edited | ⚠️ **Two acceptable variants** (below) |
| `main.py` | Edited | Final. Two-line change only |

### 4.2 The two `explorer.css` variants

| | Variant 1 (Delivery 1/2) | Variant 2 (Delivery 3 / ZIP v2) |
|---|---|---|
| Header left padding | 14px | 18px |
| Table count | small pill (yellow when unassigned) | plain right-aligned text, same right edge as row counts |
| Indent under a domain | tables 26px, columns 44px | tables 32px, columns 50px |
| Specificity | `.domain-header` | `.table-list .domain-header` |

Neither variant changes behavior. If `explorer.css` is Variant 1, the reviewer should **not** fail the work for lacking C-16.

### 4.3 Final seed (the 70 rows)

See Appendix A. Summary: 15 domains — `blog`, `code_intel`, `explorer`, `finance`, `habits`, `jobs`, `journal`, `life_os`, `media`, `medium`, `nba`, `planning`, `recipes`, `soccer`, `workout`. `explorer` contains only `explorer_table_domains` itself. `life_os` contains only `alembic_version`, which is **seeded but hidden** (it is in `HIDDEN_TABLES`, so it never appears in the explorer or the settings page).

### 4.4 API contract (final)

| Endpoint | Method | Returns |
|---|---|---|
| `/explorer` | GET | Explorer page |
| `/explorer/schema` | GET | `{table: {columns:[{name,type,is_pk}], row_count:int, domain:str}}`; `domain` is `"unassigned"` when no mapping row exists |
| `/explorer/query` | POST | Unchanged |
| `/explorer/settings` | GET | Settings page |
| `/explorer/settings/assign` | POST (`table_name`, `domain`) | One `<tr>`; 422 invalid domain, 404 unknown/hidden table, 500 save failure |
| `/explorer/settings/bulk-assign` | POST (`table_names[]`, `domain`) | All rows; unknown names silently ignored |

All settings routes are literal paths under `/explorer/settings`, so registration order relative to other routers does not matter.

---

## 5. Verification evidence

### 5.1 Environment (a sandbox — **not** the owner's stack)

| Item | Value |
|---|---|
| Database | MariaDB 10.11.14 (real server), schema `jobs`, with **stub tables for all 69 owner tables** + one extra table `brand_new_table` to represent a future table |
| App | The real routers/templates/CSS served by uvicorn, using `asyncmy` |
| Stand-ins | `database.py` (`get_db`), `core/templating.py` (copied shape from the owner's), `core/base_model.py` (copied), **`routers/_helpers.html_error` (guessed signature)**, `base.js` (`showToast` recorder), CodeMirror (stub) |
| Real artifacts used | `base.html`, `partials/error_fragment.html`, sidebar partial (from `templates.zip`) and **real HTMX 1.9.10** |
| Pinned versions | FastAPI 0.115.12, Starlette 0.46.2 (see note in §5.4) |
| Browser engine | jsdom (with polyfills for `fetch` and `document.evaluate`) — **not a real browser** |

### 5.2 Results

**Migration (run against real MariaDB via Alembic's `Operations` API)**
- Seeded **70 rows / 15 domains**; zero real tables unmapped except the deliberate `brand_new_table`; zero mapped rows without a real table.
- `downgrade()` then `upgrade()` round-trip succeeded.

**HTTP integration (real routers + real MariaDB)**
- `/explorer/schema`: 70 visible tables (71 total, minus hidden `alembic_version`), correct domain per table, `brand_new_table` → `unassigned`.
- Assign to a **new** domain `"Sports Misc"` → normalized to `sports_misc`; schema reflects it; reassign (update path); blank clears back to unassigned.
- Invalid domain (`bad;drop`) → 422 with message; unknown table → 404; hidden table (`alembic_version`) → 404.
- Bulk assign with repeated `table_names`, one bogus name ignored; bulk clear.
- **Migration not applied** (mapping table renamed away): `/explorer/schema` → 200, all 70 unassigned; settings page → 200; save → 500 with the readable message. Table restored afterwards.
- `/explorer/query`: a valid `SELECT` returns rows; `DROP TABLE x` → 400. (Unchanged code path.)

**Settings page in jsdom with real HTMX — 17 checks, all passed:** page loads 70 rows; summary counts; datalist built; text filter; "Unassigned only"; inline edit swaps the row and normalizes the domain; the new domain appears in the `<select>` and `<datalist>` without reload; "Saved." toast; invalid edit → error toast with the server message and row unchanged; checkbox selection; bulk apply touches exactly the selected rows; selection resets after a bulk swap; bulk toast; cleanup.

**Explorer tree in jsdom — 24 checks, all passed:** 15 groups; all collapsed by default; "unassigned" last and styled; counts shown; topbar badge text; expand a domain; expand a table to its columns; editor receives the `SELECT`; column click inserts the name; collapse; Expand all (70 tables) / Collapse all; search auto-expands only matches; close/re-open a domain during search; search by domain name; no-match message; clearing search; mobile `<select>` with 15 options, chips for the chosen domain only, chip click expands the tree and fills the editor.

**CSS, with the real stylesheet loaded:** headers compute `display:flex`, `cursor:pointer`, small uppercase names, and (Variant 2) `padding-left:18px`, count `margin-left:auto`.

**Audit SQL (Appendix C):** verified on MariaDB, including orphan detection (a planted `ghost_table` row was reported, then removed).

### 5.3 Owner-side confirmation received
- Screenshot after Delivery 2 shows the explorer sidebar rendering the collapsed domain list with counts, and an expanded domain with its tables and row counts → **the tree works in the owner's real browser and database** (this is the only real-environment evidence).
- Owner confirmed the style problems were a cache issue.

### 5.4 What was **not** verified
- The owner's real database and compose stack (`app_env`, the `db` host, `MARIA_DB` secret handling).
- A real browser rendering of the **dark-theme settings dropdown fix** (C-14). The owner called the settings page the one needed change but did not explicitly confirm the dropdown renders correctly.
- The real `routers/_helpers.html_error` signature, and the real `base.js` `showToast` definition.
- Alembic's `env.py` (does it import `domains.explorer.models`?).
- `alembic heads` on the owner's repo — the parent was chosen from the single provided migration file.
- **Starlette version in the owner's environment.** The code uses the legacy `TemplateResponse(name, {"request": request, …})` form, matching the rest of the codebase. On Starlette ≥ 1.0 this form is removed. This is a project-wide convention, not specific to this work, but it is a known upgrade risk.

---

## 6. Acceptance criteria

Per GOVERNANCE §4.3 the original work order had no formal criteria; they are reconstructed here from the owner's requirements. ✅ = passed; ⚠️ = could not be fully exercised outside the owner's environment (reason given); ❌ = failed.

| # | Criterion | Result | Note |
|---|---|---|---|
| AC-1 | Migration creates `explorer_table_domains` and seeds exactly the owner's final list (70 rows) | ✅ | Real MariaDB; ⚠️ not yet run on the owner's DB |
| AC-2 | `down_revision` equals the latest head | ⚠️ | Equals `s0cc3r_st4ts001` from the provided file; `alembic heads` not run |
| AC-3 | Migration is reversible | ✅ | Round-trip tested |
| AC-4 | `/explorer/schema` returns a `domain` for every table; default `unassigned` | ✅ | |
| AC-5 | A table created later appears automatically as unassigned, with no code/migration change | ✅ | `brand_new_table` |
| AC-6 | Sidebar is domain → table → column, domains collapsed by default | ✅ | jsdom + owner screenshot |
| AC-7 | Settings page can assign, reassign, clear, and bulk-assign; invalid input rejected | ✅ | HTTP + jsdom |
| AC-8 | "Unassigned" is a visible category (last in tree, first on settings, badge on link) | ✅ | |
| AC-9 | Nothing outside scope changed (DAGs, other domains, core, base templates) | ✅ | See §3.5; the `get_schema` rewrite is the one disclosed extra (§2.5) |
| AC-10 | Models register without extra wiring | ⚠️ | Mechanism confirmed from `code_intel`; `env.py` not seen (Part D, D4) |
| AC-11 | Each router ≤ 300 lines; page extends `base.html`; partials under `partials/`; separate router file; Alembic migration for the schema change | ✅ | 210 and 140 lines |
| AC-12 | HTML errors use `html_error()` | ⚠️ | Used, but signature unverified |
| AC-13 | Settings page usable in dark theme | ⚠️ | CSS delivered and adopted; no explicit owner confirmation of the dropdown |
| AC-14 | Works on the owner's production configuration | ⚠️ | Only a sandbox was available |
| AC-15 | Sidebar header visual refinements (C-16) | ⚠️ | Delivered, **not adopted by the owner**; not a pass/fail item |

**Recommended verdict rule:** pass when AC-1 … AC-9 and AC-11 are ✅, and the reviewer has read and accepted each ⚠️ above.

---

## 7. Compliance with `CONTRIBUTING.md` and `GOVERNANCE.md`

| Rule | Source | Compliant? |
|---|---|---|
| DAGs never import models/database/routers/services | CONTRIBUTING | ✅ no DAG touched |
| Page templates extend `base.html` | CONTRIBUTING | ✅ |
| New module → its own router file | CONTRIBUTING | ✅ `explorer_settings.py` |
| Schema change → Alembic migration | CONTRIBUTING | ✅ |
| No router over 300 lines | CONTRIBUTING / §1.2 | ✅ |
| HTMX partials in `templates/partials/`, never full pages | CONTRIBUTING | ✅ |
| HTML errors via `html_error()`; JSON errors via `HTTPException` | CONTRIBUTING | ⚠️ signature unverified (AC-12); JSON path unchanged |
| One `models.py` per domain, models in `domains/<name>/` | §2.1 | ✅ — **note:** §2.1 says explorer has no `models.py`; that is now false (Part D, D5) |
| A domain's `models.py` must not import another domain's | §2.2 | ✅ — the table stores other domains' table names **as strings**, importing nothing |
| Cross-domain readers | §2.2 | ✅ no new cross-domain import |
| Static: domain mount before `/static` | §2.6 | ✅ already present; unchanged |
| Templates resolved via `ChoiceLoader`, bare filenames | §2.6 | ✅ no name collision with root `templates/` (checked against `templates.zip`) |
| Pre-existing bugs are not bundled | §4.5 | ✅ — found issues are listed in §9, not fixed |
| §4.6 "done" = no unrelated behavior changed | §4.6 | ⚠️ the `get_schema` performance rewrite is a disclosed, separable deviation |

---

## 8. What went wrong and lessons

1. **A documented mechanism contradicted the code, and it could not be resolved without more source.** GOVERNANCE §2.4 says model registration lives in a `database.py` import block; the `database.py` the owner provided has none. Delivery 1 noticed this and flagged it rather than guessing, but could only hand the question back to the owner. It was settled in Delivery 2 once a real domain (`code_intel`) was provided. *Lesson:* when a governance document and the code disagree, ask for one representative domain's imports up front instead of leaving it as a caveat.
2. **A dialect-specific shortcut made the code untestable.** The MySQL `ON DUPLICATE KEY UPDATE` upsert was replaced by `merge()` once testing against a real database was possible.
3. **A defensive fallback masked errors.** `get_domain_map()` swallowed every exception and returned `{}`. It was made safer (rollback) but remains a broad catch. Part D, D3 decides its fate.
4. **A visual problem was diagnosed as CSS when it was cache.** The first reaction was to adjust styles; the right first step was to check whether the shipped CSS was being loaded. Only after loading it in a test browser was the cause clear. Result: unnecessary CSS churn (C-16) shipped in ZIP v2. *Lesson:* when a described problem matches something already shipped, ask for a hard-refresh/DevTools check first.
5. **A tooling detail cost time but is not a product issue:** the sandbox's browser engine lacks `fetch` and one XPath default that HTMX 1.9 relies on. Polyfills were added in the test harness only; nothing in the shipped code works around it.
6. **A cosmetic error shipped in the migration:** the docstring says `Create Date: 2026-09-21`; the actual date is 2026-10-02. It is a comment only (Alembic does not read it).

---

## 9. Pre-existing issues found, not fixed (per §4.5 — each needs its own ticket)

| # | Issue | Evidence | Severity |
|---|---|---|---|
| P-1 | `main.py` imports `weekly_plan_generator` and `weekly_plan_shopping` but never calls `include_router` on them. If their endpoints are intended to be live, they currently 404 | `main.py` (domains.planning import line vs. the `include_router` list) | Possibly high — **confirm intent first**; they may be mounted some other way |
| P-2 | `explorer.html` ends with one extra closing `</div>` | End of the original template; preserved as-is | Low |
| P-3 | GOVERNANCE §2.3 "Current state" and §2.1 explorer note are stale (already flagged in the document itself for §2.3) | GOVERNANCE.md | Documentation |
| P-4 | Legacy `TemplateResponse(name, context)` signature is used project-wide | All routers | Upgrade risk (Starlette ≥ 1.0) |

---

## 10. Rollback

1. `alembic downgrade s0cc3r_st4ts001` — drops `explorer_table_domains` and its index. **This discards any assignments made through the settings page.** Export first if needed: `SELECT * FROM explorer_table_domains;`
2. Delete the new files listed in §4.1 as "New".
3. `git checkout` the edited files: `domains/explorer/routers/explorer.py`, `domains/explorer/templates/explorer.html`, `domains/explorer/static/css/explorer.css`, `main.py`.
4. Restart the web container. The explorer reverts to the flat table list.

Rollback is safe in either order of steps 1 and 3: while the table is missing, the current code still serves the explorer (everything unassigned).

---

## Part D — Post-migration follow-up requirements

**Read this before doing anything in Part D.** These tasks are a *separate, later* piece of work. They exist because some cleanup is only safe, or only possible, once **every other in-flight migration has been applied** and the owner's table set has stopped changing. Per GOVERNANCE §4.5, none of this was bundled into WO#36.

### D.0 Interpretation of the request (and a stop condition)

The request was to document what must happen "after all the other migrations are completed (like adjust the `models.py`, removing the references from there)". This document reads that as:

- **`models.py`:** the only `models.py` this work touches is `domains/explorer/models.py`. The **root** `models.py` was retired in WO#22 (GOVERNANCE §2.4) and has no shims left, so there are no root-level references to remove. The "references" to clean up are therefore (a) the inaccurate registration note in `domains/explorer/models.py`, (b) the Alembic `env.py` registration, and (c) any stale imports/mentions found by the repo-wide sweep in D-4.
- **"After all the other migrations":** the in-flight Alembic revisions — at minimum `s0cc3r_st4ts001` (soccer stats), which was *unapplied* when this work was written, plus anything created since.

> **STOP CONDITION.** If the owner meant a *different* `models.py`, or a different set of "references" (for example, hard-coded domain lists inside other domains' models), **do not guess.** Report what was found and ask. Nothing in this work introduced such references.

### D.1 Gate — preconditions (all must hold before starting)

| # | Precondition | How to check |
|---|---|---|
| G-1 | Exactly one Alembic head, and it descends from `expl0r3r_d0ma1ns001` | `alembic heads` prints one hash; `alembic history` shows `s0cc3r_st4ts001 → expl0r3r_d0ma1ns001 → …` |
| G-2 | `expl0r3r_d0ma1ns001` is applied in **every** environment (local and production) | `alembic current` in each; the table exists: `SHOW TABLES LIKE 'explorer_table_domains';` |
| G-3 | Every other in-flight migration is applied in every environment | `alembic current` equals `alembic heads` in each |
| G-4 | The owner has reviewed `/explorer/settings` and confirmed the domain assignments | Owner confirmation; do not infer it |

If any gate fails, stop and report. Do not "fix" a failing gate as part of this work.

### D.2 Tasks

#### D-1 — Re-verify the migration's parent (only if it has not been applied anywhere)

**Hard rule: never edit a migration that has been applied in any environment.**

| Situation | Action |
|---|---|
| Applied in at least one environment | **Do not touch the file.** Later migrations chain after it. If two heads exist, `alembic merge heads -m "merge …"` creates a merge revision |
| Not applied anywhere, `alembic heads` is a single hash that is **not** `s0cc3r_st4ts001` | Change `down_revision` in the migration to that hash |
| Not applied anywhere, `alembic heads` prints more than one hash | `alembic merge heads -m "merge before explorer domains"`, then point `down_revision` at the merge revision |

**Acceptance:** `alembic heads` prints exactly one hash; `alembic upgrade head` succeeds on a fresh copy of the database.

#### D-2 — Reconcile the data (do not edit the seed)

1. Run the three audit queries in Appendix C in **each** environment.
2. Expected: **0 orphan rows**, and "unassigned" containing only tables the owner intentionally left unassigned.
3. For tables created by other migrations after 2026-10-02, assign them via `/explorer/settings` (preferred) or a one-off `INSERT`.
4. For orphan rows (a mapped table that no longer exists), delete them: `DELETE FROM explorer_table_domains WHERE table_name = '…';`
5. **Do not edit `_SEED` inside the applied migration.** The settings page and the table are the source of truth from the moment the migration runs. If the owner wants the new assignments *in code* (for fresh environments), create a **new** data-only migration; this is a decision (DP-4).

**Acceptance:** orphan query returns 0 rows in every environment; unassigned query returns only owner-approved tables; the row count equals the number of non-hidden tables minus the approved-unassigned ones (plus hidden seeded rows such as `alembic_version`).

#### D-3 — Remove the pre-migration tolerance (**requires owner decision DP-1**)

Two behaviors exist only to make the explorer survive the window *before* the migration is applied:

| Location | Behavior | Why it should go |
|---|---|---|
| `domains/explorer/table_domains.py::get_domain_map()` | `try/except Exception` returns `{}` and logs a warning | After the migration it **hides real database errors**: any failure here silently turns every table "unassigned" |
| `domains/explorer/routers/explorer_settings.py` | `_SAVE_FAILED_MSG` = "…has the explorer_table_domains migration been applied?" | Misleading once the table always exists |

Recommended change, if the owner agrees:
1. In `get_domain_map()`, remove the `try/except` and the `rollback()`; let errors propagate to the global 500 handler.
2. In `explorer_settings.py`, keep `_apply()`'s `try/except SQLAlchemyError` (it rolls back and logs, which remains valid), but change `_SAVE_FAILED_MSG` to `"Could not save the assignment."`.
3. Update any test that asserts the "migration not applied" behavior.

If the owner prefers to keep the tolerance, record that decision here and leave the code alone.

**Acceptance:** `/explorer/schema` returns 200 with the table present; with the table temporarily renamed it returns a 500 (error visible), **not** a 200 with everything unassigned; saving with the table missing no longer mentions "migration".

#### D-4 — `models.py`, Alembic registration, and the reference sweep

1. **Alembic registration.** Open Alembic's `env.py` (not provided during WO#36). Determine how `target_metadata` is populated and which modules are imported.
   - If models are imported **explicitly, per domain**, add `import domains.explorer.models  # noqa: F401`.
   - If they are imported through the app (`main`/`database`/routers), confirm `domains.explorer.models` is reached. It is reached **transitively** via `routers/explorer.py → table_domains.py → models.py`, and **directly** via `routers/explorer_settings.py`.
2. **Drift check.** Run `alembic check` (or the project's equivalent). Expected for `explorer_table_domains`: **no diff** (the model's `index=True` on `domain` produces `ix_explorer_table_domains_domain`, which the migration creates under that exact name). Record unrelated diffs from other domains as findings; **do not fix them here**. **Do not generate a revision file.**
   - If autogenerate wants to *drop* `explorer_table_domains`, step 1 failed — fix the registration, not the migration.
3. **Fix the docstring in `domains/explorer/models.py`.** It currently says the model registers "by being imported from this domain's routers (see routers/explorer.py)". That is inaccurate: `explorer.py` imports it only transitively through `table_domains.py`. Reword to the verified mechanism, and, if step 1 added an explicit import, say so.
4. **Repo-wide reference sweep** (read-only grep, then fix only what the grep proves):

   | Search for | Expected | If found |
   |---|---|---|
   | `HIDDEN_TABLES` imported from `domains.explorer.routers.explorer` | none | Repoint to `domains.explorer.table_domains` |
   | `from models import` / `import models` (root `models.py`) | none (retired in WO#22) | Out of scope: report as a finding |
   | Any other file defining or re-exporting `ExplorerTableDomain` | none | Remove the duplicate; the single definition is `domains/explorer/models.py` |
   | Any other module reading `explorer_table_domains` directly | none expected | Report; the table is meant to be read only through `table_domains.py` |

**Acceptance:** `alembic check` shows no diff for `explorer_table_domains`; the grep table above is all "none" or each hit is resolved/reported; `models.py` has exactly one class and an accurate docstring.

#### D-5 — Documentation amendments (**GOVERNANCE changes need owner approval — DP-2**)

Exact proposed text is in Appendix D. Summary:
- GOVERNANCE §2.1: the example "`explorer` has no `models.py`" is now false — remove it.
- GOVERNANCE amendment log: add the WO#36 entry.
- Optional new rule (needs approval): migrations that create tables should name the explorer domain for them — either by seeding `explorer_table_domains` in the same migration or by noting in the docstring that they will be assigned in `/explorer/settings`.
- CONTRIBUTING "Other Rules": add the same rule **only if** approved.

**Acceptance:** each approved edit is present; each unapproved one is absent.

#### D-6 — File the pre-existing issues as tickets (§9 of this document)

Create one ticket each for P-1 … P-4. **Do not fix them in this work.** For P-1, first confirm intent with the owner (`weekly_plan_generator` / `weekly_plan_shopping` may be mounted somewhere not visible in `main.py`).

**Acceptance:** four tickets exist, each with the evidence from §9.

#### D-7 — Settle the `explorer.css` variant and the cache-busting policy (**requires DP-3**)

1. Identify which `explorer.css` is deployed (Variant 1 or 2, §4.2). Either is acceptable. Record the one in use so docs don't describe the other.
2. `?v=2` is a **hard-coded** query string on two `<link>` tags. Decide: keep manual bumping (state this in the README), or introduce a shared mechanism later as its own work item. **Do not build one here.**

**Acceptance:** the chosen variant is recorded; the policy is stated.

#### D-8 — Cosmetic correction (optional)

The migration docstring says `Create Date: 2026-09-21`; the true date is 2026-10-02. It is a comment only. If the migration is already applied anywhere, **leave it** (the rule in D-1 outweighs a comment fix). If not applied anywhere, correct it.

### D.3 Decision points (owner input required — the agent must not decide these)

| ID | Decision | Default if the owner says "your call" |
|---|---|---|
| DP-1 | Remove the pre-migration tolerance (D-3)? | Remove it |
| DP-2 | Adopt the proposed GOVERNANCE / CONTRIBUTING rule about assigning domains to new tables (D-5)? | Adopt in GOVERNANCE; add to CONTRIBUTING only on explicit yes |
| DP-3 | Which `explorer.css` variant is canonical; keep manual `?v=` bumping (D-7)? | Record what is deployed; keep manual bumping |
| DP-4 | Should assignments made after seeding be captured in a new data migration for fresh environments (D-2)? | No — the settings page is the source of truth |

### D.4 Out of scope for the follow-up (do **not** implement without explicit approval)

These were mentioned as possible enhancements but **never agreed**:
- Suggesting a domain for an unassigned table from its name prefix (e.g. `nba_standings` → `nba`).
- Deriving domains automatically from the model modules.
- Re-sorting settings rows immediately after an edit (currently they re-sort on reload).
- A toggle to show `alembic_version` or other hidden tables.
- Exact (non-estimated) row counts. The explorer uses `information_schema.TABLE_ROWS`, which is an estimate for InnoDB; that is pre-existing behavior.
- Any change to the SQL execution path.

### D.5 Follow-up work order, in the standing template (GOVERNANCE §4.3)

```markdown
## ROLE
Cleanup engineer, not a feature builder. Minimal, reversible, verifiable changes.
Every change must trace to a task in Part D of explorer-domains-postmortem.md.

## HARD BOUNDARIES
- Do not edit any Alembic migration that is applied in any environment.
- Do not edit `_SEED` inside `expl0r3r_d0ma1ns001`.
- Do not reassign table domains on your own — assignments are the owner's.
- Do not touch DAGs, other domains, core/, base templates, or docker-compose.
- Do not fix the pre-existing issues P-1..P-4; file tickets.
- Do not implement anything in section D.4.
- Do not decide DP-1..DP-4. Ask, or follow the stated default only if the owner says "your call".
- If the code contradicts this document, STOP and report; don't improvise.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
(standard four-step procedure from GOVERNANCE §4.3)

## WORKING METHOD
Verify gate G-1..G-4 first. Then execute D-1..D-8 in order, verifying after each
behavior-changing step. Substitute checks must be stated and marked ⚠️.

## OUTPUT FORMAT
1. Files created  2. Files moved  3. Files edited (flag anything beyond literal
instructions)  4. Acceptance criteria (✅/❌/⚠️)  5. Notes

## ROLLBACK
git checkout every file listed under "Files edited". No schema change is made by this work.

## SCOPE
domains/explorer/**, alembic/env.py, alembic/versions/* (read; edit only per D-1),
GOVERNANCE.md, CONTRIBUTING.md (per DP-2), read-only repo-wide grep.

## STEPS
Part D, tasks D-1 … D-8.

## ACCEPTANCE CRITERIA
The "Acceptance" line under each task in Part D.
```

---

## Appendix A — Final seed (generated from the owner's final list)

| Domain | # | Tables |
|---|---:|---|
| `blog` | 1 | `blog_ideas` |
| `code_intel` | 3 | `code_files`, `code_projects`, `folder_readmes` |
| `explorer` | 1 | `explorer_table_domains` |
| `finance` | 3 | `accounts`, `categories`, `transactions` |
| `habits` | 3 | `habit_logs`, `habit_settings`, `habits` |
| `jobs` | 6 | `application_logs`, `job_scout_run_log`, `job_search_keywords`, `linkedin_jobs`, `staging_jobs`, `watched_companies` |
| `journal` | 2 | `journal_entries`, `weekly_syntheses` |
| `life_os` | 1 | `alembic_version` |
| `media` | 5 | `media_items`, `media_recommendations`, `streaming_services`, `tv_season_progress`, `user_media` |
| `medium` | 2 | `medium_articles`, `medium_feed_sources` |
| `nba` | 14 | `nba_box_score_advanced`, `nba_box_score_defensive`, `nba_box_score_fourfactors`, `nba_box_score_hustle`, `nba_box_score_matchup`, `nba_box_score_misc`, `nba_box_score_scoring`, `nba_box_score_tracking`, `nba_box_score_traditional`, `nba_box_score_usage`, `nba_games`, `nba_play_by_play`, `nba_players`, `nba_teams` |
| `planning` | 5 | `shopping_lists`, `user_intent`, `weekly_plan_days`, `weekly_plan_meals`, `weekly_plans` |
| `recipes` | 6 | `ingredients`, `pantry_items`, `recipe_ingredients`, `recipe_tags`, `recipe_tags_junction`, `recipes` |
| `soccer` | 9 | `soccer_bookings`, `soccer_coaches`, `soccer_competitions`, `soccer_goals`, `soccer_match_lineups`, `soccer_matches`, `soccer_raw_payloads`, `soccer_settings`, `soccer_substitutions` |
| `workout` | 9 | `body_metrics`, `equipment`, `exercises`, `workout_locations`, `workout_plan_days`, `workout_plan_exercises`, `workout_plans`, `workout_sessions`, `workout_sets` |
| **Total** | **70** | |

## Appendix B — Reviewer's verification commands

```bash
alembic heads                      # one hash; expect s0cc3r_st4ts001 before applying
alembic upgrade head
```
```sql
SELECT COUNT(*) FROM explorer_table_domains;              -- 70 immediately after the migration (before any UI edits)
SELECT domain, COUNT(*) FROM explorer_table_domains GROUP BY domain ORDER BY domain;   -- 15 domains
```
```bash
curl -s localhost:<port>/explorer/schema | python3 -c "import sys,json,collections; \
d=json.load(sys.stdin); print(collections.Counter(v['domain'] for v in d.values()))"
```
Then in the browser (hard-refresh once): `/explorer` shows collapsed domains with counts; `/explorer/settings` lists every table; create a throwaway table and confirm it appears as unassigned in both places, then drop it and confirm any mapping row for it is ignored.

## Appendix C — Audit SQL (verified on MariaDB 10.11)

`information_schema` columns are `utf8mb3`, so the comparison converts explicitly. This avoids collation errors regardless of the database's default collation.

```sql
-- 1. Unassigned tables (exist in the DB, no mapping row)
SELECT t.TABLE_NAME AS unassigned_table
FROM information_schema.TABLES t
LEFT JOIN explorer_table_domains d
  ON d.table_name COLLATE utf8mb4_general_ci
   = CONVERT(t.TABLE_NAME USING utf8mb4) COLLATE utf8mb4_general_ci
WHERE t.TABLE_SCHEMA = DATABASE() AND t.TABLE_TYPE = 'BASE TABLE' AND d.table_name IS NULL;

-- 2. Orphan mappings (mapping row, no such table)
SELECT d.table_name AS orphan_mapping
FROM explorer_table_domains d
LEFT JOIN information_schema.TABLES t
  ON t.TABLE_SCHEMA = DATABASE()
 AND CONVERT(t.TABLE_NAME USING utf8mb4) COLLATE utf8mb4_general_ci
   = d.table_name COLLATE utf8mb4_general_ci
WHERE t.TABLE_NAME IS NULL;

-- 3. Mapped row count
SELECT COUNT(*) AS mapped_rows FROM explorer_table_domains;
```

Note: the explorer UI itself hides `alembic_version` and the Airflow tables in `HIDDEN_TABLES`, so query 1 will list `dag`, `dag_run`, etc. if they live in the same schema. Those are expected and not a defect.

## Appendix D — Proposed documentation text (applies only if approved, DP-2)

**GOVERNANCE.md §2.1** — change
> Not every domain needs every subfolder (e.g. `explorer` has no `models.py`, `code_intel` has no dedicated CSS)

to
> Not every domain needs every subfolder (e.g. `code_intel` has no dedicated CSS)

**GOVERNANCE.md — new §2.8 (optional):**
> ### 2.8 Explorer Table → Domain Mapping
> The SQL Explorer groups tables by the `explorer_table_domains` table, edited at `/explorer/settings`. A table with no mapping row is shown as **unassigned**; that is a valid, transient state, not an error. A migration that creates tables should either seed `explorer_table_domains` in the same migration or state in its docstring that the tables will be assigned through the settings page. Domain labels are free text matching `[a-z0-9_]{1,50}`. Nothing outside `domains/explorer/` should read this table directly.

**GOVERNANCE.md — amendment log (add at the top):**
> - §2.1, §2.8 (new) — recorded the `explorer` domain's `models.py` and the table-domain mapping convention, after WO#36 (explorer table → domain grouping). WO#36's postmortem also records one documentation error corrected in passing: §2.4's claim that model registration lives in a `database.py` import block does not match the real code, where registration happens through router imports.

**CONTRIBUTING.md "Other Rules" (only on an explicit yes):**
> - Every migration that creates tables must say which explorer domain they belong to — either seed `explorer_table_domains` in the same migration or note in the docstring that they will be assigned at `/explorer/settings`.

## Appendix E — Reviewer checklist

- [ ] Read §0 and §3.4; understood that Delivery 3's sidebar CSS (C-16) is optional.
- [ ] `alembic heads` is a single hash and matches the migration's `down_revision` (or D-1 was applied).
- [ ] Migration applied; 70 rows; 15 domains (Appendix B).
- [ ] `/explorer` loads with domains collapsed; a domain expands to tables; a table expands to columns.
- [ ] `/explorer/settings` loads; an inline edit saves ("Saved." toast); a bad domain is rejected with a message; bulk assign works.
- [ ] A new throwaway table appears as **unassigned** in both pages, and the topbar link shows the count.
- [ ] Settings page is legible in the dark theme (filter, dropdown, bulk bar). **(⚠️ AC-13)**
- [ ] Reviewed every ⚠️ in §6 and accepted or escalated each.
- [ ] Pre-existing issues P-1 … P-4 have tickets (they are not blockers).
