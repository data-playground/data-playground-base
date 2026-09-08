# Work Order #33 — NBA Data Domain (New Build)

*This is a greenfield feature build, not a refactor. Per GOVERNANCE.md §4.2
("Role Framing Is Not Optional"), a build like this needs a different default
posture than every other work order in this program — exploratory and
creative on implementation details, not minimal/reversible/restrictive. The
architecture guardrails below exist so this new domain fits the rest of the
codebase; they are not a literal step-by-step script to follow without
judgment.*

---

## ROLE
You are a full-stack engineer building a new feature domain from scratch
inside an established codebase. Unlike every other work order in this
program, you are **not** constrained to "don't improve, don't add, pure
relocation only." You have real latitude on schema shape, endpoint design,
template layout, and implementation approach. What you're not free to do is
ignore this codebase's existing architectural conventions (domain-folder
structure, the DAG/FastAPI import boundary, the shared AI service layer) —
those exist for reasons documented in `migration_docs/GOVERNANCE.md` and
apply to new domains exactly as much as migrated ones.

## HARD BOUNDARIES (the few that are non-negotiable)
- New code lives under `domains/nba/` (models, routers, templates, static —
  only the subfolders you actually use).
- If ingestion runs as a scheduled/triggered Airflow DAG: that DAG must
  never import `models.py`, `database.py`, or any router/service — DAG
  database access goes through `airflow/dag_db.py`'s raw-SQL helpers only.
  This is an absolute rule in this codebase (`CONTRIBUTING.md`, restated in
  GOVERNANCE §2.2) and applies to brand-new domains exactly as much as
  existing ones.
- Any LLM/AI calls (e.g. generating a recap or summary) route through
  `services/ai/` (`call_gemini_text`/`call_gemini_json`/etc.) — do not write
  a new, independent provider-calling implementation. This codebase has
  already been through six duplicate implementations of "call an LLM
  provider" and consolidated them once (GOVERNANCE §2.3); don't create a
  seventh.
- Keep router files under GOVERNANCE §1.2's 300-line ceiling from the
  start — split by responsibility early rather than letting one file grow
  and needing a cleanup pass later (see the Track C work orders in this
  same program for what that cleanup looks like when it isn't done early).
- Step 1 below (requesting materials) must actually happen, and must
  happen **before** any code is written.

## WORKING METHOD
Do Step 1 first, in full, and wait for a response before writing anything.
Everything after that is your judgment call, documented in the final
report — not something to check in about at every decision point. The one
exception: if a decision would be expensive to reverse once real data
exists (e.g. a schema choice that discards raw detail in favor of
precomputed aggregates), surface it and ask before committing, the same way
you would for any action that's hard to undo.

## OUTPUT FORMAT
1. Materials requested and received (Step 1) — what was asked for, what was
   supplied, any assumptions made where something wasn't available
2. Files created (full list — models, routers, templates, static, DAG if any)
3. Files edited (`main.py`, `core/templating.py`, `templates/partials/
   sidebar.html` if included)
4. Design decisions made and why (schema shape, ingestion cadence, endpoint
   design, anything else non-obvious)
5. Acceptance criteria results
6. Notes / open questions for a follow-up pass

## ROLLBACK
Since this is new code with no prior state, rollback is simply not merging
it — no existing behavior is at risk.

---

## STEP 1 — Request materials (do this before writing any code)

Acknowledge this work order, then request the following from the project
owner:

1. **The existing NBA data extraction process.** Either the actual code (a
   script, notebook, or partial implementation), or — if nothing exists yet
   — a description of the intended data source (a specific stats API, a
   free public endpoint, something else) and any access details (API key
   location, rate limits, auth).
2. **Ingestion pattern confirmation.** Every other data-heavy domain in this
   codebase (jobs, media streaming-availability refresh, embeddings) uses a
   scheduled/triggered Airflow DAG writing into persistent tables, read by a
   FastAPI router. **Default to that pattern** unless told otherwise — but
   ask explicitly, since choosing on-demand-only (fetched live per page
   load, no DAG, no persistent table) is a real alternative with different
   tradeoffs (fresher data, no ingestion lag, but slower page loads and no
   historical record).
3. **Data granularity.** Standings? Team stats? Player stats? Box
   scores? Live/in-progress games? This determines the schema — ask rather
   than guess at scope.
4. **Whether dashboard/sidebar integration is in scope for this pass**, or
   should be a fast-follow once the core domain exists.
5. **Any existing naming/format conventions** the owner already uses
   elsewhere for this data (team abbreviations, season format, date
   handling) that should carry over for consistency.

Do not proceed past this step until at least items 1–3 are answered.

## STEP 2 — Design, then build

Once materials are in hand:

1. If the schema involves a genuinely irreversible tradeoff (see WORKING
   METHOD above), propose it and get a nod before writing the migration.
   Otherwise, design and build directly — you don't need sign-off on every
   table or column name.
2. Follow the established domain shape:
   - `domains/nba/models.py` — SQLAlchemy ORM classes, importing `Base`
     from `core.base_model` (the same pattern every other domain uses).
   - `domains/nba/routers/*.py` — one or more router files, each under 300
     lines; if the feature naturally splits (e.g. teams vs. games vs.
     standings), split from the start rather than waiting for a line-limit
     work order later.
   - `domains/nba/templates/` (+ `partials/` if needed) — extend
     `base.html` like every other page in this app.
   - `domains/nba/static/` — only if you have dedicated CSS/JS; if the
     shared `base.css` primitives cover it, skip this folder entirely.
3. If ingestion is DAG-scheduled: new DAG(s) under `airflow/dags/nba/`,
   using `dag_db.py` helpers only (see HARD BOUNDARIES). If the ingestion
   logic needs an API-calling wrapper that could plausibly also be needed
   by a router someday, look at how `services/tmdb_service.py` (router-facing)
   and `airflow/agents/media_agents.py` (DAG-facing, deliberately
   duplicating rather than importing the service — see that file's own
   docstring for why) handle the same tension, and follow whichever shape
   fits your actual reuse needs.
4. Wire it in:
   - Register the new router(s) in `main.py`.
   - Add `domains/nba/templates/` to `core/templating.py`'s `ChoiceLoader`.
   - If you added a static mount, register it **before** the general
     `/static` mount in `main.py` — Starlette matches mounts in
     registration order, and this project has already hit the bug where a
     general mount registered first silently swallows a more specific one
     (see GOVERNANCE §2.6).
   - If sidebar/dashboard integration is in scope for this pass: add a nav
     entry to `templates/partials/sidebar.html`, and (if a dashboard card
     is wanted) add a read-only query to `routers/dashboard.py` — that
     file is the one place in this codebase allowed to read across domain
     boundaries for summary purposes (GOVERNANCE §2.2).

## ACCEPTANCE CRITERIA
- [ ] Step 1 actually happened before code was written — the report shows
  what was requested and what was supplied.
- [ ] All new code lives under `domains/nba/`, following the established
  shape (no unused empty subfolders).
- [ ] If a DAG exists: confirmed it never imports `models.py`, `database.py`,
  or any router/service.
- [ ] Any AI/LLM calls route through `services/ai/`.
- [ ] Router(s) registered in `main.py`; templates path added to
  `core/templating.py`; any static mount ordered before the general
  `/static` mount.
- [ ] Every router file is under 300 lines.
- [ ] The report documents every non-obvious design decision (schema shape,
  ingestion cadence/pattern, endpoint design) so it can be reviewed as a
  whole.
- [ ] Basic functional smoke test: the new page(s) render, and (if DAG-based)
  the DAG runs and populates data, or (if on-demand) the router successfully
  fetches and displays live data.

## For the next work order (not part of this one)
WO#34 (Soccer) is a structurally similar new-domain build. Once this one's
schema/ingestion-pattern decisions are made, it's worth a quick shared read
before finalizing Soccer's own schema — not to merge the two domains (they
stay separate, per the project owner's explicit instruction), but to keep
conventions (team ID format, season format, date handling) consistent
across both.
