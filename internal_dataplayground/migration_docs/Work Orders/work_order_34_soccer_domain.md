# Work Order #34 — Soccer Data Domain (New Build)

*Structurally similar to WO#33 (NBA) — kept as a fully separate domain and
work order per explicit instruction, not merged, even though the shape of
the build is nearly identical. Same greenfield posture applies: this is a
build, not a refactor. See WO#33's ROLE section for the full framing; it's
restated briefly here for a reader who only has this file.*

---

## ROLE
You are a full-stack engineer building a new feature domain from scratch.
You have real latitude on schema shape, endpoint design, template layout,
and implementation approach — this is not a "pure relocation, don't improve
anything" work order like the rest of this program's Track B/C items. What
you're not free to do is ignore this codebase's existing architectural
conventions (domain-folder structure, the DAG/FastAPI import boundary, the
shared AI service layer).

## HARD BOUNDARIES (the few that are non-negotiable)
- New code lives under `domains/soccer/` (models, routers, templates,
  static — only what you actually use).
- If ingestion runs as a scheduled/triggered Airflow DAG: it must never
  import `models.py`, `database.py`, or any router/service — DAG database
  access goes through `airflow/dag_db.py`'s raw-SQL helpers only. Absolute
  rule in this codebase, applies to new domains too.
- Any LLM/AI calls route through `services/ai/` — do not write a new,
  independent provider-calling implementation.
- Keep router files under GOVERNANCE §1.2's 300-line ceiling from the
  start. Soccer covers more competitions than NBA covers leagues
  (Brasileirão, Premier League, Libertadores, Champions League, per the
  interests list this project already tracks) — this makes an early split
  by competition or by concern (fixtures vs. standings vs. results) more
  likely to matter than it did for NBA, not less.
- Step 1 below must happen, in full, before any code is written.

## WORKING METHOD
Same as WO#33: Step 1 first, wait for a response, then build with judgment.
Surface only genuinely hard-to-reverse decisions before committing to them.

## OUTPUT FORMAT
1. Materials requested and received (Step 1)
2. Files created (full list)
3. Files edited (`main.py`, `core/templating.py`, sidebar if included)
4. Design decisions made and why
5. Acceptance criteria results
6. Notes / open questions

## ROLLBACK
New code, no prior state — rollback is simply not merging it.

---

## STEP 1 — Request materials (do this before writing any code)

Acknowledge this work order, then request the following:

1. **The FIFA endpoint** (or whichever data source actually feeds this —
   confirm the specific provider/API) — access details, auth, rate limits,
   and either existing extraction code or a description of what's been
   tried so far.
2. **Competition scope for this first pass.** The project's own tracked
   interests list four soccer competitions (Brasileirão, Premier League,
   Libertadores, Champions League) plus the World Cup and Olympics more
   broadly. Ask explicitly whether all of these are in scope for v1, or
   whether to start with one or two and phase the rest in — don't assume
   "all of them" is the right first-pass scope just because they're all
   named as interests.
3. **Ingestion pattern confirmation** — same question as WO#33: DAG-scheduled
   (default, matches every other data domain here) vs. on-demand. If NBA
   (WO#33) has already been decided, ask whether Soccer should match that
   choice for consistency, or has different freshness/latency needs that
   argue for something different.
4. **Data granularity** — fixtures/schedule, live scores, standings/tables,
   match stats, or some combination.
5. **Whether dashboard/sidebar integration is in scope for this pass.**

Do not proceed past this step until at least items 1–4 are answered.

## STEP 2 — Design, then build

Once materials are in hand:

1. Surface genuinely irreversible schema tradeoffs before committing;
   otherwise design and build directly.
2. Follow the established domain shape:
   - `domains/soccer/models.py` — importing `Base` from `core.base_model`.
   - `domains/soccer/routers/*.py` — split by competition or by concern
     from the start if the scope spans multiple competitions (see HARD
     BOUNDARIES above on why this is more likely to matter here than for
     NBA).
   - `domains/soccer/templates/` — extends `base.html`.
   - `domains/soccer/static/` — only if the shared `base.css` primitives
     don't cover it.
3. If a WO#33 (NBA) design decision on ingestion pattern, table-naming
   convention, season/date format, or team/competition ID format already
   exists by the time this runs: read it and match it where it makes sense,
   for consistency across the two sports domains — but don't force a match
   where soccer's actual shape (multiple competitions, promotion/relegation,
   international tournaments alongside domestic leagues) genuinely differs
   from NBA's single-league shape.
4. If ingestion is DAG-scheduled: new DAG(s) under `airflow/dags/soccer/`,
   `dag_db.py` helpers only. Same service-vs-DAG-duplication question as
   WO#33 applies if an API wrapper needs to be shared.
5. Wire it in: register router(s) in `main.py`; add templates path to
   `core/templating.py`; register any static mount before the general
   `/static` mount; add sidebar/dashboard entries if in scope for this pass.

## ACCEPTANCE CRITERIA
- [ ] Step 1 happened before code was written.
- [ ] All new code lives under `domains/soccer/`.
- [ ] If a DAG exists: confirmed it never imports `models.py`, `database.py`,
  or any router/service.
- [ ] Any AI/LLM calls route through `services/ai/`.
- [ ] Router(s) registered in `main.py`; templates path added; static mount
  ordered correctly if applicable.
- [ ] Every router file under 300 lines.
- [ ] Design decisions documented, including whether/how they were aligned
  with WO#33's NBA conventions.
- [ ] Basic functional smoke test passes for whichever competitions were
  actually scoped into this pass.

## For the next work order (not part of this one)
Nothing downstream depends on this specific domain beyond the general
Dashboard-integration fast-follow noted above, if it wasn't included here.
