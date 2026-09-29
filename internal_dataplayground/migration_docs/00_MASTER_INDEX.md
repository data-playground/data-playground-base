# Life OS Restructuring — Master Index

**Purpose:** single reference for every deliverable produced across this
engagement, and its actual execution status. Update the STATUS column as
work orders are run — this file is meant to stay accurate, not just be a
historical log.

**Last reviewed:** WO#16, WO#21, WO#22, and now **WO#23–31 and WO#35** have
executed and been reviewed against their postmortems (WO#32–34 remain in
progress and are intentionally not part of this pass). **32 of 35 work
orders are executed and reviewed; 3 remain drafted (WO#32–34).** Top
priority is unchanged in substance and, if anything, sharper now: WO#22's
Task 3 (`configure_mappers()` run against a real, dependency-installed
environment) remains the single largest open gap in the entire program.
Two new items from this pass deserve equal attention before anything here
is treated as production-ready: (1) `GOVERNANCE.md` §3.3 is now confirmed
stale by four independent postmortems and should be corrected immediately
— see item 26 below; (2) WO#35 discovered a real, cross-cutting
infrastructure bug (`docker-compose.yml`'s `web` service was missing
`AIRFLOW_SECRET_KEY`, silently breaking every Airflow-trigger feature in
the app, not just medium's) — already fixed, but worth its own governance
follow-up, see item 27. See "Where This Actually Stands" at the bottom for
the full current picture.

---

## Foundational Documents

- `GOVERNANCE.md` — the standing architectural rules this entire program
  works against. Amended four times now: for the DAG Reorganization table
  (pre-WO#21), by WO#21 itself (§2.5 rewrite, plus a §2.2 cross-reference
  fix), by WO#22 (§2.4 marked historical/closed), and in this pass
  (§3.3's Migration Debt Tracker corrected to historical/closed after four
  independent postmortems caught it listing already-migrated domains as
  outstanding; a new §2.7 added codifying the secret/env-var-wiring rule
  WO#35's `AIRFLOW_SECRET_KEY` discovery surfaced).
  **Still stale, not yet fixed:** §2.3's "Current state" prose still
  describes six independent AI-provider-call implementations as an open
  problem, even though WO#11–16 have since migrated five of them and
  documented the sixth (`finance_upload.py`'s SDK call) as a deliberate,
  permanent exception. See item 24 in the Deferred section below.
- `CONTRIBUTING.md` — day-to-day conventions for anyone writing new code in
  this repo, domain or otherwise.

---

## Domain Migration Work Orders

| # | Domain | File | Status |
|---|---|---|---|
| 1 | Habits | `work_order_01_habits_domain.md` | ✅ Executed, reviewed |
| 2 | Blog + Code Intel | `work_order_02_blog_code_intel_domain.md` | ✅ Executed, reviewed |
| 3 | Jobs | `work_order_03_jobs_domain.md` | ✅ Executed, reviewed |
| 4 | Explorer | `work_order_04_explorer_domain.md` | ✅ Executed, reviewed |
| 5 | Finance | `work_order_05_finance_domain.md` | ✅ Executed, reviewed |
| 6 | Journal | `work_order_06_journal_domain.md` | ✅ Executed, reviewed |
| 7 | Recipes + Pantry | `work_order_07_recipes_pantry_domain.md` | ✅ Executed, reviewed |
| 8 | Workout | `work_order_08_workout_domain.md` | ✅ Executed, reviewed |
| 9 | Media | `work_order_09_media_domain.md` | ✅ Executed, reviewed |
| 10 | Planning | `work_order_10_planning_domain.md` | ✅ Executed, reviewed |

*(Execution order constraint and per-domain review notes 1–10: unchanged
from the prior review — not reproduced here to keep this revision focused
on what actually changed.)*

---

## AI Service Layer Work Orders (Track A)

| # | Scope | File | Status |
|---|---|---|---|
| 11 | Foundation (`services/ai/base.py`, `keys.py`) | `work_order_11_ai_service_foundation.md` | ✅ Executed, reviewed |
| 12 | Recipe agents migration | `work_order_12_ai_service_recipe_agents.md` | ✅ Executed, reviewed |
| 13 | Batch 3 migration | `work_order_13_ai_service_batch3_REWRITE.md` | ✅ Executed, reviewed |
| 14 | Groq provider | `work_order_14_ai_service_groq.md` | ✅ Executed, reviewed |
| 15 | Cerebras provider | `work_order_15_ai_service_cerebras.md` | ✅ Executed, reviewed |
| 16 | Capstone | `work_order_16_ai_service_capstone.md` | ✅ Executed, reviewed — see `WO16-postmortem-OVERVIEW-and-program-closeout.md`. **Open items:** `services/ai/base.py`/`keys.py`/`__init__.py` still unverified against real source (G.1 — six consecutive work orders have now built on unverified stubs); `GOVERNANCE.md` §2.3 "current state" prose still stale, needs a rewrite reflecting the completed migration (see item 24 below). |

*Track A is now fully executed for the first time. The series was strictly
sequential; with WO#16 done, nothing further is gated on it, but see the
two open items above before treating the track as risk-free.*

---

## Frontend Consolidation Work Orders

| # | Scope | File | Status |
|---|---|---|---|
| 17 | Toast consolidation | `work_order_17_frontend_toast_consolidation.md` | ✅ Executed, reviewed |

---

## DAG Reorganization & Cleanup Work Orders

| # | Scope | File | Status |
|---|---|---|---|
| 18 | DAG file relocation into domain subfolders | `work_order_18_dag_reorganization.md` | ✅ Executed, reviewed |
| 19 | Dead code sweep + router line-limit lint | `work_order_19_misc_cleanup.md` | ✅ Executed, reviewed |
| 20 | Legacy shim removal | `work_order_20_shim_removal.md` | ✅ Executed, reviewed |

*The two follow-on commitments these three work orders' own postmortems
made — the GOVERNANCE.md §2.5 amendment (from WO#18) and the `models.py`
end-state decision plus stale DAG header cleanup (from WO#10 and WO#20) —
no longer sit unapplied. They are now WO#21 and WO#22, both executed — see
the Track B table immediately below.*

---

## Track B — Loose Ends Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 21 | Apply the GOVERNANCE.md §2.5 amendment (DAG relocation risk reframing) | `work_order_21_governance_dag_amendment.md` | ✅ Executed, reviewed — see `WO21-domains-migration-postmortem-governance-dag-amendment.md`. §2.5 rewrite confirmed directly against the live `GOVERNANCE.md`; one authorized follow-up (a stale §2.2 cross-reference) also landed and was independently diffed against true source. | WO#18 |
| 22 | `models.py` end-state + 13 stale DAG header comments + `configure_mappers()` verification | `work_order_22_models_dag_headers_configure_mappers.md` | ✅ Executed — Tasks 1 & 2 complete and verified (`WO22-domains-migration-postmortem-models-dag-headers-mappers.md`); `GOVERNANCE.md` §2.4 confirmed updated to match. **Task 3 (`configure_mappers()` live check) blocked by sandbox environment (no network, no `sqlalchemy`, no `domains/`/`core/`/`routers/`/`gcp_secrets.py` available) — open follow-up ticket. Do not treat this row as fully closed until Task 3 is run against the real deployed container.** | WO#10, WO#18, WO#20 |

Both closed independently of each other, as expected. The third original
Track B item (moving `life_os_staging_promoter.py` and
`life_os_refresh_streaming_availability.py` into their DAG subfolders) was
already closed by the project owner directly, outside any work order,
before either of these ran.

---

## Track C — Router Line-Limit Remediation Work Orders

| # | Domain | File | Status | Depends on |
|---|---|---|---|---|
| 23 | Habits | `work_order_23_habits_router_split.md` | ✅ Executed — see `WO23-habits-router-split-postmortem.md`. 11 routes split cleanly across 3 files, zero collisions. **New finding: a cross-router registration-order shadowing risk** (`DELETE /habits/log` can be shadowed by `DELETE /habits/{habit_id}` if `main.py` registers the wrong router first — confirmed via negative-control test). Five owner-agreed amendments applied on top of the base split (two are real, tested behavior changes — "done" count now excludes deactivated habits; new habits sort to the bottom); authorization for these is described but not verbatim-quoted, weaker than the WO#29 standard. `main.py` edit specified as a diff only, not yet applied to a real file. | none |
| 24 | Code Intel (3 files) | `work_order_24_code_intel_router_split.md` | ✅ Executed — see `WO24-code-intel-router-split-postmortem.md`. Parts A & C fully resolved. **Part B (`ci_files.py`) left at 324 lines — the original WO's own STEPS and ACCEPTANCE CRITERIA directly conflict; correctly reported, not unilaterally resolved. Open item: reviewer must choose Option A (document exception) or Option B (move `update_comment_status` out) before this can close.** | none |
| 25 | Journal | `work_order_25_journal_router_split.md` | ✅ Executed — see `WO25-journal-router-split-postmortem.md`. Clean 3-way split; privacy-boundary comments and the deferred `domains.planning.models` cross-domain import confirmed untouched. **Flags a real design question needing owner sign-off:** the journal→planning link doesn't follow the sanctioned string-`relationship()` pattern — promote it to one (needs a joint migration) or promote the existing import to top-level; not a router-split WO's call to make unilaterally. Independently confirmed GOVERNANCE.md §3.3 staleness (see item 26). | none |
| 26 | Recipes (2 files) | `work_order_26_recipes_router_split.md` | ✅ Executed — see `WO26-recipes-router-split-postmortem.md`. Both splits land cleanly. **Governance concern: 6 of 9 review-round fixes were bundled directly into the split diff, including reversing the original WO's explicit "do not fix" instruction on a pre-existing `__import__(...)` oddity — disclosed openly, with a recommended 2-commit split, but no verbatim owner-authorization quote on record for that specific override.** Open item: obtain that quote or split the commit before treating as fully compliant. Independently confirmed GOVERNANCE.md §3.3 staleness. | none |
| 27 | Blog | `work_order_27_blog_router_split.md` | ✅ Executed, reviewed — see `WO27-blog-router-split-postmortem.md`. Clean on every axis: zero route collisions, DAG/router import boundary re-confirmed via grep, scratch artifact explicitly flagged as non-deliverable. No open items beyond standard sandbox-verification caveats. Model example for this batch. | none |
| 28 | Media (2 files) | `work_order_28_media_router_split.md` | ✅ Executed — see `WO28-media-router-split-postmortem.md`. Same shape of conflict as WO#24: both resulting files (316, 309 lines) still exceed the ceiling because the pinned function sets don't fit under it — correctly disclosed, not hidden. **Exemplary handling of 5 pre-existing dead imports found along the way: explicitly deferred to a standalone ticket, not bundled in** (useful contrast with WO#26). Open item: a proposed Part C follow-up split is explicitly not self-authorized — needs its own work order. Independently confirmed GOVERNANCE.md §3.3 staleness. | none |
| 29 | Workout (3 files) | `work_order_29_workout_router_split.md` | ✅ Executed, reviewed — see `WO29-workout-router-split-postmortem.md`. Parts A/B clean. **Part C (`workout_log.py`) correctly held for sign-off since it reverses a WO#8 HARD BOUNDARY — authorization is verbatim-quoted ("sure, let's separate them. And provide the files") and explicitly scoped to this file only, not treated as blanket precedent. This is the authorization-discipline standard the rest of the program should match.** Flags a real `main.py` merge-coordination risk across all of Track C — see the Deferred section. Independently confirmed GOVERNANCE.md §3.3 staleness (4th corroboration). | none |
| 30 | Planning — verification only | `work_order_30_planning_router_verification.md` | ✅ Executed — see `WO30-planning-router-split-postmortem.md`. **The master index's prior "likely resolves as a no-op" framing was wrong** — `weekly_plan.py` was genuinely still at 413 lines and the contingency split was required (now split into `weekly_plan.py`/`weekly_plan_confirm.py`). **Two real `main.py` findings:** (1) the split itself initially left the new router unregistered — fixed; (2) a separate, pre-existing production bug, unrelated to this WO, where `weekly_plan_generator.router` and `weekly_plan_shopping.router` were never registered at all, meaning `POST /plan/generate` and `GET /plan/{id}/shopping` have been 404ing in production — also fixed, but explicitly labeled as its own change per §4.5's spirit, with rollback instructions to un-bundle it. Live functional re-verification against a running instance is the one remaining gate. | none |

Covers the real 14-file list WO#19's lint script produced. All eight now
report back complete. Two cross-cutting items surfaced by this batch — see
the Deferred section, items 26–28 — apply across several of these rows
simultaneously and should be resolved once, not per-domain.

---

## Track D — Agent Folder Reorganization

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| — | Pre-flight risk analysis (real consumer map, tiering) | `track_D_agent_reorg_scoping_analysis.md` | ✅ Complete | WO#18 |
| 31 | Tier 1 reorg — DAG-only agent modules | `work_order_31_agent_reorg_tier1.md` | ✅ Executed — see `WO31-reorg-agent-dags-postmortem.md`. Six modules moved into `airflow/agents/jobs/` and `airflow/agents/media/`, content byte-identical. **One legitimate scope completion, not creep:** `life_os_daily_digest.py` wasn't in the original file list but imports one of the six moved modules — left unfixed it would have broken a scheduled DAG at its next run; correctly reasoned as satisfying the WO's own "zero stale references" acceptance criterion rather than unrelated work. DAG/FastAPI boundary reasoning confirmed sound throughout. | Track D analysis |

The analysis found `airflow/agents/` does **not** have one uniform risk
profile the way `airflow/dags/` did: some modules are DAG-only (safe-ish —
Tier 1, drafted as WO#31), some are router-only and arguably misplaced
under `airflow/agents/` at all (`recipe_agents.py`, `weekly_agents.py` —
Tier 2, deliberately left unscoped), and one is consumed by both DAGs and
routers and can't be moved incrementally (`blog_agents.py` — Tier 3,
deliberately left unscoped). Only Tier 1 is drafted. Tiers 2 and 3 each
need their own dedicated future scoping pass — see the analysis document's
Part 3 for why drafting them further now would mean guessing.

---

## Track E — New Domains & Analysis Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 32 | Blog Scout prompt / interest-alignment review | `work_order_32_blog_scout_prompt_review.md` | 📝 Drafted — blocked on a JSON export from the owner | none |
| 33 | NBA data domain (new build) | `work_order_33_nba_domain.md` | 📝 Drafted — Step 1 (materials request) not yet actioned | none |
| 34 | Soccer data domain (new build) | `work_order_34_soccer_domain.md` | 📝 Drafted — Step 1 (materials request) not yet actioned | loosely coordinate with WO#33 |
| 35 | Medium article extraction (revisit) | `work_order_35_medium_domain_revisit.md` | ✅ Executed — see `WO35-domains-medium-creation.md`. Both phases completed correctly (real assessment before any code, direction confirmed, then a clean own-domain build under `domains/medium/`). Notably **declined an owner request mid-build** to automate the GraphQL flank via browser-TLS-impersonation tooling built to defeat anti-bot detection — correct judgment call, reasoned that manual-vs-automated cadence doesn't change what the tool itself is built to do. **Surfaced a serious, cross-cutting infrastructure bug** (see the banner at the top of this document and item 27 below) — `docker-compose.yml`'s `web` service was missing `AIRFLOW_SECRET_KEY`, silently breaking every Airflow-trigger feature app-wide, not just medium's; fixed and owner-confirmed working. **Open item:** four Alembic migrations exist for this domain but none are confirmed applied to production — this is the one blocking item before calling the domain done. | none |

WO#33 and WO#34 are genuine greenfield builds and were deliberately written
with a looser, exploratory posture (per GOVERNANCE §4.2) rather than this
program's usual restrictive refactor template — they specify architectural
guardrails, not literal steps, and their first instruction is to request
the materials/decisions needed before writing any code. WO#35 is different
in kind (an existing implementation to revisit, not a from-scratch build)
and ran as two phases — assess what exists and get a direction confirmed
before any code is written — and has now executed. WO#32 is the closest to
executable of the remaining three; it only needs the JSON export the
project owner is already producing, and both open questions the original
idea for this item raised (which agent function the Scout DAG calls;
whether a baseline interests list exists) are already resolved directly
inside that work order.

---

## Deferred / Not Yet Scoped — What's Actually Left

As of this revision, **35 work orders exist in total. 32 are executed and
reviewed** (WO#1–31, plus WO#35 — the full run through Track C, Track D's
Tier 1, and Medium is now clean); **3 remain drafted (WO#32–34)**, all in
progress and blocked on owner-supplied materials rather than scoping gaps.
What's below is what's genuinely still either (a) blocked on something
only the project owner can supply, (b) deliberately left unscoped because
drafting it further would mean guessing, or (c) a real gap surfaced by
review that no WO currently owns:

**A. Needs real data or a decision, not more speculation:**

1. ~~Router splitting for `media_recommend.py`, `ci_readme.py`.~~
   **Now executed** — see WO#23–30 in the Track C table above, one per
   domain, matching WO#19's real 14-file lint result. Several rows carry
   their own open decision points — see items 29–33 below.

2. **Agent folder reorganization** (`airflow/agents/*.py`, mirroring
   WO#18's DAG treatment) — **partially resolved.** See the Track D table
   above: a real consumer map exists, it found three distinct risk tiers
   rather than one uniform profile, and only the safest tier (DAG-only
   modules) has executed, as **WO#31, now done**. The other two tiers
   remain deliberately unscoped — Tier 2 (`recipe_agents.py`,
   `weekly_agents.py`) probably doesn't belong under `airflow/agents/` at
   all, and Tier 3 (`blog_agents.py`) can't move incrementally since it's
   consumed by both DAGs and routers simultaneously. Neither should be
   attempted without its own dedicated scoping pass.

**B. Explicitly parked per instruction, not touched:**

3. The four backlog ideas (GOVERNANCE.md §5: adaptive dashboard, digest
   email, in-Docker coding environments, new domains) — untouched, as
   directed.

4. **NBA data extraction + own page + dashboard card (new).** Drafted as
   WO#33 — see the Track E table above. Still blocked: WO#33's own Step 1
   is a request for the extraction code/process and an ingestion-pattern
   decision, neither supplied yet.

5. **Soccer data extraction (FIFA endpoint) + own page + dashboard card
   (new).** Drafted as WO#34, same blocked-on-materials status as item 4.

6. ~~Medium article extraction — revisit existing process (new).~~ **Now
   executed — WO#35.** The original implementation had been lost, so
   Phase 1 assessed via owner recollection instead (explicitly permitted
   by the WO's own text); own-domain direction confirmed before Phase 2
   built `domains/medium/`. See the Track E table above and item 27 below
   for the significant infrastructure bug this domain's build surfaced.
   **Still open:** four Alembic migrations for this domain exist but are
   not confirmed applied to production.

**C. Real open items surfaced by executed work, not previously tracked here:**

*(Items 7–19: unchanged from the prior review — Jobs' undeployed
`UniqueConstraint`/`Index` pair, Recipes' undeployed `needs_review`
migration, Finance's `account_options.html` finding, Journal's companion
report, Media's orphaned shims, the AI Service Layer stub-file
verification, and the rest. Not reproduced here to keep this revision
focused on what changed; see "Still-open pre-deploy blockers" below for
the two that remain genuinely open.)*

20. **Shim removal (WO#20, note 14).** ~~`models.py`'s end-state decision
    (delete vs. reduce to a registry, Option 2 recommended) is fully
    specified but not yet executed.~~ **Now executed — WO#22, Task 1.**
    Direct reading of the actual file (rather than trusting this note's
    own "Option 2 recommended" framing) found that literal Option 2, taken
    as written, wouldn't actually work — a registry file nothing imports
    doesn't guarantee mapper registration. WO#22 resolved this properly:
    root `models.py` was **deleted**, and the mapper-registration
    guarantee now lives as an explicit, module-scope import block in
    `database.py`. Confirmed directly against the live `GOVERNANCE.md`
    §2.4, which now reflects this as historical/closed. **Still open:**
    the `routers/`/`templates/`/`static/` filesystem audit (confirm
    nothing domain-specific was left behind at the old flat locations)
    remains unaddressed, unchanged since WO#10 first flagged it — no WO
    in this series has executed it yet.

21. **DAG reorganization (WO#18, note 15).** 2 DAG files that sat at the
    flat `airflow/dags/` root — closed directly by the project owner,
    outside any work order. `GOVERNANCE.md §2.5`'s pre-committed amendment
    **is now executed — WO#21**, confirmed directly against the live
    document. 13 stale header-comment paths (cosmetic, one more than the
    originally-tracked 12 — `life_os_staging_promoter.py` inherited the
    same stale pattern) **are now all fixed — WO#22, Task 2**, confirmed
    via a post-edit grep and `ast.parse` sweep across all 13 files.

22. **Router line-limit remediation (WO#19, note 16).** 14 files need
    splitting, none of which touch `services/ai/` or `blog_agents.py`.
    **Fully drafted as WO#23–30** — see item 1 above and the Track C
    table. Still not executed. One file, `weekly_plan.py`, turned out on
    direct inspection to likely already be resolved by WO#10's own
    follow-up split (the "413 lines" figure this note originally carried
    almost certainly describes the file *before* that split) — WO#30
    verifies this rather than assuming either the stale figure or a clean
    resolution.

23. **Blog Scout prompt review (new idea, category B).** **Drafted as
    WO#32.** Both open questions this item originally raised (which agent
    function the Scout DAG calls; whether a canonical interests baseline
    exists) are resolved directly inside WO#32 itself. Blocked only on the
    JSON export the project owner is providing.

24. **`GOVERNANCE.md` §2.3 "current state" prose is stale.** *(New, first
    surfaced by review of WO#16.)* The section still opens with "six
    independent implementations of 'call an LLM provider' exist" as a
    live, unresolved problem — but WO#11 through WO#16 have since migrated
    five of the original six callers into `services/ai/providers/`
    (`gemini.py`, `groq.py`, `cerebras.py`) and explicitly documented the
    sixth (`finance_upload.py`'s `google-genai` SDK call) as a deliberate,
    permanent exception rather than remaining debt. No work order has
    updated this prose to match. Recommend folding this into whatever
    future pass finally resolves WO#16's own open items (G.1's stub-file
    verification, G.4's five-vs-six discrepancy) — it's a small,
    low-risk documentation fix once those land, not urgent on its own.

25. **AI Service Layer foundation files still unverified against real
    source.** *(Restated, not new — carried forward explicitly because
    WO#16 did not close it.)* `services/ai/base.py`, `services/ai/keys.py`,
    and `services/ai/__init__.py` have been reconstructed stubs since
    WO#11, never diffed against the real repository, across all six work
    orders in this track. WO#16's own mandatory pre-execution gate called
    for this to be checked before it built anything further on top —
    it was not resolved, only disclosed again. This should be treated as
    a standing blocker on trusting Track A's output in production, not a
    closed item.

**D. Cross-cutting items resolved directly in `GOVERNANCE.md` this pass:**

26. ~~`GOVERNANCE.md` §3.3 Migration Debt Tracker staleness.~~ **Resolved.**
    §3.3 was independently caught as stale by four separate Track C
    postmortems (WO#25, #26, #28, #29) — it still listed `finance`,
    `journal`, `recipes`+`pantry`, `workout`, `media`, and `planning` as
    un-migrated well after WO#5–10 had shipped them. `GOVERNANCE.md` §3.3
    has now been rewritten to mark this historical/closed, mirroring how
    §2.4 was handled after WO#20/#22.

27. ~~`AIRFLOW_SECRET_KEY` missing from `docker-compose.yml`'s `web`
    service.~~ **Bug fixed (during WO#35); rule now codified.** WO#35's
    build was the first feature to actually exercise
    `trigger_airflow()`'s live credential path end-to-end, and in doing so
    found this had been silently breaking *every* Airflow-trigger feature
    already shipped (blog's Scout/Creator/Finalizer/Idea Expander,
    code_intel's README Writer and Narrate/Comment/Improve DAGs) — not
    just medium's. Fixed and owner-confirmed working. `GOVERNANCE.md` now
    has a new §2.7 codifying the generalizable rule: verify a secret is
    threaded through *every* `docker-compose.yml` service that will
    actually call it, not just the service that owns the secret's other
    config values.

28. ~~`main.py` merge-coordination risk across Track C's eight `main.py`
    edits.~~ **Resolved — confirmed directly by the project owner:** the
    merge across WO#23–30's `main.py` changes has been coordinated and
    applied. No further action needed on this specific point. (The
    underlying lesson — audit every router-split WO's `main.py`
    registration explicitly rather than assuming it composes — is still
    worth folding into the standing work-order template per WO#29 §10.4
    and WO#30 §9.6's own recommendation, but it's no longer a live risk
    for this batch.)

**E. Track C open items — each needs a specific owner decision, not a
mechanical fix (surfaced during this review, none blocking the rest of the
program):**

29. **`ci_files.py` line count (WO#24, Part B).** Left at 324 lines because
    the original work order's own STEPS ("keep these 7 endpoints +
    docstrings together") and its ACCEPTANCE CRITERIA ("all files ≤300")
    directly conflict — correctly reported rather than unilaterally
    resolved. **Needs your decision:** Option A (document a 24-line
    exception in `GOVERNANCE.md` §1.2) or Option B (move
    `update_comment_status` into its own small file).

30. **Recipes' `__import__(...)` fix authorization (WO#26).** The original
    work order explicitly said "do not fix" this pre-existing style
    oddity — it was fixed anyway, bundled into the same diff as the
    router split, at what the postmortem describes as "the requester's
    explicit direction," but with no verbatim quote on record the way
    WO#29's Part C authorization has. **Needs either:** a confirming quote
    for the record, or splitting this fix into its own commit per the
    postmortem's own §7 recommendation, before treating the bundled diff
    as fully compliant.

31. **`media_recommend.py` / `media.py` line counts (WO#28).** Same shape
    of conflict as item 29 — both resulting files (316, 309 lines) still
    exceed the ceiling because the pinned function sets don't fit under
    it. A Part C follow-up split is proposed (move
    `recommendation_history` and `update_notes` out) but explicitly
    **not self-authorized** by the postmortem — needs its own work order
    if you want it done.

32. **Journal → Planning cross-domain import pattern (WO#25).** The
    `save_entry()` link to `domains.planning.models` is a defensive,
    function-local import, not the sanctioned string-`relationship()`
    pattern §2.2 describes elsewhere. **Needs your decision:** promote to
    a real `relationship()` (requires a joint work order touching both
    domains, per the `blog`+`code_intel` precedent), or promote the
    existing import to a normal top-level one and drop the
    `try/except Exception: pass` guard. Not a call a router-split WO
    should make unilaterally, and correctly left alone.

33. **`weekly_plan_generator`/`weekly_plan_shopping` registration gap
    (WO#30) — needs its own tracked ticket even though it's already
    fixed.** This was a real, pre-existing production bug (both routers
    were imported in `main.py` but never registered, so `POST
    /plan/generate` and `GET /plan/{id}/shopping` were 404ing) found and
    fixed in the same patch as WO#30's own required `main.py` edit, but
    explicitly labeled as a separate change per §4.5's spirit. File it as
    its own line item in whatever change-tracking this repo uses — don't
    let it get absorbed into "WO#30" in history, since it predates that
    work order and isn't planning-router-split work in substance.

34. **Habits' cross-router registration-order shadowing risk (WO#23).**
    `DELETE /habits/log` can be shadowed by `DELETE /habits/{habit_id}` if
    `main.py` registers `habits_settings` before `habits` — confirmed via
    a negative-control test. No fix needed (the correct order is already
    documented in both files' docstrings and, per your confirmation in
    item 28, the real `main.py` merge has already been coordinated
    correctly) — but this is a *pattern*, not a one-off: any future router
    split that separates literal-path routes from a parametric `/{id}`
    route into different files is exposed to the same risk. Worth adding
    an automated route-order regression test and/or folding the check
    into the standing work-order template, per WO#23 §7.6's own
    recommendation.

---

## Roadmap — Recommended Next Tracks

### Track A — AI Service Layer capstone
✅ **Executed — WO#16.** Nothing else in this document is gated on it.
Two open items remain before this track can be trusted unconditionally in
production: the reconstructed-stub verification (item 25 above) and the
`GOVERNANCE.md` §2.3 documentation staleness (item 24 above). Neither
blocks other tracks; both should be picked up as their own small,
low-risk follow-ups.

### Track B — Loose ends
✅ **Executed — WO#21 and WO#22** (see the Track B table above). WO#21 is
fully closed. WO#22 is closed for Tasks 1–2; **Task 3
(`configure_mappers()` against a real environment) remains open and is now
the single highest-priority item in the whole program** — see "Where This
Actually Stands" below. The original third item in this track (moving the
two straggler DAG files) was already closed by the project owner directly,
outside any work order, before WO#21/#22 ran.

### Track C — Router line-limit remediation
✅ **Executed — WO#23 through WO#30** (see the Track C table above), one
per domain, matching the real 14-file list WO#19's lint script produced.
`main.py` merge across all eight rows has been coordinated and applied by
the project owner directly — no outstanding conflict risk. Five open
decision points remain, none blocking: `ci_files.py`'s and
`media_recommend.py`/`media.py`'s line-count exceptions (items 29, 31),
the recipes `__import__` fix's authorization trail (item 30), the
journal↔planning import pattern (item 32), and the standalone bug ticket
WO#30 asked to have filed (item 33). See the Deferred section above for
each.

### Track D — `airflow/agents/*.py` reorganization
🔶 **Tier 1 executed, Tiers 2–3 deliberately unscoped.** The pre-flight
risk analysis this track called for (`track_D_agent_reorg_scoping_analysis.md`)
found the directory doesn't have one uniform risk profile — DAG-only
modules, router-only modules that are arguably misplaced under
`airflow/agents/` at all, and one dual-consumed module (`blog_agents.py`)
that can't move incrementally. **WO#31 (the DAG-only tier) has now
executed** and is reviewed above. The other two tiers still need their own
dedicated future scoping pass — see that analysis document's Part 3 before
treating this track as fully "done."

### Track E — New domains and analysis
🔶 **WO#35 executed; WO#32–34 remain in progress, blocked on materials by
design.** Per instruction, these work orders were written so that
whichever agent runs them acknowledges the work order and requests what it
needs as its own first step, rather than this document guessing at
NBA/Soccer extraction code or an existing Medium implementation it hadn't
seen. **WO#35 (Medium) has completed both phases** — assessment, then a
clean own-domain build — and, as a byproduct, surfaced the
`AIRFLOW_SECRET_KEY` infrastructure bug now fixed and codified in
`GOVERNANCE.md` §2.7. WO#32 (Blog Scout review) is closest to executable
of the remaining three — it only needs the JSON export the project owner
is already producing. WO#33 (NBA) and WO#34 (Soccer) are genuine
greenfield builds, deliberately written with a looser, exploratory posture
per GOVERNANCE §4.2 rather than the restrictive refactor template the rest
of this program uses — they specify architectural guardrails (stay inside
`domains/<name>/`, route AI calls through `services/ai/`, respect the
DAG/FastAPI boundary), not literal steps.

**Cross-track note:** with Track C, Track D's Tier 1, and WO#35 now
executed, the only work still outstanding anywhere in this document is
WO#22's Task 3, WO#32–34 (each waiting on a specific named input), and the
handful of owner-decision items in the Deferred section above. There is no
sequencing constraint left — everything remaining is either blocked on you
or ready to run independently.

---

## Where This Actually Stands

**35 work orders + `GOVERNANCE.md` are drafted. 32 are executed and
reviewed:** the full domain-migration backlog (WO#1–10), the complete AI
Service Layer series (WO#11–16), WO#17 (toast consolidation), WO#18–20
(DAG reorg, dead code + lint, shim removal), Track B in full (WO#21–22),
all eight Track C router splits (WO#23–30), Track D's Tier 1 (WO#31), and
WO#35 (Medium). All thirty-two pass their alignment and governance audits
— none were rejected, though several carry disclosed open items rather
than an unconditional clean pass (see the Deferred section's items 25 and
29–34 for the specific ones still live). **3 remain drafted and blocked on
owner-supplied materials: WO#32–34.**

**Top priority, unchanged in substance but now the single largest open
item in the whole program: run the `configure_mappers()` check for real.**
This is **WO#22, Task 3**, and it was genuinely attempted — not skipped —
but blocked by a sandboxed session lacking network access and the real
application source (`domains/`, `core/`, `routers/`, `gcp_secrets.py` were
never part of that engagement's materials). The reasoning for why this
matters hasn't changed: every domain's shim is gone (WO#20) and
`models.py` itself is now deleted (WO#22, Task 1), so the *only* thing left
providing the implicit mapper-registration guarantee is `database.py`'s
new import block, and that block's correctness has never been confirmed in
a real environment across this entire 32-work-order-deep program. The
concrete next action: run `docker compose exec web python -c "import
sqlalchemy.orm as orm; import main; orm.configure_mappers()"` against the
live deployed container, and fold the result back into WO#22's postmortem
and this index.

**Second: two `GOVERNANCE.md` amendments landed this pass.** §3.3's
Migration Debt Tracker was corrected (marked historical/closed, matching
§2.4's treatment) after four independent postmortems caught it listing
already-migrated domains as outstanding. A new §2.7 codifies the
secret/environment-variable-wiring rule WO#35's `AIRFLOW_SECRET_KEY`
discovery surfaced. Neither required touching any code.

**Third: five owner-decision items from Track C, none blocking.** Two
line-count exceptions need Option A/B calls (`ci_files.py`, item 29;
`media_recommend.py`/`media.py`, item 31), one bundled fix needs an
authorization quote or a commit split (recipes' `__import__`, item 30),
one cross-domain import pattern needs a direction (journal↔planning, item
32), and one pre-existing production bug WO#30 found and fixed needs its
own standalone ticket filed so it doesn't get absorbed into "WO#30" in
history (item 33). None of these block Track D Tier 2/3, WO#32–34, or
WO#22's Task 3 — they're independent.

**Fourth: two AI Service Layer follow-ups, both low-risk, neither urgent
on its own:** swap in the real `services/ai/base.py`, `keys.py`, and
`__init__.py` before trusting WO#11–16's output unconditionally in
production (item 25), and refresh `GOVERNANCE.md` §2.3's stale "six
implementations" framing once that verification lands (item 24 — still
open, distinct from the §3.3/§2.7 amendments made this pass).

**Still-open pre-deploy blockers, unchanged from the last review:** the
Jobs domain's undeployed `UniqueConstraint`/`Index` pair (item 7) and the
Recipes domain's undeployed `needs_review` Alembic migration (item 11).
Neither was touched in this review pass. Both are cheap to close and get
more entangled the longer they're left.

**A pattern worth naming explicitly, now confirmed independently across
several separate reviews of this program:** WO#17's `base.css` oversight,
WO#20's original router-source claim, WO#15's shipped content truncation,
and a stale line-count figure for `weekly_plan.py` that survived in this
very index until a direct file read caught it (WO#30). This batch added
another data point in the same family, from a different angle: four
separate postmortems (WO#25, #26, #28, #29) each independently rediscovered
the same `GOVERNANCE.md` §3.3 staleness rather than any one of them
catching it and the others trusting that finding — a sign that "verify
directly" is being practiced well at the individual-postmortem level but
that cross-postmortem findings aren't yet being pooled efficiently. Worth
keeping in mind for how future batches of postmortems get reviewed
together, not just how each one is checked in isolation.
