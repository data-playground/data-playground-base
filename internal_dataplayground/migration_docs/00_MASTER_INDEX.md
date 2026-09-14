# Life OS Restructuring — Master Index

**Purpose:** single reference for every deliverable produced across this
engagement, and its actual execution status. Update the STATUS column as
work orders are run — this file is meant to stay accurate, not just be a
historical log.

**Last reviewed:** This revision adds Work Orders #21–35, drafted across
Track B (loose ends), Track C (router line-limit remediation), Track D
(agent folder reorganization — Tier 1 only), and Track E (new domains and
analysis). None of #21–35 have been executed yet — they're tracked the same
way WO#16 was tracked before it ran: drafted, reviewed, parallel-safe,
waiting on either execution or, for several of them, a specific named input
from the project owner. The Roadmap and Deferred sections below have been
cleaned up to reflect this — most of what they used to describe as
"recommended for future scoping" now has an actual file. The prior review
(covering WO#1–13, #17, and #20) is preserved unchanged everywhere it still
applies.

---

## Foundational Documents

- `GOVERNANCE.md` — the standing architectural rules this entire program
  works against. Amended once already (see the DAG Reorganization table
  below); amended again in this pass (WO#21).
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
| 16 | Capstone | `work_order_16_ai_service_capstone.md` | 📝 Drafted, not executed |

*This series is strictly sequential — unchanged. WO#16 is the only item
left in Track A and, per the Roadmap below, no longer blocks anything else
in this program.*

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

*Two follow-on commitments these three work orders' own postmortems made —
the GOVERNANCE.md §2.5 amendment (from WO#18) and the `models.py` end-state
decision plus stale DAG header cleanup (from WO#10 and WO#20) — sat
unapplied for a while. They're now drafted as WO#21 and WO#22; see the
Track B table immediately below.*

---

## Track B — Loose Ends Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 21 | Apply the GOVERNANCE.md §2.5 amendment (DAG relocation risk reframing) | `work_order_21_governance_dag_amendment.md` | 📝 Drafted, not executed | WO#18 |
| 22 | `models.py` end-state + 13 stale DAG header comments + `configure_mappers()` verification | `work_order_22_models_dag_headers_configure_mappers.md` | 📝 Drafted, not executed | WO#10, WO#18, WO#20 |

Both are independent of each other and of every other track in this
document — run in either order, or in parallel. The third original Track B
item (moving `life_os_staging_promoter.py` and
`life_os_refresh_streaming_availability.py` into their DAG subfolders) was
already closed by the project owner directly, outside any work order,
before this drafting pass started — confirmed by reading current source.

---

## Track C — Router Line-Limit Remediation Work Orders

| # | Domain | File | Status | Depends on |
|---|---|---|---|---|
| 23 | Habits | `work_order_23_habits_router_split.md` | 📝 Drafted, not executed | none |
| 24 | Code Intel (3 files) | `work_order_24_code_intel_router_split.md` | 📝 Drafted, not executed | none |
| 25 | Journal | `work_order_25_journal_router_split.md` | 📝 Drafted, not executed | none |
| 26 | Recipes (2 files) | `work_order_26_recipes_router_split.md` | 📝 Drafted, not executed | none |
| 27 | Blog | `work_order_27_blog_router_split.md` | 📝 Drafted, not executed | none |
| 28 | Media (2 files) | `work_order_28_media_router_split.md` | 📝 Drafted, not executed | none |
| 29 | Workout (3 files) | `work_order_29_workout_router_split.md` | 📝 Drafted, not executed — Part C blocked on owner sign-off | none |
| 30 | Planning — verification only | `work_order_30_planning_router_verification.md` | 📝 Drafted, not executed — likely resolves as a no-op | none |

Covers the real 14-file list WO#19's lint script produced (not the
2-file guess an earlier draft of this document carried). All eight are
confirmed parallel-safe with each other and with every other track — none
share a file. WO#29's Part C (`workout_log.py`) is explicitly **not**
authorized to execute without a fresh owner sign-off, since the only
available split would reverse a structural decision WO#8 deliberately made.
WO#30 was drafted verification-first because direct inspection of the
current source suggests `weekly_plan.py` may already be resolved by an
earlier, unrelated follow-up split — its own line-count figure elsewhere in
this document (see item 22 below) was stale.

---

## Track D — Agent Folder Reorganization

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| — | Pre-flight risk analysis (real consumer map, tiering) | `track_D_agent_reorg_scoping_analysis.md` | ✅ Complete | WO#18 |
| 31 | Tier 1 reorg — DAG-only agent modules | `work_order_31_agent_reorg_tier1.md` | 📝 Drafted, not executed | Track D analysis |

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
| 35 | Medium article extraction (revisit) | `work_order_35_medium_domain_revisit.md` | 📝 Drafted — two-phase; Phase 1 Step 1 (materials request) not yet actioned | none |

WO#33 and WO#34 are genuine greenfield builds and were deliberately written
with a looser, exploratory posture (per GOVERNANCE §4.2) rather than this
program's usual restrictive refactor template — they specify architectural
guardrails, not literal steps, and their first instruction is to request
the materials/decisions needed before writing any code. WO#35 is different
in kind (an existing implementation to revisit, not a from-scratch build)
and runs as two phases — assess what exists and get a direction confirmed
before any code is written. WO#32 is the closest to executable of the four;
it only needs the JSON export the project owner is already producing, and
both open questions the original idea for this item raised (which agent
function the Scout DAG calls; whether a baseline interests list exists) are
already resolved directly inside that work order.

---

## Deferred / Not Yet Scoped — What's Actually Left

As of this revision, **35 work orders now exist in total** — WO#1–20 from
before, plus WO#21–35 drafted in this pass (see the four new tables above).
Most of what this section used to list as "not yet scoped" now has an
actual file. What's below is what's genuinely still either (a) blocked on
something only the project owner can supply, or (b) deliberately left
unscoped because drafting it further would mean guessing:

**A. Needs real data or a decision, not more speculation:**

1. ~~Router splitting for `media_recommend.py`, `ci_readme.py`.~~
   **Now drafted** — see WO#23–30 in the Track C table above, one per
   domain, matching WO#19's real 14-file lint result. (An earlier draft of
   this document pointed at a planned `track_C_router_line_limit_briefs.md`
   file; that name was never produced — the individual per-domain work
   orders supersede that plan.)

2. **Agent folder reorganization** (`airflow/agents/*.py`, mirroring
   WO#18's DAG treatment) — ~~still genuinely unscoped~~. **Partially
   resolved.** See the Track D table above: a real consumer map now exists,
   it found three distinct risk tiers rather than one uniform profile, and
   only the safest tier (DAG-only modules) is drafted, as WO#31. The other
   two tiers remain deliberately unscoped.

**B. Explicitly parked per instruction, not touched:**

3. The four backlog ideas (GOVERNANCE.md §5: adaptive dashboard, digest
   email, in-Docker coding environments, new domains) — untouched, as
   directed.

4. **NBA data extraction + own page + dashboard card (new).** ~~Kept as
   its own, separate work order per explicit instruction.~~ **Now drafted
   as WO#33** — see the Track E table above. Still blocked: WO#33's own
   Step 1 is a request for the extraction code/process and an
   ingestion-pattern decision, neither supplied yet.

5. **Soccer data extraction (FIFA endpoint) + own page + dashboard card
   (new).** **Now drafted as WO#34**, same blocked-on-materials status as
   item 4.

6. **Medium article extraction — revisit existing process (new).** **Now
   drafted as WO#35**, structured as a two-phase work order (assess the
   existing implementation and decide own-domain-vs-fold-into-blog before
   building) since, unlike NBA/Soccer, this isn't greenfield. Still blocked
   on the existing implementation being shared.

**C. Real open items surfaced by executed work, not previously tracked here:**

*(Items 7–19: unchanged from the prior review — Jobs' undeployed
`UniqueConstraint`/`Index` pair, Recipes' undeployed `needs_review`
migration, Finance's `account_options.html` finding, Journal's companion
report, Media's orphaned shims, the AI Service Layer stub-file
verification, and the rest. Not reproduced here to keep this revision
focused on what changed; see "Still-open pre-deploy blockers" below for the
two that remain genuinely open.)*

20. **Shim removal (WO#20, note 14).** ~~`models.py`'s end-state decision
    (delete vs. reduce to a registry, Option 2 recommended) is fully
    specified but not yet executed.~~ **Now drafted as WO#22, Task 1** —
    worth flagging explicitly: reading the actual current file directly
    (rather than trusting this note's own "Option 2 recommended" framing)
    found that literal Option 2, taken as written, wouldn't actually work —
    a registry file nothing imports doesn't guarantee mapper registration.
    WO#22 resolves this properly; see that work order's own HARD BOUNDARIES
    for the correction. The `routers/`/`templates/`/`static/` filesystem
    audit (confirm nothing domain-specific was left behind at the old flat
    locations) remains open, unchanged since WO#10 first flagged it.

21. **DAG reorganization (WO#18, note 15).** ~~2 DAG files still sit at the
    flat `airflow/dags/` root~~ — closed directly by the project owner,
    outside any work order. `GOVERNANCE.md §2.5`'s pre-committed amendment
    ~~is still ready and unapplied~~ **is now drafted as WO#21.** 12 stale
    header-comment paths (cosmetic) ~~also still unfixed~~ — direct review
    found a 13th (`life_os_staging_promoter.py`, added after the original
    count and inheriting the same stale pattern) — **all 13 are now
    covered by WO#22, Task 2.**

22. **Router line-limit remediation (WO#19, note 16).** ~~14 files need
    splitting, none of which touch `services/ai/` or `blog_agents.py`.~~
    **Now fully drafted as WO#23–30** — see item 1 above and the Track C
    table. One file, `weekly_plan.py`, turned out on direct inspection to
    likely already be resolved by WO#10's own follow-up split (the "413
    lines" figure this note originally carried almost certainly describes
    the file *before* that split) — WO#30 verifies this rather than
    assuming either the stale figure or a clean resolution.

23. **Blog Scout prompt review (new idea, category B).** ~~Full scoping
    brief... in track_E_new_domains_and_analysis_briefs.md.~~ **Now
    drafted as WO#32.** Both open questions this item originally raised
    (which agent function the Scout DAG calls; whether a canonical
    interests baseline exists) are resolved directly inside WO#32 itself.
    Blocked only on the JSON export the project owner is providing.

---

## Roadmap — Recommended Next Tracks

### Track A — AI Service Layer capstone
*Unchanged — ongoing, not part of this revision.* WO#16 is drafted and
ready; nothing else in this document blocks it, and it no longer blocks
anything else either.

### Track B — Loose ends
✅ **Drafted — WO#21 and WO#22** (see the Track B table above). Both are
independent of each other and of every other track; run in either order.
The original third item in this track (moving the two straggler DAG files)
was already closed by the project owner directly, outside any work order,
before this drafting pass started.

### Track C — Router line-limit remediation
✅ **Drafted — WO#23 through WO#30** (see the Track C table above), one
per domain, matching the real 14-file list WO#19's lint script produced.
All eight are confirmed parallel-safe with each other and with every other
track. One exception: WO#29's Part C (`workout_log.py`) is explicitly not
authorized to execute without a fresh owner sign-off, since the only
available split would reverse a structural decision WO#8 deliberately made.
WO#30 (`weekly_plan.py`) may resolve as a no-op — direct inspection
suggests it's likely already fixed by an earlier, unrelated follow-up
split.

### Track D — `airflow/agents/*.py` reorganization
🔶 **Partially drafted.** The pre-flight risk analysis this track called
for is done (`track_D_agent_reorg_scoping_analysis.md`) and found the
directory doesn't have one uniform risk profile — DAG-only modules,
router-only modules that are arguably misplaced under `airflow/agents/` at
all, and one dual-consumed module (`blog_agents.py`) that can't move
incrementally. WO#31 drafts only the DAG-only tier. The other two tiers are
deliberately left unscoped rather than guessed at — see that analysis
document's Part 3 before treating this track as "done."

### Track E — New domains and analysis
📝 **Drafted but blocked on materials, by design.** Per instruction, these
work orders were written so that whichever agent runs them acknowledges the
work order and requests what it needs as its own first step, rather than
this document guessing at NBA/Soccer extraction code or an existing Medium
implementation it hasn't seen. WO#32 (Blog Scout review) is closest to
executable — it only needs the JSON export the project owner is already
producing. WO#33 (NBA) and WO#34 (Soccer) are genuine greenfield builds,
deliberately written with a looser, exploratory posture per GOVERNANCE
§4.2 rather than the restrictive refactor template the rest of this program
uses — they specify architectural guardrails (stay inside `domains/<name>/`,
route AI calls through `services/ai/`, respect the DAG/FastAPI boundary),
not literal steps. WO#35 (Medium) runs as two phases — assess the existing
implementation and get a direction confirmed before any code is written,
since the right architecture depends entirely on what that assessment
finds.

**Cross-track note:** every work order in Tracks B, C, D (Tier 1), and E is
confirmed parallel-safe with Track A's remaining WO#16 and with each other —
none share a file. There is no sequencing constraint left anywhere in this
document; the only real gates remaining are the sign-off WO#29's Part C
needs and the materials WO#32–35 are each waiting on.

---

## Where This Actually Stands

**35 work orders + `GOVERNANCE.md` are drafted.** 19 are executed and
reviewed, unchanged from before this pass: the full domain-migration
backlog (WO#1–10), WO#11–15 of the AI Service Layer, WO#17 (toast
consolidation), and WO#18–20 (DAG reorg, dead code + lint, shim removal).
All nineteen pass their alignment and governance audits — none were
rejected. **16 more are drafted but not yet executed: WO#16, and the new
WO#21–35.**

**Top priority is unchanged in substance but now has a home: run the
`configure_mappers()` check.** This used to be an ad hoc instruction in
this section; it's now formalized as **WO#22, Task 3**, which runs after
that work order's Task 1 (`models.py`'s end state) and Task 2 (DAG header
cleanup) land. The reasoning hasn't changed: every domain's shim is gone,
so the only thing left providing the implicit mapper-registration guarantee
is `main.py`'s router-import order, and that's never been independently
confirmed in a real environment across this entire program. Run WO#22 and
this closes for good.

**Second: work the newly-drafted tracks — B, C, D (Tier 1), and E are all
ready or close to it.** WO#21–22 (Track B), WO#23–30 (Track C), and WO#31
(Track D, Tier 1) are confirmed parallel-safe with each other, with WO#16,
and with everything else — none share a file. WO#32–35 (Track E) are
drafted but each is blocked on a specific, named input (a JSON export, or a
materials request that hasn't been answered yet) rather than a scoping gap
— see the Track E table and the Roadmap section above for exactly what each
is waiting on.

**Third: before trusting WO#13, #14, or #15's output in production, swap
in the real `services/ai/base.py`, `keys.py`, and `__init__.py`.**
Unchanged from the prior review — this is Track A's own concern and wasn't
touched in this pass.

**Still-open pre-deploy blockers, unchanged from the last review:** the
Jobs domain's undeployed `UniqueConstraint`/`Index` pair (item 7) and the
Recipes domain's undeployed `needs_review` Alembic migration (item 11).
Neither was touched in this drafting pass. Both are cheap to close and get
more entangled the longer they're left.

**A pattern worth naming explicitly, now confirmed at least four times
independently** — WO#17's `base.css` oversight, WO#20's original
router-source claim, WO#15's shipped content truncation, and, at smaller
stakes, this document's own prior note carrying a stale line count for
`weekly_plan.py` that a direct read of the current file called into
question before WO#30 was drafted: materials or claims presented as
complete, unmodified, or accurate, when they weren't or might not be,
caught only by someone actually checking rather than by the original
process. This pass tried to hold itself to that standard — every new work
order above that references a specific file's current state (line counts,
which imports exist, which headers are stale) was written from a direct
read of that file, not from this document's own earlier figures, and one
place where those two sources actually disagreed (`weekly_plan.py`,
item 22) is called out rather than silently resolved either way. Worth
continuing to build into the standing work-order template itself, per
WO#17's own §7.4 recommendation.
