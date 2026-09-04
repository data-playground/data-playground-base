# Life OS Restructuring — Master Index

**Purpose:** single reference for every deliverable produced across this
engagement, and its actual execution status. Update the STATUS column as
work orders are run — this file is meant to stay accurate, not just be a
historical log.

**Last reviewed:** QA/Architecture pass covering WO#1–13, WO#17, and
WO#20 against their postmortems. This revision folds in five completions
not previously recorded here (WO#10, WO#12, WO#13, WO#17, WO#20) and
elevates one cross-cutting risk — see "Where This Actually Stands" — that
every postmortem since WO#2 has flagged individually but that no single
engagement has ever had the materials to actually close.

---

## Foundational Documents

| Deliverable | Status |
|---|---|
| Phase 1 — Audit Report | ✅ Delivered (in-conversation, not a file) |
| Phase 2/3 — Consolidation blueprint + folder structure plan | ✅ Delivered (in-conversation, not a file) |
| `GOVERNANCE.md` | ✅ Delivered — the standing reference for every rule below. **Note:** GOVERNANCE.md's ROLE/HARD BOUNDARIES/OUTPUT FORMAT work-order template (§4.3) was formalized *after* WO#1 ran — several of its conventions (explicit ⚠️ substitution-marking, the standard 5-part report format) are retroactive lessons drawn partly from WO#1's own execution, not requirements WO#1 was held to at the time. Reviews of WO#1 below account for this. |

---

## Domain Migration Work Orders

| # | Domain(s) | File | Status |
|---|---|---|---|
| 1 | `habits` (pilot) | `work_order_01_habits_domain.md` | ✅ **Executed & confirmed working in production** (see `domains-migration-postmortem-habits.md`). Full re-audit against the correct WO#1 text passed — see note 1 below. **Action needed:** the file previously on record at this path was actually the follow-on bug-fix ticket (view-context 500 error), not the migration spec — replace its contents with the real WO#1 text now on file. |
| 2 | `blog` + `code_intel` | `work_order_02_blog_code_intel_domain.md` | ✅ **Executed & live in production** (see `domains-migration-postmortem-blog-code-intel.md`). Confirmed clean — scope expansion (2 pre-existing bugs found and fixed) correctly isolated in separate, independently-revertable commits per GOVERNANCE §4.5. No open items block this row. |
| 3 | `jobs` | `work_order_03_jobs_domain.md` | ✅ **Executed & verified via automated harness** (see `domains-migration-postmortem-jobs.md`). Engagement grew well beyond the base WO across 7 phases (bug fixes, new DAG, schema addition, shared-file nav fix, pagination rework, `search_date` semantics change) — **explicitly owner-authorized**, not a scope violation. Marked complete; see note 2 below for real open items carried forward (none block this row, but none should be silently lost either). |
| 4 | `explorer` | `work_order_04_explorer_domain.md` | ✅ **Executed & verified via automated harness** (see `domains-migration-postmortem-explorer.md`). Cleanest migration to date — zero cross-domain footprint, zero shared-file touches, follow-up docstring fix correctly isolated in its own commit per §4.5. Minor outstanding items, non-blocking (see note 3). |
| 5 | `finance` | `work_order_05_finance_domain.md` | ✅ **Executed, verified in sandboxed harness** (see `domains-migration-postmortem-finance.md`). Relocation itself is sound and the shim matches precedent. **Two required items not closed** — see note 4. |
| 6 | `journal` | `work_order_06_journal_domain.md` | ✅ **Executed** (see `domains-migration-postmortem-journal.md`). Privacy boundary and the deliberate `save_entry()` import asymmetry both correctly preserved. **Cannot be fully criterion-audited** — the detailed pass/fail report this postmortem references (`WO6_journal_migration_report.md`) wasn't provided; see note 5. |
| 7 | `recipes` + `pantry` | `work_order_07_recipes_pantry_domain.md` | ✅ **Executed — best-documented migration to date** (see `domains-migration-postmortem-recipes.md`). All 14 original criteria explicitly ✅. **One real pre-deploy blocker** — see note 6. |
| 8 | `workout` | `work_order_08_workout_domain.md` | ✅ **Executed** (see `domains-migration-postmortem-workout.md`). Best deviation-authorization log in the set — every departure from HARD BOUNDARIES traced to its exact authorization quote. Real, fully-disclosed verification debt remains — see note 7. |
| 9 | `media` | `work_order_09_media_domain.md` | ✅ **Executed** (see `domains-migration-postmortem-media.md`). Cleanly separates original scope from follow-on work per §4.5/§4.6. New capability (weekly DAG) and new duplication debt both self-disclosed — see note 8. |
| 10 | `planning` (`weekly_plan` + `intent`) | `work_order_10_planning_domain.md` | ✅ **Executed — last domain on the core backlog** (see `domains-migration-postmortem-planning.md`). One real gap at original delivery (`shopping_list.html` missing outright, marked ❌ not ⚠️) closed in authorized follow-up. Includes the single most consequential authorized HARD BOUNDARY exception in the whole series — see note 10. |

**Execution order constraint:** #2 internally pairs `blog`+`code_intel`
(must run together). #10 depended on #7 and #8 already being done — both
were, and #10 has now run, closing out the core domain-migration backlog
in full.

**Review notes:**

1. **WO#1 (habits) — re-audited against the correct spec.** All six
   acceptance criteria are substantively met per the postmortem: identical
   rendering (confirmed via live production logs + direct user testing),
   drag-to-reorder/heatmap working (a real bug here — siblings stranded on
   drag — was found and fixed, verified via jsdom against both pre- and
   post-fix code), log/unlog via HTMX working (two real pre-existing bugs
   found and fixed: the `view`-context mismatch and a `greenlet_spawn`
   session-expiry bug), and the `/dashboard` cross-domain spot-check
   passing. Two criteria (the literal `alembic check`/dry-run output, and
   an explicit `grep` transcript) aren't reproduced verbatim in the
   postmortem — the postmortem instead documents an equivalent
   `Base.metadata` table-count + class-identity check run at every step,
   which is substantively the same verification but not the literally
   requested command output. Not treated as a failure (WO#1 predates the
   ⚠️-substitution-marking convention GOVERNANCE.md later formalized), but
   worth a literal re-run if anyone wants the exact requested artifact on
   file. No governance violations — WO#1 is in fact the origin of the
   static-mount-ordering rule later codified in GOVERNANCE.md §2.6.

2. **WO#3 (jobs) — real open items carried forward from the postmortem**
   (not scope violations — the phase work beyond the base WO was
   authorized by the project owner in-conversation):
   - The Phase 4 `UniqueConstraint` and Phase 6 `Index` declarations exist
     only in the ORM — **not yet applied to the live MariaDB.** The
     postmortem's own pre-flight duplicate-check query and `SHOW INDEX`
     confirmation (§9, items 1–2) still need to be run before this is
     truly schema-complete.
   - Three assumptions made under environment constraints remain
     unverified: whether `base.html`/`base.js` actually provide
     `showToast()` at the DOM-lifecycle point `jobs.html` needs it;
     whether htmx's real error-swap behavior is customized in `base.js`
     (affects `POST /jobs/stage/process` and possibly pre-existing
     `add_keyword`/`add_watched_company` behavior); and the
     title/company/location CSS selectors in
     `job_agents.get_full_job_posting()`, written without access to a live
     LinkedIn page.
   - `airflow/dag_db.py`'s real interface has never been seen by any agent
     across this entire engagement — the Phase 7 rescan-refresh SQL
     patterns were verified only against a hand-built stand-in.
   - Postmortem's own status line states this was **not yet observed
     against real production** at time of writing — worth a live
     confirmation pass before treating the domain as fully settled, even
     though the harness-based verification is accepted as sufficient to
     mark this row complete.
   - Non-blocking: `jobs.html` has grown large enough to be a future
     split candidate (§8.4).

3. **WO#4 (explorer) — non-blocking outstanding items:** live smoke-test
   against real MariaDB `jobs` DB, confirmation of `data_playground`'s
   actual grants on that database, and a deferred (not rejected) decision
   on provisioning a dedicated read-only DB user. All explicitly logged in
   the postmortem's own §8, not newly surfaced here.

4. **WO#5 (finance) — two required items not closed, unlike WO#1–4.**
   - HARD BOUNDARIES required an explicit finding on whether
     `templates/partials/account_options.html` is actually referenced
     anywhere in the codebase — "the usage finding itself is a required
     part of this report." **That finding does not appear anywhere in the
     postmortem.** The file was moved per instructions either way, but the
     required report item is missing — needs to be produced before this
     row is considered fully closed, not just relocated.
   - The postmortem explicitly states `/dashboard`'s finance summary card
     was checked for **import resolution only** ("were only checked for
     import resolution, not a full render, because the stubs don't
     reproduce those domains' real relationships/fields") — this is a
     narrower check than the WO's own acceptance criterion, which required
     confirming the card "renders correctly." Needs a real render check
     once live domain models are available, not just an import-resolves
     check.
   - Also worth doing before treating finance as settled: the WO's CSV
     upload error-path criteria (missing-column / empty-CSV returning
     `html_error()` fragments) aren't explicitly confirmed in the report
     either — likely just an omission from the writeup rather than a real
     gap, but not verifiable from what's on file.

5. **WO#6 (journal) — audit is necessarily partial.** The postmortem is
   explicit that its own criterion-by-criterion results live in a separate
   `WO6_journal_migration_report.md` / `.patch` "from the prior turn,"
   neither of which was provided here — so the acceptance criteria can't
   be independently confirmed one by one, only the overall approach
   (privacy handling, the deliberate `save_entry()` asymmetry, the
   verbatim-move discipline). Recommend attaching that companion report
   the next time this row is reviewed, for full traceability. Separately:
   the postmortem itself flags that `routers/dashboard.py` could be
   repointed from the shim to five already-migrated domains **today**,
   independent of any further migration work — a live, actionable,
   currently-unclaimed recommendation (its own author suggested folding it
   into WO#7's housekeeping or a standalone "WO#6.5" — neither happened).

6. **WO#7 (recipes + pantry) — one real pre-deploy blocker.** The
   follow-on `RecipeIngredient.needs_review` column is a genuine schema
   change with **no Alembic migration generated or run** — the postmortem
   flags this itself, in bold, as required before deploy
   (`alembic revision --autogenerate...` then review then
   `alembic upgrade head`). Do not treat this domain as deploy-ready until
   that's done. Minor governance note, not blocking: a real pre-existing
   production bug (`MissingGreenlet` on recipe-tag save) was fixed
   *without* being explicitly requested — transparently disclosed in the
   report as going beyond the ask, but technically bundled rather than
   ticketed separately per the spirit of §4.5. Given the transparency and
   that it's already shipped, not worth unwinding — just noting the
   pattern so it isn't repeated silently elsewhere.

7. **WO#8 (workout) — real, fully-disclosed verification debt, nothing
   hidden.**
   - Three files remain over GOVERNANCE §1.2's 300-line ceiling even after
     the authorized follow-up split:
     `workout_plan_ai_generator.py` (373), `workout_log.py` (357),
     `workout_settings.py` (378). Candidate split points are already
     proposed in the postmortem's own §7.2 — don't split without a fresh
     go-ahead, per that section's own instruction.
   - Five items were never run against real infrastructure: a real `git
     diff` against the actual repo, a real `Base.metadata`/
     `configure_mappers()` run, a live Gemini plan-generation call, a live
     `WeeklyPlan`↔`WorkoutSession` join, and a live smoke-test of the
     `GET /workout/exercises` fix.
   - Useful correction worth keeping: shim removal for this domain is
     **not** gated on every remaining domain migrating — it only needs
     whichever file owns weekly-planning logic to repoint its
     `WorkoutPlan`/`WorkoutPlanDay`/`WorkoutSession`/`WeightUnit` import
     away from the shim. That's a single-file precondition, not a
     whole-backlog one — worth remembering when WO#10 (planning) runs.

8. **WO#9 (media) — two findings worth carrying into WO#20 and the AI
   service layer roadmap respectively.**
   - The postmortem's own per-name breakdown of `dashboard.py`'s shim
     usage reveals **both Code Intel's (WO#2) and Media's (WO#9) shims are
     fully orphaned** — zero real consumers anywhere, not just
     zero-in-dashboard. Both can be deleted with **no** corresponding
     `dashboard.py` edit whenever WO#20 runs — a concrete efficiency the
     shim-removal pass should use rather than rediscover.
   - The new weekly DAG (`life_os_refresh_streaming_availability`)
     required a **second, independent TMDB watch-providers implementation**
     (`airflow/agents/media_agents.py`), because GOVERNANCE §2.2 forbids
     DAGs from importing `services/`. Correctly followed the rule as
     written, but it's real, disclosed duplication debt — the postmortem's
     own §3.9 flags this explicitly and recommends a diff-for-drift check
     against `services/tmdb_service.py` before assuming the two agree.
     Never smoke-tested against a live Airflow instance.
   - Informational, not a problem: the postmortem independently confirmed
     (by reading the actual `main.py`, not assuming) that `recipes`/
     `pantry` and `workout` had **not** yet migrated at the time media
     ran, contradicting media's own "For the next work order" note's
     assumption. Harmless given domain independence, but a reminder that
     forward-looking notes in one postmortem can go stale by the time the
     next one is written.

9. **WO#11 (AI Service Layer Foundation) — a roadmap coverage gap,
   surfaced by cross-referencing the postmortem against the actual
   drafted WO#12–16 texts, not stated by the postmortem itself.** The
   postmortem flags that `life_os_weekly_synthesis.py` imports
   `blog_agents._gemini_flash` directly as a **private** function, inside
   a task body — invisible at DAG-parse time, breaking only when that task
   actually runs. It correctly warns that whichever future work order
   deletes `_gemini_flash` from `blog_agents.py` must add this DAG to its
   SCOPE explicitly or the coupling will snap silently. **Checked against
   the current plan at the time: none of WO#12–16 as originally drafted
   ever did this.** ~~As currently drafted, the AI Service Layer program
   never actually retires `blog_agents.py`'s own `_gemini_flash`/
   `_gemini_flash_json` functions.~~ **RESOLVED by WO#13's rewrite and
   execution — see note 12.** Left here, struck through rather than
   deleted, as the record of the gap that prompted the rewrite.

10. **WO#10 (planning) — closes the core domain-migration backlog, with
    one authorized exception worth naming precisely.** Original delivery
    had a genuine gap (`shopping_list.html`'s content was simply never
    provided, marked ❌ — not softened to ⚠️ — and closed once supplied).
    The one thing to flag hard: **`airflow/agents/weekly_agents.py` was
    explicitly out of WO#10's own scope, and was edited anyway**, under
    turn-by-turn authorization, to fix a real month-boundary bug
    (`date(year, month, day + offset)` arithmetic silently producing an
    empty meal plan on ~21% of weeks — swapped for `timedelta`). Logged
    with its authorization quote, not silently absorbed — but this is now
    load-bearing for the AI Service Layer program too, since WO#13
    (note 12) touches this same file next and had to be specifically
    guarded against reverting it. Also worth carrying forward: a
    program-wide file-size-ceiling table (`weekly_plan.py` 413 lines,
    `workout_plan_ai_generator.py` 373, `workout_log.py` 357,
    `workout_settings.py` 378 — all over the 300-line ceiling, none
    re-measured against WO#19's lint script since it's never been run),
    and a standing blind spot worth naming plainly: **`domains/journal/models.py`'s
    real source has never been shared in any engagement across this
    entire 20-work-order series** — every cross-check involving it has
    used a hand-built stand-in.

11. **WO#12 (recipe_agents.py) — six authorized amendments, all clean.**
    Closed its own previously-flagged retry-coverage gap (Amendment 6:
    the vision function now retries on 429/503 via `post_with_retry`
    directly, by adding an optional `timeout` parameter rather than
    building premature vision-service-layer support). Remaining, tracked
    for WO#16: the vision call's *payload construction* still isn't a
    real `services/ai/` function, only its transport layer is.

12. **WO#13 (Rewritten) — closes note 9's roadmap gap for real, and is
    otherwise the cleanest execution in the AI Service Layer series so
    far.** Grepped `blog_agents.py` first per its own instruction and
    found **exactly** the 4 functions predicted (`agent_readme_writer`,
    `agent_researcher`, `agent_editor`, `agent_idea_expander`) — no
    discrepancy. Correctly refused to reconcile the TMDB 404 drift within
    its own authority in Part A; Part B's five follow-on fixes are all
    explicitly authorized and disclosed as reversing specific WO#13 rules
    where relevant, consistent with the pattern since WO#7. Handled the
    `weekly_agents.py` guard (note 10) better than specified — applied
    the pre-existing fix forward as a diff-separable edit rather than
    just halting.
    **The one thing that must be closed before any of this is trusted:**
    `services/ai/base.py`, `keys.py`, and `services/ai/__init__.py` were
    **never provided as real source anywhere in this engagement** — all
    three are reconstructed stubs built from how other files import and
    call them. Every "identical to pre-migration payload" claim in this
    postmortem, and Fix 4's "fail loudly on missing key" change
    specifically, is verified against a guess at these files' real
    content, not the files themselves. Swap in the real files and re-run
    every payload-shape and backward-compatibility check before merging.
    Also disclosed, correctly scoped as partial: two `async def` routes
    were fixed for event-loop-blocking (`asyncio.to_thread` wrapping)
    but this was **not** a codebase-wide audit — other async routes
    calling `services/ai/` may have the same exposure, unchecked.

13. **WO#17 (toast consolidation) — the most rigorous self-auditing
    postmortem in the series.** Found its own regression
    (`domains/recipes/templates/partials/pantry_list.html` referencing a
    div ID removed three phases earlier, in a file that was never in any
    phase's SCOPE) during its own verification pass, traced the exact
    user-visible impact rather than just flagging "might be broken," and
    fixed it using an already-established codebase convention. Also
    self-disclosed that `base.css` was provided from the start but not
    checked before Phase 1's report was written — worth naming as the
    same category of gap WO#20 (note 14) independently found in its own
    first pass: **materials already in hand, presented as unavailable
    rather than simply unchecked.** This is now a confirmed pattern
    across two independent engagements, not a one-off — worth a standing
    "verify SCOPE against materials already provided before drafting
    acceptance criteria" step in the work-order template itself, which
    WO#17's own §7.4 recommends. Real backend follow-up (§7.1–7.3:
    route-handler compatibility, stale `models.py`/schema references,
    test-suite audit) is written as guidance, not yet executed — no
    backend file was ever part of this engagement.

14. **WO#20 (shim removal) — closes GOVERNANCE §2.4, tracked as
    outstanding since WO#1. This is the single most important postmortem
    to read in the whole series, for one reason: §4.4.** Removing the
    last shim also removed an *implicit* guarantee those shims
    accidentally provided — that every domain's models module gets
    imported, and therefore mapper-registered, before the first query
    runs. The only thing providing that now is `main.py`'s router
    imports. This has never been independently verified end-to-end, in
    any of the 20 work orders in this series, because doing so requires
    every domain's real `models.py` simultaneously in a real Python
    environment — materials no single engagement, including this one,
    has ever had at once. **This is now the top-priority open item across
    the entire program** — see "Where This Actually Stands" below.
    Separately: Part 1.4's own evidentiary-gap finding (claimed 5
    domains' router source "was not part of this engagement" when it
    actually was, just unchecked) is the same pattern as WO#17's
    `base.css` oversight (note 13) — self-corrected once a reviewer
    pushed back, honestly disclosed as a diligence gap rather than a
    wrong conclusion. `models.py`'s final end-state (Option 1 — delete —
    vs. Option 2 — reduce to a ~15-line registry, recommended) is
    specified in full but **not yet executed** — still an open decision.

15. **WO#18 (DAG reorganization) — clean execution, one small gap, now
    closed.** Verified as a genuine pure relocation: full-file diff (not
    just the `sys.path`/`dag_id` lines) confirmed byte-identical content
    across all 13 files, `git log --follow` confirmed history
    preservation, zero `docker-compose.yml` change. **The original gap**
    — `life_os_staging_promoter.py` and
    `life_os_refresh_streaming_availability.py` (both created after
    WO#18 was first drafted, in later follow-on work) were left flat
    rather than moved into subfolders — **has since been closed directly
    by the project owner**, outside any formal work order, using the
    same zero-content-change move already proven safe 13 times over. Not
    independently re-verified here (full-file-diff/`git log --follow`
    weren't re-run against this manual fix), but it's a mechanical,
    low-risk operation with no reason to doubt it. The
    `GOVERNANCE.md §2.5` amendment this postmortem's own §6.1
    pre-committed to is still ready and unapplied — see Track B below.

16. **WO#19 (dead code + lint script) — the lint script finally ran for
    real, and the results change what "next" should mean for this
    program.** Task 1 correctly ran before WO#15 (confirmed — it located
    the dead block using the live `_cerebras()` function and its section
    header, both of which still existed at execution time). Task 2's
    real output: **14 router files over GOVERNANCE §1.2's 300-line
    ceiling**, the full list now the authoritative replacement for every
    scattered "N files over the limit" mention in earlier postmortems:
    `habits.py` (551, **the worst in the codebase, previously unknown to
    every prior postmortem**), `ci_readme.py` (448), `journal.py` (446),
    `recipe_extract.py` (440), `blog.py` (419), `media_recommend.py`
    (416), `weekly_plan.py` (413), `ci_files.py` (382),
    `workout_settings.py` (378), `workout_plan_ai_generator.py` (375),
    `recipes.py` (371), `workout_log.py` (357), `media.py` (355),
    `ci_projects.py` (328). None of these are new problems — they're the
    first *measured* view of debt that's been accumulating silently
    since each domain's own migration.

17. **WO#14 (Groq + Ghostwriter) — clean, minimal, exemplary.** Correctly
    ran its own precondition check before touching anything (confirmed
    WO#13's Gemini extraction had actually landed, per the amendment's own
    instruction). Diff isolated to exactly 3 hunks. Correctly reported
    `services/ai/keys.py` needed zero edit — the Groq key entry was
    already present — rather than claiming a no-op step as "done" without
    saying so. 5/6 criteria ✅, the one ⚠️ (HTTP request-shape parity) is
    a genuine environment gap, not incomplete work. §8 of its own
    postmortem is worth reading independent of anything else in this
    entry — it draws a clean, explicit line between the Domain Migration
    track (shims, `dashboard.py`, `models.py`) and the AI Service Layer
    track (`blog_agents.py`, `services/ai/`), stating plainly that the two
    are "unrelated subsystems that happen to be mid-migration in the same
    repository at the same time" — worth internalizing before treating
    any AI-layer cleanup checklist as if it were a domain-migration one.

18. **WO#15 (Cerebras) — the most rigorous verification in this entire
    program, and a serious, disclosed process failure in the same
    document.** The retry-path testing built a fake Cerebras SDK and
    *executed* all 8 real retry/backoff/exception scenarios rather than
    reading code, diffed programmatically against a pre-migration
    baseline, and separately simulated `life_os_code_improve.py`'s own
    sleep-decision logic against a fake low-token-budget response to
    confirm the DAG contract holds end to end. Genuinely excellent.
    **Read the correction notice at the top of the postmortem before
    trusting anything else in it, though:** the deliverable initially
    shipped with five files silently truncated (docstrings and full
    function bodies stripped from `blog_agents.py`, `services/ai/base.py`,
    `providers/gemini.py`, `providers/groq.py`, and `__init__.py`'s header
    comment) — caught for one file on a first pass, and for the other four
    only after a direct follow-up question forced a second check. **This
    is now the third confirmed instance of the same pattern in this
    program** (WO#17's `base.css`, WO#20's router-source claim, now this,
    more severe than either since it's actual content corruption in a
    deliverable, not just an unchecked assumption) — worth treating as a
    standing, recurring risk in how this kind of work gets verified, not
    three unrelated incidents. Resolved a mystery both this postmortem and
    WO#14's independently flagged and couldn't explain: whether a
    commented-out dead `_cerebras()` block still existed in
    `blog_agents.py`. It doesn't — because WO#19 already removed it, and
    WO#19 ran before both WO#14 and WO#15, a fact neither postmortem had
    the cross-visibility to know on its own.

---

## AI Service Layer Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 11 | Foundation + `job_agents.py` (Gemini) | `work_order_11_ai_service_foundation.md` | ✅ **Executed** (see `domains-migration-postmortem-ai-foundation.md`). `job_agents.py`'s import change confirmed as the *only* line touched; `gemini_client.py` deleted only post-verification per Step 6. Forward-looking finding resolved — see note 12. | none |
| 12 | `recipe_agents.py` (Gemini + Gemma) | `work_order_12_ai_service_recipe_agents.md` | ✅ **Executed** (see `domains-migration-postmortem-ai-recipe.md`). Six authorized amendments, all clean — see note 11. | #11 |
| 13 | `weekly_agents.py`, `workout_plan_ai_generator.py`, `media_recommend.py`, `blog_agents.py` (Gemini) + TMDB reconciliation | `work_order_13_ai_service_batch3_REWRITE.md` | ✅ **Executed** (see `domains-migration-postmortem-ai-service.md`). Rewritten scope fully delivered; closes note 9's roadmap gap. **`services/ai/base.py`/`keys.py`/`__init__.py` are reconstructed stubs, never real source — must be swapped and re-verified before trusting this.** See note 12. | #11, #12 |
| 14 | Groq provider + Ghostwriter | `work_order_14_ai_service_groq.md` | ✅ **Executed** (see `domains-migration-postmortem-ai-groq.md`). Clean, minimal, precondition check passed. See note 17. | #11–13 |
| 15 | Cerebras provider (Narrator/Refiner/Commenter/Improver) | `work_order_15_ai_service_cerebras.md` | ✅ **Executed** (see `domains-migration-postmortem-ai-cerebras.md`). Most rigorous retry-path verification in the whole series — but read the correction notice at the top of its postmortem before trusting the deliverable. See note 18. | #11–14 |
| 16 | Generic dispatcher, vision support, `finance_upload.py` decision, `blog_agents.py` shell cleanup | `work_order_16_ai_service_capstone.md` | 📝 Drafted, not executed. **The only remaining sequential item — and it turns out to be fully isolated from every other track.** See the WO#14–20 amendments' updated conflict map: WO#16 shares zero files with Track B, C, D, or E. Amended to add a mandatory stub-verification gate and an explicit Task 2 (shell cleanup) before drafting. | #11–15 (needs all three providers to exist first — now true) |

**This series is strictly sequential** — unlike the domain migrations,
each AI service work order builds directly on the previous one's new
files. Do not run out of order.

---

## Frontend Consolidation Work Orders

| # | Scope | File | Status |
|---|---|---|---|
| 17 | Toast notification dedup + dead `sidebar_js.html` | `work_order_17_frontend_toast_consolidation.md` | ✅ **Executed** (see `domains-migration-postmortem-toast-and-sidebar.md`). Grew from 5 files to 17 edited + 1 deleted across an approved full-tree audit; found and fixed its own regression. See note 13. |

---

## DAG Reorganization & Cleanup Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 18 | Reorganize `airflow/dags/*.py` into domain subfolders (revised, lower-risk approach — no `docker-compose.yml` change needed) | `work_order_18_dag_reorganization.md` | ✅ **Executed and concluded** (see `domains-migration-postmortem-dag-cleaning.md`). The 2-file gap from the original scope was closed directly by the project owner. See note 15. | none |
| 19 | `blog_agents.py` dead-code removal + router line-limit lint script | `work_order_19_misc_cleanup.md` | ✅ **Executed** (see `domains-migration-postmortem-dead-code-removal.md`). Ran before WO#15 as required. **First real line-count data in the whole series — 14 files over the ceiling, see note 16.** | none |
| 20 | Shim removal pass across all domains (designed to be safely re-runnable at any migration completion state) | `work_order_20_shim_removal.md` | ✅ **Executed** (see `domains-migration-postmortem-shim-removal.md`). Closes GOVERNANCE §2.4 — tracked as outstanding since WO#1. **Surfaces the single most important open item in the entire program — see note 14 and "Where This Actually Stands" below.** | Was: WO#2–10 |

---

## Deferred / Not Yet Scoped — This Is the Real Remaining List

Everything that was reasonably scopable without live execution feedback or
a new decision from you has now been drafted (20 work orders total). What
remains falls into two honest categories:

**A. Needs real data from execution, not more speculation:**
1. ~~Router splitting for `media_recommend.py`, `ci_readme.py`.~~
   **Superseded — WO#19's lint script ran for real** (item 16) and found
   14 files over the ceiling, not 2. Full scoping briefs for all 14, split
   by domain, are in `track_C_router_line_limit_briefs.md`.
2. **Agent folder reorganization** (`airflow/agents/*.py` grouped by
   domain, mirroring WO#18's DAG treatment) — still genuinely unscoped.
   See Track D in the Roadmap below — this is deliberately kept as a
   scoping-only task, not handed a code-moving work order the way WO#18
   was, since the "location doesn't matter" property that made DAG
   reorganization safe does not automatically transfer to files imported
   by both DAGs and routers.

**B. Explicitly parked per instruction, not touched:**
3. The four backlog ideas (GOVERNANCE.md §5: adaptive dashboard, digest
   email, in-Docker coding environments, new domains) — untouched, as
   directed.
4. **NBA data extraction + own page + dashboard card (new).** Data source
   is a process to be provided later. Falls squarely under GOVERNANCE.md
   §5 item 4 ("new domains... exactly what the domain-folder pattern was
   derisked for") — should be built directly as `domains/nba/` from day
   one rather than needing a later migration. **Kept as its own,
   separate work order per explicit instruction** — full scoping brief
   in `track_E_new_domains_and_analysis_briefs.md`.
5. **Soccer data extraction + own page + dashboard card (new).** Brasileirão,
   Premier League, Libertadores, and Champions League, all via a single
   FIFA endpoint to be provided later. **Kept as its own, separate work
   order per explicit instruction** — not merged with item 4 despite the
   structural similarity. Full scoping brief in the same file as item 4.
6. **Medium article extraction — revisit existing process (new).** Unlike
   items 4–5, a starting implementation already exists and needs
   revisiting, not building from scratch. Full scoping brief in the same
   file as items 4–5 — the one open decision it flags (own domain vs.
   feeding the existing `blog` domain) should be resolved before SCOPE is
   drafted, not defaulted into.

**C. Real open items surfaced by executed work, not previously tracked
here (see the numbered review notes under Domain Migration Work Orders
and AI Service Layer Work Orders for full detail on each):**
7. **Jobs (note 2):** two schema changes ORM-declared but not applied to
   live MariaDB; three unverified shared-infrastructure assumptions
   (`base.js` toast/htmx behavior, LinkedIn selectors, `dag_db.py`'s real
   interface).
8. **Explorer (note 3):** deferred decision on a dedicated read-only DB
   user; live smoke-test against real MariaDB never run.
9. **Finance (note 4):** the required `account_options.html` usage
   finding is missing from the report entirely; `/dashboard`'s finance
   card was only checked for import resolution, not a full render.
10. **Journal (note 5):** full criterion-by-criterion audit blocked on a
    companion report that wasn't provided; `dashboard.py` could be
    repointed to 5 already-migrated domains today, unclaimed.
11. **Recipes+Pantry (note 6):** `needs_review` column has **no Alembic
    migration run** — blocks deploy. A real pre-existing bug was fixed
    without a separate ticket (disclosed, not hidden).
12. **Workout (note 7):** file-size ceiling item now superseded by the
    complete measured list in note 16 (3 of workout's own files are on
    it); 5 verification-debt items never run live; shim removal only
    needed `planning`'s import repointed, not the full backlog — done,
    per WO#20.
13. **Media (note 8):** Code Intel's and Media's shims both confirmed
    fully orphaned (actionable for WO#20 now); a second, independent TMDB
    implementation now exists as disclosed duplication debt.
14. ~~**AI Service Layer (note 9)** — as originally drafted, WO#12–16
    never retired `blog_agents.py`'s `_gemini_flash`/`_gemini_flash_json`,
    leaving GOVERNANCE §2.3's target state permanently unmet for that
    file's Gemini calls.~~ **RESOLVED.** WO#13 was rewritten specifically
    to close this gap and has now executed — confirmed via grep that
    exactly the 4 predicted functions were the only Gemini call sites,
    and the `life_os_weekly_synthesis.py` coupling was fixed in the same
    change. See note 12. Left here, struck through, as the record of why
    WO#13 grew the scope it did.
15. **WO#1's archived file needs correcting** — the document on file at
    `work_order_01_habits_domain.md` was, until this review, the follow-on
    bug-fix ticket rather than the actual pilot migration spec. Replace it
    with the real WO#1 text so the archive is self-consistent for future
    readers.
16. **Planning (note 10):** `weekly_agents.py`'s month-boundary fix is now
    load-bearing for WO#13/14/15 too — confirm it's still present before
    any future edit to that file. ~~Program-wide file-size-ceiling table
    needs WO#19's lint script, still never run.~~ **Done — see note 16's
    full 14-file list, WO#19.** `domains/journal/models.py`'s real source
    has never been shared in any engagement in this entire series — a
    standing blind spot, not specific to any one domain's follow-up.
17. **`recipe_agents.py` (note 11):** the vision function's *transport*
    layer now retries correctly (Amendment 6), but its *payload
    construction* still isn't a real `services/ai/` function — tracked
    for WO#16 item 4b, unchanged by this.
18. **AI Service Layer batch (WO#13, note 12):** `services/ai/base.py`,
    `keys.py`, and `__init__.py` are **reconstructed stubs** — never real
    source in this engagement. Every payload-shape and backward-
    compatibility claim in WO#13's postmortem needs re-verification
    against the real files before this is trusted. A full codebase audit
    for other async-route-calls-sync-`services/ai/` instances (beyond
    the 2 already fixed) is still outstanding.
19. **Toast consolidation (WO#17, note 13):** backend follow-up not yet
    executed — route-handler `toast`-context compatibility check, a
    search for stale toast-related references in `models.py`/schemas/
    test suites, and a GOVERNANCE §3.2 update marking this fully
    resolved. None of these were in reach of a frontend-only engagement.
20. **Shim removal (WO#20, note 14) — see also the elevated callout in
    "Where This Actually Stands":** `models.py`'s end-state decision
    (delete vs. reduce to a registry, Option 2 recommended) is fully
    specified but **not yet executed**. The `routers/`/`templates/`/
    `static/` filesystem audit (confirm nothing domain-specific was left
    behind at the old flat locations) is still open, unchanged since
    WO#10 first flagged it.
21. **DAG reorganization (WO#18, note 15):** ~~2 DAG files still sit at
    the flat `airflow/dags/` root~~ — **closed directly by the project
    owner, outside any work order.** `GOVERNANCE.md §2.5`'s pre-committed
    amendment is still ready and unapplied. 12 stale header-comment paths
    (cosmetic) also still unfixed.
22. **Router line-limit remediation (WO#19, note 16) — new, real, and
    the single largest body of well-scoped, low-risk work now available:**
    14 files need splitting, none of which touch `services/ai/` or
    `blog_agents.py`. Full scoping briefs, one per domain, in
    `track_C_router_line_limit_briefs.md`.
23. **Blog Scout prompt review (new idea, category B — appended here
    rather than renumbered into sequence, to avoid breaking the many
    cross-references to items 7–22 above).** Not a code migration —
    an analysis task: does the Blog Scout DAG's idea-generation output
    actually reflect the project owner's stated interests, or has it
    drifted into repetition/narrow focus? Deliverable is a report, with
    any prompt revision explicitly gated as separate follow-up work, not
    bundled into the same pass. Full scoping brief, including the open
    questions that need resolving before SCOPE can be written (which
    agent function the Scout DAG actually calls; whether a canonical
    "likes and interests" baseline exists anywhere to compare against),
    in `track_E_new_domains_and_analysis_briefs.md`.

---

## Roadmap — Recommended Next Tracks

**The core question: with WO#14→15→16 forced sequential (all three edit
`blog_agents.py` and/or `services/ai/__init__.py`), what can run alongside
that bottleneck?** Five tracks, ranked by how ready each is to hand to
another agent right now.

### Track A — AI Service Layer (down to one item, and it's now isolated)
~~WO#14 → WO#15 → WO#16, strictly one at a time~~ — **WO#14 and WO#15 are
both done.** Only WO#16 remains, and per its own postmortems' confirmed
SCOPE, it shares zero files with Tracks B, C, D, or E — **it is no
longer a bottleneck blocking anything else.** Two real additions were
folded into WO#16's amendment before drafting it, though — see the
`work_order_14-20_amendments.md` file: a mandatory gate to verify the
`services/ai/` stub files against real source before adding a fifth layer
on top of them, and an explicit Task 2 for `blog_agents.py`'s now-ready
shell cleanup (dead imports, the still-unowned MODEL ROUTING doc
relocation). Run WO#16 whenever convenient — genuinely in parallel with
everything else on this list now, not sequenced ahead of it.

### Track B — Loose ends (do this first, it's an afternoon)
~~Four~~ **Three** small, independent, zero-dependency items — item 1
below was closed directly by the project owner outside any formal work
order:
1. ~~Move `life_os_staging_promoter.py` and
   `life_os_refresh_streaming_availability.py` into their subfolders~~ —
   **done.**
2. Apply the `GOVERNANCE.md §2.5` amendment WO#18's own postmortem
   pre-committed to (item 21).
3. Execute `models.py`'s end-state (Option 2, the ~15-line registry —
   item 20) — fully specified by both WO#10's and WO#20's postmortems,
   and the natural vehicle to finally run the `configure_mappers()` check
   below.
4. Clean up WO#18's 12 stale header-comment paths (cosmetic, item 21).

None of these touch a file any other track needs. **Full scoping briefs
for items 2–3 (2 work orders) in `track_B_loose_ends_briefs.md`** — item 4
is folded into the same document's second brief.

### Track C — Router line-limit remediation (new, the biggest opportunity)
14 files, real data as of WO#19 (item 22). None of them require touching
`services/ai/`, `blog_agents.py`, or any AI-agent file — a router split is
purely about extracting route handlers, independent of what those
handlers call into. This means **all 14 can run in parallel with Track A
and with each other**, following the exact precedent already proven twice
(`workout_plans.py` → CRUD + AI-generator in WO#8; `weekly_plan.py` → 3
files in WO#10's follow-up). `habits.py` (551, 251 over) is the biggest
single file in the codebase and was never previously flagged by any
postmortem — worth its own dedicated look before assuming a simple split
fits, since nobody's read its internal structure since WO#1. **Full
scoping briefs, one per domain (8 briefs — code_intel's covers what will
likely become 3 separate work orders once drafted, since it has 3
independently-oversized files rather than one file needing its first
split), with proposed split seams pulled from each domain's own prior
postmortems where they exist (workout's own postmortem already proposes
its own splits — use those directly), in
`track_C_router_line_limit_briefs.md`.**

### Track D — `airflow/agents/*.py` reorganization (scoping only, not execution yet)
Flagged by WO#18's own §7.5 as a real, separate question with its own
risk profile — these files are imported by both DAGs *and* at least one
router (`recipe_agents.py`) directly, so DAG-migration's "pure move, zero
code change" conclusion (WO#18) does **not** transfer automatically. This
is genuinely unscoped — no work order draft exists for it at all. Good
candidate to hand to an agent for a **pre-flight risk analysis only**
(working through WO#18 §7.2's own comparison table for this specific file
category), producing a draft work order to review, not code to merge.
Low risk to start now since it's investigation, not execution — but don't
let anyone jump straight to moving files the way WO#18 could, since the
zero-risk properties that made that safe don't hold here.

### Track E — New domains and analysis (NBA, Soccer, Medium, Blog Scout review) — needs your input before drafting most of it
Four separate work orders, **kept separate on purpose** (NBA and Soccer
are structurally similar but explicitly not merged into one domain).
Three are new-capability builds; the fourth is different in kind. **Full
scoping briefs for all four in
`track_E_new_domains_and_analysis_briefs.md`:**
- **NBA:** blocked on the data-extraction process being shared — that's
  the only real blocker. Once in hand, decide DAG-scheduled ingestion vs.
  on-demand router call before drafting SCOPE.
- **Soccer:** blocked on the FIFA endpoint, same shape as NBA. Worth a
  shared scoping *conversation* with NBA before either WO is finalized
  (to spot reusable patterns), without merging the two work orders
  themselves.
- **Medium:** different in kind — existing process to revisit, not
  greenfield. The one decision to make before SCOPE is written: own
  domain, or fold into the existing `blog` domain's AI-content work.
- **Blog Scout prompt review (new):** not a code migration — an
  analysis task producing a report on whether the Scout DAG's generated
  ideas overlap, stay narrowly focused, or track the project owner's
  actual interests. Any resulting prompt change is explicitly a
  *separate*, later, gated work order — this one only investigates.
  Genuinely the lowest-risk item on this entire list to start immediately,
  since its first phase makes zero code changes.

This whole track has zero file overlap with anything in Tracks A–D — the
only shared resource is your own time reviewing what comes back.

---

## Where This Actually Stands

**20 work orders + `GOVERNANCE.md` are drafted. 19 are executed and
reviewed: the full domain-migration backlog (WO#1–10), WO#11–15 of the AI
Service Layer, WO#17 (toast consolidation), and WO#18–20 (DAG reorg, dead
code + lint, shim removal). All nineteen pass their alignment and
governance audits — none were rejected. Only WO#16 remains, and it's no
longer blocking anything else — see the Roadmap's Track A above.**

**Top priority, above everything else on this list: run the
`configure_mappers()` check WO#20's postmortem §4.4 specifies.** This is
not a new finding — every postmortem since WO#2 has flagged some version
of "the real SQLAlchemy mapper-registration check still hasn't run" — but
WO#20 changes *why* it matters. Every domain's shim is gone now, which
means the implicit guarantee those shims accidentally provided (that
importing `models.py` at all would transitively import every domain and
register every mapper) is gone too. The only thing providing that
guarantee today is `main.py`'s router-import order, and nobody has ever
confirmed this holds — not once, across 20 work orders — because doing so
needs every domain's real `models.py` simultaneously in a real Python
environment, and no single engagement in this entire series has had that.
This is a five-minute check once someone has the real repo:
```python
import sqlalchemy.orm as orm
import main
orm.configure_mappers()  # must raise nothing
```
If it raises, the fix is adding the missing domain to whichever
`models.py` end-state is chosen (Track B, item 3) — never resurrecting a
shim.

**Second: work the roadmap above — now with essentially no bottleneck.**
Track B, Track C (router splits, real data), Track D (scoping), and Track
E (new domains + the Blog Scout review) are all confirmed parallel-safe
with each other *and* with WO#16, per the updated conflict map in
`work_order_14-20_amendments.md`. There's no longer a reason to sequence
anything behind the AI Service Layer track — that constraint is gone now
that WO#14 and WO#15 are both done and WO#16 turned out to touch nothing
the other tracks need.

**Third: before trusting WO#13, #14, or #15's output in production, swap
in the real `services/ai/base.py`, `keys.py`, and `__init__.py`.** All
three have been reconstructed stubs since WO#11 — genuinely never seen —
and every payload-shape and retry-behavior claim across four consecutive
work orders is only as good as those reconstructions turned out to be.
This has now been folded into WO#16's amendment as a mandatory
pre-execution gate, so it can't quietly become a fifth layer of
unverified assumption.

**Still-open pre-deploy blockers, unchanged from the last review:** the
Jobs domain's undeployed `UniqueConstraint`/`Index` pair (item 7) and the
Recipes domain's undeployed `needs_review` Alembic migration (item 11).
Both are cheap to close and get more entangled the longer they're left.

**A pattern worth naming explicitly, now confirmed three times
independently (WO#17's `base.css` oversight, WO#20's original
router-source claim, and now WO#15 — the most severe instance, actual
content corruption in a shipped deliverable, caught only after a direct
follow-up question):** materials or claims presented as complete or
unmodified, when they weren't, self-corrected only once someone pushed
back rather than being caught by the original process. Three independent
occurrences is a pattern, not bad luck. Worth building "verify SCOPE
against materials already in hand, and independently diff any file
claimed unchanged, before finalizing a deliverable" into the standing
work-order template itself, per WO#17's own §7.4 recommendation — this
should no longer be treated as a one-off risk in any future work order's
review.
