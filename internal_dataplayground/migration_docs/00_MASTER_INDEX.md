# Life OS Restructuring — Master Index

**Purpose:** single reference for every deliverable produced across this
engagement, and its actual execution status. Update the STATUS column as
work orders are run — this file is meant to stay accurate, not just be a
historical log.

---

## Foundational Documents

| Deliverable | Status |
|---|---|
| Phase 1 — Audit Report | ✅ Delivered (in-conversation, not a file) |
| Phase 2/3 — Consolidation blueprint + folder structure plan | ✅ Delivered (in-conversation, not a file) |
| `GOVERNANCE.md` | ✅ Delivered — the standing reference for every rule below |

---

## Domain Migration Work Orders

| # | Domain(s) | File | Status |
|---|---|---|---|
| 1 | `habits` (pilot) | `work_order_01_habits_domain.md` |  📝 Drafted, **not yet executed** |
| 2 | `blog` + `code_intel` | `work_order_02_blog_code_intel_domain.md` | 📝 Drafted, **not yet executed** |
| 3 | `jobs` | `work_order_03_jobs_domain.md` | 📝 Drafted, **not yet executed** |
| 4 | `explorer` | `work_order_04_explorer_domain.md` | 📝 Drafted, **not yet executed** |
| 5 | `finance` | `work_order_05_finance_domain.md` | 📝 Drafted, **not yet executed** |
| 6 | `journal` | `work_order_06_journal_domain.md` | 📝 Drafted, **not yet executed** |
| 7 | `recipes` + `pantry` | `work_order_07_recipes_pantry_domain.md` | 📝 Drafted, **not yet executed** |
| 8 | `workout` | `work_order_08_workout_domain.md` | 📝 Drafted, **not yet executed** |
| 9 | `media` | `work_order_09_media_domain.md` | 📝 Drafted, **not yet executed** |
| 10 | `planning` (`weekly_plan` + `intent`) | `work_order_10_planning_domain.md` | 📝 Drafted, **not yet executed**. Depends on #7 and #8 having run first (repoints imports at their real locations) — **must run after #7 and #8**, not before. |

**Execution order constraint:** #2 internally pairs `blog`+`code_intel`
(must run together). #10 depends on #7 and #8 already being done. Every
other domain (#3, #4, #5, #6, #9) is independent and can run in any order,
including in parallel across separate AI sessions if useful — this was
one of the explicit goals of the domain-folder pattern.

---

## AI Service Layer Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 11 | Foundation + `job_agents.py` (Gemini) | `work_order_11_ai_service_foundation.md` | 📝 Drafted, not executed | none |
| 12 | `recipe_agents.py` (Gemini + Gemma) | `work_order_12_ai_service_recipe_agents.md` | 📝 Drafted, not executed | #11 |
| 13 | `weekly_agents.py`, `workout_plans.py`, `media_recommend.py` + generalize `call_gemini_json` | `work_order_13_ai_service_batch3.md` | 📝 Drafted, not executed | #11, #12 |
| 14 | Groq provider + Ghostwriter | `work_order_14_ai_service_groq.md` | 📝 Drafted, not executed | #11–13 (independent of Groq specifically, but written assuming the series order) |
| 15 | Cerebras provider (Narrator/Refiner/Commenter/Improver) | `work_order_15_ai_service_cerebras.md` | 📝 Drafted, not executed | #11–14 |
| 16 | Generic dispatcher, vision support, `finance_upload.py` decision | `work_order_16_ai_service_capstone.md` | 📝 Drafted, not executed | #11–15 (needs all three providers to exist first) |

**This series is strictly sequential** — unlike the domain migrations,
each AI service work order builds directly on the previous one's new
files. Do not run out of order.

---

## Frontend Consolidation Work Orders

| # | Scope | File | Status |
|---|---|---|---|
| 17 | Toast notification dedup + dead `sidebar_js.html` | `work_order_17_frontend_toast_consolidation.md` | 📝 Drafted, not executed. Independent of everything else — can run any time. |

---

## DAG Reorganization & Cleanup Work Orders

| # | Scope | File | Status | Depends on |
|---|---|---|---|---|
| 18 | Reorganize `airflow/dags/*.py` into domain subfolders (revised, lower-risk approach — no `docker-compose.yml` change needed) | `work_order_18_dag_reorganization.md` | 📝 Drafted, not executed | none |
| 19 | `blog_agents.py` dead-code removal + router line-limit lint script | `work_order_19_misc_cleanup.md` | 📝 Drafted, not executed | none |
| 20 | Shim removal pass across all domains (designed to be safely re-runnable at any migration completion state) | `work_order_20_shim_removal.md` | 📝 Drafted, not executed | Meaningful only once some/all of WO#2–10 have run — safe to run early too, will just report "not applicable yet" per domain |

---

## Deferred / Not Yet Scoped — This Is the Real Remaining List

Everything that was reasonably scopable without live execution feedback or
a new decision from you has now been drafted (20 work orders total). What
remains falls into two honest categories:

**A. Needs real data from execution, not more speculation:**
1. **Router splitting for `workout_plans.py`, `media_recommend.py`,
   `ci_readme.py`** — GOVERNANCE.md §1.2 uses `workout_plans.py` as a
   worked example of a file needing the CRUD/AI-generator split (the
   treatment WO#10 actually gave `weekly_plan.py`), but WO#8 moved it
   as one file without splitting it — an inconsistency I flagged in
   WO#20's closing note rather than silently resolving. I don't have
   verified current line counts for any of these three files, so I'm not
   going to guess at split boundaries — run WO#19's lint script first,
   then this becomes a small, well-defined follow-up work order.
2. **Agent folder reorganization** (`airflow/agents/*.py` grouped by
   domain, mirroring WO#18's DAG treatment) — flagged as an open question
   in WO#18's closing note. Unlike DAGs, these files are imported as
   regular Python modules by both DAGs and FastAPI routers, so the
   "location doesn't matter" argument that made DAG reorganization safe
   does NOT automatically apply here. This needs its own risk analysis,
   not a copy-paste of WO#18's reasoning.

**B. Explicitly parked per your instruction, not touched:**
3. The four backlog ideas (GOVERNANCE.md §5: adaptive dashboard, digest
   email, in-Docker coding environments, new domains) — untouched, as
   directed.

---

## Where This Actually Stands

**20 work orders + `GOVERNANCE.md` are drafted. 1 (`habits`) is executed
and reviewed. 19 are not yet run.** I've now reached the point where
continuing to draft would mean either guessing at things I don't have data
for (category A above) or working on things you've asked me not to touch
yet (category B). Neither is a good use of "keep going."

**My honest recommendation:** this is the moment to shift from drafting to
running. Pick any of the fully independent, no-sequencing-required work
orders — #3, #4, #5, #6, #9, #17, #18, #19 are all safe to hand to an AI
session right now with no prerequisites — and start collecting real
execution reports the way WO#1 produced one. That feedback is what makes
the next round of decisions (the router-splitting question especially)
grounded instead of speculative, and it's the only way to validate whether
all this planning actually holds up in practice.

If you'd rather I keep drafting instead, I need a specific direction — the
two "needs real data" items above genuinely can't be scoped further
without either running WO#19 first or you telling me what you already know
about those files' current state.
