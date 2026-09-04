# Amendments to WO#14–20

**Read this before running any of WO#14 through WO#20 as originally
drafted.** Apply these as targeted patches against the existing documents
— per GOVERNANCE.md §4.1's own standing rule, this file gives exact
before/after text, not full-document regeneration, since these are files
whose full current state I only have as a point-in-time snapshot from
drafting time.

Every item below traces to a specific fact surfaced by a postmortem that
didn't exist when the corresponding WO was drafted. Nothing here changes
*intent* — only paths, exclusion lists, and dependency ordering that drifted.

---

## Parallel-execution / file-conflict map (added once WO#13 and WO#17 were started concurrently)

Built by checking every remaining WO's SCOPE against what the rewritten
WO#13 actually touches (`services/ai/providers/gemini.py`,
`weekly_agents.py`, `workout_plan_ai_generator.py`, `media_recommend.py`,
`blog_agents.py`, `life_os_weekly_synthesis.py`, `media_agents.py`,
`services/tmdb_service.py` read-only). Use this before starting any two
work orders at once — a "no conflict" verdict means no shared file, not
necessarily no logical dependency; both are checked separately below.

| Pair | Shared file(s) | Safe in parallel? |
|---|---|---|
| WO#13 + WO#17 | none | ✅ |
| WO#13 + WO#20 (or WO#10 Part 4) | none | ✅ |
| WO#13 + WO#19 Task 2 (lint script) only | none — read-only against routers, writes one new file | ✅, if split from Task 1 |
| WO#13 + WO#18 | `airflow/dags/life_os_weekly_synthesis.py` — WO#13 edits content, WO#18 moves location | ❌ — run WO#13's edit to this file first, then WO#18 |
| WO#13 + WO#19 Task 1 | `airflow/agents/blog_agents.py` | ❌ |
| WO#13 + WO#14 | `blog_agents.py`, `services/ai/__init__.py`; also a real dependency (WO#14 assumes WO#13's Gemini extraction already landed) | ❌ — strictly sequential |
| WO#13 + WO#15 | `blog_agents.py`, `services/ai/__init__.py`; also depends on WO#14 being done first | ❌ |
| WO#13 + WO#16 | `services/ai/providers/gemini.py`; also needs Groq+Cerebras (WO#14/#15) to exist first for the dispatcher design | ❌ |
| WO#17 + WO#18, #19, #20 | none | ✅ all |
| WO#18 + WO#19 (either task) | none — WO#18 never touches `airflow/agents/` | ✅ |
| WO#18 + WO#20 | none | ✅ |
| WO#19 + WO#20 | none | ✅ |
| WO#14 + WO#15 | `blog_agents.py`; strict dependency (#15 needs #14 done) | ❌ |
| WO#14 + WO#16 | `services/ai/__init__.py`; dependency (#16 needs #14+#15 done) | ❌ |
| WO#15 + WO#16 | `services/ai/__init__.py`; same dependency | ❌ |
| WO#14/#15/#16 + WO#17/#18/#19/#20 | none | ✅ all |

**Practical reading, as originally written (superseded below — kept for
history):** once WO#13 lands, #18, #19 (both tasks), and #20 can all run
simultaneously with each other and with anything else in this row — then
#14 → #15 → #16 must run strictly sequentially, one at a time, nothing
else touching `blog_agents.py` or `services/ai/__init__.py` concurrently
with any of the three.

---

## Updated state: WO#14 and WO#15 are now both done — WO#16 is the only thing left, and it turns out to be isolated

This changes the practical answer materially. WO#16's actual SCOPE
(`services/ai/__init__.py`, `services/ai/providers/gemini.py`,
`services/ai/README.md`, `recipe_agents.py`'s one vision function,
`finance_upload.py`'s comment-only edit, plus the new Task 2 added above —
`blog_agents.py`'s shell cleanup) **shares zero files with Track B, Track
C, or Track E as scoped in their brief documents.** The only thing WO#16
now conflicts with is itself — i.e., don't run two copies of WO#16
concurrently, and don't let anything else touch `blog_agents.py` or
`services/ai/__init__.py` while WO#16's own Task 2 is in flight.

| Pair | Shared file(s) | Safe in parallel? |
|---|---|---|
| WO#16 + Track B (either brief) | none | ✅ |
| WO#16 + Track C (any/all 8 briefs) | none — router splits don't touch `services/ai/`, `blog_agents.py`, `recipe_agents.py`, or `finance_upload.py` | ✅ |
| WO#16 + Track D (scoping only) | none — no file writes at all in this phase | ✅ |
| WO#16 + Track E (all 4 briefs) | none — including the Blog Scout review, whose first phase is explicitly read-only against `blog_agents.py`'s prompt text, not an edit | ✅ |
| Track B + Track C + Track D + Track E, all against each other | none, with one soft note: if Track E's Medium brief (E3) ends up folded into the `blog` domain, coordinate with Track C's Brief C5 (`blog.py`'s line-limit split) if both are in flight — a new feature landing mid-split makes for a hard-to-review diff, not a hard conflict | ✅, with that one coordination note |

**Practical reading, current:** there is no longer a meaningful
sequential bottleneck outside WO#16 itself. WO#16 can run at the same time
as literally everything else drafted from the Track B/C/D/E briefs — the
only discipline required is the same one that's applied throughout this
whole series: don't have two separate efforts editing the same file
concurrently. Given WO#16's own scope, that constraint doesn't touch any
of the other tracks at all.

---

## WO#14 (Groq + Ghostwriter)

**Why:** runs after the rewritten WO#13, which now removes
`blog_agents.py`'s Gemini-shaped functions before WO#14 touches the file.
WO#14's own diff-scope language should account for a smaller file.

**Add to HARD BOUNDARIES**, after the existing "`_gemini_flash()`,
`_gemini_flash_json()`, and `_cerebras()`, and every other function in
`blog_agents.py` remain completely untouched" line:

> **By the time this WO executes, `blog_agents.py` will already have had
> its Gemini-shaped functions and helpers (`_gemini_flash`,
> `_gemini_flash_json`, `_gemini_key`) removed by WO#13** — that WO now
> owns this extraction, ahead of this one. "Every other function... is
> completely untouched" in this WO's acceptance criteria means untouched
> relative to *WO#13's post-migration state* of the file, not the
> original pre-any-AI-migration state. Confirm this is the state you're
> starting from before diffing — if `_gemini_flash` still exists in the
> file when you start, WO#13 hasn't actually landed yet; stop and report
> rather than proceeding out of order.

**No change to STEPS or ACCEPTANCE CRITERIA content** — `_groq_llama`/
`agent_ghostwriter` are independent of the Gemini pieces; only the
baseline the "nothing else changed" diff is measured against shifts.

---

## WO#15 (Cerebras)

**Why:** same reasoning as WO#14, one layer deeper — by the time this
runs, `blog_agents.py` will have lost both its Gemini pieces (WO#13) and
its Groq piece (WO#14).

**Add to HARD BOUNDARIES**, same location as WO#14's addition above:

> **By the time this WO executes, `blog_agents.py` will already have had
> its Gemini functions (WO#13) and its Groq function (WO#14) removed.**
> The file you're diffing "everything else is unchanged" against is
> smaller than the version this WO was originally drafted against —
> confirm you're starting from a state where `_gemini_flash` and
> `_groq_llama` are both already gone before proceeding.

**No change to the Cerebras extraction steps themselves** — same
reasoning as WO#14.

---

## WO#16 (Capstone)

**Why:** three things changed since this was drafted — the file rename
from WO#8's split; the fact that by WO#16's time *all* of `blog_agents.py`'s
provider functions are now migrated (WO#13/#14/#15 all confirmed executed);
and two new, real findings from WO#14's and WO#15's own postmortems that
nothing in the series currently owns.

**Change, in HARD BOUNDARIES:**
```diff
- Do not migrate `job_agents.py`, `recipe_agents.py`,
- `weekly_agents.py`, `workout_plans.py`, `media_recommend.py`, or
- `blog_agents.py`'s Ghostwriter/Narrator/Refiner/Commenter/Improver
- functions to use the new generic dispatcher.
+ Do not migrate `job_agents.py`, `recipe_agents.py`,
+ `weekly_agents.py`, `workout_plan_ai_generator.py`, `media_recommend.py`,
+ or any of `blog_agents.py`'s functions (Researcher, Editor, Idea
+ Expander, Readme Writer, and Ghostwriter via WO#13/#14's Gemini/Groq
+ extraction; Narrator, Refiner, Commenter, Improver via WO#15's Cerebras
+ extraction) to use the new generic dispatcher.
```

Same correction wherever `workout_plans.py` appears elsewhere in the
document (SCOPE's file list, ACCEPTANCE CRITERIA's "confirm zero changes"
item) — replace with `workout_plan_ai_generator.py`.

**New — add a mandatory pre-execution gate, before Step 1:**

> **Before touching anything: `services/ai/base.py`, `services/ai/keys.py`,
> and `services/ai/__init__.py` have been reconstructed stubs since
> WO#11 — never verified against real source across four consecutive work
> orders (WO#11, #13, #14, #15), each of which added to the same
> reconstructed `__init__.py` export list. This compounds with every
> passing WO. Swap in the real files and re-run the payload-shape and
> retry-path checks from WO#11, #13, #14, and #15's postmortems against
> them *before* WO#16 adds a fifth layer on top of an unverified
> foundation.** If the real files aren't available in this environment
> either, state that explicitly as a ⚠️ carried forward, not resolved —
> do not let a fifth consecutive WO quietly inherit this gap without
> naming it again.
>
> **Separately, verify any claim of "file included unmodified for
> context" in your own deliverable independently, don't trust the label.**
> WO#15's own postmortem shipped five files with silently truncated
> content — caught only after a direct follow-up question, not by the
> original self-check. This is now a confirmed, recurring pattern across
> this program (WO#17, WO#20, WO#15), not a one-off. Diff every file you
> claim is unchanged against its actual prior state before asserting it.

**New — add as an explicit, separately-labeled Task 2 (mirroring WO#19's
own two-task-bundle precedent — keep this diff-separable from Task 1's
dispatcher/vision/finance_upload work, don't fold it in silently):**

> ## Task 2 — `blog_agents.py` shell-level cleanup (new, confirmed by
> both WO#14 and WO#15's postmortems, owned by neither)
>
> With Gemini, Groq, and Cerebras all extracted, clean up what's left:
> - Remove now-dead `import os` (used only by the now-deleted
>   `_cerebras_key()`) and `import time` (already effectively dead before
>   WO#15, confirmed still dead after). `import requests` was already
>   flagged dead by WO#14's postmortem and remains so — confirm via grep
>   nothing added a new usage before removing.
> - Remove the now-empty `# ── KEY HELPERS ──` section header.
> - Resolve `_CEREBRAS_INTER_REQUEST_SLEEP = 65` — WO#15's postmortem
>   flags it as an orphaned constant with no in-file consumer, possibly
>   vestigial (`life_os_code_improve.py` has its own separate
>   `INTER_REQUEST_DELAY_SEC`). Confirm before deleting rather than
>   assuming.
> - Relocate the module docstring's MODEL ROUTING table and ROUTING
>   RATIONALE section into `services/ai/README.md`, per GOVERNANCE §2.3's
>   own stated target state ("moved from blog_agents.py's header comment
>   ... applies project-wide") — this WO's own Step 5 already creates
>   `services/ai/README.md`; fold this relocation into that same step
>   rather than treating it as a separate file creation.
> - **Do not touch** the pre-existing duplicate `_estimate_tokens()`
>   definition (flagged by both WO#14 and WO#15, predates this whole
>   series) — report it per GOVERNANCE §4.5, same as every prior WO in
>   this chain has correctly declined to fix it.
>
> Confirm final state: `grep -n "^def _"` in `blog_agents.py` should
> return only `_detect_file_type` and (still, until reported/fixed
> separately) the duplicate `_estimate_tokens`.

**No change** to the vision-support work (Step 3, `recipe_agents.py`) or
the `finance_upload.py` documentation decision (Step 4) — both unaffected.

---

## Note: the dead-block mystery WO#14 and WO#15 both flagged is resolved

Both postmortems independently reported being unable to confirm whether
the commented-out prior `_cerebras()` implementation block still existed
in `blog_agents.py` — neither could explain the discrepancy from what the
original WO#15 text assumed. **It's not a mystery: WO#19's Task 1 already
removed it**, and WO#19 ran before both WO#14 and WO#15. Neither
postmortem had visibility into WO#19's own postmortem to make this
connection. Recorded here so nobody re-opens this as an open question —
WO#15's own §6.2 checklist item asking "does this block exist, and if not,
was WO#19's sub-task moot?" is answered: it existed, WO#19's task was real
and necessary, and it's already done.

---

## WO#17 (Frontend Toast Consolidation)

**Why:** the conditional "may be under `domains/<name>/templates/`"
language in SCOPE is now partly resolved by fact, and one file's location
was already wrong in the original draft independent of anything new (blog
was migrated in WO#2, before WO#17 was even drafted).

**Change, in SCOPE's template list:**
```diff
- `templates/recipes.html` (local `#recipe-toast` div +
-   `showRecipeToast()` function)
- `templates/pantry.html` (local `#pantry-toast` div +
-   `showPantryToast()` function)
- `templates/blog.html` (local `showToast()` function redefinition — uses
-   the shared `#toast` element already, just redefines the function
-   unnecessarily)
- `templates/jobs.html` (same pattern as `blog.html`)
- `templates/weekly_plan_view.html` (same pattern as `blog.html`)
+ `domains/recipes/templates/recipes.html` (local `#recipe-toast` div +
+   `showRecipeToast()` function) — confirmed path, WO#7 has run
+ `domains/recipes/templates/pantry.html` (local `#pantry-toast` div +
+   `showPantryToast()` function) — confirmed path, WO#7 has run
+ `domains/blog/templates/blog.html` (local `showToast()` redefinition) —
+   confirmed path, WO#2 has run (this was already inaccurate in the
+   original draft, independent of anything discovered since)
+ `domains/jobs/templates/jobs.html` (same pattern) — confirmed path,
+   WO#3 has run
+ `weekly_plan_view.html` — **confirmed path, no longer conditional.**
+   WO#10 (planning) has run: this file is at
+   `domains/planning/templates/weekly_plan_view.html`. One more thing
+   worth knowing before editing it: per WO#10's own authorized follow-up
+   work, `weekly_plan.py` was further split into a third router,
+   `weekly_plan_shopping.py` — this doesn't affect a template edit
+   directly, but if this toast-consolidation touches any inline
+   `<script>` logic that assumes a particular router owns a particular
+   endpoint, confirm against the current 3-router split rather than the
+   2-router shape assumed when this file's toast code was first written.
```

**No change** to the actual toast-consolidation logic, the
`sidebar_js.html` investigation, or the 2600ms timing-harmonization
disclosure requirement — all unaffected.

---

## WO#18 (DAG Reorganization)

**Why:** two DAGs now exist that didn't when WO#18 was drafted — both
created in follow-on work (jobs' Phase 3, media's follow-on item 2), and
both, per their own postmortems, were placed with WO#18's eventual
structure already in mind. This is the most concrete, mechanical
correction in this set — worth applying before WO#18 runs, since running
it against the original 13-file list would silently leave 2 files
unaccounted for.

**Change, in SCOPE:**
```diff
  `airflow/dags/jobs/`:
  - `life_os_job_scout.py`
  - `life_os_job_scout_ats.py`
  - `life_os_daily_digest.py`
+ - `life_os_staging_promoter.py` — new since WO#18 was drafted (added in
+   WO#3's authorized follow-on Phase 3). Per the jobs postmortem, no
+   special handling needed — same `sys.path`/`dag_id` characteristics as
+   every other DAG in this list.

  `airflow/dags/media/`:
  - `life_os_generate_embeddings.py`
+ - `life_os_refresh_streaming_availability.py` — new since WO#18 was
+   drafted (added in WO#9's authorized follow-on work). Per the media
+   postmortem's own §3.9 item 3: this file was already placed under
+   `airflow/dags/media/` from the start, matching this WO's target
+   structure — it needs **zero move**, only confirmation that it's
+   already correctly located. Don't skip verifying it just because
+   there's nothing to do.
```

**Change, in ACCEPTANCE CRITERIA**, every instance of "thirteen DAG
files" → "fifteen DAG files," and the first criterion becomes:

```diff
- All thirteen DAG files exist at their new subfolder locations and no
- longer at their original flat location in `airflow/dags/`
+ All fifteen DAG files (the original thirteen, plus
+ `life_os_staging_promoter.py` and
+ `life_os_refresh_streaming_availability.py`) exist at their new
+ subfolder locations. The latter two should already be correctly placed
+ — confirm rather than move.
```

**No change** to the DESIGN DECISION section's reasoning (recursive DAG
discovery, absolute `sys.path` inserts, `dag_id`-not-file-path keying) —
it applies identically to the two new files.

---

## WO#19 (Dead Code Removal + Lint Script)

**Why:** Task 1's instructions locate the dead commented-out `_cerebras()`
block using landmarks — "directly above the live, working `_cerebras()`
function" and "ends just before the `# ── CEREBRAS MODEL IDs ──` section
header" — that WO#15 deletes. If WO#19 runs after WO#15, those landmarks
are gone and Task 1 can't be executed as written.

**Add, to the top of Task 1 (Dead Code Removal)'s steps, before Step 1:**

> **Ordering dependency, not previously stated:** this task's location
> instructions below rely on the live `_cerebras()` function and the
> `# ── CEREBRAS MODEL IDs ──` section header as landmarks — both are
> deleted by WO#15 (Cerebras provider migration). **Run this task before
> WO#15**, or confirm WO#15 hasn't run yet before starting. If WO#15 has
> already landed, these instructions need rewriting to locate the dead
> block by a landmark that survives — e.g. its own commented-out `# def
> _cerebras(` opening line and docstring, independent of what follows it
> — do not attempt to apply the original landmark-based instructions
> against a file where they no longer exist.

**Add, to Task 2's context** (no functional change to the script itself):

> `workout_plans.py`'s fate — previously an open question this task's own
> closing note flagged as needing the lint script's real data — is now
> partially resolved independent of running this script: WO#8's follow-up
> already split it into `workout_plans_crud.py` (190 lines, compliant) and
> `workout_plan_ai_generator.py` (373 lines, **confirmed still over the
> 300-line ceiling**). This task's lint script will now report on two
> files where it previously would have reported on one — expected, not a
> bug in the script.

---

## WO#20 (Shim Removal Cleanup Pass)

**Why this amendment changed since it was first written:** the original
version of this amendment seeded WO#20 with per-domain findings current
as of the media (WO#9) postmortem, and correctly flagged Journal,
Recipes, and Workout as blocked on `planning` (WO#10) migrating. **WO#10
has since run, and its own postmortem resolved all three blocking
conditions directly** — the table below is corrected accordingly.

**More importantly: WO#10's postmortem Part 4 is now a more complete,
more current final-cleanup specification than this WO#20 draft.** It has
real, verified answers (not seeded hypotheses) for `models.py`'s exact
end state (§4.2 — confirmed via AST parse: 249 lines, zero remaining
`ClassDef` nodes, purely 10 shim imports), the full `dashboard.py`
breakdown (§4.3), a complete cross-domain relationship inventory verified
via real `configure_mappers()` (§4.4), confirmed final states for
`core/templating.py` and `main.py` (§4.5–4.6), a `routers/`/`templates/`/
`static/` audit checklist (§4.7), a program-wide file-size-ceiling table
(§4.8), and its own 12-item final verification checklist (§4.11) that
substantially overlaps WO#20's own ACCEPTANCE CRITERIA. **Recommend
treating WO#10's Part 4 as the operative final-cleanup work order going
forward, and either retiring WO#20 as a standalone document or reducing
it to a thin pointer at WO#10 Part 4** — maintaining two overlapping
specs risks exactly the kind of drift this whole amendment file exists to
catch. If WO#20 is kept as a separate document anyway (e.g. because its
mechanical mid-run idempotency check, absent from WO#10 Part 4, is worth
preserving), at minimum replace its own per-domain discovery steps with a
direct reference to WO#10 §4.2–§4.4 rather than re-deriving the same
findings a third time.

**Add, as a new section immediately after SCOPE, before STEPS** (revised
from the original amendment — three rows changed from "blocked" to
"unblocked," Planning added, and every row's evidence upgraded from
"inferred" to "confirmed directly against real source" per WO#10 §4.3):

> ## Known findings as of this amendment (confirm, don't re-derive from
> scratch — and see the note above recommending WO#10 Part 4 as the
> primary source instead of this table)
>
> | Domain | Shim exists? | Non-dashboard consumer? | Action this seeds |
> |---|---|---|---|
> | Habits | Yes | No | Repoint `dashboard.py`'s `Habit`, `HabitLog`, `HabitSettings` import; remove shim |
> | Blog | Yes | No | Repoint `dashboard.py`'s `BlogIdea`, `BlogIdeaStatus` import; remove shim |
> | Code Intel | Yes | No (confirmed, WO#10 §4.3) | Remove shim; **no `dashboard.py` edit needed** |
> | Jobs | Yes | No | Repoint `dashboard.py`'s `Job`, `ApplicationLog`, `ApplicationStatus`, `StagingJob`, `StagingJobStatus` import; remove shim |
> | Explorer | N/A — no models, no shim | — | Nothing to do, ever |
> | Finance | Yes | No | Repoint `dashboard.py`'s `Transaction` import; remove shim |
> | Journal | Yes | **No longer** — WO#10 repointed `save_entry()`'s local import from the shim to `domains.planning.models` directly (its Step 8/§1.3) | Repoint `dashboard.py`'s `JournalEntry`, `WeeklySynthesis` import; remove shim. **This domain is now unblocked** — the WO#6-era condition ("wait for planning") is satisfied. |
> | Recipes+Pantry | Yes | **No longer** — WO#10 repointed `weekly_plan.py`'s module-level and local (`_generate_shopping_list()`) imports to `domains.recipes.models` directly (§1.3) | Remove shim; **no `dashboard.py` edit needed** (confirmed zero references, WO#10 §4.3). **Unblocked.** |
> | Workout | Yes | **No longer** — WO#10 repointed `weekly_plan.py`'s `WorkoutPlan`/`WorkoutPlanDay`/`WorkoutSession`/`WeightUnit` imports to `domains.workout.models` directly (§1.3) | Remove shim; **no `dashboard.py` edit needed** (confirmed, WO#10 §4.3). **Unblocked.** |
> | Media | Yes | No | Remove shim; **no `dashboard.py` edit needed** |
> | Planning | Yes | No (trivially — nothing consumes planning's own shim except possibly future code) | Remove shim; **no `dashboard.py` edit needed** (confirmed, WO#10 §4.3) |
>
> **Practical implication, corrected from the original amendment:** every
> single domain is now unblocked. **5 of 10 (Code Intel, Recipes, Workout,
> Media, Planning) can be closed with zero `dashboard.py` edit. The other
> 5 (Habits, Blog, Jobs, Finance, Journal) each need one specific,
> already-known `dashboard.py` import line repointed** (exact replacement
> lines are given in WO#10's postmortem §4.3). **There is no longer any
> reason to defer a full WO#20 run** — the readiness check WO#10's own
> Part 4 §4.1 specifies (`grep -rn "from models import" ... | grep -v
> "^./models.py:"` returning only `dashboard.py`) is the one thing worth
> re-confirming against the real repository before running the full pass,
> since every postmortem in this series — including WO#10's own — has
> flagged its confidence as bounded by whatever files it was actually
> given.

**No change** to the mechanical STEPS or ACCEPTANCE CRITERIA — the table
above (and, more importantly, WO#10 Part 4 itself) is a head start on
Steps 1–3's findings, not a replacement for actually running the
readiness check first.

---

## Summary of what does *not* need amending

- **WO#12** (recipe_agents.py) — now audited against its postmortem. No
  path or scope drift affecting WO#13–20: WO#12's follow-on amendments
  (model-constant fix, `_gemini_key()` removal, vision-call retry
  coverage) are all self-contained within `recipe_agents.py` and don't
  touch any file WO#13–20 also touch.
- **WO#14, WO#15, WO#18, WO#19** — no further changes beyond what's
  already in this document.
- **The vision-support and `finance_upload.py` portions of WO#16** —
  unaffected by anything above.
- **WO#17's toast-consolidation logic itself and the `sidebar_js.html`
  investigation** — only file paths changed (see the `weekly_plan_view.html`
  update above), not the work itself. The 2600ms disclosure requirement is
  unaffected.

## Superseded by this update

- **WO#20's original "blocked on WO#10" framing for Journal, Recipes, and
  Workout** — WO#10 has run and resolved all three. See the rewritten
  WO#20 section above.
- **WO#20 as a standalone document, more broadly** — WO#10's own Part 4
  is now the more complete, more current final-cleanup specification. See
  the note at the top of the WO#20 section above.
