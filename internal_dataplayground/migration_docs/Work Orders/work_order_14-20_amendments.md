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

**Why:** two things changed — the file rename from WO#8's split, and the
fact that by WO#16's time *all* of `blog_agents.py`'s provider functions
will already be migrated (via WO#13/#14/#15), not just the subset the
original exclusion list named.

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

**No change** to the vision-support work (Step 3, `recipe_agents.py`) or
the `finance_upload.py` documentation decision (Step 4) — both unaffected.

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
+ `weekly_plan_view.html` — **path still conditional.** WO#10 (planning)
+   is in progress as of this amendment. If it has landed by the time this
+   WO executes, the file is at
+   `domains/planning/templates/weekly_plan_view.html`; otherwise it's
+   still at `templates/weekly_plan_view.html`. Locate it directly rather
+   than assuming either.
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

**Why:** WO#20 was drafted to *discover* each domain's shim status from
scratch. Nine domains have since run, and their postmortems already did
significant parts of that discovery — most usefully, the media
postmortem's own per-name breakdown of `dashboard.py`'s import block.
Re-discovering this from zero would be wasted effort; seeding WO#20 with
what's already known lets a partial run close real ground immediately.

**Add, as a new section immediately after SCOPE, before STEPS:**

> ## Known findings as of this amendment (confirm, don't re-derive from
> scratch)
>
> The table below is seeded from postmortem evidence already on file. Use
> it as a starting hypothesis for Step 1–3's per-domain checks, not a
> substitute for actually running them — a "known" row can still be wrong
> if something changed since its postmortem was written.
>
> | Domain | Shim exists? | Non-dashboard consumer? | Action this seeds |
> |---|---|---|---|
> | Habits | Yes | No | Repoint `dashboard.py`'s `Habit`, `HabitLog`, `HabitSettings` import; remove shim |
> | Blog | Yes | No | Repoint `dashboard.py`'s `BlogIdea`, `BlogIdeaStatus` import; remove shim |
> | Code Intel | Yes | **None found anywhere** (media postmortem §3.2 — zero dashboard.py usage, likely zero anywhere, though this specific "anywhere" claim is inherited from an earlier postmortem's weaker check, not re-verified as rigorously as media's own) | Remove shim; **no `dashboard.py` edit needed** — but re-run the full-repo consumer grep for these class names once, since the "zero anywhere" claim is less rigorously sourced than media's |
> | Jobs | Yes | No | Repoint `dashboard.py`'s `Job`, `ApplicationLog`, `ApplicationStatus`, `StagingJob`, `StagingJobStatus` import; remove shim |
> | Explorer | N/A — no models, no shim | — | Nothing to do, ever |
> | Finance | Yes | No | Repoint `dashboard.py`'s `Transaction` import; remove shim |
> | Journal | Yes | **Yes — `save_entry()`'s local import of `WeeklyPlanDay`/`WeeklyPlan`/`WeeklyPlanStatus`, deliberately left pointing at the shim per WO#6's own instruction** | **Do not remove this shim until `planning` (WO#10) migrates and repoints that specific local import** (per WO#6's own instruction and WO#10's own Step 8) — repointing `dashboard.py`'s `JournalEntry`/`WeeklySynthesis` import alone is *not* sufficient to retire this shim |
> | Recipes+Pantry | Yes | **Yes — `weekly_plan.py`'s module-level and local (`_generate_shopping_list()`) imports of `Recipe`/`RecipeMealType`/`Ingredient`/`PantryItem`/`RecipeIngredient`** | **Same — blocked on WO#10**, per the recipes postmortem's own Part C §C.2 |
> | Workout | Yes | **Yes — `weekly_plan.py`'s `WorkoutPlan`/`WorkoutPlanDay`/`WorkoutSession`/`WeightUnit` imports** | **Same — blocked on WO#10**, per the workout postmortem's own §7.1's precise (single-file) condition |
> | Media | Yes | **None found anywhere** (media postmortem §1.5/§3.2 — the most rigorously checked of any domain in this set) | Remove shim; **no `dashboard.py` edit needed** |
> | Planning | Not yet migrated as of this amendment | — | Not applicable yet — WO#10 in progress |
>
> **Practical implication:** Code Intel and Media can be fully closed out
> by WO#20 *today*, without waiting for `planning`. Habits, Blog, Jobs,
> and Finance can also be closed today, each with a specific,
> already-known `dashboard.py` repoint. Journal, Recipes, and Workout are
> **genuinely blocked on WO#10** — not because of any uncertainty, but
> because their real non-dashboard consumer is planning's own router,
> which doesn't exist at its final location yet. Don't run this task's
> "act" step for those three until WO#10's postmortem confirms the
> relevant imports were actually repointed as instructed.

**No change** to the mechanical STEPS or ACCEPTANCE CRITERIA — the table
above is a head start on Steps 1–3's findings, not a replacement for
running them.

---

## Summary of what does *not* need amending

- **WO#12** (recipe_agents.py) — no path or scope drift found; its own
  postmortem hasn't been reviewed yet, so this isn't a final word, but
  nothing in WO#13–20's own text depends on WO#12 having deviated from
  its draft.
- **The vision-support and `finance_upload.py` portions of WO#16** —
  unaffected by anything above.
- **WO#17's toast-consolidation logic itself, `sidebar_js.html`
  investigation, and 2600ms disclosure requirement** — only file paths
  changed, not the work.
