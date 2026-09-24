# Postmortem — Work Order #31: `airflow/agents/` Tier 1 Reorganization

**Work order:** `work_order_31_agent_reorg_tier1.md`
**Track:** D — Agent Folder Reorganization
**Status:** ✅ Executed. This postmortem is the record a reviewer should use
to pass or reject the migration, and the reference document any future
work order touching `airflow/agents/` should read first.

---

## 1. Summary

WO#31 relocated the six **Tier 1 (DAG-only)** modules out of the flat
`airflow/agents/` directory into two new subfolders — `airflow/agents/jobs/`
and `airflow/agents/media/` — mirroring the domain-subfolder pattern WO#18
already established for `airflow/dags/`. Each of the six files' content is
byte-identical at its new location; only the four (in practice, five — see
§4) DAG files that import them were edited, and only their import
statements changed.

This mirrors WO#18's DAG relocation in spirit but **not** in risk profile:
WO#18 was a pure directory move with zero import changes needed (DAGs
resolve `sys.path` absolutely, not relative to their own file location).
WO#31 is not pure — six modules changing location means every consumer's
import line has to change in the same commit, which is exactly what
Track D's own scoping analysis (`track_D_agent_reorg_scoping_analysis.md`)
flagged as WO#31's added risk versus WO#18.

Tier 2 (`recipe_agents.py`, `weekly_agents.py`) and Tier 3
(`blog_agents.py`) were explicitly out of scope and remain untouched — see
§7.

---

## 2. Scope as originally drafted

Recapped here so a reviewer doesn't need to cross-reference the source
work order separately.

### 2.1 Files moved (per WO#31 §SCOPE)
| Old path | New path |
|---|---|
| `airflow/agents/job_agents.py` | `airflow/agents/jobs/job_agents.py` |
| `airflow/agents/job_ats_agents.py` | `airflow/agents/jobs/job_ats_agents.py` |
| `airflow/agents/job_resume_context.py` | `airflow/agents/jobs/job_resume_context.py` |
| `airflow/agents/job_dedup.py` | `airflow/agents/jobs/job_dedup.py` |
| `airflow/agents/job_scout_health.py` | `airflow/agents/jobs/job_scout_health.py` |
| `airflow/agents/media_agents.py` | `airflow/agents/media/media_agents.py` |

### 2.2 Files the work order named for editing (import lines only)
- `airflow/dags/jobs/life_os_job_scout.py`
- `airflow/dags/jobs/life_os_job_scout_ats.py`
- `airflow/dags/jobs/life_os_staging_promoter.py`
- `airflow/dags/media/life_os_refresh_streaming_availability.py`

### 2.3 Explicitly out of scope (per WO#31 §HARD BOUNDARIES)
- `recipe_agents.py`, `weekly_agents.py` (Tier 2 — router-only, likely
  misplaced under `airflow/agents/` at all; needs its own destination
  decision, not drafted)
- `blog_agents.py` (Tier 3 — dual-consumed by both DAGs and FastAPI
  routers; cannot move incrementally)
- `email_client.py` — confirmed to stay at `airflow/agents/email_client.py`
  top-level per the Track D analysis's own reasoning

### 2.4 Acceptance criteria as drafted
1. All six files exist only at their new path; history preserved.
2. Each of the four named DAGs' import line updated, nothing else changed
   ("one line per file, or two for DAGs importing more than one of the
   six").
3. `life_os_job_scout` / `life_os_job_scout_ats` still resolve
   `search_linkedin_jobs`, `score_job_batch`, `fetch_all_watched_companies`,
   etc.
4. `life_os_staging_promoter` still resolves `get_full_job_posting`,
   `extract_linkedin_job_id`, `RESUME_MARKDOWN`, `KEY_STRENGTHS_TO_WEIGHT`.
5. `life_os_refresh_streaming_availability` still resolves
   `get_tmdb_watch_providers`.
6. Grep confirms **zero** remaining references to any of the six modules'
   old import path anywhere in the repo.
7. `email_client.py` confirmed untouched.

---

## 3. What was executed

### 3.1 Files moved
All six files moved with **byte-identical content** — no function name,
signature, or logic changed anywhere. This satisfies WO#31's HARD
BOUNDARY ("content is byte-identical except the file's own location") and
Acceptance Criterion 1's content half. History preservation
(`git log --follow`) could not be verified in this session — see §5.

### 3.2 Files edited
Import lines only, in the tasks listed. No `sys.path` changes were needed
or made, per WO#31 §STEPS item 3 (the existing
`sys.path.insert(0, '/opt/airflow/project/airflow')` line already makes
`airflow/agents/` importable; a subfolder under it just adds a path
segment to the import statement).

| File | Import lines changed | Old → New |
|---|---|---|
| `life_os_job_scout.py` | 5 | `agents.job_agents` (×2), `agents.job_dedup`, `agents.job_resume_context`, `agents.job_scout_health` → `agents.jobs.*` |
| `life_os_job_scout_ats.py` | 5 | `agents.job_ats_agents`, `agents.job_dedup`, `agents.job_agents`, `agents.job_resume_context`, `agents.job_scout_health` → `agents.jobs.*` |
| `life_os_staging_promoter.py` | 4 | `agents.job_agents` (×2), `agents.job_resume_context`, `agents.job_scout_health` → `agents.jobs.*` |
| `life_os_refresh_streaming_availability.py` | 1 | `agents.media_agents` → `agents.media.media_agents` |
| `life_os_daily_digest.py` | 1 | `agents.job_scout_health` → `agents.jobs.job_scout_health` — **not in WO#31's original file list; see §4** |

`email_client.py`'s import in `life_os_daily_digest.py`
(`from agents.email_client import send_email`) was left unchanged, as
required.

### 3.3 Verification performed
A repo-wide grep for the six modules' old import forms
(`from agents.job_agents import`, `from agents.job_ats_agents import`,
`from agents.job_resume_context import`, `from agents.job_dedup import`,
`from agents.job_scout_health import`, `from agents.media_agents import`,
and their `import agents.<name>` equivalents) returned zero hits after
the edits in §3.2. See §5 for what this check could and couldn't cover in
this session.

---

## 4. Deviation from drafted scope — agreed amendment

This is the one substantive difference between what WO#31 specified and
what was actually needed, and it's the piece a reviewer should weigh most
carefully.

**Finding:** `airflow/dags/jobs/life_os_daily_digest.py` — not listed
anywhere in WO#31's SCOPE — contains
`from agents.job_scout_health import get_health_summary`. This surfaced
during the WO#31 §STEPS item 4 grep-verification pass (the step that
exists specifically to catch consumers the SCOPE list missed), not during
drafting.

**Why this matters:** `job_scout_health.py` is one of the six Tier 1
modules being relocated. Left unedited, `life_os_daily_digest.py` would
import from a path that no longer exists the moment the six-file move
lands — the daily digest DAG would fail at import time on its very next
scheduled run (`0 13 * * *`), silently, until someone checked Airflow's
DAG-parse errors.

**Resolution applied:** the import was updated to
`from agents.jobs.job_scout_health import get_health_summary`, identical
in form to every other edit in this work order (import line only, no
other change to the file).

**Why this was treated as in-bounds rather than "STOP and report":**
WO#31's own Acceptance Criterion 6 requires zero remaining references to
any of the six old import paths **anywhere in the repo** — not just in
the four named files. That criterion is stricter than, and supersedes,
the incomplete file list in §SCOPE/§STEPS. Fixing the omitted consumer is
what satisfies the work order's own stated acceptance bar; leaving it
broken to honor an incomplete file list would have meant shipping WO#31
against its own criteria. This is analogous to the precedent
GOVERNANCE.md §4.5 sets for pre-existing bugs discovered during
verification, with one difference worth being explicit about: §4.5 is
about bugs unrelated to the migration, deliberately left for a separate
ticket. This is not that — it's a direct, mechanical consequence of the
very six-file move WO#31 orders (the same class of change as the four
listed edits, found by the exact verification step WO#31 itself
prescribes), so completing it is fixing an incomplete SCOPE list for this
migration, not doing unrelated "while we're at it" work.

**What a reviewer should confirm to accept this as correct rather than
scope creep:**
- [ ] The edit to `life_os_daily_digest.py` is exactly one import line,
  nothing else in the file changed (diff should be a single line).
- [ ] No other file in the provided corpus imports any of the six moved
  modules under the old path (see §5 for the limits of what was checked).
- [ ] The reviewer agrees the alternative — leaving
  `life_os_daily_digest.py` broken to match the literal SCOPE list — is
  the wrong outcome for a "migration," which by GOVERNANCE.md §4.6's own
  definition of done should not leave any unrelated behavior changed
  (here: broken).

A secondary, non-functional deviation: WO#31's Acceptance Criterion 2
expected "one line per file, or two for DAGs importing more than one of
the six." In practice `life_os_job_scout.py` and `life_os_job_scout_ats.py`
each import from four of the six Tier 1 modules across three separate
task functions, and `life_os_staging_promoter.py` imports from three
across two task functions — so their real diffs are 4–5 lines, not 1–2.
This is not a scope problem, just a documentation undercount in the
original work order (it was drafted from the Track D consumer map's
file-level granularity, not a line-level import audit). Flagged here so
the number doesn't read as a red flag on review, and so the next
Tier-reorg work order sizes its own diff expectations from an actual
per-task import count rather than a per-file guess.

---

## 5. Verification limits (read before treating any ✅ above as final)

This execution was performed against the file contents provided in
conversation, not against a live checkout of the repository or a running
Airflow instance. Concretely:

- **No `git mv` was run** and no `git log --follow` history check was
  possible — the deliverable is the moved/edited file contents, ready to
  be applied with `git mv` + these edits against the real repo. History
  preservation must be confirmed against the real repo after applying.
- **No live Airflow scheduler or DAG parse was available** — "still
  resolves `X`" in Acceptance Criteria 3–5 was checked by static
  read/grep (confirming the moved files still define those names, and
  the edited DAGs still import them correctly), not by an actual Python
  import or DAG-bag parse. Mark these ⚠️, not ✅, until confirmed against
  a real environment.
- **The repo-wide grep in §3.3 covers only the files provided in this
  conversation** — every DAG, agent module, router, and doc file that was
  shared. It cannot see routers, services, or any other file not shared
  in this session. Per GOVERNANCE.md §2.2, routers should never import
  `airflow/agents/*` at all (that's the DAG/FastAPI boundary the whole
  project enforces), so no router hits are expected — but "expected" is
  not "confirmed," and this should be re-grepped against the full repo
  before this postmortem is signed off as final.

---

## 6. Acceptance criteria — final status

| # | Criterion | Result | Note |
|---|---|---|---|
| 1 | Six files exist only at new path, history preserved | ⚠️ | Content confirmed byte-identical; history check needs the real repo (§5) |
| 2 | Each named DAG's import line updated, nothing else changed | ✅ (scope corrected) | Diff sizes larger than the WO expected per file — see §4's second deviation; `life_os_daily_digest.py` also required — see §4's primary deviation |
| 3 | `life_os_job_scout` / `life_os_job_scout_ats` resolve their six-module imports | ⚠️ | Static check only — see §5 |
| 4 | `life_os_staging_promoter` resolves its imports | ⚠️ | Static check only — see §5 |
| 5 | `life_os_refresh_streaming_availability` resolves `get_tmdb_watch_providers` | ⚠️ | Static check only — see §5 |
| 6 | Zero remaining references to any old import path, repo-wide | ✅ (within what was shared) | See §5 for scope of the grep |
| 7 | `email_client.py` untouched | ✅ | Confirmed |

---

## 7. Explicitly not done — Tier 2 and Tier 3

Restated here because it's load-bearing for how the follow-up section
below (§8) should be read: **this work order did not touch, and this
postmortem does not claim anything about,** `recipe_agents.py`,
`weekly_agents.py`, or `blog_agents.py`. All three still live at the flat
`airflow/agents/` root, unchanged. Any future reviewer or agent reading
this postmortem should not infer that the full `airflow/agents/`
reorganization is complete — only Tier 1 is.

---

## 8. Follow-up work required after the remaining migrations complete

This section is the standing checklist for **after** Tier 2 and Tier 3 of
the agent reorg (and any other in-flight migration touching these files)
are done. None of the items below are blocking WO#31's own acceptance —
they're what closes out the *track*, not this *work order*. This mirrors
how WO#18's DAG relocation spawned its own follow-up items (the
GOVERNANCE.md §2.5 amendment and the stale-header/`models.py`/
`configure_mappers()` cleanup, both tracked separately as WO#21 and
WO#22) rather than being folded back into WO#18 itself. Treat this list
the same way: each item becomes its own future work order, not an
amendment to WO#31.

### 8.1 GOVERNANCE.md needs a new subfolder-convention amendment
GOVERNANCE.md §2.5 documents the `airflow/dags/<domain>/` subfolder
convention (from WO#18) as settled, permanent architecture. It says
nothing about `airflow/agents/<domain>/` — that convention now exists
(`jobs/`, `media/`) but is undocumented as a standing rule. Once Tier 2
and Tier 3 are resolved (so the final shape of `airflow/agents/` is
known — including where Tier 2's modules actually end up, which per the
Track D analysis is genuinely still open, "probably isn't inside
`airflow/agents/` at all"), GOVERNANCE.md needs a new subsection
(§2.5.1, or wherever it best fits alongside §2.5) stating:
- The subfolder convention for `airflow/agents/` and which modules are
  DAG-only vs. dual-consumed vs. router-only, so a future contributor
  doesn't have to re-derive the Track D tiering from scratch.
- Whether dual-consumed modules like `blog_agents.py` (if Tier 3 is ever
  resolved) get a different treatment than DAG-only modules, and why.
- An explicit cross-reference to the Track D scoping analysis document,
  the same way §2.5 cross-references WO#18's own postmortem for its
  evidence trail.

### 8.2 Stale path references across the repo
Prose references (docstrings, comments, other migration docs) that name
these modules by their **old bare path** — as opposed to just their
module/function name, which is still accurate — will go stale as Tier 2
and Tier 3 land and the picture of `airflow/agents/` changes further.
This is the same class of cleanup WO#22 Task 2 performs for the 13 stale
DAG-header paths left over from WO#18. Known candidates from what was
reviewed in this work order alone:
- `MASTER_INDEX.md`'s Track D table and its "Deferred" section item 2 —
  once Tier 2/3 are scoped and/or executed, this needs its status
  updated the same way item 21 was updated after WO#18 closed (see
  §8.3 below for the immediate part of this that applies now).
- The Track D scoping analysis document itself
  (`track_D_agent_reorg_scoping_analysis.md`) — its consumer map was
  accurate as of WO#31's drafting; re-verify it against the real repo
  before drafting a Tier 2 or Tier 3 work order from it, the same
  caution WO#30 applied to a stale line-count figure elsewhere in this
  project.
- Any doc that cites a Tier 1 module's old flat path
  (`airflow/agents/job_agents.py` rather than
  `airflow/agents/jobs/job_agents.py`) needs a one-time repo-wide sweep
  once this postmortem is accepted — not deferred to the Tier 2/3
  completion point, since Tier 1's paths are already final. Recommend
  doing this sweep as part of accepting this postmortem rather than
  waiting, since it's cheap and unrelated to Tier 2/3's own scoping.

### 8.3 `MASTER_INDEX.md` status update (immediate, not deferred)
Unlike §8.1 and §8.2's later items, this one doesn't need to wait for
Tier 2/3:
- WO#31's row in the Track D table should move from
  "📝 Drafted, not executed" to "✅ Executed, reviewed" once this
  postmortem is accepted.
- The Track D narrative paragraph ("Only Tier 1 is drafted...") should be
  updated to "Tier 1 is executed; Tiers 2 and 3 remain unscoped," matching
  the pattern already used for WO#18/WO#20/etc. elsewhere in that
  document.
- The Roadmap section's Track D entry should be updated similarly.

### 8.4 Tier 2 and Tier 3 scoping (not this work order's job — restated)
Per WO#31's own closing note and Track D's analysis: Tier 2
(`recipe_agents.py`, `weekly_agents.py`) needs a **destination decision**
first (the Track D analysis's own language is that the right destination
"probably isn't inside `airflow/agents/` at all" — these are router-only
modules arguably misplaced there in the first place), and Tier 3
(`blog_agents.py`) needs a fundamentally different move strategy since it
can't be relocated incrementally (it's imported by both DAGs and FastAPI
routers simultaneously — a single-commit move would need to update every
consumer on both sides atomically, which is a materially different and
riskier operation than WO#31's DAG-only move). Neither should be
attempted as a casual follow-on; each needs its own dedicated scoping
pass, per the original work order's explicit instruction.

### 8.5 `models.py` — expected finding, needs explicit verification
This was raised as an example of the kind of downstream check this
section should cover, so it's addressed directly: **no evidence in the
material reviewed for WO#31 suggests `models.py` references any of the
six Tier 1 agent modules, or should.** Per GOVERNANCE.md §2.2's DAG/
FastAPI boundary — which this entire project treats as absolute — files
under `domains/*/models.py` (and the FastAPI side generally) should
**never** import from `airflow/agents/*` in either direction; that
boundary is the whole reason Tier 1 modules were safe to move
incrementally in the first place (nothing outside the Airflow layer
imports them). If a live grep against the real repo turns up a
`models.py` (or any FastAPI-side file) importing `agents.job_agents`,
`agents.media_agents`, or any of the other four under either the old or
new path, **that is not a stale reference to fix quietly** — it's a
DAG/FastAPI boundary violation and, per GOVERNANCE.md §4.5, should be
filed as its own standalone bug ticket with its own fix and
verification, not folded into an agent-reorg work order's diff.
Recommended action: run this grep once against the real repo as part of
accepting this postmortem (it doesn't need to wait for Tier 2/3), and
record the result — "confirmed clean" or "ticket filed" — directly in
this section so the next reader doesn't have to re-derive whether this
was ever checked.

### 8.6 `configure_mappers()` / mapper-registration — not applicable, noted to prevent confusion
WO#22's `configure_mappers()` verification (per MASTER_INDEX.md, that
work order's Task 3) exists because SQLAlchemy's mapper registry depends
on every domain's `models.py` having been imported somewhere before the
first query runs — a concern specific to ORM class registration.
`airflow/agents/*` modules carry no SQLAlchemy models and are never
imported by anything in that registration path. There is no
`configure_mappers()`-equivalent check needed for the agent reorg. This
is stated explicitly only because the models.py question in §8.5 sits
right next to WO#22's own scope in this project's history, and it's easy
to assume the two verifications are linked when they aren't.

### 8.7 Definition of done for the full (all-tier) agent reorg
Modeled on GOVERNANCE.md §4.6's "what 'done' means for a domain
migration," adapted for this track. The `airflow/agents/` reorganization
as a whole (Tiers 1–3) is done when:
- Every module named in the Track D analysis lives at its decided final
  location (which, per §8.4, is not necessarily still under
  `airflow/agents/` for Tier 2).
- Every consumer (DAG or router) imports each module from its final
  location — zero remaining old-path references anywhere in the repo,
  not just in the files any one tier's work order happened to list
  (§4 of this postmortem is the precedent for why that check matters).
- GOVERNANCE.md documents the final convention (§8.1).
- `MASTER_INDEX.md` reflects every tier's real status (§8.3, extended to
  Tiers 2 and 3 once they execute).
- The `models.py` / FastAPI-boundary check in §8.5 has been run and its
  result recorded, for the final state of all three tiers, not just
  Tier 1.
- No unrelated behavior changed — any pre-existing bug surfaced along the
  way (including anything found by §8.5) has its own separate ticket,
  per GOVERNANCE.md §4.5, not bundled into whichever tier's diff
  happened to surface it.

---

## 9. Rollback

Unchanged from WO#31's own instruction: `git checkout` on every file
listed in §3.1 and §3.2 (including `life_os_daily_digest.py`, per the
agreed amendment in §4) restores the pre-WO#31 state.

---

## 10. Reviewer sign-off checklist

- [ ] §4's amendment (`life_os_daily_digest.py`) reviewed and agreed as
  correct scope, not scope creep.
- [ ] §5's verification limits are acceptable, or the ⚠️ items in §6 have
  since been confirmed against the real repo and can be upgraded to ✅.
- [ ] §8.3's `MASTER_INDEX.md` update applied.
- [ ] §8.5's `models.py`/boundary grep run against the real repo and its
  result recorded in this document.
- [ ] §8.2's immediate stale-path sweep (Tier 1's own old paths only,
  not Tier 2/3-dependent items) applied.

Once all five are checked, WO#31 can be marked ✅ Executed, reviewed in
`MASTER_INDEX.md`, and this document's status line at the top updated
accordingly.
