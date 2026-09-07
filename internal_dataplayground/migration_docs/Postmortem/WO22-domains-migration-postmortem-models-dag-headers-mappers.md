# WO#22 Postmortem — `models.py` End State, Stale DAG Header Cleanup, and `configure_mappers()` Verification

**Status:** ⚠️ **Executed with one open item.** Tasks 1 and 2 are complete and
independently verified against the real file contents. Task 3 — the
`configure_mappers()` check that every postmortem since WO#2 has flagged as
outstanding — was attempted for real in this session and is correctly
blocked by environment, not by a defect in Tasks 1–2's work. **Do not mark
this row ✅ in `00_MASTER_INDEX.md` until Task 3 is re-run in the real
deployed environment** — see §8 and §9 below. This is the single most
important sentence in this document for the reviewer.

**Work order:** `work_order_22_models_dag_headers_configure_mappers.md`
**Depends on:** WO#20 (shim removal) — its precondition ("every domain
shim removed, `dashboard.py` already repointed") was taken as given per
the work order's own "Current state (confirmed by direct read of the
file)" framing, and was independently re-confirmed by this session's own
grep (§3.1) rather than re-trusted blindly.
**Session type:** Single AI coding session, chat-based, no direct access
to the live repo, live database, or installed dependencies. This
constraint shapes almost every "what to verify before trusting this"
caveat in this document — flagged explicitly rather than glossed over,
per the standing lesson from WO#15/#17/#20 (see §5).

---

## 1. Executive Summary

Three independent, boundary-scoped tasks were executed in sequence, as
the work order required (Task 3 last, against the state Tasks 1–2
produced):

| Task | What it was | Result |
|---|---|---|
| 1 | Decide and execute `models.py`'s final end-state | ✅ **Delete-and-centralize** branch taken (grep-confirmed zero real consumers); `models.py` deleted; a 10-line domain-import registry added to `database.py` at module scope |
| 2 | Fix 13 stale DAG header-comment paths | ✅ All 13 corrected (14 line-edits total — `life_os_blog_scout.py` needed two); the 2 files needing no change were confirmed untouched |
| 3 | Run `configure_mappers()` against the post-change state | ⚠️ **Attempted for real, blocked by environment** — no network (couldn't install SQLAlchemy) and no access to `domains/`, `core/`, `routers/`, or `gcp_secrets.py` (never part of this engagement's materials). Follow-up ticket filed; **this is the same top-priority gap `00_MASTER_INDEX.md` has flagged since WO#2, still open** |

No schema changes, no behavior changes, no router/template/static edits.
Diff surface is exactly: `models.py` (deleted), `database.py` (one block
added), `migration_docs/GOVERNANCE.md` §2.4 (one paragraph added,
optional per the WO), and 13 DAG files (one or two comment lines each).

---

## 2. What The Work Order Asked For (As Originally Drafted)

Recapping the literal spec, so §3 and §4 below can be checked against it
line by line rather than against this session's own memory of what it did.

**ROLE:** senior refactoring engineer doing final cleanup — none of the
three tasks is a feature change.

**HARD BOUNDARIES (all tasks):**
- Only touch files listed in each task's own SCOPE.
- Task 3 runs last, against the state Tasks 1–2 produced.
- Task 1 is explicitly pre-authorized to diverge from `00_MASTER_INDEX.md`'s
  literal "Option 2 — shrink to a ~15-line registry" recommendation, *if*
  reading the actual file shows that recommendation wouldn't accomplish
  anything — framed by the WO itself as "a correction of prior reasoning,
  not a scope deviation."
- Task 2 is cosmetic only — no `sys.path`/`dag_id`/schedule/logic changes.
- Task 3 is verify-and-report only — no preemptive fixing if it raises.

**Task 1 SCOPE:** `models.py`, `database.py`, `migration_docs/GOVERNANCE.md`
§2.4 only (optional).
**Task 1 STEPS:** grep the repo for `from models import` / bare `import
models`; branch on the result (real consumer → shrink to registry; zero
consumers → delete `models.py` and centralize the import block in
`database.py`); don't touch anything else in `database.py`; optionally
close out GOVERNANCE §2.4.

**Task 2 SCOPE:** 13 named DAG files, one header-comment line each
(`life_os_blog_scout.py` needs both of its two occurrences fixed).
**Task 2 STEPS:** edit exactly the stale path comment in each; confirm (don't
edit) `life_os_idea_expander.py` (no header comment exists) and
`life_os_refresh_streaming_availability.py` (already correct).

**Task 3 SCOPE:** verification only, no files edited.
**Task 3 STEPS:** in the real app environment, run
`import sqlalchemy.orm as orm; import main; orm.configure_mappers()`;
report the exact result verbatim; if it raises, file a separate ticket
and mark the criterion ⚠️, not ❌ — no fix attempted here.

**OUTPUT FORMAT** required by the WO: files created / moved / edited,
acceptance criteria per task, and Notes. **It did not ask for a zip
archive of the changed files** — that was requested by the user in a
later turn of this same conversation, after the written report was
delivered. See §4, item D.

---

## 3. What Was Actually Done, Task By Task

### 3.1 Task 1 — `models.py` End State

**Grep, run for real** against the reconstructed repo (all 15 DAG files,
`main.py`, `database.py`, `models.py`, `CONTRIBUTING.md`, `GOVERNANCE.md`,
`docker-compose.yml`, `Dockerfile`, `Dockerfile.airflow`,
`requirements.txt` — i.e., every file this session had access to):

```
$ grep -rn "from models import" .
./models.py:9:# other file still doing `from models import Base` keeps working unchanged.
./CONTRIBUTING.md:23:from models import BlogIdea, BlogIdeaStatus
./CONTRIBUTING.md:39:from models import BlogIdeaStatus

$ grep -rEn "(^|[^.[:alnum:]_])import models($|[^.[:alnum:]_])" .
(no matches)
```

All three hits are inert, not real consumers:
- `models.py:9` is a **Python comment** inside `models.py` itself,
  describing the historical shim pattern retrospectively — it is not
  executed and does not import anything.
- `CONTRIBUTING.md:23` and `:39` are both inside that document's own
  **"WRONG — never do this in a DAG"** example code block — illustrative
  anti-pattern text for a human reader, never executed.

No hit appears in `main.py`, any of the 15 DAG files, or the executable
part of `database.py`. This is exactly the outcome the work order itself
predicted ("expected — this is what direct reading of `main.py`/
`dashboard.py`/every router suggests"), now grep-confirmed rather than
assumed.

**Branch taken: zero real consumers → delete-and-centralize**, per the
work order's own Step 2 instructions for this exact outcome.

**Concrete changes:**
1. `models.py` deleted from the repo root. Confirmed via `ls models.py` →
   `No such file or directory`.
2. `database.py` — a 10-line import block was added at **module scope**,
   immediately after the existing top-of-file imports and before the
   `Settings` class definition (not inside `init_db()`'s function body).
   One import per domain that has a real `models.py`:
   `habits, blog, code_intel, jobs, finance, journal, recipes, workout,
   media, planning` — ten domains, matching the full, closed
   domain-migration backlog per `00_MASTER_INDEX.md`. `explorer` is
   correctly omitted, per that domain's own migration record ("no ORM
   classes at all").
3. Nothing else in `database.py` was touched — `Settings`, `init_db()`,
   `get_db()`, and `__all__` are byte-identical to the version given to
   this session, aside from the new block and its explanatory comment.
4. **Placement rationale (module scope, not inside `init_db()`):**
   `main.py`'s very first local import is `from database import init_db`
   — `database.py` is therefore imported unconditionally and *before*
   `init_db()` is ever awaited during the FastAPI lifespan. Module-scope
   placement means the import side-effect (mapper registration) fires at
   `database.py`'s own import time, which is strictly earlier and cannot
   be skipped, whereas function-scope placement would only fire the first
   time `init_db()` is actually called. Module scope is the stronger
   guarantee.
5. **Optional GOVERNANCE.md §2.4 edit — exercised.** The WO explicitly
   offered this as "only if convenient." One closing paragraph was added
   marking §2.4 historical/closed, pointing at this work order. This is
   not a scope deviation — it's the WO's own explicitly-offered option,
   taken.

### 3.2 Task 2 — Stale DAG Header-Comment Paths

All 13 files in the WO's table were edited, one atomic string-replace
each, with the old string required to match **exactly once** in the file
(this is itself the diff-scope guarantee — see §5):

```
airflow/dags/blog/life_os_blog_creator.py            (line 1)
airflow/dags/blog/life_os_blog_finalizer.py          (line 1)
airflow/dags/blog/life_os_blog_scout.py              (lines 1 AND 12 — both occurrences)
airflow/dags/code_intel/life_os_readme_writer.py     (line 1)
airflow/dags/code_intel/life_os_code_narrate.py      (line 1)
airflow/dags/code_intel/life_os_code_comment.py      (line 1)
airflow/dags/code_intel/life_os_code_improve.py      (line 1)
airflow/dags/jobs/life_os_job_scout.py               (line 1)
airflow/dags/jobs/life_os_job_scout_ats.py           (line 1)
airflow/dags/jobs/life_os_daily_digest.py            (line 1)
airflow/dags/jobs/life_os_staging_promoter.py        (line 1 — the 13th, not in the master index's tracked count of 12)
airflow/dags/journal/life_os_weekly_synthesis.py     (line 1)
airflow/dags/media/life_os_generate_embeddings.py    (line 1)
```

`life_os_blog_scout.py`'s duplicated module-docstring-with-header
structure (two near-identical docstring blocks pasted back to back — the
second an evolved v2 of the first) was left exactly as found. Both stale
path comments inside it were corrected; the duplication itself was not
touched, per the WO's explicit instruction to note it, not fix it (see
§9, item 3).

Confirmed untouched, as instructed:
- `life_os_idea_expander.py` — grep for the header pattern returns
  nothing; the file genuinely never had one.
- `life_os_refresh_streaming_availability.py` — its docstring's first
  line already reads the correct current path; nothing to change.

### 3.3 Task 3 — `configure_mappers()` Verification

Run for real, not assumed unavailable:

```
$ pip install sqlalchemy fastapi pydantic-settings --break-system-packages
ERROR: Could not find a version that satisfies the requirement sqlalchemy
ERROR: No matching distribution found for sqlalchemy
  (this session's network egress is disabled — see its own environment notice)

$ python3 -c "import sqlalchemy.orm as orm; import main; orm.configure_mappers()"
ModuleNotFoundError: No module named 'sqlalchemy'
```

Even setting the missing dependency aside, `main.py` itself cannot import
in this session — it pulls in `domains.*`, `core.*`, `routers.dashboard`,
and `gcp_secrets`, and a direct check confirmed **none of these were ever
part of this engagement's materials**:

```
domains:        MISSING
core:            MISSING
routers:         MISSING
gcp_secrets.py:  MISSING
```

**This is not a bug in the application.** It is the exact, specific gap
`00_MASTER_INDEX.md`'s "Where This Actually Stands" section names as the
single top-priority open item across the entire 20-work-order program: no
engagement, including this one, has ever had every domain's real
`models.py` simultaneously alongside a real, dependency-installed Python
environment. This session did not fabricate a "raised nothing" result to
manufacture a clean pass — the honest result is "cannot be run here,"
reported verbatim, with a follow-up ticket (§9).

---

## 4. Deviations & Agreements Log — What Changed From The Literal Spec, And When

This section exists specifically so a reviewer doesn't have to
cross-reference the WO text against the report by memory. Every departure
from the WO's literal text is listed here, tagged by type.

| # | Item | What the WO literally said | What actually happened | Type |
|---|---|---|---|---|
| A | Task 1 branch choice | Branch on grep result: shrink-to-registry *or* delete-and-centralize | Delete-and-centralize taken, because grep came back zero real consumers | **Not a deviation** — this is the WO's own pre-specified conditional logic, correctly followed given the grep result it produced. Recorded here only so the reviewer can see the condition was actually checked, not assumed. |
| B | GOVERNANCE.md §2.4 edit | "Optional, only if convenient... skip if this feels like scope creep" | Done — one closing paragraph added | **Option exercised, not a deviation.** The WO explicitly offered this as agent's discretion. |
| C | Task 2 diff-scope verification method | Acceptance criterion literally says: "`git diff` on each of the 13 shows exactly one changed comment line" | No real `git` repository existed for this session (files were reconstructed from pasted document text, not cloned). Substituted: (1) each edit was a `str_replace` requiring an *exact, unique* match on only the stale comment text, which is a structural guarantee no other line could have changed in that same operation; (2) a full `ast.parse()` syntax check across every `.py` file, confirming no corruption; (3) a tail-line check on every file confirming each ends on its real closing logic, not mid-statement. | **Disclosed substitution**, in the same spirit as WO#1's `Base.metadata` check standing in for a literal `alembic` dry-run output. Equivalent-strength evidence, not identical to the literal requested artifact. Flagged here explicitly — see §5. |
| D | Deliverable format | WO's OUTPUT FORMAT specifies a written report (files created/moved/edited, acceptance criteria, Notes) | A `.zip` archive of the full changed file tree, under the real `internal_dataplayground/` folder structure, was additionally produced and delivered | **Agreed after the fact, at the user's explicit request**, in a later turn of this same conversation (*"Provide all files in a ZIP following the project's structure"*). Not something this session added on its own initiative. |
| E | Workspace artifact | N/A — not part of any instruction | An early `mkdir -p .../{blog,jobs,code_intel,media,journal}` command was run in a shell that doesn't brace-expand, literally creating a directory named `{blog,jobs,code_intel,media,journal}` inside the working tree | **Self-introduced error, self-caught and corrected** before the zip was delivered. Disclosed here rather than silently fixed and left unmentioned, per the standing "materials/claims presented as complete when they weren't" pattern flagged repeatedly in `00_MASTER_INDEX.md` (WO#15, #17, #20) — this session is naming its own instance of the same category proactively. |
| F | Task 3 execution | "In the real application environment... execute [the check]" | Could not be executed in the real application environment — attempted in this session's own sandboxed shell instead, which lacks both network access and the application source | **Environment substitution, disclosed, not silently skipped.** Per the WO's own "HANDLING PRE-EXISTING BUGS" rule extended to this case: reported the exact blocking condition, did not attempt a fix, filed a follow-up ticket, marked ⚠️ not ❌. |

**Net effect for the reviewer:** items A and B are not really deviations
at all — they're either conditional-logic-followed-correctly or
explicitly-offered-discretion-exercised. Items C, D, E, and F are the
real "what changed from a literal reading of the WO" list, and none of
them touch the actual production diff (`database.py`, `models.py`, the 13
DAG files) — they're all about *verification method* or *delivery format*,
not about what code changed.

---

## 5. Verification Methodology — Read This Before Trusting Any ✅ Above

Being explicit about *how* each claim above was checked, because this
program's own postmortem history (WO#15's silently-truncated files,
WO#17's unchecked `base.css`, WO#20's unchecked router-source claim) shows
that unstated verification methodology is exactly where false confidence
creeps in.

- **Grep (Task 1):** run for real, against every file this session had in
  its working tree — not assumed, not guessed at from memory of the
  documents.
- **File deletion / creation (Task 1):** confirmed via direct filesystem
  check (`ls models.py` → not found), not just "the edit tool said it
  succeeded."
- **Header edits (Task 2):** confirmed via a second, independent `grep`
  pass over all 13 target files plus both excluded files, run *after* the
  edits, not just trusted from the edit tool's own success message.
- **No-corruption check (Task 2, and incidentally Task 1's `database.py`
  edit too):** `ast.parse()` was run against **every** `.py` file in the
  reconstructed tree (17 files) and every one parsed without a
  `SyntaxError`. This catches truncation, mismatched quotes/braces, or
  any accidental content loss from an edit — the exact failure mode
  WO#15's postmortem flagged as having gone undetected until a direct
  follow-up question forced a second check.
- **Task 3:** the blocking condition (`ModuleNotFoundError`, then the
  missing-directory check) was reproduced and shown verbatim, not
  paraphrased or assumed.
- **What was *not* verified, and should not be treated as verified:**
  - No live MariaDB connection was ever available, so nothing about
    actual query behavior, actual mapper resolution, or actual
    `relationship()` string-name resolution across domains was tested —
    only that the Python source itself is syntactically intact and that
    the import graph *as read* implies the guarantee holds.
  - No real `git` history exists for this session's reconstructed files,
    so `git log --follow`-style history-preservation claims (as WO#18's
    postmortem was able to make) are **not** possible here and were not
    attempted.
  - The 10-domain list in `database.py`'s new registry block was taken
    directly from `00_MASTER_INDEX.md`'s own statement that the "core
    domain-migration backlog" (WO#1–10) is closed in full. This session
    did not independently confirm all ten `domains/<name>/models.py`
    files actually exist with that exact import path, because none of
    them were provided as materials to this session, at all, in any
    form. **This is the same category of gap as Task 3** — trust in the
    domain list is only as good as `00_MASTER_INDEX.md`'s own claim, not
    independently re-derived here.

---

## 6. Files Changed

| Path | Change | Verified how |
|---|---|---|
| `models.py` | Deleted | Filesystem check |
| `database.py` | 10-line domain-import registry block added at module scope, plus one explanatory comment; nothing else touched | `ast.parse()`, tail-check, manual review of full file |
| `migration_docs/GOVERNANCE.md` | §2.4 — one closing paragraph appended, marking the section historical/closed | Manual review |
| `airflow/dags/blog/life_os_blog_creator.py` | Header comment, line 1 | grep + ast.parse + tail-check |
| `airflow/dags/blog/life_os_blog_finalizer.py` | Header comment, line 1 | same |
| `airflow/dags/blog/life_os_blog_scout.py` | Header comment, lines 1 and 12 (both occurrences) | same |
| `airflow/dags/code_intel/life_os_readme_writer.py` | Header comment, line 1 | same |
| `airflow/dags/code_intel/life_os_code_narrate.py` | Header comment, line 1 | same |
| `airflow/dags/code_intel/life_os_code_comment.py` | Header comment, line 1 | same |
| `airflow/dags/code_intel/life_os_code_improve.py` | Header comment, line 1 | same |
| `airflow/dags/jobs/life_os_job_scout.py` | Header comment, line 1 | same |
| `airflow/dags/jobs/life_os_job_scout_ats.py` | Header comment, line 1 | same |
| `airflow/dags/jobs/life_os_daily_digest.py` | Header comment, line 1 | same |
| `airflow/dags/jobs/life_os_staging_promoter.py` | Header comment, line 1 | same |
| `airflow/dags/journal/life_os_weekly_synthesis.py` | Header comment, line 1 | same |
| `airflow/dags/media/life_os_generate_embeddings.py` | Header comment, line 1 | same |
| `airflow/dags/blog/life_os_idea_expander.py` | **Not edited** — confirmed no header comment exists | grep (no match) |
| `airflow/dags/media/life_os_refresh_streaming_availability.py` | **Not edited** — confirmed already correct | manual review |

No other file (routers, templates, static assets, `main.py`,
`CONTRIBUTING.md`, `docker-compose.yml`, `Dockerfile*`,
`requirements.txt`) was modified.

---

## 7. Acceptance Criteria — Full Results

### Task 1
- [x] Grep command and output reproduced verbatim (§3.1). ✅
- [x] Branch taken stated explicitly: delete-and-centralize. ✅
- [x] `models.py` no longer exists at repo root; the registry block exists
  in `database.py`, confirmed to run at import time (module scope,
  ahead of `init_db()`'s own definition in the same file). ✅
- [ ] (Kept-branch criterion — not applicable, since the delete branch
  was taken.)
- [⚠️] "The `configure_mappers()` check in Task 3 passes... that's the
  real proof either branch worked" — **cannot be confirmed yet.** Task 1's
  own file-level work is done and internally consistent, but this
  specific acceptance criterion is explicitly gated on Task 3, which is
  itself ⚠️. **Do not mark Task 1 fully ✅ independent of Task 3 — the WO
  wrote it that way on purpose.**

### Task 2
- [x] All 13 files' header comments now match their actual current path. ✅
- [x] Each edit changed exactly one comment line (two, for
  `life_os_blog_scout.py`) — no `sys.path`/`dag_id`/schedule/logic changes
  anywhere, guaranteed structurally by `str_replace`'s exact-unique-match
  requirement and independently spot-checked via `ast.parse` + tail
  review. ✅
- [x] `life_os_idea_expander.py` and
  `life_os_refresh_streaming_availability.py` confirmed untouched. ✅

**Task 2 is the one task in this WO with no outstanding dependency — it
can be marked fully ✅ on its own.**

### Task 3
- [x] Result reported verbatim (§3.3). ✅
- [ ] "If it raised nothing: this closes the mapper-registration
  verification gap" — did not raise nothing; could not be run at all. Not
  applicable.
- [x] "If it raised something: a follow-up ticket exists... this task's
  own criterion is marked ⚠️, not ❌" — done (§9). ✅ *(criterion about
  process-followed, not about the check having passed)*

---

## 8. Reviewer Checklist (per `GOVERNANCE.md` §4.4)

Walking the standing 4-point review order explicitly, for whoever signs
off on this:

1. **Did every HARD BOUNDARY get respected?** Yes — cross-checked "Files
   edited" (§6) against each task's SCOPE line by line. No router,
   template, static, schema, or DAG logic/schedule/`sys.path` file was
   touched. The one file touched outside a task's literal file list
   (`GOVERNANCE.md`) was explicitly pre-authorized as optional by Task 1
   itself.
2. **Are all ❌/⚠️ items genuinely out of the agent's control?** Yes for
   the one ⚠️ (Task 3) — it is a missing-materials/missing-network problem
   identical in kind to the gap named in `00_MASTER_INDEX.md`'s own
   "Where This Actually Stands" section, not incomplete work. There are no
   ❌ items in this WO.
3. **Does the Notes/Open-Items section surface anything needing its own
   ticket?** Yes — see §9. Filed separately, not folded into this diff.
4. **Only after 1–3: do the acceptance criteria that matter functionally
   actually pass?** Task 2: yes, unconditionally. Task 1: yes, at the
   file/diff level — but Task 1's own acceptance criteria explicitly tie
   its "real proof" to Task 3, which has not passed yet. **This WO should
   be recorded as "Executed, Task 3 pending live verification" — not as a
   clean, unconditional ✅ — until someone runs the check in the actual
   deployed container.**

---

## 9. Open Items / Follow-Up Tickets

1. **[Blocking full sign-off] Run `configure_mappers()` for real.**
   Ticket: *"Run `configure_mappers()` against the live repo with
   `MARIA_DB` and dependencies installed, per WO#22 Task 3."* Natural
   vehicle: `docker compose exec web python -c "import sqlalchemy.orm as
   orm; import main; orm.configure_mappers()"` — the `web` container
   already has every dependency installed and every domain's real source
   mounted (`docker-compose.yml`'s `web` service mounts `.:/app`). This
   closes the single most-repeated open item across the entire
   20-work-order program (flagged in some form since WO#2).
2. **`00_MASTER_INDEX.md` update** (the WO's own "for the next work
   order" note): record that Task 1 took the **delete-and-centralize**
   branch, not the "Option 2 — shrink to registry" language currently on
   record, and update Track B item 3 accordingly once item 1 above is
   also closed.
3. **`life_os_blog_scout.py`'s duplicated docstring block** — cosmetic,
   trivial, explicitly out of this WO's scope (Task 2 Step 2 says fix the
   stale paths, don't clean up the duplication). Left as its own
   candidate for a future one-line-scope cleanup ticket.
4. **Domain-list trust gap (§5, last bullet):** the 10-domain list now
   hard-coded into `database.py`'s registry was taken from
   `00_MASTER_INDEX.md`'s own claim that WO#1–10 are all executed, not
   independently re-derived from the real `domains/` tree (never provided
   to this session). Worth a five-minute `ls domains/*/models.py` sanity
   check the next time someone has real repo access, purely as a
   belt-and-suspenders check alongside item 1.

---

## 10. Post-Migration Maintenance Requirements — What To Do After Future Domain Work

**This section is the standing operating procedure for `database.py`'s
new registry block, going forward.** It replaces the old two-step
"add a shim to root `models.py`, then update `dashboard.py`" contract
that `GOVERNANCE.md` §2.4 used to describe. Any future agent picking up
domain-related work should read this section first and treat it as a
checklist, not prose to skim.

### 10.1 The contract, stated plainly

`database.py` now contains the **single, centralized guarantee** that
every domain's ORM classes are registered with SQLAlchemy's mapper
registry before the app can serve a request. That guarantee is exactly as
strong as the accuracy of its import block — nothing more. If a domain's
`models.py` exists but isn't imported there, its classes are not
mapper-registered, and any `relationship("ClassName", ...)` string
reference pointing at one of its classes will fail at query time with a
confusing, hard-to-trace error — not at import time, and not obviously
tied back to "a registry line is missing." This is precisely the failure
mode `GOVERNANCE.md` §2.2's cross-domain string-relationship pattern
depends on being prevented.

### 10.2 When a **new domain** is added (Track E: NBA, Soccer, and
possibly Medium — per `00_MASTER_INDEX.md`'s "Deferred / Not Yet Scoped"
section)

For every new domain that defines real ORM classes in its own
`domains/<name>/models.py`:

1. Add **one line** to the import block in `database.py`:
   ```python
   from domains.<name> import models as _<name>_models  # noqa: F401
   ```
   placed alphabetically alongside the existing ten, for scanability.
2. Re-run the `configure_mappers()` check (§9, item 1) — every time, not
   just once historically. A new domain can introduce a
   `relationship("ClassName", ...)` typo or a genuinely missing
   registration that only this check catches, and it catches it cheaply
   (no live DB needed, just an import-time check).
3. If the new domain follows the "no ORM classes" pattern (`explorer`'s
   precedent) — e.g., a domain that's purely a UI/reporting layer over
   other domains' data — **do not add a line for it.** Confirm this
   explicitly in the work order's own report ("Domain X has no
   `models.py` — omitted from the registry, per the `explorer` precedent")
   rather than silently having no line and leaving a reviewer to wonder
   whether it was missed.
4. **Medium specifically** has an open, undecided branch per
   `00_MASTER_INDEX.md` item 6: own domain vs. folding into the existing
   `blog` domain's AI-content work. If it becomes its own domain, it needs
   a registry line per the steps above. If it folds into `blog`, it needs
   **no separate line** — `blog`'s own `models.py` is already registered,
   and Medium-related classes would live inside it. **Whoever resolves
   that open decision should update this checklist's assumption, not
   silently pick one path.**

### 10.3 When a domain is **deprecated, merged, or removed** — "removing
the references"

This is the symmetric case, and it's easy to miss because nothing breaks
loudly if it's skipped *until* someone hits a mapper error much later:

1. If a domain's `domains/<name>/models.py` is deleted or merged into
   another domain's `models.py` (e.g., a hypothetical future decision to
   fold two overly-granular domains together, in the spirit of
   `GOVERNANCE.md` §1.2's "consider whether it's actually two domains that
   were merged prematurely" note), **remove that domain's corresponding
   import line from `database.py`'s registry block.** An import line
   pointing at a module that no longer exists will raise `ImportError` on
   the very next app startup — this is a loud, fail-fast failure mode,
   which is actually the safest possible outcome; do not "fix" it by
   leaving a stale line commented out indefinitely. Delete it cleanly.
2. Re-run `configure_mappers()` again after the removal — a merge can
   just as easily break a `relationship()` string reference as an
   addition can, if the merged classes were renamed in the process.
3. Search for any other `from domains.<removed-name>.models import ...`
   references across the codebase (any router, any DAG's comment
   examples, any documentation) before considering the removal complete
   — the same class of "stale reference" problem this WO's own Task 2
   existed to clean up for DAG header comments.
4. Update `GOVERNANCE.md` §2.1's domain list and `00_MASTER_INDEX.md`'s
   own tracking tables to reflect the removal — don't let the documentation
   drift the way the DAG header comments and the "12 vs. 13" file-count
   both drifted after WO#18.

### 10.4 Recommendation: make this a standing, automated check

Given that this exact verification gap (`configure_mappers()` never run
against the real app) has now been flagged as outstanding across
**every** postmortem since WO#2, this session recommends — but has not
executed, since it's outside this WO's scope — adding a lightweight CI
step or pytest test that imports `main` and calls
`sqlalchemy.orm.configure_mappers()` on every push, mirroring the spirit
of `GOVERNANCE.md` §1.2's existing CI-checkable-lint-step precedent for
router line limits. This would turn "did anyone remember to update the
registry" from a manual, easy-to-forget checklist step into an automatic,
fail-fast gate — exactly the kind of structural fix this program has
favored elsewhere (e.g., the static-mount-ordering rule born from WO#1).
Flagging this as a recommendation for whoever scopes the next
infrastructure-adjacent work order, not executing it here.

### 10.5 Quick-reference checklist (copy this into the next relevant work order)

- [ ] New domain has real ORM classes? → add one import line to
  `database.py`, alphabetically placed.
- [ ] New domain has no ORM classes (like `explorer`)? → explicitly state
  "no line added, no-ORM precedent" in the report — don't leave it
  ambiguous.
- [ ] Domain removed/merged? → delete its import line; grep for any other
  stale references to it project-wide.
- [ ] Either way → re-run `configure_mappers()` before calling the change
  done. This is not optional and not a "nice to have" — it is the only
  check that actually proves the registry is correct.
- [ ] Update `GOVERNANCE.md` §2.1 and `00_MASTER_INDEX.md` if the domain
  list itself changed.

---

## 11. Rollback

`git checkout` on every file in §6's "Change" column:
`models.py` (restores it), `database.py`, `migration_docs/GOVERNANCE.md`,
and the 13 DAG files. No database migration, no schema change, no router
change — rollback is a pure file-content revert with no other side
effects to unwind.

---

## 12. Recommended `00_MASTER_INDEX.md` Updates (Carried Forward From The WO's Own Closing Note)

The work order's own final section ("For the next work order — not part
of this one") asked that, once this lands, `00_MASTER_INDEX.md` be
updated to record the Task 1 decision in place of its current "Option 2"
language, closing out Track B item 3. Restating it here so it isn't lost
between this postmortem and that document:

- Replace: *"Execute `models.py`'s end-state (Option 2, the ~15-line
  registry — item 20)..."*
- With: *"`models.py` deleted; mapper-registration guarantee centralized
  in `database.py` at module scope (WO#22, delete-and-centralize branch —
  see `domains-migration-postmortem-wo22-models-dag-headers-mappers.md`).
  `configure_mappers()` live-run still pending — see that postmortem's
  §9."*
- Do **not** mark Track B item 3, or this WO's row, as fully ✅ until the
  `configure_mappers()` live run (§9, item 1) is actually performed and
  its result is folded back into both this postmortem and the master
  index. This is the one gate standing between "diff looks right" and
  "migration confirmed successful."
