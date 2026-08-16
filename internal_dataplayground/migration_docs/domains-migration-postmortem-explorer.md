# Explorer Domain Migration — Post-Mortem & Forward Requirements (WO#4)

**Status:** Complete and verified via automated harness. Not yet observed against
the real production MariaDB at the time of writing — see Section 8, "Outstanding
manual actions," before considering this fully live.
**Scope of this document:** (1) a post-mortem of the `explorer` domain-folder
migration (Work Order #4) — the fourth, and per the original roadmap, last of the
priority-batch migrations (after habits, blog+code_intel, jobs); (2) a **final
cleanup checklist** to execute only once every remaining domain
(`finance`, `journal`, `recipes`/`pantry`, `workout`, `media`, `planning`) has also
migrated, following the exact deferred-batching principle established in the habits,
blog/code_intel, and jobs postmortems' own final-cleanup sections — written to be
thorough enough to stand on its own as a requirements source, without requiring the
reader to cross-reference all three prior postmortems to get the full picture; (3) a
record of decisions made (and explicitly not made) in this engagement, so they aren't
silently rediscovered or re-litigated later.

This document assumes no prior conversation context, per the convention set by every
prior postmortem in this series. If you're an agent picking up the final cross-domain
cleanup pass, or the next domain migration, read Section 5 (verification
methodology/lessons) and Part 2 (final cleanup requirements) in full before touching
any files.

---

## PART 1 — POST-MORTEM

## 1. Summary

WO#4 migrated the `explorer` module (the read-only SQL Explorer / BigQuery-style
query UI) out of the flat `routers/` / `templates/` / `static/` layout and into
`domains/explorer/`, following the exact precedent of the three prior domain
migrations. This is the smallest and structurally simplest of the four priority
domains: one router, one template, one CSS file, zero partials, and — unlike every
prior domain — **zero ORM or Pydantic model classes**, since `explorer.py` reads
database schema dynamically via raw `information_schema` queries rather than
importing any SQLAlchemy model. There was accordingly no `models.py` extraction
step and no re-export shim to add, for the first time in this migration series.

The relocation itself shipped as a pure, minimal move — the smallest diff of the
four priority-domain migrations. Verification (specifically, the standing
"the moved router's diff must show *only* the two import changes" acceptance
check) surfaced one real, pre-existing documentation/security-claim issue in the
moved file. That issue was fixed separately, as its own small, independently
verified follow-up commit, *after* the migration itself was confirmed complete —
same "don't bundle fixes into the migration diff" discipline established by every
prior domain (GOVERNANCE.md §4.5).

## 2. Scope & Objective

**In scope, and delivered:**
- **Router:** `routers/explorer.py` → `domains/explorer/routers/explorer.py`.
- **Template:** `templates/explorer.html` → `domains/explorer/templates/explorer.html`.
- **Static:** `static/css/explorer.css` → `domains/explorer/static/css/explorer.css`.
- **Config:** `core/templating.py` (new `ChoiceLoader` entry), `main.py` (import
  repointed to `domains.explorer.routers`; new `/static/explorer` mount registered
  **before** the general `/static` mount, per the Starlette registration-order rule
  established in WO#1).
- **Follow-up** (separately committed, after the migration itself was verified
  complete): one docstring-only correction in the moved `explorer.py` (Section 4.1).

**Explicitly out of scope / not applicable to this domain, confirmed during
scoping:**
- No `models.py` extraction — this domain has no model classes.
- No shim in `models.py` — nothing else in the repo imports `routers.explorer` or
  `routers/explorer.py` (confirmed via grep — see Section 4.3), so there was never
  anything for another file to import from this domain in the first place.
- No DAG involvement of any kind.
- No cross-domain SQLAlchemy `relationship()` risk (see Part 2, Section 3) — there
  is no `domains/explorer/models.py` to create such a risk.

## 3. What shipped (diffstat)

Migration commit:
```
 core/templating.py                                       | 1 +
 domains/explorer/__init__.py                              | 0 (new, empty)
 domains/explorer/routers/__init__.py                       | 0 (new, empty)
 {routers => domains/explorer/routers}/explorer.py         | 3 +--
 {static => domains/explorer/static}/css/explorer.css      | 0 (pure rename)
 {templates => domains/explorer/templates}/explorer.html   | 2 +-
 main.py                                                    | 4 +-
 7 files changed, 6 insertions(+), 4 deletions(-)
```

Follow-up commit (Section 4.1, kept separate per GOVERNANCE.md §4.5):
```
 domains/explorer/routers/explorer.py | 11 ++++++++---
 1 file changed, 8 insertions(+), 3 deletions(-)
```
(docstring only — no functional line touched; re-verified against the full
harness after this commit, see Section 5.)

## 4. Issues discovered during migration (root cause analysis)

### 4.1 — Docstring claimed a DB-level enforcement layer that was never actually provisioned (pre-existing, not introduced by this migration)

**Symptom:** none observable at runtime. Found by the mandatory close read the
"diff must show *only* the two import changes" acceptance criterion forces on
whoever performs the move — not by any test failure.

**Root cause:** `explorer.py`'s module docstring claimed:
> Uses a dedicated read-only DB session to enforce permissions at the DB level
> (configure the MariaDB user with SELECT-only grants — see notes below)

But `get_db()` (`database.py`) is the **same** session/credential (`data_playground`
user) every other router in the app uses. There is no separate read-only DB user, no
separate session factory anywhere in the codebase as provided to this engagement,
and no "notes below" section elsewhere in the file for the docstring's own
cross-reference to point to. `init-db/01-create-airflow-db.sql` — the only DB init
script available in this engagement — grants `data_playground` `ALL PRIVILEGES` on
at least the `airflow` database; the actual grants on the `jobs` database (the one
`explorer` actually queries) were never confirmed against a live instance in this
engagement (see Section 8, item 2).

**Net effect:** as of today, the keyword blocklist (`BLOCKED_PATTERN` /
`_validate_sql()`) is the **only** enforcement layer protecting this read-only
endpoint from write operations. A bug in that regex, or a new blocked keyword that
should have been added and wasn't, has **no database-level backstop**. This is the
same category of issue as the habits postmortem's §4.4 ("'enforced at the DB level'
comments are not enforcement; verify and declare") — here applied to a security
claim rather than a data-integrity constraint, which raises the practical stakes of
leaving it unaddressed.

**Decision (owner's call, recorded here for continuity):** take the cheaper of the
two available options — correct the docstring to stop overclaiming, rather than
provisioning a real SELECT-only MariaDB user in this engagement. The "provision a
real read-only user" option was deliberately **not rejected**, just deferred — see
Section 8, item 1, and Section 9.

**Fix**, shipped as its own separately-verified, docstring-only commit:
```diff
 Security:
-  - Keyword blocklist rejects any query containing write operations
-  - Uses a dedicated read-only DB session to enforce permissions at the DB level
-    (configure the MariaDB user with SELECT-only grants — see notes below)
+  - Keyword blocklist (BLOCKED_PATTERN, below) rejects any query containing
+    write operations. This is currently the ONLY enforcement layer — this
+    endpoint uses the same `get_db` session/credentials as every other
+    router in the app, NOT a dedicated read-only DB user. A bug in
+    BLOCKED_PATTERN or _validate_sql() is not currently backstopped at the
+    database level. (A previous version of this docstring claimed a
+    SELECT-only MariaDB grant was configured; that was never actually
+    provisioned. Provisioning one — and pointing this router at a session
+    bound to it — remains a real option if this ever needs a second layer
+    of defense; not done here to keep this change a docstring-only fix.)
    - Row cap of 500 rows prevents accidental full-table dumps
```

**Verification:** diffed against both the pre-fix moved file and the original
pre-migration baseline — confirmed zero functional lines changed (no change to
`BLOCKED_PATTERN`, `ROW_CAP`, `HIDDEN_TABLES`, `_validate_sql()`,
`_infer_column_type()`, or any route body). Full 6-check harness (Section 5) re-run
after this commit and confirmed identical pass results to before it, proving the fix
was behaviorally inert as intended.

### 4.2 — No functional bugs found

Unlike jobs (ATS status-highlight bug, unbounded query causing a 60+ second hang,
two sidebar navigation bugs) and blog/code_intel (non-functional folder-README
persistence, a tuple/string return-type mismatch), **no functional bugs** were
discovered in `explorer.py` during this migration — only the documentation issue in
Section 4.1. The file's small size, self-containedness, and lack of any ORM/session
lifecycle logic (no `commit()`/`rollback()`-then-reuse pattern, since it never
mutates ORM state) likely explains the difference — there was simply far less
surface area for the kind of drift the other three migrations' verification passes
caught.

### 4.3 — Confirmed: no external consumers of the old import path

Grepped the entire codebase available to this engagement for `routers.explorer`,
`routers/explorer`, and the old static path `/static/css/explorer.css`. The only
match for the former was the moved file's own now-stale top-of-file comment
(`# routers/explorer.py`) — a cosmetic detail, deliberately left untouched to keep
the migration diff at exactly the two import-line changes the acceptance criteria
required (see Section 9 for why this wasn't escalated). Zero matches for the old
static path anywhere. This confirms explorer really is domain-isolated in the way
the work order assumed, unlike jobs (which required a `templates/partials/
sidebar.html` detour) or blog/code_intel (which required updating `routers/
dashboard.py`'s shim-based import).

## 5. Verification methodology

No formal pytest suite exists yet in this repository (same starting point as every
prior migration — see blog/code_intel §5, jobs §5). The following throwaway harness
was built for this engagement:

- A **real FastAPI app** built from only the moved `explorer` router (same
  "don't boot the whole multi-domain `main.py`" approach used by every prior
  domain's harness), exercised via `httpx.AsyncClient` + `ASGITransport`.
- **Departure from precedent, and the main methodological finding worth recording
  for future domains:** this is the first domain where the in-memory-SQLite
  stand-in used by every prior migration's harness does **not** work.
  `explorer`'s `/schema` endpoint queries MariaDB's `information_schema.TABLES` /
  `information_schema.COLUMNS` directly — SQLite has no equivalent system views, so
  swapping in SQLite would silently exercise materially different SQL than
  production, not a faithful stand-in (unlike habits/blog/code_intel/jobs, where
  `Base.metadata.create_all()` against SQLite reproduces the real schema closely
  enough to trust). Used a **mock `AsyncSession`** instead — one that recognizes the
  exact SQL shapes `explorer.py` issues (`information_schema.TABLES`,
  `information_schema.COLUMNS`, the `TABLE_ROWS` count query, and the `/query`
  endpoint's raw-pass-through vs. `ROW_CAP`-wrapped-subquery shapes) and returns
  canned, MariaDB-shaped rows. This lets the **real** router code — imports,
  dependency wiring, `HIDDEN_TABLES` filtering, `_infer_column_type()`, JSON
  shaping, `_validate_sql()`, the LIMIT-injection/capping logic — run completely
  unmodified end-to-end. Nothing about `explorer.py`'s own logic was faked or
  bypassed; only the database layer beneath it was.
- **6 checks run, all passing both before and after the Section 4.1 docstring fix**
  (re-run specifically to confirm the fix was inert on behavior, not just assumed
  so from reading the diff):
  1. `GET /explorer` — full-page render, all 4 panels present (table browser,
     editor pane, results pane, quick-query chips), updated static CSS link
     confirmed present.
  2. `GET /explorer/schema` — correct JSON shape; `HIDDEN_TABLES` (tested with
     `alembic_version`) correctly excluded; column types/PK flags correctly
     inferred.
  3. `POST /explorer/query` with an explicit `LIMIT` — not wrapped, `capped: false`.
  4. `POST /explorer/query` without a `LIMIT` — wrapped in the `ROW_CAP` subquery,
     `capped: true`.
  5. `POST /explorer/query` with a direct blocked keyword (`DROP`) — HTTP 400 with
     the correct "Write operations are not permitted" message. **The single most
     important check**, per the work order's own emphasis, since it confirms the
     security-relevant regex/validation logic moved without alteration.
  6. `POST /explorer/query` with a blocked keyword hidden after a `--` comment —
     HTTP 400, proving the comment-stripping step inside `_validate_sql()` still
     runs correctly post-move (this specific case wasn't covered by any prior
     domain's harness and is a genuinely new check added in this engagement).
- **Environment note** (same one already recorded in the jobs postmortem §5): this
  sandbox's default `pip install` resolves a `starlette`/`fastapi` pairing that
  breaks this codebase's `TemplateResponse(name, context)` calling convention.
  Verification here re-used the same known-good pin already documented there:
  `starlette==0.36.3`, `fastapi==0.109.2`.
- **Not verified in this engagement, and not verifiable without one:** a live
  smoke-test against the real MariaDB `jobs` database. The mock-session harness
  proves the *move* is behavior-preserving; it cannot prove `explorer.py`'s
  `information_schema` queries are correct MariaDB syntax against the real schema
  — though this was equally true, equally unverified, and equally out of scope
  *before* this migration too, so it is not a regression risk introduced here. See
  Section 8, item 3.

## 6. Deployment

- **`web` (FastAPI):** bind-mounts the project root and runs `uvicorn --reload`,
  same as every prior migration — file moves/edits under `domains/explorer/` are
  picked up automatically. `docker compose restart web` recommended as a clean
  checkpoint, not required.
- **Alembic:** not applicable — zero schema touched, since this domain has no models
  at all.
- **No new dependency added.**
- **Airflow:** untouched, no restart needed — explorer has no DAG involvement of any
  kind, unlike jobs (which added a DAG) or journal (which is paired with one).

## 7. What went well

- The smallest, cleanest diff of the four priority-domain migrations: 7 files, 6
  insertions, 4 deletions for the migration itself — no bugfixes had to be layered
  into the same diff, unlike blog/code_intel and jobs, where the real diffstat was
  substantially inflated by in-scope bugfixes shipped in the same engagement.
- The "the moved router's `git diff` must show *only* the two import changes"
  acceptance criterion did its job exactly as designed: it forced a close read of
  the entire file, not just its import block, and caught the stale/overclaiming
  docstring (Section 4.1) — the same mechanism that caught real functional bugs in
  the two prior migrations, functioning correctly here even though this time the
  finding was a documentation/security-claim issue rather than a runtime bug.
- The docstring fix was kept as its own separately-committed, separately-verified
  change rather than folded into the migration commit, per the standing rule
  (GOVERNANCE.md §4.5, "bugs found during migration are not migration work") — kept
  the migration diff reviewable as pure relocation and the fix independently
  revertable.
- No shared, non-domain-owned file needed a fix as a side effect of this migration —
  no equivalent of jobs' `sidebar.html` detour or blog/code_intel's `dashboard.py`
  shim update was necessary, since nothing outside `domains/explorer/` ever
  referenced this domain in the first place (Section 4.3).

## 8. Outstanding manual actions

Recorded per the same convention as the jobs postmortem §9 — these are real open
items independent of migration sequencing, not "eventually, once all domains
migrate" items:

1. **Decide whether to provision a real SELECT-only MariaDB user for explorer's DB
   session** (Section 4.1). Not done in this engagement — the docstring was
   corrected instead to stop overclaiming an enforcement layer that doesn't exist.
   If picked up later: this is a `database.py` change (a second `get_db`-shaped
   dependency bound to the restricted credential, with `explorer.py`'s routes
   swapping their `Depends(get_db)` for the new dependency), **not** an
   `explorer.py` internals change beyond that one swap — the router's own query
   logic doesn't need to know which credential its session uses.
2. **Confirm `data_playground`'s actual grants on the `jobs` database** —
   `SHOW GRANTS FOR 'data_playground'@'%';` — never checked against a live instance
   in this engagement. The only init script available
   (`init-db/01-create-airflow-db.sql`) only shows `ALL PRIVILEGES` granted on the
   `airflow` database; the `jobs` database's actual grants (the one explorer
   queries) are unknown. Worth confirming regardless of whether item 1 is picked
   up — "what `data_playground` can currently do to `jobs`" is a fact worth simply
   knowing, independent of whether a second credential is ever added.
3. **Live smoke-test against the real MariaDB `jobs` database once deployed** —
   `GET /explorer`, `GET /explorer/schema`, a couple of `POST /explorer/query`
   calls including one blocked-keyword one. The mock-session harness in Section 5
   proves the *move* is behavior-preserving; it does not prove `explorer.py`'s
   `information_schema` queries are correct against the real, live schema (though
   again, this was equally unverified before this migration too — not a regression
   risk, just a pre-existing gap worth closing).

## 9. Considered and not pursued this round

- **Provisioning a dedicated read-only MariaDB user** (Section 4.1/8, item 1) —
  considered, deliberately deferred, **not rejected outright**. Revisit if/when a
  second defense layer feels worth the setup cost; the docstring fix in the
  meantime ensures nobody reading the code is misled about what protection
  currently exists.
- **Editing `core/templating.py`'s stale docstring** (it named habits by name as if
  the only fallback, now one of five) — flagged during this engagement; the project
  owner made this edit independently, outside the scope of this work order, so no
  action was needed here. Noted for completeness/continuity, the same way the jobs
  postmortem's "Considered and Rejected" section records decisions even when they
  didn't require code changes from the migrating agent.
- **A banner-comment header above each import group in `main.py`'s router-import
  block** — proposed as an optional readability improvement. The project owner's
  actual `main.py` already groups migrated-vs-not-yet-migrated imports with
  per-line `# WO<n>` tags, which was judged sufficient on its own merits. Not
  pursued further; flagged here only so it isn't independently re-proposed later
  without this context.
- **Fixing the moved file's stale `# routers/explorer.py` header comment** — left
  in place to keep the migration diff at exactly the two import-line changes the
  acceptance criteria specified. Genuinely trivial; fine to fix whenever this file
  is next touched for any other reason, not worth a standalone ticket.

---

## PART 2 — REQUIREMENTS FOR FINAL CLEANUP (execute only once ALL domain-folder migrations are complete)

This section follows the exact structure and reasoning established in the
blog/code_intel and jobs postmortems' own Part 2 sections. **Do not execute any of
this mid-way through the migration program** — every item below assumes **every**
domain (habits, blog, code_intel, jobs, explorer, **and** every domain still in
GOVERNANCE.md §3.3's backlog: `finance`, `journal`, `recipes`/`pantry`, `workout`,
`media`, `planning`) has already been extracted out of the flat `routers/` /
`templates/` / `models.py` layout.

Explorer's own footprint in this final-cleanup pass is small — it never had a
`models.py` presence to remove — but this section is written to summarize the
**full** final-cleanup picture across every domain migrated so far, not just
explorer's slice of it, so a future agent can use this single document as a
complete requirements source without needing to cross-reference all three prior
postmortems line-by-line.

### 1. How to know you're actually ready

Same check specified in every prior postmortem:
```bash
grep -rn "from models import" --include="*.py" .
```
**Target end-state:** nothing except `models.py`'s own internal comments/imports.
As of the end of WO#4 (explorer), this command's expected output is **unchanged**
from the jobs postmortem's own snapshot — explorer contributed **zero** new lines
to this grep's output, since it never imported from or exported to `models.py` in
the first place:
```
routers/dashboard.py:...:from models import (
models.py:...  (Base, in a comment)
models.py:...  (comment, inside the Jobs shim docstring)
models.py:...  (comment, inside the Blog shim docstring)
models.py:...  (comment, inside the Code Intel shim docstring)
models.py:...  (comment, inside the Habits shim docstring)
```
i.e., exactly one real remaining consumer (`routers/dashboard.py`) plus shim
docstring comments, for every domain migrated so far (habits, blog, code_intel,
jobs) — **explorer does not appear here at all, and never will**, since it has no
models. Confirm this list has shrunk to zero real consumers (`dashboard.py`
included) before proceeding — per GOVERNANCE.md §3.3, `finance`, `journal`,
`recipes`/`pantry`, `workout`, `media`, and `planning` still need to migrate before
that's true.

### 2. `models.py` final cleanup — explorer needs NO entry here; full picture reproduced for continuity

**Unlike every other domain, there is no "explorer shim" to remove from
`models.py`, because one was never added** — `explorer.py` never imported anything
from `models.py`, and `models.py` never imported anything from
`domains/explorer/`. When the eventual `models.py` cleanup pass happens (per
whichever of the two end-states the blog/code_intel postmortem Part 2 §2 lays out —
full deletion vs. reduction to a pure import-registry), **explorer requires zero
corresponding edits there.** Stated explicitly here so whoever does the final pass
doesn't spend time looking for an explorer shim that was never there.

For the domains that *do* have shims — reproduced here so this document is
self-contained and doesn't require cross-referencing three other files to get the
full current picture:

| Domain | Shim contents (current, as of end of WO#4) |
|---|---|
| Habits | `Habit`, `HabitCreate`, `HabitLog`, `HabitLogResponse`, `HabitResponse`, `HabitSettings`, `HabitUpdate` |
| Blog | `BlogProjectType`, `BlogIdeaStatus`, `DIFFICULTY_LEVELS`, `BlogIdea`, `BlogIdeaCreate`, `BlogIdeaResponse` |
| Code Intel | `ReadmeStatus`, `FolderReadmeStatus`, `CommentedStatus`, `ImprovementStatus`, `CodeProject`, `CodeFile`, `FolderReadme`, `CodeProjectCreate`, `CodeProjectResponse`, `CodeFileResponse`, `FolderReadmeResponse`, `FolderReadmeCreate` |
| Jobs | `ApplicationStatus`, `Job`, `ApplicationLog`, `JobSearchKeyword`, `WatchedCompany`, `JobScoutRunLog`, `JobResponse`, `ApplicationLogCreate`, `ApplicationLogResponse`, `StagingJobStatus`, `StagingJob`, `StagingJobCreate`, `StagingJobResponse` |
| **Explorer** | **— none. No shim exists or is needed.** |

Per the batching principle established in the habits postmortem's own final-cleanup
§1 ("do this for all domains in the same pass, not incrementally, to avoid leaving
the app in a state where some shims are gone and others aren't for no principled
reason") — **do not remove any of the above shims in isolation.** Wait until every
domain, including the ones still in the migration backlog, can have its shim (where
one exists) removed in the same pass.

`routers/dashboard.py` is still the one confirmed real external consumer of any of
these shims (see the jobs postmortem Part 2 §3 for its current import block and
what it needs to become at cleanup time). **Explorer adds nothing to
`dashboard.py`'s import list** — `dashboard.py` never read anything
explorer-related, since explorer has no data model for a cross-domain dashboard to
aggregate.

### 3. Critical risk: SQLAlchemy string-based relationship resolution — explorer is a non-issue here

Per the critical warning first raised in the blog/code_intel postmortem Part 2 §3,
and re-confirmed in the jobs postmortem Part 2 §4: any `relationship()` call using a
**string** class name only resolves correctly if every module defining a referenced
class has been imported by *something* before the first query touches that
relationship.

**Explorer adds zero risk here.** It has no ORM models at all, so there is nothing
for any other domain's `relationship()` to reference, and nothing in
`domains/explorer/` that itself needs another domain's models imported first.
**Skip `domains/explorer/` entirely** when re-running the
```bash
grep -rn 'relationship(' domains/*/models.py
```
check both prior postmortems specify — `domains/explorer/models.py` does not exist
and, under the current design (raw `information_schema` introspection, no ORM
usage), never will.

### 4. `core/templating.py` final cleanup

Explorer's entry (`FileSystemLoader("domains/explorer/templates")`) is already
correctly appended to the `ChoiceLoader` list as of this migration. At final
cleanup time, per the blog/code_intel postmortem Part 2 §4, the eventual full list
should look like:
```python
templates.env.loader = ChoiceLoader([
    FileSystemLoader("templates"),               # shared/core only: base.html, dashboard.html, 404.html, 500.html
    FileSystemLoader("domains/habits/templates"),
    FileSystemLoader("domains/blog/templates"),
    FileSystemLoader("domains/code_intel/templates"),
    FileSystemLoader("domains/jobs/templates"),
    FileSystemLoader("domains/explorer/templates"),
    # ... one line per remaining domain (finance, journal, recipes, workout, media, planning)
])
```
Nothing further needed for explorer specifically at final-cleanup time beyond
confirming this line survives whatever final-form `core/templating.py` the last
domain's migration converges on — same confirmation every other already-migrated
domain's entry needs. Also worth resolving at final-cleanup time (carried over from
the blog/code_intel postmortem, not explorer-specific): `templates/desktop.ini`, a
stray Windows Explorer artifact noted-but-not-investigated across two prior
postmortems now.

### 5. `main.py` final cleanup

Explorer's router import (`from domains.explorer.routers import explorer`) and
static mount (`app.mount("/static/explorer", ...)`, registered before the general
`/static` mount) are both already correctly in place as of this migration. By the
end of the full migration program, `main.py`'s router-import block should be
entirely `from domains.X.routers import Y` — zero `from routers import ...` lines
except for whatever stays genuinely cross-cutting (`routers/dashboard.py` — see
Section 6 below). Preserve whatever relative include-order constraints exist for
path-matching specificity — explorer introduced none (it has no
overlapping-path routers, unlike the `ci_files`/`ci_readme`-before-`ci_projects`
precedent from code_intel).

### 6. Fate of `routers/`, `templates/`, `static/`

- `routers/explorer.py`, `templates/explorer.html`, `static/css/explorer.css` are
  now **gone** from their old flat locations — confirmed via the move itself,
  nothing left behind.
- `routers/dashboard.py` almost certainly stays at the top level (or becomes its
  own thin non-domain "core" module) rather than moving into a domain folder — same
  reasoning as every prior postmortem: it has no models of its own and
  structurally aggregates every other domain.
- `static/` (top-level) likely keeps whatever is genuinely global (shared CSS/JS
  used by `base.html`, etc.) even after every domain-specific asset has moved out.
- **Explorer does not block deleting the flat `routers/`, `templates/`, or
  `static/css/` directories** once every other remaining domain reaches the same
  state — confirm this is still true (no stray explorer-related file was
  accidentally left behind or re-created at the top level) before deleting, same
  check every prior domain's postmortem specifies for itself.

### 7. Inferred remaining domain boundaries — unchanged from the jobs postmortem, reproduced here for continuity

Explorer's migration doesn't change the shape of the remaining backlog (`finance`,
`journal`, `recipes`/`pantry`, `workout`, `media`, `planning`) — see the jobs
postmortem Part 2 §7 for the full candidate-domain table, still accurate as of this
writing. **One addition worth carrying forward specifically because of what this
migration surfaced:** explorer's own experience (Part 1, Section 5 —
`information_schema` queries not being portable to a SQLite verification stand-in)
is a signal worth checking for early in any of the remaining domains too. If any of
them turn out to issue raw `information_schema` (or other MariaDB-system-table)
queries similar to explorer's, plan that domain's verification harness accordingly
from the start — build the mock-`AsyncSession` pattern in up front — rather than
discovering the SQLite mismatch mid-migration the way this engagement did.

### 8. Related but separate initiatives — sequencing recommendation

Unchanged from the blog/code_intel postmortem Part 2 §8 (AI provider-layer
consolidation, formal pytest suite, docstring/documentation consistency pass) —
explorer doesn't add or remove anything from that list. **One addition specific to
what this migration surfaced:** the mock-`AsyncSession` verification pattern (Part
1, Section 5) is worth formalizing as a reusable test fixture once the eventual
pytest suite gets built — specifically for any domain/router that talks to MariaDB
system tables/views directly rather than through the ORM, since the standard
in-memory-SQLite fixture pattern doesn't cover that case.

**Also worth carrying into that same documentation-consistency pass:** the
docstring/security-claim issue found in Section 4.1 of Part 1 is a *pattern*, not a
one-off — the habits postmortem's §4.4 already caught an "enforced at the DB level"
claim that wasn't (a data-integrity constraint that time). A dedicated pass across
every domain's docstrings and comments, specifically grepping for phrases like
"enforced at," "configured," "provisioned," "dedicated," or "guaranteed," and
cross-checking each one against what's actually wired up in code, is worth
scheduling once the structure is final — this is now the **second** instance of
this exact failure mode across four completed migrations, which is enough of a
pattern to plan for deliberately rather than keep discovering opportunistically.

### 9. Final verification checklist (run after the LAST domain migration + this cleanup)

Same standing checklist as the blog/code_intel postmortem Part 2 §9 and the jobs
postmortem Part 2 §8, reproduced and extended here:

- [ ] `grep -rn "from models import" --include="*.py" .` returns nothing (or only
  the registry file's own `Base` import, if `models.py` was kept per Option 2).
- [ ] `grep -rn 'relationship(' domains/*/models.py` — every string-named class's
  module is covered by the registry mechanism chosen. (`domains/explorer/models.py`
  will not exist and correctly should not appear in this grep's output.)
- [ ] `Base.metadata` table-identity check: same total table count before/after the
  **entire** migration series, same identity checks for every migrated class.
  (Explorer contributes zero tables either way — this check is entirely unaffected
  by explorer specifically, but must still cover every *other* migrated domain.)
- [ ] Every remaining router's `Jinja2Templates(...)` instantiation has been
  replaced with `from core.templating import templates` — explorer's is already
  done.
- [ ] Every domain's static mount (if any) is registered before the general
  `/static` mount — explorer's is already done and confirmed correctly ordered.
- [ ] Full regression suite passes against the final structure, including
  explorer's 6 checks from Part 1, Section 5 above — **adapted to run against the
  real MariaDB `jobs` database at that point, not the mock session** used for this
  engagement's verification.
- [ ] **Explorer-specific:** confirm the outstanding items in Part 1, Section 8
  above (the DB-grant question, the live smoke-test, and the read-only-user
  decision) have been either resolved or explicitly, consciously deferred again —
  don't let them silently fall out of scope just because explorer's own migration
  is long since "done."
- [ ] `docker compose restart web` (bind-mount + `--reload` means no rebuild/`down`
  needed, consistent with every migration so far) — re-confirm this assumption
  still holds if the `Dockerfile`/`docker-compose.yml` volume configuration changes
  for any reason during the remaining migrations.

---

## Reference — Files touched in WO#4 (explorer domain, final state)

**New:**
- `domains/explorer/__init__.py`, `domains/explorer/routers/__init__.py`

**Moved (then edited in place per the diffs in Part 1):**
- `domains/explorer/routers/explorer.py` (moved from `routers/explorer.py`;
  import-swap edit in the migration commit + separate follow-up docstring edit,
  Section 4.1)
- `domains/explorer/templates/explorer.html` (moved from `templates/explorer.html`;
  one-line static-path edit)
- `domains/explorer/static/css/explorer.css` (moved from `static/css/explorer.css`;
  pure rename, byte-identical)

**Edited in place (shared files — applied as targeted diffs, not full-file
replacement, per the standing rule from the habits postmortem §4.1):**
- `core/templating.py` — `domains/explorer/templates` added to `ChoiceLoader`
- `main.py` — explorer router import repointed to `domains.explorer.routers`;
  `/static/explorer` mount added before the general `/static` mount

**Untouched, confirmed via grep across everything available in this engagement:**
- Every other file in the repo — explorer has no cross-domain consumers, no shim,
  no DAG involvement, and (unlike jobs' `sidebar.html` detour, or blog/code_intel's
  `dashboard.py` shim update) no shared non-domain file needed a fix as a side
  effect of this migration.

**Endpoints covered by the regression harness used in this engagement (reuse this
list's shape for verifying any future change to this domain):**
`GET /explorer`, `GET /explorer/schema`, `POST /explorer/query` (valid w/ `LIMIT`,
valid w/o `LIMIT`, blocked keyword direct, blocked keyword hidden after a
comment).

**Outstanding items carried forward (see Part 1, Section 8 for full detail):**
1. Decide on provisioning a dedicated read-only MariaDB user for explorer's session.
2. Confirm `data_playground`'s real grants on the `jobs` database.
3. Live smoke-test against the real MariaDB `jobs` database post-deploy.
