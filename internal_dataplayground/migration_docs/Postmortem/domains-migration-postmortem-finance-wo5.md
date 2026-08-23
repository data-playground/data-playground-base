# Finance Domain Migration — Post-Mortem & Outstanding Work

**Work order:** WO#5 (finance)
**Status:** Complete, verified in a sandboxed harness (see §4)
**Audience:** future work-order authors and the agents executing them — this doc is meant to be read alongside `GOVERNANCE.md`, not as a replacement for it.

---

## 1. What happened

WO#5 relocated the `finance` domain (models, 4 routers, 4 templates + 4
partials, 1 stylesheet) out of the monolithic top-level `models.py` /
`routers/` / `templates/` / `static/` locations and into
`domains/finance/`, following the same pattern established by WO#1
(habits), WO#2 (blog, code_intel), WO#3 (jobs), and WO#4 (explorer). No
schema, endpoint, or behavior changes were made — this was a pure
location refactor, per the work order's ROLE constraint.

`models.py` now re-exports the 9 finance classes (`AccountType`,
`Category`, `Account`, `Transaction`, `AccountCreate`, `AccountResponse`,
`CategoryCreate`, `CategoryResponse`, `TransactionResponse`) from
`domains.finance.models` via a shim import, tagged
`# TODO: remove after all cross-references are updated` — identical in
spirit to the jobs/blog/code_intel/habits shims already sitting above it.

## 2. Environment caveat (read this before trusting any "✅" from WO#5)

The agent that executed WO#5 did **not** have live filesystem access to
the actual repository — only the file contents that were pasted into
that conversation. Every acceptance criterion was verified by
reconstructing the scope files exactly as given, applying the edits, and
running the result in a **sandboxed harness**: real SQLAlchemy 2.0, real
FastAPI, real Jinja2, an in-memory SQLite DB — but with hand-written
**stub** packages standing in for `domains.jobs.models`,
`domains.blog.models`, `domains.code_intel.models`, and
`domains.habits.models`, since their real source was out of scope and
not provided.

This means:
- Import-graph correctness, SQLAlchemy mapper configuration, and the
  finance HTTP endpoints themselves were **genuinely** exercised and are
  trustworthy.
- Cross-domain consumers that depend on the *real* shape of other
  domains' models (chiefly `routers/dashboard.py`, which reads from
  `Job`, `ApplicationLog`, `StagingJob`, `BlogIdea`, `Habit`, `HabitLog`,
  `HabitSettings` in addition to `Transaction`) were only checked for
  **import resolution**, not a full render, because the stubs don't
  reproduce those domains' real relationships/fields.
- A repo-wide `grep` for lingering `from models import <FinanceClass>`
  call sites was only run over the files provided in that conversation,
  not the actual full repository.

**Action for whoever applies this migration for real:** after copying
the WO#5 output into the live repo, run `pytest` (if a suite exists),
boot the app against the real MariaDB, and hit `/finance`,
`/finance/ledger`, `/finance/upload`, `/finance/settings`, and
`/dashboard` by hand once. Don't treat the WO#5 report's ✅ marks as a
substitute for that.

This same caveat will apply to every future domain-migration work order
run the same way — **future agents should say so explicitly in their own
reports**, the same way this one does, rather than letting a passing
sandboxed test read as "verified against production."

## 3. Migration status as of WO#5

| Domain | Work order | Status |
|---|---|---|
| habits | WO#1 | ✅ migrated |
| blog | WO#2 | ✅ migrated |
| code_intel | WO#2 | ✅ migrated |
| jobs | WO#3 | ✅ migrated |
| explorer | WO#4 | ✅ migrated |
| **finance** | **WO#5** | **✅ migrated (this doc)** |
| journal | WO#6 (proposed, not started) | ⬜ not started |
| recipes / ingredients / pantry / tags | — no WO yet | ⬜ not started |
| workout / equipment / exercises / plans / body metrics | — no WO yet | ⬜ not started |
| media / streaming services / recommendations | — no WO yet | ⬜ not started |
| weekly planning / user intent / shopping list | — no WO yet | ⬜ not started |
| dashboard | n/a — see §5.5 | intentionally not a domain |

The four "no WO yet" rows are **not** small. Looking at the current
`models.py`, they account for roughly half of its remaining line count
(everything after `WeeklySynthesis`): `Ingredient`, `RecipeTag`, `Recipe`,
`RecipeIngredient`, `PantryItem` (recipe module); `WorkoutLocation`,
`Equipment`, `Exercise`, `WorkoutPlan`, `WorkoutPlanDay`,
`WorkoutPlanExercise`, `WorkoutSession`, `WorkoutSet`, `BodyMetric`
(workout module); `StreamingService`, `MediaItem`, `UserMedia`,
`TVSeasonProgress`, `MediaRecommendation` (media module); `UserIntent`,
`WeeklyPlan`, `WeeklyPlanDay`, `WeeklyPlanMeal`, `ShoppingList` (weekly
planning module). Each of these has real cross-module foreign keys
(e.g. `WeeklyPlanMeal.recipe_id → recipes.id`,
`WeeklyPlanDay.workout_session_id → workout_sessions.id`,
`WeeklyPlanDay.journal_entry_id → journal_entries.id`) that **span what
would become separate domain packages** — unlike finance, which was
self-contained. Whoever scopes those work orders needs to decide up
front how cross-domain foreign keys are handled (string-based
`ForeignKey("other_domain_table.id")` references work fine across
modules as long as both sides register on the same shared `Base`, as
proven by this migration's `Base.metadata` identity check — but the
*import* graph between domain packages needs a decision: do
`domains/weekly_planning/models.py` and `domains/recipes/models.py`
import from each other, or does one own the FK and the other stay
string-only? This should be settled before WO#N for those domains is
written, not discovered mid-migration.)

## 4. Verification method used in WO#5 (recommended for future WOs)

For consistency and comparable confidence across work orders, WO#5 used
this checklist. Recommend future domain-migration WOs (starting with
WO#6/journal) adopt the same one and report against it explicitly:

1. **Import identity check** — `domain.models.X is models.X` for every
   moved class, proving the shim actually re-exports the same object
   (not a re-defined lookalike).
2. **Mapper configuration check** — `sqlalchemy.orm.configure_mappers()`
   run against the full (stubbed, if needed) model graph, catching
   FK/relationship breakage that a plain import wouldn't.
3. **Table-registry diff** — `Base.metadata.tables.keys()` compared
   before/after the migration; should be byte-for-byte identical.
4. **Router import test** — every moved router imported through its new
   package path, asserting the route table (`path`, `methods`) matches
   pre-migration.
5. **Live HTTP smoke test** — a real FastAPI app + in-memory SQLite DB +
   `httpx.ASGITransport`, hitting every endpoint in the acceptance
   criteria (including error paths) and asserting on response status and
   key markup/fragments.
6. **Static grep pass** — for orphaned/renamed static assets (CSS links,
   included partials) and for lingering `from models import <Class>`
   call sites outside the shim.

Known friction point to reuse, not rediscover: SQLite's `aiosqlite`
driver only auto-generates rowids for a plain `INTEGER` primary key, not
`BigInteger` — any table with a `BigInteger` primary key (e.g.
`transactions.id`, `workout_sessions.id`, `workout_sets.id`,
`user_media.id`) will throw `NOT NULL constraint failed` on insert
against an in-memory SQLite test DB unless the column's type is swapped
to plain `Integer` for the test harness only (see §5.2 — this is a
harness workaround, not a model change).

## 5. Cleanup to do **only after every domain above is migrated**

Do not attempt any of the following until the "no WO yet" rows in §3 are
all ✅ — doing it earlier will break the many other modules that still
import from top-level `models.py`.

### 5.1 Collapse `models.py`
Once every class has a real home in some `domains/*/models.py`,
`models.py` will contain nothing but re-export shims. At that point:
- Update every remaining `from models import X` call site across the
  repo to import `X` from its actual owning domain module directly.
- Delete the shim blocks from `models.py` one domain at a time as its
  call sites are updated (don't do this in one big-bang change — it's
  the same "keep the shim until nothing points at it" strategy this
  migration itself is following).
- Decide the fate of `models.py` once it's empty: either delete it
  entirely, or leave a one-line `# All models now live under domains/`
  stub so anyone who reflexively types `from models import ...` gets a
  clear `ImportError`/comment pointing them at the right place instead
  of a confusing `ModuleNotFoundError`.

### 5.2 Apply the SQLite/`BigInteger` fix for real (not just in a harness)
If/when this codebase gets an automated test suite, every
`BigInteger`-keyed table will hit the autoincrement issue noted in §4.
Fix it once, in the model layer, rather than re-discovering the
workaround in every future domain's tests:
```python
from sqlalchemy import BigInteger, Integer
id: Mapped[int] = mapped_column(
    BigInteger().with_variant(Integer, "sqlite"),
    primary_key=True, autoincrement=True,
)
```
This has zero effect on the real MariaDB backend (still `BIGINT
AUTO_INCREMENT`) and makes every table usable in fast in-memory SQLite
tests. Candidates: `transactions.id`, `workout_sessions.id`,
`workout_sets.id`, `user_media.id`.

### 5.3 Delete the superseded top-level files
Every domain migration to date (WO#1–WO#5) produced *new* files under
`domains/` but — because the executing agents didn't have write access
to actually delete anything in the real repo — the **old** top-level
files were left in place, to be deleted by whoever applies the patch.
Audit for, and delete, the pre-migration originals once each domain's
move is confirmed working in the real repo:
- `routers/finance_summary.py`, `routers/finance_ledger.py`,
  `routers/finance_upload.py`, `routers/finance_settings.py`
- `templates/finance.html`, `templates/finance_ledger.html`,
  `templates/finance_upload.html`, `templates/finance_settings.html`
- `templates/partials/upload_result.html`,
  `templates/partials/account_list.html`,
  `templates/partials/category_list.html`,
  `templates/partials/account_options.html`
- `static/css/finance.css`
- (and the equivalent originals for every other completed domain — this
  is worth a one-time repo-wide audit rather than trusting each WO's
  report individually, since none of them could confirm deletion.)

### 5.4 Re-check `core/templating.py` and static mounts
The `ChoiceLoader` root `FileSystemLoader("templates")` and the plain
`app.mount("/static", ...)` in `main.py` should **not** be removed even
after every domain is migrated — `base.html`, `404.html`, `500.html`,
and any other genuinely shared/global templates and assets were never
in scope for any per-domain WO and are expected to stay at the top
level permanently. Don't let a future "finish the migration" pass
delete these roots by mistake.

### 5.5 Decide what happens to `routers/dashboard.py` and `routers/_helpers.py`
- `dashboard.py` is intentionally cross-cutting (it reads from every
  domain to build the hub view) and is **not** a candidate for its own
  domain folder — flag this explicitly so a future WO doesn't try to
  "migrate" it into `domains/dashboard/`.
- `routers/_helpers.py` (currently just `html_error()`, used by
  `finance_upload.py` and likely others) is genuinely shared,
  non-domain-specific code. It shouldn't be deleted when the last domain
  migrates — either leave a slim `routers/` package containing only this
  kind of shared utility, or relocate it to `core/http_helpers.py`
  alongside `core/templating.py` and `core/base_model.py`. Whoever
  writes the "final cleanup" work order should make this decision
  explicitly rather than leaving `_helpers.py` orphaned.

### 5.6 Add real test coverage
No automated tests were found for the finance domain (or apparently any
domain) in what was provided to WO#5. Once §5.2's SQLite fix is in place
project-wide, the `smoke_test.py` pattern used to verify WO#5 (real
FastAPI app + in-memory SQLite + `httpx.ASGITransport`, seeding data and
asserting on live HTTP responses) is a reasonable starting point for a
real `pytest` suite — it caught real issues (the `BigInteger` quirk, an
Jinja2/Starlette version-compatibility bug) that pure static review
wouldn't have.

## 6. Deferred/tracked debt specific to finance

These were noticed during WO#5 but explicitly left untouched per its
ROLE (a location-only refactor, not a cleanup pass):

1. `finance_upload.py`'s Gemini categorisation call uses the
   `google-genai` SDK directly instead of the raw-REST pattern used
   elsewhere in the codebase — pre-existing, tracked debt per
   `GOVERNANCE.md §2.3`, not something WO#5 introduced.
2. `_get_active_categories()` is duplicated near-verbatim across
   `finance_ledger.py`, `finance_summary.py`, and `finance_upload.py`.
3. `Transaction.category` is a plain string, not a foreign key to
   `Category.id` — renaming or deleting a category does not cascade to
   existing transactions, and there's no DB-level integrity between the
   two. (Noted here because it's adjacent to #2 and worth fixing in the
   same pass, if the category-handling code is being touched anyway.)

See the reply in-conversation for a recommended sequencing of these
plus the SQLite fix from §5.2.

## 7. Notes for whoever writes WO#6 (journal)

- `journal.py` / `JournalEntry` / `WeeklySynthesis` carry a hard privacy
  constraint documented directly in the model docstrings: `content`,
  `gratitude`, and `challenges` must never be sent to any external AI
  call — only `mood_score`/`energy_score` may leave the app. This is the
  same category of "don't just move it, understand what it's protecting
  first" constraint that explorer's `BLOCKED_PATTERN` logic carried in
  WO#4, and should get the same explicit callout in WO#6's HARD
  BOUNDARIES section.
- `WeeklySynthesis` is written by an Airflow DAG
  (`life_os_weekly_synthesis`), which — per the standing DAG-relocation
  rule already used in WO#5 and earlier — stays untouched and out of
  scope for the domain-folder move itself.
