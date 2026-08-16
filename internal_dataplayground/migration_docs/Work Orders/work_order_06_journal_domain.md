# Work Order #6 — Domain Migration: `journal`

*Single-router domain. Its DAG (`life_os_weekly_synthesis`) stays untouched
per the standing DAG-relocation rule. This domain carries a real privacy
constraint documented directly in its own code — treat it with the same
seriousness as a security boundary, not just a style note.*

---

## ROLE
You are a senior refactoring engineer performing a structural code migration.
Your job is NOT to improve, optimize, or modernize the code you move — only
to relocate it correctly and verify it still behaves identically. Resist the
urge to "clean up while you're in there." Flag improvement opportunities as
a NOTES section at the end instead of acting on them.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below. If you believe you
  need to touch a file outside this list to complete the task, STOP and
  report why — do not proceed and do not guess.
- No schema changes. No renamed tables, columns, or endpoints. No behavior
  changes. This is a location refactor only.
- If a step's instructions conflict with what you find in the actual code
  (e.g. a file/class isn't where the work order says it is), stop and report
  the discrepancy rather than improvising a fix.
- You may create empty `__init__.py` package markers as needed to support
  new import paths — this is expected scaffolding, not a scope expansion,
  and does not need to be flagged as a deviation (a one-line mention in the
  report's "Files created" section is enough).
- **PRIVACY BOUNDARY — treat as seriously as a security constraint:**
  `JournalEntry.content`, `.gratitude`, and `.challenges` are documented
  in the existing code as fields that must NEVER be sent to any external
  AI API — only `mood_score` and `energy_score` (numeric fields) feed the
  weekly synthesis DAG. This constraint is enforced entirely by convention
  today (comments and docstrings in `models.py`, `routers/journal.py`, and
  the DAG), not by code-level restriction. **Do not change, remove, or
  weaken any of these existing privacy comments/docstrings during the
  move** — carry them forward verbatim into the new file location. If you
  notice this constraint could be more robustly enforced (e.g.
  programmatically rather than by convention), do NOT implement that —
  note it under "Notes" as a future hardening idea, since implementing it
  would be a behavior change, not a location refactor.
- `routers/journal.py` contains a **local, in-function import from another
  not-yet-migrated domain**: inside `save_entry()`, there's a `try` block
  doing `from models import WeeklyPlanDay as _WPD, WeeklyPlan as _WP,
  WeeklyPlanStatus as _WPS` to link a journal entry back to its weekly
  plan day, wrapped in a broad `except Exception: pass` so a linking
  failure never breaks the journal save itself. **Leave this import
  exactly as `from models import ...` — do NOT redirect it to
  `domains.journal.models` or attempt to create a `domains.planning`
  reference.** The `planning` domain (`weekly_plan.py` + `intent.py`) has
  not been migrated yet as of this work order, so those classes still live
  in root `models.py`, and that's exactly where this import should keep
  pointing until `planning` has its own migration. This is a deliberate,
  temporary asymmetry — not an oversight — and should be called out as an
  exception if anyone reviews the diff and wonders why one import wasn't
  touched like the others.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline (the code
   as it existed before your changes) to confirm it is not a regression you
   introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket:
   which endpoint/file, the exact error, and root cause if you found one.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue —
   explain the distinction in the one-line reason.

## WORKING METHOD
Execute steps in the order listed. After each step that changes running
behavior (not pure file moves), pause and self-verify against the relevant
acceptance criteria before continuing to the next step. Do not defer all
verification to the end.

If an acceptance criterion requires a resource not listed in SCOPE (e.g. a
config file, external service, or database not provided), do not skip the
criterion silently. Perform the closest verification achievable with what
you have, state explicitly what the substitute check was and why, and mark
the result ⚠️ rather than ✅ or leaving it blank.

## OUTPUT FORMAT
End with a report in exactly this structure:
1. **Files created** (list — include any scaffolding `__init__.py` files here)
2. **Files moved** (old path → new path)
3. **Files edited** (path — one-line description of the change; if a change
   goes beyond what a step literally asked for but was necessary to make
   that step work, say so explicitly and explain why)
4. **Acceptance criteria results** (checklist, ✅/❌/⚠️ per item, with a
   one-line reason for any ❌ or ⚠️ — including substitute-verification
   explanations per the rule above)
5. **Notes / things I noticed but did not act on** (optional — improvement
   ideas, pre-existing bugs found per the rule above, inconsistencies, risks
   spotted while working, explicitly out of scope for this task)

## ROLLBACK
This work order operates on files already tracked in git. If acceptance
criteria fail and cannot be quickly fixed, the safe rollback is `git
checkout` on every file listed in "Files created / moved / edited" above —
do not attempt a partial manual revert.

---

## SCOPE

**Models to extract from `models.py`:**
`JournalEntry`, `WeeklySynthesis`

**Router:**
- `routers/journal.py`

**Templates:**
- `templates/journal.html`
- `templates/journal_synthesis.html`
- `templates/partials/journal_no_entry.html`
- `templates/partials/journal_entry_saved.html`
- `templates/partials/journal_form.html`
- `templates/partials/journal_readonly.html`
- `templates/partials/synthesis_detail.html`

**Static:**
- `static/css/journal.css`

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:**
- `routers/dashboard.py` (imports `JournalEntry`, `WeeklySynthesis` from
  `models` for the journal summary and latest-synthesis cards — must keep
  working via shim, same pattern as every prior domain)
- `routers/weekly_plan.py` and `models.py`'s `WeeklyPlanDay` (has a plain
  `ForeignKey("journal_entries.id", ...)` column — **not** a
  `relationship()` requiring class resolution, so no cross-domain
  relationship handling is needed here, unlike the blog/code_intel case in
  WO#2). Do not edit this file or column.
- `airflow/dags/life_os_weekly_synthesis.py` (DAG — out of scope per
  standing rule, uses `dag_db.py` raw SQL only, never imports `models.py`)

---

## STEPS

1. **Create `domains/journal/models.py`.** Move `JournalEntry` and
   `WeeklySynthesis` there verbatim, **including every existing privacy
   comment and docstring** (the `# ── PRIVATE TEXT — NEVER SENT TO
   EXTERNAL AI ──` block inside `JournalEntry`, and `WeeklySynthesis`'s
   docstring about being "built exclusively from numeric scores"). Import
   `Base` from `core.base_model`.

2. **In `models.py`:** delete the two moved class bodies and replace with
   a re-export shim: `from domains.journal.models import JournalEntry,
   WeeklySynthesis`. Tag it `# TODO: remove after all cross-references are
   updated`.

3. **Move the router:** `routers/journal.py` →
   `domains/journal/routers/journal.py`. Update its `from models import
   JournalEntry, WeeklySynthesis` to `from domains.journal.models import
   JournalEntry, WeeklySynthesis`. Update its `templates =
   Jinja2Templates(directory="templates")` line to `from core.templating
   import templates`. **Leave the local `from models import WeeklyPlanDay
   as _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS` import inside
   `save_entry()` completely unchanged** — see HARD BOUNDARIES above.

4. **Move templates**, preserving the `partials/` subfolder structure, into
   `domains/journal/templates/` per the SCOPE list above.

5. **Move `static/css/journal.css`** to
   `domains/journal/static/css/journal.css`. Update the `<link
   rel="stylesheet" href="/static/css/journal.css">` references inside
   `journal.html` and `journal_synthesis.html` to
   `/static/journal/css/journal.css`.

6. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/journal/templates/` as an additional search root, alongside
   the roots already added in WO#1–5.

7. **In `main.py`:**
   - Update the router import/include to its new path (`from
     domains.journal.routers import journal`).
   - Add the new static mount: `app.mount("/static/journal",
     StaticFiles(directory="domains/journal/static"), name="journal_static")`.
     Register it **before** the general `/static` mount, per the ordering
     rule in GOVERNANCE.md §2.6.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /journal` renders identically — streak badge, 3-month calendar,
  synthesis history mini-cards, today's entry form or read-only view
  depending on lock state
- [ ] `GET /journal/{date}` renders identically for a past date, including
  the "too old to backdate" state (`journal_no_entry.html`) when applicable
- [ ] `POST /journal` (save/update today's entry) still returns
  `journal_entry_saved.html` correctly, and still correctly refuses writes
  to a locked entry (403 path)
- [ ] `PATCH /journal/{entry_id}/lock` still returns the locked-badge HTML
  fragment correctly
- [ ] `GET /journal/synthesis/history` renders `journal_synthesis.html`
  correctly for all syntheses
- [ ] `GET /journal/synthesis/latest` still returns the correct JSON shape
  for the dashboard card
- [ ] `GET /journal/synthesis/{week_start_date}` still returns
  `synthesis_detail.html` correctly
- [ ] **Privacy verification (required, not optional):** confirm by reading
  the moved `domains/journal/models.py` and
  `domains/journal/routers/journal.py` that `content`, `gratitude`, and
  `challenges` are never referenced anywhere outside this domain's own
  templates/router — i.e. confirm no accidental new code path was
  introduced that could expose them. This is a read-and-confirm check, not
  a functional test, but must be explicitly reported as done.
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  same table count before/after, `models.JournalEntry is
  domains.journal.models.JournalEntry`, no `InvalidRequestError` on mapper
  configuration
- [ ] `routers/dashboard.py`'s existing `JournalEntry` / `WeeklySynthesis`
  imports (via `from models import ...`) still resolve, and `/dashboard`'s
  journal stat card and latest-synthesis panel render correctly
- [ ] Confirm the local `from models import WeeklyPlanDay as _WPD, ...`
  import inside `save_entry()` is untouched (`git diff` on this specific
  block should show no change) and still functions — trigger a save while
  a confirmed weekly plan day exists for today, if feasible in this
  environment, to confirm the plan-day linking still works; otherwise mark
  ⚠️ with an explanation of what couldn't be tested
- [ ] `grep -r "from models import"` for `JournalEntry` and
  `WeeklySynthesis` across the repo returns only the shim's own lines in
  `models.py`, `routers/dashboard.py`'s import, and the deliberately
  unchanged local import inside `save_entry()` — nothing else should need
  updating

---

## For the next work order (not part of this one)

Per GOVERNANCE.md §3.3, **Work Order #7 = `recipes` + `pantry`** is next —
these two share ingredient/pantry data the same way `blog` + `code_intel`
shared a live FK in WO#2, so they migrate together. Note in advance:
`recipes` has real relationship complexity (`Recipe.ingredients`,
`Recipe.tags` via the `recipe_tags_junction` association table,
`Ingredient.pantry_item`) that will need the same "string-based
relationship, no direct cross-file import" treatment as WO#2, and the
association table (`recipe_tags_junction = Table(...)`, not a mapped
class) will need explicit handling since it's not a class that can simply
be "moved" the same way an ORM model is — it's a `sqlalchemy.Table` object
tied to `Base.metadata` directly.
