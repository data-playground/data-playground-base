# Work Order #4 — Domain Migration: `explorer`

*This is the smallest of the four priority domains — one router, one
template, one CSS file, no partials, no DAGs, and (unlike every prior
domain) no ORM classes to move at all. `explorer.py` reads the database
schema dynamically via raw `information_schema` queries rather than
importing any SQLAlchemy model — so there is no `models.py` extraction step,
no shim, and no cross-domain relationship to worry about. This should be the
fastest of the domain moves so far.*

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
- `explorer.py` intentionally has no dependency on `models.py` — do not add
  one. Its `HIDDEN_TABLES` set and `_infer_column_type()` helper are
  read-only introspection logic; leave them exactly as they are.
- Do NOT change the `BLOCKED_PATTERN` regex, the `ROW_CAP` constant, or any
  other part of the read-only-query enforcement logic. This file is a
  security boundary (blocks INSERT/UPDATE/DELETE/DROP/etc. at the
  application layer) — even a whitespace-level accidental change here needs
  to be treated as high-risk and called out explicitly if it happens for any
  reason during the move.

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

**Models to extract from `models.py`:** none. This domain has no ORM or
Pydantic classes — skip any model-extraction step entirely.

**Router:**
- `routers/explorer.py`

**Template:**
- `templates/explorer.html`

**Static:**
- `static/css/explorer.css`

**Core/config files to edit:**
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:**
- None. No other file in the repo imports from `routers/explorer.py` or
  references its module path — this domain has no external consumers to
  verify against, unlike every prior work order.

---

## STEPS

1. **Move the router:** `routers/explorer.py` → `domains/explorer/routers/explorer.py`.
   This file only imports `from database import get_db` (unchanged path) and
   standard library / SQLAlchemy — there are no `models` imports to update.
   Update its `templates = Jinja2Templates(directory="templates")` line to
   `from core.templating import templates`, matching the pattern from
   WO#1/WO#2/WO#3.

2. **Move the template:** `templates/explorer.html` →
   `domains/explorer/templates/explorer.html`. This template extends
   `base.html` and has no `partials/` includes — it's a single self-contained
   file.

3. **Move the static asset:** `static/css/explorer.css` →
   `domains/explorer/static/css/explorer.css`. Update the reference inside
   `explorer.html`:
   - `<link rel="stylesheet" href="/static/css/explorer.css">` →
     `/static/explorer/css/explorer.css`

4. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/explorer/templates/` as an additional search root, alongside the
   roots already added in WO#1 (habits), WO#2 (blog, code_intel), and WO#3
   (jobs).

5. **In `main.py`:**
   - Update the router import/include to its new path (`from
     domains.explorer.routers import explorer`).
   - Add the new static mount: `app.mount("/static/explorer",
     StaticFiles(directory="domains/explorer/static"), name="explorer_static")`.
     Register it **before** the general `/static` mount, per the ordering
     lesson from WO#1 (Starlette matches `Mount` routes in registration
     order; the general mount would otherwise silently swallow the more
     specific one).

---

## ACCEPTANCE CRITERIA

- [ ] `GET /explorer` renders identically to before the move — table
  browser panel, editor pane, results pane, quick-query chip bar all present
- [ ] `GET /explorer/schema` still returns the same JSON shape (per-table
  columns + row counts), with `HIDDEN_TABLES` still correctly excluded from
  the response
- [ ] `POST /explorer/query` with a valid `SELECT` still executes and
  returns `{rows, columns, row_count, capped}` correctly
- [ ] `POST /explorer/query` with a blocked keyword (e.g. `INSERT`,
  `DROP`) still returns HTTP 400 with the correct "Write operations are not
  permitted" message — **this is the most important check in this work
  order**, since it confirms the security-relevant regex/validation logic
  moved without alteration
- [ ] `POST /explorer/query` without a `LIMIT` clause still gets wrapped in
  the `ROW_CAP`-enforcing subquery and returns `capped: true`
- [ ] `git diff` on the moved `explorer.py` shows **only** the two import
  changes (Step 1) and no other line differs — paste or describe the diff
  in your report to make this easy to confirm at a glance
- [ ] No other file in the repo needed updating as a result of this move
  (confirmed via the "Not in scope" note above — if you find a reference to
  `routers.explorer` or `routers/explorer.py` anywhere else, report it, since
  it would contradict this domain's assumed isolation)

---

## For the next batch (not part of this one)

With `habits` (pilot), `blog` + `code_intel`, `jobs`, and `explorer` done,
the four priority domains from the roadmap are complete. Remaining domains
(`finance`, `journal`, `recipes`, `workout`, `media`, `planning`) can move in
whatever order is convenient — none of them are on the priority list, and
per your note they've been tested but aren't in heavy use yet. Suggested
grouping to keep future work orders reasonably sized:
- `finance` (4 routers: summary, ledger, upload, settings — similar shape to
  `jobs`)
- `journal` (1 router, but paired conceptually with the
  `life_os_weekly_synthesis` DAG — DAG stays out of scope per the same rule
  as jobs/blog/code_intel)
- `recipes` + `pantry` (routers: recipe_extract, recipe_discovery, pantry,
  recipes — these already share ingredient/pantry models, similar to the
  blog/code_intel pairing)
- `workout` (4 routers: workout, workout_log, workout_plans, workout_settings)
- `media` (4 routers: media, media_search, media_recommend, media_settings)
- `planning` (weekly_plan.py + intent.py — these share `UserIntent` context
  and should move together, plus note `weekly_plan.py` is already past the
  300-line rule and may need splitting as part of that move, not just
  relocation)

`dashboard` stays last, since it's the cross-domain reader that touches
every other domain's models — it should only move (if it ever does) once
everything it reads from is already settled in its final location.
