# Work Order #25 — Journal Domain Router Split (`domains/journal/routers/journal.py`)

---

## ROLE
You are a senior refactoring engineer splitting an oversized router file by
responsibility. Location-and-organization refactor only — no behavior
change, no schema change.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- **PRIVACY BOUNDARY — treat as seriously as a security constraint.** This
  file (and its models) documents that `content`, `gratitude`, and
  `challenges` are never sent to any external AI call — only numeric scores
  feed the weekly synthesis DAG. Every comment enforcing this must move
  verbatim to wherever the relevant code ends up — do not paraphrase,
  shorten, or drop any of them while splitting.
- Do **not** touch `save_entry()`'s local `from domains.planning.models
  import WeeklyPlanDay as _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS`
  import — this is a deliberate, already-reviewed cross-domain reference
  (see WO#6 and WO#10's own HARD BOUNDARIES) and is unrelated to line count.
- Only touch `domains/journal/routers/journal.py`, files you create under
  `domains/journal/routers/`, and `main.py` if a new router needs
  registration (it won't — all three resulting files share the existing
  `/journal` prefix under one router object, unless you choose to register
  new sub-routers, in which case update `main.py` accordingly and say so).

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.

## WORKING METHOD
Run `scripts/check_router_line_limits.py` first; confirm the real current
line count before proceeding.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results
5. Notes

## ROLLBACK
`git checkout` on every file touched.

---

## SCOPE
- `domains/journal/routers/journal.py` (split)
- New: `domains/journal/routers/_calendar.py`
- New: `domains/journal/routers/journal_synthesis.py`

## STEPS

1. Run the lint script; confirm the real current line count.

2. Move the calendar-building helpers (`_build_calendar_months`,
   `_mood_class`, `_calculate_streak`, `_get_calendar_dates`,
   `_get_calendar_data`) into `_calendar.py`. `_build_calendar_months` in
   particular is large on its own (three-month grid construction) and has
   zero dependency on the synthesis or entry-CRUD logic.

3. Move the synthesis-only endpoints (`synthesis_history`,
   `latest_synthesis_json`, `synthesis_detail`) into
   `journal_synthesis.py` — these already render a separate page
   (`journal_synthesis.html`) and partial (`synthesis_detail.html`), so
   this is a clean, already-implied boundary.

4. Keep `journal_home`, `journal_date`, `save_entry`, `lock_entry` in
   `journal.py` — these are the actual daily-journaling surface.

5. Carry forward every privacy comment verbatim (see HARD BOUNDARIES).

## ACCEPTANCE CRITERIA
- [ ] All three resulting files under 300 lines.
- [ ] `GET /journal`, `GET /journal/{date}`, `POST /journal`, `PATCH
  /journal/{id}/lock` unchanged.
- [ ] `GET /journal/synthesis/history`, `GET /journal/synthesis/latest`,
  `GET /journal/synthesis/{week_start}` unchanged.
- [ ] The privacy-boundary comments on `content`/`gratitude`/`challenges`
  are present verbatim wherever they ended up — confirm and quote them in
  your report.
- [ ] The `save_entry()` → `domains.planning.models` local import still
  resolves and functions (trigger a save while a plan day exists, if
  feasible; otherwise mark ⚠️ and say what couldn't be tested).

## For the next work order (not part of this one)
Can run fully in parallel with every other Track C work order, Track A,
Track B, and Track D.
