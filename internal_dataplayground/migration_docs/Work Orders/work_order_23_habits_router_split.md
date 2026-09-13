# Work Order #23 — Habits Domain Router Split (`domains/habits/routers/habits.py`)

*Per WO#19's lint script, this is the largest router in the codebase and was
never previously flagged by any prior work order in this program.*

---

## ROLE
You are a senior refactoring engineer splitting an oversized router file by
responsibility. This is a location-and-organization refactor only — no
behavior change, no renamed endpoints, no schema change.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- Only touch `domains/habits/routers/habits.py`, files you create under
  `domains/habits/routers/`, and the registration lines in `main.py`.
- If your own read of the file's current line count materially disagrees
  with what's expected (i.e. it's already comfortably under 300), stop and
  report rather than manufacturing a split that isn't needed.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
Don't fix, reproduce against baseline to confirm it's not new, report under
Notes, mark the related criterion ⚠️ not ❌.

## WORKING METHOD
Run `scripts/check_router_line_limits.py` first and record the real current
line count before doing anything else. Only proceed with the split if it
confirms the file is still over 300 lines.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results (✅/❌/⚠️ + reason for non-✅)
5. Notes

## ROLLBACK
`git checkout` on every file touched.

---

## SCOPE
- `domains/habits/routers/habits.py` (split)
- New: `domains/habits/routers/_shared.py`
- New: `domains/habits/routers/habits_settings.py`
- `main.py`, `core/templating.py` (only if a new router needs template
  access — it will)

## STEPS

1. Run the lint script; confirm the real current line count. If already
   under 300, stop and report — no split needed.

2. Move the pure-calculation helpers that have no route decorator —
   `_get_grace_period`, `_get_logged_dates_for_habit`, `_calculate_streak`,
   `_build_habit_view`, `_get_today_logged_ids`, `_sort_habits_for_display`,
   `_get_week_dates` — into a new `domains/habits/routers/_shared.py`. This
   mirrors the precedent already established in
   `domains/workout/routers/_shared.py` for the same kind of
   duplicated/reusable helper consolidation.

3. Split the remaining route handlers into two files by concern:
   - `habits.py` (keep): the daily check-in surface — `habits_page` (GET
     `/habits`), `log_habit` (POST `/habits/log`), `unlog_habit` (DELETE
     `/habits/log`), `habit_progress` (GET `/habits/progress`),
     `habit_heatmap` (GET `/habits/heatmap/{habit_id}`). This is the
     hot-path file someone opens most often — keep it lean.
   - `habits_settings.py` (new): `habits_settings` (GET
     `/habits/settings`), `create_habit`, `reorder_habits`, `update_habit`,
     `deactivate_habit`, `update_grace_period`. Same `/habits` prefix — this
     is safe since none of these paths collide with the ones staying in
     `habits.py` (list every route from both files in your report to make
     this explicit).

4. Both files import the shared helpers from `_shared.py` rather than
   redefining them.

5. Register the new `habits_settings` router in `main.py` alongside the
   existing `habits.router` include.

## ACCEPTANCE CRITERIA
- [ ] Both `habits.py` and `habits_settings.py` are under 300 lines.
- [ ] Every route enumerated from both files, confirmed zero path collisions.
- [ ] `GET /habits`, `POST /habits/log`, `DELETE /habits/log`, `GET
  /habits/progress`, `GET /habits/heatmap/{id}` still work identically.
- [ ] `GET /habits/settings`, habit create/update/deactivate/reorder, and
  grace-period update still work identically.
- [ ] `Base.metadata` / mapper identity unaffected (this work order touches
  no models).

## For the next work order (not part of this one)
This can run fully in parallel with every other Track C work order, Track A,
Track B, and Track D — no shared files with any of them.
