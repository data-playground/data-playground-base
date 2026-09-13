# Work Order #30 — Planning Domain Router Verification (`domains/planning/routers/weekly_plan.py`)

*Unlike the other seven Track C work orders, direct reading of the current
`weekly_plan.py` suggests this one may already be resolved — this work
order is written as verification-first with a contingency split, not a
guaranteed split.*

---

## ROLE
You are verifying whether a previously-flagged oversized file has already
been resolved by unrelated prior work, and doing a small contingency split
only if it hasn't.

## HARD BOUNDARIES
- Do not split anything until Step 1 confirms it's actually still needed.
- If a split is needed: no endpoint's URL path, HTTP method, request/
  response shape, or template name changes.
- Only touch `domains/planning/routers/weekly_plan.py`, a new file under
  `domains/planning/routers/` if the contingency split applies, and
  `00_MASTER_INDEX.md` (correcting the stale line-count figure — see
  ACCEPTANCE CRITERIA).

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.

## WORKING METHOD
This work order is verification-first by design — do not skip straight to
splitting because the master index says this file is oversized.

## OUTPUT FORMAT
1. Files created (if any)
2. Files moved (if any)
3. Files edited (path — description)
4. Acceptance criteria results
5. Notes

## ROLLBACK
`git checkout` on any file touched.

---

## Background

`00_MASTER_INDEX.md` carries a "413 lines" figure for `weekly_plan.py` from
WO#19's lint run. WO#10's own authorized follow-up split the original
monolithic file into three (`weekly_plan.py` — day/meal CRUD and plan
lifecycle; `weekly_plan_generator.py` — the one AI-calling handler;
`weekly_plan_shopping.py` — shopping-list generation/view). The "413 lines"
figure almost certainly describes the pre-split file, not the current
`weekly_plan.py` (which now contains only `_get_monday`,
`_sync_plan_status`, `plan_hub`, `plan_new_form`, `confirm_plan`,
`plan_view`, `override_day`, `update_meal_status`) — reading the current
content directly suggests it's meaningfully shorter than 300 lines already.

## SCOPE
- `domains/planning/routers/weekly_plan.py` (verification, contingency split)
- New (only if contingency applies): `domains/planning/routers/weekly_plan_confirm.py`
- `00_MASTER_INDEX.md` (correction only, if the stale figure is confirmed
  wrong)

## STEPS

1. Run `scripts/check_router_line_limits.py` (or equivalent) against
   `domains/planning/routers/weekly_plan.py` specifically. This is the
   only step required if the file is already under 300 lines.

2. **If it reports under 300 lines:** close this work order with "not
   applicable — already resolved by WO#10's follow-up split." Correct the
   master index's tracked figure for this file in your report (and, if
   convenient, directly in `00_MASTER_INDEX.md`).

3. **If it's still over 300** (contingency, in case the earlier split
   wasn't as complete as a read-through suggests): split `confirm_plan`
   (the longest single handler — builds day/meal/workout-session rows in
   a 7-day loop) out into its own `weekly_plan_confirm.py`, keeping
   `plan_hub`, `plan_new_form`, `plan_view`, `override_day`,
   `update_meal_status`, and `_sync_plan_status` in `weekly_plan.py`.

## ACCEPTANCE CRITERIA
- [ ] Real current line count reported and reconciled against the master
  index's stale "413" figure — state explicitly whether that figure was
  confirmed stale.
- [ ] If already resolved: no code changes made; master index correction
  noted (and applied, if you chose to make it directly).
- [ ] If the contingency split was needed: `GET /plan`, `GET /plan/new`,
  `POST /plan/confirm`, `GET /plan/{id}`, `PATCH /plan/{id}/day/{date}`,
  `PATCH /plan/meal/{meal_id}` all unchanged; `weekly_plan_generator.py`'s
  and `weekly_plan_shopping.py`'s existing route registrations in
  `main.py` are unaffected by this further split.

## For the next work order (not part of this one)
This is the last of the eight Track C work orders. Once all eight report
back, the 14-file line-limit backlog from WO#19 is fully closed (or
explicitly, individually accounted for where a file turned out not to need
work).
