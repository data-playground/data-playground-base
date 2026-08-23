# domains/workout/routers/_shared.py
"""
Workout Tracker — shared router-level helpers.

Internal helper module: leading underscore means this is not a route file
and is never passed to app.include_router(). It follows the existing
in-file pattern already used elsewhere in this codebase — underscore-
prefixed helper functions living next to their callers (see
workout_settings.py's own _fetch_locations()/_location_list_ctx(), or
weekly_plan.py's _generate_shopping_list()) — just promoted one level up
because these two helpers are duplicated ACROSS router files rather than
reused within a single one.

Consolidated here as an explicitly-authorized follow-up to Work Order #8
(not part of the original migration diff):
  - _get_previous_best(): was defined identically in both workout.py and
    workout_log.py.
  - parse_weight_unit(): the `WeightUnit.KG if x == "kg" else WeightUnit.LB`
    pattern, repeated 4x across workout_log.py and workout_settings.py.
"""

from typing import Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.workout.models import WeightUnit, WorkoutSet


async def _get_previous_best(
    db: AsyncSession, exercise_id: int, exclude_session_id: Optional[int] = None
) -> Optional[WorkoutSet]:
    """
    Most recent working set for an exercise, excluding warmups and,
    optionally, the current session. Used to populate 'Previous best'
    displays on both the main workout page and the set-logging endpoint.
    """
    stmt = (
        select(WorkoutSet)
        .where(WorkoutSet.exercise_id == exercise_id)
        .where(WorkoutSet.is_warmup == False)
        .where(WorkoutSet.weight_used != None)
        .order_by(desc(WorkoutSet.created_at))
    )
    if exclude_session_id:
        stmt = stmt.where(WorkoutSet.session_id != exclude_session_id)
    result = await db.execute(stmt.limit(1))
    return result.scalar_one_or_none()


def parse_weight_unit(raw: str) -> WeightUnit:
    """
    Parses an already-normalized (stripped) form-submitted weight-unit
    string into the WeightUnit enum. Matches the exact behavior of the 4
    call sites this replaces: anything other than the literal string
    "kg" defaults to LB.
    """
    return WeightUnit.KG if raw == "kg" else WeightUnit.LB
