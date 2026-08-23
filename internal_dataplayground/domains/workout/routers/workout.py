# domains/workout/routers/workout.py
"""
Workout Tracker — Main Views

Endpoints:
  GET /workout                     → Main page (active session, today's plan, recent activity)
  GET /workout/history/{exercise_id} → Last 10 sets for an exercise (HTMX partial)
  GET /workout/progress            → Progress charts page
  GET /workout/history             → Full session history page
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, desc, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.workout.models import (
    WorkoutPlan, WorkoutPlanDay, WorkoutPlanExercise,
    WorkoutSession, WorkoutSet, Exercise, BodyMetric, WeightUnit,
)
from domains.workout.routers._shared import _get_previous_best

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout", tags=["Workout"])


async def _get_active_plan(db: AsyncSession) -> Optional[WorkoutPlan]:
    result = await db.execute(
        select(WorkoutPlan).where(WorkoutPlan.is_active == True).limit(1)
    )
    return result.scalar_one_or_none()


async def _get_active_session(db: AsyncSession) -> Optional[WorkoutSession]:
    """Returns today's open session if one exists."""
    today = datetime.date.today()
    result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.session_date == today)
        .where(WorkoutSession.ended_at == None)
        .order_by(desc(WorkoutSession.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()

# NOTE: _get_previous_best() used to be defined locally in this file (and,
# identically, in workout_log.py). Consolidated into
# domains/workout/routers/_shared.py as an explicitly-authorized follow-up
# to Work Order #8 — see that module's docstring.


@router.get("", response_class=HTMLResponse)
async def workout_home(request: Request, db: AsyncSession = Depends(get_db)):
    today = datetime.date.today()
    active_plan = await _get_active_plan(db)
    active_session = await _get_active_session(db)

    # ── Today's suggested plan day ─────────────────────────────────────────
    suggested_day: Optional[WorkoutPlanDay] = None
    plan_exercises_with_prev: list[dict] = []

    if active_plan and active_plan.days:
        sessions_count_result = await db.execute(
            select(func.count(WorkoutSession.id))
            .where(WorkoutSession.plan_id == active_plan.id)
            .where(WorkoutSession.ended_at != None)
        )
        completed_sessions = sessions_count_result.scalar() or 0
        day_idx = completed_sessions % len(active_plan.days)
        suggested_day = active_plan.days[day_idx]

        if active_session and active_session.plan_day_id:
            for d in active_plan.days:
                if d.id == active_session.plan_day_id:
                    suggested_day = d
                    break

        for plan_ex in suggested_day.exercises:
            prev = await _get_previous_best(
                db,
                plan_ex.exercise_id,
                exclude_session_id=active_session.id if active_session else None,
            )
            sets_logged = 0
            if active_session:
                sets_result = await db.execute(
                    select(func.count(WorkoutSet.id))
                    .where(WorkoutSet.session_id == active_session.id)
                    .where(WorkoutSet.exercise_id == plan_ex.exercise_id)
                    .where(WorkoutSet.is_warmup == False)
                )
                sets_logged = sets_result.scalar() or 0

            plan_exercises_with_prev.append({
                "plan_exercise": plan_ex,
                "exercise": plan_ex.exercise,
                "prev_best": prev,
                "sets_logged": sets_logged,
            })

    # ── Recent sessions ────────────────────────────────────────────────────
    recent_result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.ended_at != None)
        .order_by(desc(WorkoutSession.session_date), desc(WorkoutSession.created_at))
        .limit(5)
    )
    recent_sessions = recent_result.scalars().all()

    # ── Body metrics sparkline ─────────────────────────────────────────────
    cutoff = today - datetime.timedelta(days=30)
    metrics_result = await db.execute(
        select(BodyMetric)
        .where(BodyMetric.metric_date >= cutoff)
        .order_by(BodyMetric.metric_date)
    )
    body_metrics = metrics_result.scalars().all()

    latest_metric: Optional[BodyMetric] = body_metrics[-1] if body_metrics else None

    # ── All exercises for the quick-add search ─────────────────────────────
    exercises_result = await db.execute(
        select(Exercise)
        .where(Exercise.is_custom == False)
        .order_by(Exercise.primary_muscle_group, Exercise.name)
    )
    all_exercises = exercises_result.scalars().all()

    custom_result = await db.execute(
        select(Exercise).where(Exercise.is_custom == True).order_by(Exercise.name)
    )
    custom_exercises = custom_result.scalars().all()

    return templates.TemplateResponse("workout.html", {
        "request": request,
        "active_module": "workout",
        "active_plan": active_plan,
        "active_session": active_session,
        "suggested_day": suggested_day,
        "plan_exercises": plan_exercises_with_prev,
        "recent_sessions": recent_sessions,
        "body_metrics": body_metrics,
        "latest_metric": latest_metric,
        "today": today,
        "all_exercises": all_exercises,
        "custom_exercises": custom_exercises,
    })


@router.get("/history/{exercise_id}", response_class=HTMLResponse)
async def exercise_history(
    exercise_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the last 10 working sets for an exercise as an HTMX partial.
    Shown in the quick-log panel as a reference for progressive overload.
    """
    exercise = await db.get(Exercise, exercise_id)
    if not exercise:
        return HTMLResponse("Exercise not found", status_code=404)

    result = await db.execute(
        select(WorkoutSet, WorkoutSession.session_date)
        .join(WorkoutSession, WorkoutSet.session_id == WorkoutSession.id)
        .where(WorkoutSet.exercise_id == exercise_id)
        .where(WorkoutSet.is_warmup == False)
        .where(WorkoutSession.ended_at != None)
        .order_by(desc(WorkoutSession.session_date), WorkoutSet.set_number)
        .limit(10)
    )
    rows = result.all()

    return templates.TemplateResponse("partials/workout/exercise_history.html", {
        "request": request,
        "exercise": exercise,
        "rows": [{"set": r[0], "date": r[1]} for r in rows],
    })


@router.get("/progress", response_class=HTMLResponse)
async def workout_progress(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Progress page: weight-over-time chart for each exercise logged ≥3 times.
    """
    freq_result = await db.execute(
        select(
            WorkoutSet.exercise_id,
            func.count(WorkoutSet.id).label("set_count"),
            func.max(WorkoutSet.created_at).label("last_logged"),
        )
        .where(WorkoutSet.is_warmup == False)
        .where(WorkoutSet.weight_used != None)
        .group_by(WorkoutSet.exercise_id)
        .having(func.count(WorkoutSet.id) >= 3)
        .order_by(desc("last_logged"))
    )
    freq_rows = freq_result.all()

    progress_data = []
    for row in freq_rows:
        exercise = await db.get(Exercise, row.exercise_id)
        if not exercise:
            continue

        prog_result = await db.execute(
            select(
                WorkoutSession.session_date,
                func.max(WorkoutSet.weight_used).label("max_weight"),
                WorkoutSet.weight_unit,
            )
            .join(WorkoutSession, WorkoutSet.session_id == WorkoutSession.id)
            .where(WorkoutSet.exercise_id == row.exercise_id)
            .where(WorkoutSet.is_warmup == False)
            .where(WorkoutSet.weight_used != None)
            .where(WorkoutSession.ended_at != None)
            .group_by(WorkoutSession.session_date, WorkoutSet.weight_unit)
            .order_by(WorkoutSession.session_date)
        )
        prog_rows = prog_result.all()

        progress_data.append({
            "exercise": exercise,
            "data": [
                {
                    "date": str(r.session_date),
                    "weight": float(r.max_weight),
                    "unit": r.weight_unit,
                }
                for r in prog_rows
            ],
        })

    cutoff = datetime.date.today() - datetime.timedelta(days=90)
    metrics_result = await db.execute(
        select(BodyMetric)
        .where(BodyMetric.metric_date >= cutoff)
        .order_by(BodyMetric.metric_date)
    )
    body_metrics = metrics_result.scalars().all()

    return templates.TemplateResponse("workout_progress.html", {
        "request": request,
        "active_module": "workout",
        "progress_data": progress_data,
        "body_metrics": body_metrics,
    })


@router.get("/history", response_class=HTMLResponse)
async def session_history(
    request: Request,
    db: AsyncSession = Depends(get_db),
    page: int = 1,
):
    """Full session history — paginated, 20 sessions per page."""
    per_page = 20
    offset = (page - 1) * per_page

    total_result = await db.execute(
        select(func.count(WorkoutSession.id)).where(WorkoutSession.ended_at != None)
    )
    total = total_result.scalar() or 0

    result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.ended_at != None)
        .order_by(desc(WorkoutSession.session_date), desc(WorkoutSession.created_at))
        .offset(offset)
        .limit(per_page)
    )
    sessions = result.scalars().all()

    return templates.TemplateResponse("workout_history.html", {
        "request": request,
        "active_module": "workout",
        "sessions": sessions,
        "page": page,
        "total": total,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })
