# routers/workout_log.py
"""
Workout Tracker — Session Logging

Endpoints:
  POST   /workout/sessions/start               → Start a new session
  POST   /workout/sessions/{id}/sets           → Log a single set (fast path)
  PATCH  /workout/sessions/{id}/end            → End a session
  GET    /workout/sessions/{id}                → Full session detail
  DELETE /workout/sessions/{id}/sets/{set_id} → Delete an incorrectly logged set
  POST   /workout/body-metrics                 → Log body weight (upsert)
  GET    /workout/body-metrics                 → Last 90 days of body metrics (JSON)
"""

import datetime
import logging
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import (
    BodyMetric, Exercise, WeightUnit,
    WorkoutPlan, WorkoutPlanDay, WorkoutSession, WorkoutSet,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout/sessions", tags=["Workout"])
templates = Jinja2Templates(directory="templates")


async def _get_previous_best(
    db: AsyncSession, exercise_id: int, exclude_session_id: Optional[int] = None
) -> Optional[WorkoutSet]:
    """Most recent working set for an exercise, excluding the current session."""
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


@router.post("/start", response_class=HTMLResponse)
async def start_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a new workout session.
    Accepts optional form fields: plan_id, plan_day_id, location_id, weight_unit.
    Returns the active session header partial for HTMX swap.
    """
    form = await request.form()
    today = datetime.date.today()

    # Check if an open session already exists for today
    existing_result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.session_date == today)
        .where(WorkoutSession.ended_at == None)
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        # Return the existing session rather than creating a duplicate
        return templates.TemplateResponse("partials/workout/active_session_header.html", {
            "request": request,
            "session": existing,
            "toast": "Session already in progress.",
        })

    plan_id = int(form.get("plan_id")) if form.get("plan_id") else None
    plan_day_id = int(form.get("plan_day_id")) if form.get("plan_day_id") else None
    location_id = int(form.get("location_id")) if form.get("location_id") else None
    weight_unit_raw = str(form.get("weight_unit", "lb")).strip()
    weight_unit = WeightUnit.KG if weight_unit_raw == "kg" else WeightUnit.LB

    # Default to active plan's location if not specified
    if not location_id and plan_id:
        plan = await db.get(WorkoutPlan, plan_id)
        if plan and plan.location_id:
            location_id = plan.location_id

    session = WorkoutSession(
        plan_id=plan_id,
        plan_day_id=plan_day_id,
        location_id=location_id,
        session_date=today,
        started_at=datetime.datetime.utcnow(),
        weight_unit=weight_unit,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return templates.TemplateResponse("partials/workout/active_session_header.html", {
        "request": request,
        "session": session,
        "toast": "Session started. Let's go.",
    })


@router.post("/{session_id}/sets", response_class=HTMLResponse)
async def log_set(
    session_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Logs a single set during an active session.
    This is the hot path — called on every set tap. Must be fast.
    Returns the set_logged_row partial with 'previous best' comparison.
    """
    form = await request.form()

    session = await db.get(WorkoutSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at:
        raise HTTPException(status_code=400, detail="Session already ended")

    exercise_id_raw = form.get("exercise_id", "")
    if not exercise_id_raw:
        raise HTTPException(status_code=422, detail="exercise_id is required")
    exercise_id = int(exercise_id_raw)

    reps_raw = form.get("reps_completed", "0")
    reps = int(reps_raw) if reps_raw else 0
    if reps < 1:
        raise HTTPException(status_code=422, detail="reps_completed must be ≥ 1")

    weight_raw = form.get("weight_used", "").strip()
    weight_used = Decimal(weight_raw) if weight_raw else None

    weight_unit_raw = str(form.get("weight_unit", session.weight_unit.value)).strip()
    weight_unit = WeightUnit.KG if weight_unit_raw == "kg" else WeightUnit.LB

    rpe_raw = form.get("rpe", "").strip()
    rpe = int(rpe_raw) if rpe_raw else None
    if rpe is not None and not (1 <= rpe <= 10):
        rpe = None

    is_warmup = form.get("is_warmup", "").lower() in ("true", "1", "on", "yes")

    # Auto-increment set_number for this exercise in this session
    set_count_result = await db.execute(
        select(func.count(WorkoutSet.id))
        .where(WorkoutSet.session_id == session_id)
        .where(WorkoutSet.exercise_id == exercise_id)
    )
    set_number = (set_count_result.scalar() or 0) + 1

    workout_set = WorkoutSet(
        session_id=session_id,
        exercise_id=exercise_id,
        set_number=set_number,
        reps_completed=reps,
        weight_used=weight_used,
        weight_unit=weight_unit,
        rpe=rpe,
        is_warmup=is_warmup,
    )
    db.add(workout_set)
    await db.commit()
    await db.refresh(workout_set)

    # Previous best for the motivational comparison
    prev_best = await _get_previous_best(db, exercise_id, exclude_session_id=session_id)
    exercise = await db.get(Exercise, exercise_id)

    return templates.TemplateResponse("partials/workout/set_logged_row.html", {
        "request": request,
        "workout_set": workout_set,
        "exercise": exercise,
        "prev_best": prev_best,
        "session": session,
    })


@router.patch("/{session_id}/end", response_class=HTMLResponse)
async def end_session(
    session_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Closes the session. Calculates duration, accepts fatigue_rating and notes.
    Returns the session_summary partial.
    """
    form = await request.form()
    session = await db.get(WorkoutSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at:
        raise HTTPException(status_code=400, detail="Session already ended")

    now = datetime.datetime.utcnow()
    session.ended_at = now

    # Calculate duration from started_at if available
    if session.started_at:
        session.duration_minutes = int((now - session.started_at).total_seconds() / 60)

    fatigue_raw = form.get("fatigue_rating", "").strip()
    if fatigue_raw and fatigue_raw.isdigit():
        rating = int(fatigue_raw)
        session.fatigue_rating = rating if 1 <= rating <= 5 else None

    notes_raw = str(form.get("notes", "")).strip()
    if notes_raw:
        session.notes = notes_raw

    await db.commit()
    await db.refresh(session)

    return templates.TemplateResponse("partials/workout/session_summary.html", {
        "request": request,
        "session": session,
    })


@router.get("/{session_id}", response_class=HTMLResponse)
async def session_detail(
    session_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Full session detail — used in the history page and post-session review.
    Groups sets by exercise for display.
    """
    session = await db.get(WorkoutSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Group sets by exercise for template rendering
    sets_by_exercise: dict[int, dict] = {}
    for ws in session.sets:
        if ws.exercise_id not in sets_by_exercise:
            sets_by_exercise[ws.exercise_id] = {
                "exercise": ws.exercise,
                "sets": [],
            }
        sets_by_exercise[ws.exercise_id]["sets"].append(ws)

    exercise_groups = list(sets_by_exercise.values())

    return templates.TemplateResponse("partials/workout/session_detail.html", {
        "request": request,
        "session": session,
        "exercise_groups": exercise_groups,
    })


@router.delete("/{session_id}/sets/{set_id}", response_class=HTMLResponse)
async def delete_set(
    session_id: int,
    set_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Removes an incorrectly logged set.
    Returns an empty 200 so HTMX removes the row with hx-swap="outerHTML".
    """
    ws = await db.get(WorkoutSet, set_id)
    if not ws or ws.session_id != session_id:
        raise HTTPException(status_code=404, detail="Set not found")

    await db.delete(ws)
    await db.commit()
    return HTMLResponse("")


# ── Body metrics ───────────────────────────────────────────────────────────────

body_metrics_router = APIRouter(prefix="/workout/body-metrics", tags=["Workout"])


@body_metrics_router.post("", response_class=HTMLResponse)
async def log_body_metric(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Upserts a body metric entry for a given date (defaults to today).
    Returns the updated sparkline data as JSON for the chart.
    """
    form = await request.form()
    date_raw = str(form.get("metric_date", "")).strip()
    try:
        metric_date = datetime.date.fromisoformat(date_raw) if date_raw else datetime.date.today()
    except ValueError:
        metric_date = datetime.date.today()

    weight_raw = str(form.get("weight", "")).strip()
    weight = Decimal(weight_raw) if weight_raw else None

    weight_unit_raw = str(form.get("weight_unit", "lb")).strip()
    weight_unit = WeightUnit.KG if weight_unit_raw == "kg" else WeightUnit.LB

    bf_raw = str(form.get("body_fat_pct", "")).strip()
    body_fat_pct = Decimal(bf_raw) if bf_raw else None

    notes_raw = str(form.get("notes", "")).strip()

    # Upsert — one entry per day
    existing_result = await db.execute(
        select(BodyMetric).where(BodyMetric.metric_date == metric_date)
    )
    existing = existing_result.scalar_one_or_none()

    if existing:
        if weight is not None:
            existing.weight = weight
            existing.weight_unit = weight_unit
        if body_fat_pct is not None:
            existing.body_fat_pct = body_fat_pct
        if notes_raw:
            existing.notes = notes_raw
        metric = existing
    else:
        metric = BodyMetric(
            metric_date=metric_date,
            weight=weight,
            weight_unit=weight_unit,
            body_fat_pct=body_fat_pct,
            notes=notes_raw or None,
        )
        db.add(metric)

    await db.commit()
    await db.refresh(metric)

    return templates.TemplateResponse("partials/workout/body_metric_saved.html", {
        "request": request,
        "metric": metric,
    })


@body_metrics_router.get("", response_class=JSONResponse)
async def get_body_metrics(db: AsyncSession = Depends(get_db)):
    """Returns last 90 days of body metrics as JSON for chart rendering."""
    cutoff = datetime.date.today() - datetime.timedelta(days=90)
    result = await db.execute(
        select(BodyMetric)
        .where(BodyMetric.metric_date >= cutoff)
        .order_by(BodyMetric.metric_date)
    )
    metrics = result.scalars().all()
    return [
        {
            "date": str(m.metric_date),
            "weight": float(m.weight) if m.weight else None,
            "unit": m.weight_unit.value,
            "body_fat_pct": float(m.body_fat_pct) if m.body_fat_pct else None,
        }
        for m in metrics
    ]
