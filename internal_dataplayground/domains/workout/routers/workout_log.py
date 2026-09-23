# domains/workout/routers/workout_log.py
"""
Workout Tracker — Session Logging

WO#29 Part C (sign-off obtained): this module used to also export
`body_metrics_router` (prefix /workout/body-metrics). WO#8's own HARD
BOUNDARIES said not to merge or further split that two-router-in-one-file
structure during that migration — this split explicitly reverses that
precedent, with the project owner's sign-off, because the file had grown
to 357 lines. `body_metrics_router` (log_body_metric, get_body_metrics)
now lives in workout_body_metrics.py. Both routers are still registered
as separate app.include_router() calls in main.py, exactly as before.

Endpoints:
  POST   /workout/sessions/start               → Start a new session
  POST   /workout/sessions/{id}/sets           → Log a single set (fast path)
  PATCH  /workout/sessions/{id}/end            → End a session
  GET    /workout/sessions/{id}                → Full session detail
  DELETE /workout/sessions/{id}/sets/{set_id} → Delete an incorrectly logged set
"""

import datetime
import logging
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.workout.models import Exercise, WorkoutPlan, WorkoutSession, WorkoutSet
from domains.workout.routers._shared import _get_previous_best, parse_weight_unit

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout/sessions", tags=["Workout"])

# NOTE: _get_previous_best() used to be defined locally in this file (and,
# identically, in workout.py). Consolidated into
# domains/workout/routers/_shared.py as an explicitly-authorized follow-up
# to Work Order #8 — see that module's docstring.


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

    existing_result = await db.execute(
        select(WorkoutSession)
        .where(WorkoutSession.session_date == today)
        .where(WorkoutSession.ended_at == None)
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        return templates.TemplateResponse("partials/workout/active_session_header.html", {
            "request": request,
            "session": existing,
            "toast": "Session already in progress.",
        })

    plan_id = int(form.get("plan_id")) if form.get("plan_id") else None
    plan_day_id = int(form.get("plan_day_id")) if form.get("plan_day_id") else None
    location_id = int(form.get("location_id")) if form.get("location_id") else None
    weight_unit_raw = str(form.get("weight_unit", "lb")).strip()
    weight_unit = parse_weight_unit(weight_unit_raw)

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
    weight_unit = parse_weight_unit(weight_unit_raw)

    rpe_raw = form.get("rpe", "").strip()
    rpe = int(rpe_raw) if rpe_raw else None
    if rpe is not None and not (1 <= rpe <= 10):
        rpe = None

    is_warmup = form.get("is_warmup", "").lower() in ("true", "1", "on", "yes")

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
