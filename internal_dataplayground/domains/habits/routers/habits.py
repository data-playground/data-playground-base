# routers/habits.py
"""
Habit Tracker Module — daily check-in surface

Endpoints (this file):
  GET    /habits                         → Main check-in page
  POST   /habits/log                     → Toggle habit on (idempotent)
  DELETE /habits/log                     → Toggle habit off
  GET    /habits/progress                → Progress bar partial
  GET    /habits/heatmap/{habit_id}      → 365-day activity JSON for SVG heatmap

Settings / CRUD endpoints (settings page, create, reorder, update,
deactivate, grace-period) live in habits_settings.py under the same /habits
prefix. Calculation helpers (streak, display sort, week dates, ...) live in
_shared.py.

ROUTE-ORDER CONSTRAINT: `DELETE /habits/log` (here) shares a path shape with
`DELETE /habits/{habit_id}` (habits_settings.py). Starlette matches in
registration order and does not fall through on a failed int conversion, so
main.py must include habits.router BEFORE habits_settings.router. Before the
WO#23 split this was guaranteed by definition order within one file.
"""

import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, delete, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.habits.models import Habit, HabitLog
from domains.habits.routers._shared import (
    _build_habit_view,
    _get_grace_period,
    _get_logged_dates_for_habit,
    _get_today_logged_ids,
    _get_week_dates,
    _sort_habits_for_display,
)
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/habits", tags=["Habits"])


# ── MAIN CHECK-IN PAGE ────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def habits_page(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Main daily check-in page.
    Renders active habits sorted: incomplete first, completed last.
    Includes weekly overview grid and streak rankings.
    """
    today = datetime.date.today()
    grace_period = await _get_grace_period(db)

    # Fetch all active habits
    result = await db.execute(
        select(Habit)
        .where(Habit.is_active == True)
        .order_by(Habit.sort_order)
    )
    habits = result.scalars().all()

    today_logged_ids = await _get_today_logged_ids(db, today)

    # Build view dicts with streak + today_logged attached
    habit_views = []
    for habit in habits:
        view = await _build_habit_view(db, habit, today, grace_period, today_logged_ids)
        habit_views.append(view)

    # Sort: incomplete first, completed last
    habit_views = _sort_habits_for_display(habit_views)

    # Weekly overview — current week dates and which habits were logged each day
    week_dates = _get_week_dates()
    week_start = week_dates[0]
    week_end = week_dates[-1]

    # Fetch all logs for the current week in one query
    week_logs_result = await db.execute(
        select(HabitLog.habit_id, HabitLog.logged_date)
        .where(HabitLog.logged_date >= week_start)
        .where(HabitLog.logged_date <= week_end)
    )
    # Build a set of (habit_id, date) tuples for O(1) lookup in template
    week_logged_set = {(row[0], row[1]) for row in week_logs_result.all()}

    # Count only habits shown on this page. today_logged_ids can include
    # habits that were logged and later deactivated, which would push the
    # count above total_active.
    completed_today = sum(1 for v in habit_views if v["today_logged"])
    total_active = len(habits)

    return templates.TemplateResponse("habits.html", {
        "request": request,
        "active_module": "habits",
        "habit_views": habit_views,
        "today": today,
        "week_dates": week_dates,
        "week_logged_set": week_logged_set,
        "completed_today": completed_today,
        "total_active": total_active,
        "grace_period": grace_period,
    })


# ── TOGGLE ON — POST /habits/log ──────────────────────────────────────────────

@router.post("/log", response_class=HTMLResponse)
async def log_habit(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Marks a habit as done for a given date (defaults to today).
    Idempotent — if already logged, returns success without error.
    Returns the updated habit card partial for HTMX outerHTML swap.
    """
    form = await request.form()
    habit_id = int(form.get("habit_id"))
    date_str = form.get("logged_date", "")
    logged_date = (
        datetime.date.fromisoformat(date_str)
        if date_str else datetime.date.today()
    )
    notes = str(form.get("notes", "")).strip() or None

    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")

    # Insert — silently handle duplicate (unique constraint = already logged)
    try:
        log_entry = HabitLog(habit_id=habit_id, logged_date=logged_date, notes=notes)
        db.add(log_entry)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        # Already logged — that's fine, treat as success

    # Re-fetch: both commit() and rollback() can expire `habit`'s attributes
    # (rollback always does; commit does too unless the session explicitly
    # sets expire_on_commit=False). Accessing an expired ORM object's
    # attributes without an explicit awaited reload raises "greenlet_spawn
    # has not been called" under async SQLAlchemy, so refresh it via an
    # awaited call before _build_habit_view touches habit.id below.
    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")

    grace_period = await _get_grace_period(db)
    today = datetime.date.today()
    today_logged_ids = await _get_today_logged_ids(db, today)
    view = await _build_habit_view(db, habit, today, grace_period, today_logged_ids)

    return templates.TemplateResponse(
        "partials/habit_card.html",
        {"request": request, "view": view, "habit": habit, "today": today},
    )


# ── TOGGLE OFF — DELETE /habits/log ──────────────────────────────────────────

@router.delete("/log", response_class=HTMLResponse)
async def unlog_habit(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Removes the log for a habit on a given date.
    Used to un-check an accidentally marked habit.
    Returns the updated habit card partial.
    """
    form = await request.form()
    habit_id = int(form.get("habit_id"))
    date_str = form.get("logged_date", "")
    logged_date = (
        datetime.date.fromisoformat(date_str)
        if date_str else datetime.date.today()
    )

    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")

    await db.execute(
        delete(HabitLog)
        .where(HabitLog.habit_id == habit_id)
        .where(HabitLog.logged_date == logged_date)
    )
    await db.commit()

    # Re-fetch: commit() can expire `habit`'s attributes depending on
    # session configuration — see log_habit() for the full explanation.
    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail=f"Habit {habit_id} not found")

    grace_period = await _get_grace_period(db)
    today = datetime.date.today()
    today_logged_ids = await _get_today_logged_ids(db, today)
    view = await _build_habit_view(db, habit, today, grace_period, today_logged_ids)

    return templates.TemplateResponse(
        "partials/habit_card.html",
        {"request": request, "view": view, "habit": habit, "today": today},
    )


# ── PROGRESS BAR — GET /habits/progress ──────────────────────────────────────

@router.get("/progress", response_class=HTMLResponse)
async def habit_progress(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Returns just the progress bar partial.
    Called automatically after every card toggle via hx-on::after-request
    on the habits grid — keeps the "X of Y done" count in sync without
    reloading the full page.
    """
    today = datetime.date.today()

    total_result = await db.execute(
        select(func.count(Habit.id)).where(Habit.is_active == True)
    )
    total_active = total_result.scalar() or 0

    # Join to Habit so logs belonging to deactivated habits aren't counted
    # (otherwise "done" can exceed total_active).
    done_result = await db.execute(
        select(func.count(HabitLog.id))
        .join(Habit, Habit.id == HabitLog.habit_id)
        .where(HabitLog.logged_date == today)
        .where(Habit.is_active == True)
    )
    completed_today = done_result.scalar() or 0

    return templates.TemplateResponse(
        "partials/habit_progress.html",
        {
            "request": request,
            "completed_today": completed_today,
            "total_active": total_active,
        },
    )


# ── HEATMAP DATA ──────────────────────────────────────────────────────────────

@router.get("/heatmap/{habit_id}")
async def habit_heatmap(habit_id: int, db: AsyncSession = Depends(get_db)):
    """
    Returns 365 days of activity data as JSON for the SVG heatmap.
    Each entry is {"date": "YYYY-MM-DD", "count": 0|1}.
    count is always 0 or 1 — the unique constraint prevents multi-log days.
    """
    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404, detail="Habit not found")

    logged_dates = await _get_logged_dates_for_habit(db, habit_id, days=365)

    today = datetime.date.today()
    # Build exactly 365 days ending today
    data = []
    for i in range(364, -1, -1):
        d = today - datetime.timedelta(days=i)
        data.append({"date": d.isoformat(), "count": 1 if d in logged_dates else 0})

    return JSONResponse(content=data)
