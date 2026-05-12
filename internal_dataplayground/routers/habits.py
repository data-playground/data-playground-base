# routers/habits.py
"""
Habit Tracker Module

Endpoints:
  GET    /habits                         → Main check-in page
  POST   /habits/log                     → Toggle habit on (idempotent)
  DELETE /habits/log                     → Toggle habit off
  GET    /habits/heatmap/{habit_id}      → 365-day activity JSON for SVG heatmap
  GET    /habits/settings                → Settings page (all habits incl. inactive)
  POST   /habits/new                     → Create a new habit
  PATCH  /habits/reorder                 → Save new sort_order for all habits
  PATCH  /habits/{habit_id}              → Update habit fields
  DELETE /habits/{habit_id}              → Soft-delete (set is_active=False)
  PATCH  /habits/settings/grace-period   → Update grace_period_days config

Streak algorithm:
  Walks backwards from yesterday. Counts consecutive days where a log exists,
  allowing up to grace_period_days missed days within the window before
  breaking the streak. Today is excluded (the day is not over yet).

Sort order:
  Incomplete habits render first (ascending sort_order), completed habits
  render last (ascending sort_order within completed group). This is a
  display sort applied in the router — not a DB sort.
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, delete, text, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Habit, HabitLog, HabitSettings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/habits", tags=["Habits"])
templates = Jinja2Templates(directory="templates")


# ── HELPERS ───────────────────────────────────────────────────────────────────

async def _get_grace_period(db: AsyncSession) -> int:
    """Reads grace_period_days from the single habit_settings row."""
    result = await db.execute(select(HabitSettings).limit(1))
    settings = result.scalar_one_or_none()
    return settings.grace_period_days if settings else 1


async def _get_logged_dates_for_habit(
    db: AsyncSession, habit_id: int, days: int = 400
) -> set[datetime.date]:
    """
    Returns a set of logged dates for a habit over the last `days` days.
    Used by both the streak calculation and the heatmap endpoint.
    """
    cutoff = datetime.date.today() - datetime.timedelta(days=days)
    result = await db.execute(
        select(HabitLog.logged_date)
        .where(HabitLog.habit_id == habit_id)
        .where(HabitLog.logged_date >= cutoff)
    )
    return {row[0] for row in result.all()}


def _calculate_streak(logged_dates: set[datetime.date], grace_period: int) -> int:
    """
    Counts consecutive logged days walking backwards from yesterday.

    Grace period: up to `grace_period` consecutive missed days are tolerated
    before the streak breaks. Setting grace_period=0 requires strict
    consecutive days.

    Today is intentionally excluded — the day isn't over yet, and including
    it would make the streak feel inflated before the user has actually
    completed their habits.

    Args:
        logged_dates: Set of dates that have a log entry.
        grace_period: Number of consecutive misses allowed before streak breaks.

    Returns:
        Integer streak count (0 if yesterday was not logged and grace=0).
    """
    if not logged_dates:
        return 0

    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    streak = 0
    missed_in_window = 0
    cursor = yesterday

    # Walk backwards day by day for up to 2 years (safety cap)
    for _ in range(730):
        if cursor in logged_dates:
            streak += 1
            missed_in_window = 0  # reset miss counter on a logged day
        else:
            missed_in_window += 1
            if missed_in_window > grace_period:
                break
            # Within grace window — keep walking but don't increment streak
        cursor -= datetime.timedelta(days=1)

    return streak


async def _build_habit_view(
    db: AsyncSession,
    habit: Habit,
    today: datetime.date,
    grace_period: int,
    logged_today_ids: set[int],
) -> dict:
    """
    Attaches today_logged and streak to a habit for template rendering.
    Returns a plain dict so it can be used both in the full page and partials.
    """
    logged_dates = await _get_logged_dates_for_habit(db, habit.id, days=400)
    streak = _calculate_streak(logged_dates, grace_period)
    return {
        "habit": habit,
        "today_logged": habit.id in logged_today_ids,
        "streak": streak,
    }


async def _get_today_logged_ids(db: AsyncSession, today: datetime.date) -> set[int]:
    """Returns the set of habit_ids that have been logged today."""
    result = await db.execute(
        select(HabitLog.habit_id).where(HabitLog.logged_date == today)
    )
    return {row[0] for row in result.all()}


def _sort_habits_for_display(habit_views: list[dict]) -> list[dict]:
    """
    Incomplete habits first (ascending sort_order), completed habits last
    (ascending sort_order within completed group). This is the display rule:
    checking off a habit moves it to the bottom of the page.
    """
    incomplete = [h for h in habit_views if not h["today_logged"]]
    complete = [h for h in habit_views if h["today_logged"]]
    incomplete.sort(key=lambda h: h["habit"].sort_order)
    complete.sort(key=lambda h: h["habit"].sort_order)
    return incomplete + complete


def _get_week_dates(start_on_sunday: bool = True) -> list[datetime.date]:
    """Returns the 7 dates of the current week, starting Sunday."""
    today = datetime.date.today()
    # weekday(): Monday=0, Sunday=6
    days_since_sunday = (today.weekday() + 1) % 7
    sunday = today - datetime.timedelta(days=days_since_sunday)
    return [sunday + datetime.timedelta(days=i) for i in range(7)]


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

    completed_today = len(today_logged_ids)
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

    grace_period = await _get_grace_period(db)
    today = datetime.date.today()
    today_logged_ids = await _get_today_logged_ids(db, today)
    view = await _build_habit_view(db, habit, today, grace_period, today_logged_ids)

    return templates.TemplateResponse(
        "partials/habit_card.html",
        {"request": request, **view, "today": today},
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

    grace_period = await _get_grace_period(db)
    today = datetime.date.today()
    today_logged_ids = await _get_today_logged_ids(db, today)
    view = await _build_habit_view(db, habit, today, grace_period, today_logged_ids)

    return templates.TemplateResponse(
        "partials/habit_card.html",
        {"request": request, **view, "today": today},
    )

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
 
    done_result = await db.execute(
        select(func.count(HabitLog.id))
        .where(HabitLog.logged_date == today)
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


# ── SETTINGS PAGE ─────────────────────────────────────────────────────────────

@router.get("/settings", response_class=HTMLResponse)
async def habits_settings(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Settings page: all habits including inactive ones, grace period config,
    and per-habit activity heatmaps (loaded lazily via JS on expand).
    """
    result = await db.execute(
        select(Habit).order_by(Habit.is_active.desc(), Habit.sort_order)
    )
    habits = result.scalars().all()

    grace_period = await _get_grace_period(db)

    return templates.TemplateResponse("habits_settings.html", {
        "request": request,
        "active_module": "habits_settings",
        "habits": habits,
        "grace_period": grace_period,
    })


# ── CREATE HABIT ──────────────────────────────────────────────────────────────

@router.post("/new", response_class=HTMLResponse)
async def create_habit(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Creates a new habit from form data.
    Returns the updated full habit list partial for HTMX swap.
    """
    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="Habit name is required")

    # Place new habit at end of current active list
    count_result = await db.execute(
        select(Habit).where(Habit.is_active == True)
    )
    current_count = len(count_result.scalars().all())

    habit = Habit(
        name=name,
        description=str(form.get("description", "")).strip() or None,
        icon=str(form.get("icon", "")).strip() or "✅",
        color=str(form.get("color", "")).strip() or "#7c6fff",
        is_active=True,
        sort_order=current_count + 1,
    )
    db.add(habit)
    await db.commit()
    await db.refresh(habit)

    # Return updated full list for settings page
    result = await db.execute(
        select(Habit).order_by(Habit.is_active.desc(), Habit.sort_order)
    )
    habits = result.scalars().all()
    grace_period = await _get_grace_period(db)

    return templates.TemplateResponse(
        "partials/habit_settings_list.html",
        {"request": request, "habits": habits, "grace_period": grace_period,
         "toast": f"'{habit.name}' added."},
    )


# ── REORDER — PATCH /habits/reorder ──────────────────────────────────────────

@router.patch("/reorder", response_class=HTMLResponse)
async def reorder_habits(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Saves a new sort_order for all habits.
    Expects JSON body: [{"id": 1, "sort_order": 0}, ...]
    Called when user clicks "Save Order" after drag-and-drop in settings.
    """
    data = await request.json()
    for item in data:
        habit = await db.get(Habit, int(item["id"]))
        if habit:
            habit.sort_order = int(item["sort_order"])
    await db.commit()

    return HTMLResponse('<p style="font-size:10px;color:var(--green);">✓ Order saved.</p>')


# ── UPDATE HABIT — PATCH /habits/{habit_id} ───────────────────────────────────

@router.patch("/{habit_id}", response_class=HTMLResponse)
async def update_habit(
    habit_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Updates habit fields from form data.
    Returns the updated habit row partial for the settings list.
    """
    form = await request.form()
    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404)

    if "name" in form and form.get("name", "").strip():
        habit.name = str(form.get("name")).strip()
    if "description" in form:
        habit.description = str(form.get("description", "")).strip() or None
    if "icon" in form:
        habit.icon = str(form.get("icon", "")).strip() or None
    if "color" in form:
        habit.color = str(form.get("color", "")).strip() or None
    if "is_active" in form:
        habit.is_active = form.get("is_active") in ("true", "1", "on", "True")

    await db.commit()
    await db.refresh(habit)

    return templates.TemplateResponse(
        "partials/habit_settings_row.html",
        {"request": request, "habit": habit},
    )


# ── SOFT DELETE — DELETE /habits/{habit_id} ───────────────────────────────────

@router.delete("/{habit_id}", response_class=HTMLResponse)
async def deactivate_habit(
    habit_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Soft-deletes a habit by setting is_active=False.
    Logs are preserved for historical data.
    Returns empty response — HTMX removes the row.
    """
    habit = await db.get(Habit, habit_id)
    if not habit:
        raise HTTPException(status_code=404)

    habit.is_active = False
    await db.commit()

    return HTMLResponse("")


# ── GRACE PERIOD CONFIG ───────────────────────────────────────────────────────

@router.patch("/settings/grace-period", response_class=HTMLResponse)
async def update_grace_period(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Updates the global grace_period_days setting.
    0 = strict consecutive days. 1 = one missed day allowed.
    """
    form = await request.form()
    try:
        days = int(form.get("grace_period_days", 1))
        days = max(0, min(days, 7))  # clamp to 0–7 range
    except (ValueError, TypeError):
        days = 1

    result = await db.execute(select(HabitSettings).limit(1))
    settings = result.scalar_one_or_none()
    if settings:
        settings.grace_period_days = days
    else:
        settings = HabitSettings(grace_period_days=days)
        db.add(settings)

    await db.commit()

    return HTMLResponse(
        f'<span style="font-size:10px;color:var(--green);">'
        f'✓ Grace period set to {days} day{"s" if days != 1 else ""}.</span>'
    )
