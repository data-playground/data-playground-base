# routers/habits_settings.py
"""
Habit Tracker Module — settings / CRUD surface

Endpoints (this file), all under the same /habits prefix as habits.py:
  GET    /habits/settings                → Settings page (all habits incl. inactive)
  POST   /habits/new                     → Create a new habit
  PATCH  /habits/reorder                 → Save new sort_order for all habits
  PATCH  /habits/{habit_id}              → Update habit fields
  DELETE /habits/{habit_id}              → Soft-delete (set is_active=False)
  PATCH  /habits/settings/grace-period   → Update grace_period_days config

Daily check-in endpoints live in habits.py. Shared helpers live in _shared.py.

ROUTE-ORDER CONSTRAINTS:
  1. `PATCH /habits/reorder` must stay defined ABOVE `PATCH /habits/{habit_id}`
     in this file — otherwise "reorder" is captured as a habit_id and fails
     int validation with a 422 instead of reaching reorder_habits().
  2. main.py must include habits.router (which owns `DELETE /habits/log`)
     BEFORE this router, or `DELETE /habits/{habit_id}` will swallow it.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.habits.models import Habit, HabitSettings
from domains.habits.routers._shared import _get_grace_period
from core.templating import templates

router = APIRouter(prefix="/habits", tags=["Habits"])


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

    # Place new habit at the bottom: one past the highest existing sort_order.
    # Deliberately spans ALL habits (including inactive ones) so a later
    # reactivation can never tie with a newer habit, and unlike a row count
    # it stays unique after deactivations or drag-and-drop reordering.
    max_result = await db.execute(
        select(func.coalesce(func.max(Habit.sort_order), 0))
    )
    next_sort_order = max_result.scalar_one() + 1

    habit = Habit(
        name=name,
        description=str(form.get("description", "")).strip() or None,
        icon=str(form.get("icon", "")).strip() or "✅",
        color=str(form.get("color", "")).strip() or "#7c6fff",
        is_active=True,
        sort_order=next_sort_order,
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
