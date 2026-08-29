# domains/planning/routers/weekly_plan.py
"""
Weekly Planning System — Day/meal CRUD and plan lifecycle.

Endpoints:
  GET   /plan                    → Plan hub (current week + history)
  GET   /plan/new                → Plan generation page (pre-flight settings)
  POST  /plan/confirm            → Confirm draft → generates shopping list
  GET   /plan/{id}                → Full week view
  PATCH /plan/{id}/day/{date}    → Override a day (rest, skip, note)
  PATCH /plan/meal/{meal_id}     → Update meal status (eaten/swapped/off-plan)

NOTE (WO#10 split): `POST /plan/generate` lives in the sibling module
`weekly_plan_generator.py` (the one handler that calls out to
`airflow.agents.weekly_agents`). `GET /plan/{id}/shopping` and its
`_generate_shopping_list()` helper live in `weekly_plan_shopping.py`.
All three routers share the `/plan` prefix; their route paths don't
overlap (verified — see the WO#10 report's route-enumeration check).

Correction: an earlier version of this docstring listed
`POST /plan/{id}/shopping/regenerate` as an endpoint. No such route has
ever existed in this file — checked against the actual code, and
against `weekly_agents.py` and `shopping_list.html` for any reference
that might justify it. Removed as a stale doc/code mismatch rather than
implemented, since nothing calls for it.
"""
import datetime
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.planning.models import (
    FitnessGoal, PlanDayStatus, PlanMealStatus, PlanMealType,
    UserIntent,
    WeeklyPlan, WeeklyPlanDay, WeeklyPlanMeal, WeeklyPlanStatus,
)
from domains.workout.models import WorkoutPlan, WorkoutSession, WeightUnit
from domains.planning.routers.weekly_plan_shopping import _generate_shopping_list

log = logging.getLogger(__name__)
router = APIRouter(prefix="/plan", tags=["Weekly Plan"])

from core.templating import templates


def _get_monday(for_date: datetime.date) -> datetime.date:
    """Returns the Monday of the week containing for_date."""
    return for_date - datetime.timedelta(days=for_date.weekday())


# ── Hub ────────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def plan_hub(request: Request, db: AsyncSession = Depends(get_db)):
    today = datetime.date.today()
    this_monday = _get_monday(today)

    # Current or most recent plan
    current_result = await db.execute(
        select(WeeklyPlan)
        .where(WeeklyPlan.week_start_date == this_monday)
        .limit(1)
    )
    current_plan = current_result.scalar_one_or_none()

    # Past plans (last 8 weeks)
    history_result = await db.execute(
        select(WeeklyPlan)
        .where(WeeklyPlan.week_start_date < this_monday)
        .order_by(desc(WeeklyPlan.week_start_date))
        .limit(8)
    )
    past_plans = history_result.scalars().all()

    # Intent summary
    intent_result = await db.execute(select(UserIntent).limit(1))
    intent = intent_result.scalar_one_or_none()

    return templates.TemplateResponse("weekly_plan_hub.html", {
        "request": request,
        "active_module": "plan",
        "current_plan": current_plan,
        "past_plans": past_plans,
        "intent": intent,
        "today": today,
        "this_monday": this_monday,
    })


# ── Pre-flight (generation page shell — actual generation lives in weekly_plan_generator.py) ──

@router.get("/new", response_class=HTMLResponse)
async def plan_new_form(request: Request, db: AsyncSession = Depends(get_db)):
    """Pre-flight form: lets user flag unavailable days before generation."""
    today = datetime.date.today()
    this_monday = _get_monday(today)
    next_monday = this_monday + datetime.timedelta(weeks=1)

    # Default to next week if this week already has a plan
    existing = await db.execute(
        select(WeeklyPlan).where(WeeklyPlan.week_start_date == this_monday)
    )
    target_monday = next_monday if existing.scalar_one_or_none() else this_monday

    intent_result = await db.execute(select(UserIntent).limit(1))
    intent = intent_result.scalar_one_or_none()

    # Build the 7 dates so user can flag days
    week_dates = [
        target_monday + datetime.timedelta(days=i)
        for i in range(7)
    ]

    active_plan_result = await db.execute(
        select(WorkoutPlan).where(WorkoutPlan.is_active == True).limit(1)
    )
    active_plan = active_plan_result.scalar_one_or_none()

    return templates.TemplateResponse("weekly_plan_new.html", {
        "request": request,
        "active_module": "plan",
        "intent": intent,
        "target_monday": target_monday,
        "week_dates": week_dates,
        "active_plan": active_plan,
        "fitness_goals": list(FitnessGoal),
    })


@router.post("/confirm", response_class=HTMLResponse)
async def confirm_plan(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Saves the reviewed/edited draft as a confirmed WeeklyPlan.
    Creates WeeklyPlanDay and WeeklyPlanMeal rows.
    Creates WorkoutSession stubs for workout days.
    Generates the shopping list.
    """
    form = await request.form()

    week_start_raw = str(form.get("week_start_date", "")).strip()
    try:
        week_start = datetime.date.fromisoformat(week_start_raw)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid week_start_date")

    week_end = week_start + datetime.timedelta(days=6)

    # Delete existing draft for this week if any
    existing_result = await db.execute(
        select(WeeklyPlan)
        .where(WeeklyPlan.week_start_date == week_start)
        .where(WeeklyPlan.status == WeeklyPlanStatus.DRAFT)
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        await db.delete(existing)
        await db.flush()

    # Load intent snapshot
    intent_result = await db.execute(select(UserIntent).limit(1))
    intent = intent_result.scalar_one_or_none()
    intent_snapshot = {}
    if intent:
        intent_snapshot = {
            "fitness_goal": intent.fitness_goal.value,
            "weekly_workout_days": intent.weekly_workout_days,
            "macro_preference": intent.macro_preference,
            "cooking_time_preference": intent.cooking_time_preference,
        }

    # Create plan
    plan = WeeklyPlan(
        week_start_date=week_start,
        week_end_date=week_end,
        status=WeeklyPlanStatus.CONFIRMED,
        intent_snapshot=intent_snapshot,
    )
    db.add(plan)
    await db.flush()

    workouts_planned = 0
    meals_planned = 0
    recipe_ids_needed = []  # for shopping list

    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    for i in range(7):
        day_num = i + 1
        plan_date = week_start + datetime.timedelta(days=i)
        is_workout = form.get(f"day_{day_num}_is_workout") == "true"
        plan_day_id = form.get(f"day_{day_num}_plan_day_id", "")

        # Create workout session stub if workout day
        session_id = None
        if is_workout:
            active_plan_result = await db.execute(
                select(WorkoutPlan).where(WorkoutPlan.is_active == True).limit(1)
            )
            active_plan = active_plan_result.scalar_one_or_none()

            session = WorkoutSession(
                plan_id=active_plan.id if active_plan else None,
                plan_day_id=int(plan_day_id) if plan_day_id.isdigit() else None,
                session_date=plan_date,
                weight_unit=WeightUnit.LB,
                # Note: started_at is NULL until user actually starts the session
            )
            db.add(session)
            await db.flush()
            session_id = session.id
            workouts_planned += 1

        plan_day = WeeklyPlanDay(
            weekly_plan_id=plan.id,
            plan_date=plan_date,
            day_number=day_num,
            workout_session_id=session_id,
            is_rest_day=not is_workout,
            day_status=PlanDayStatus.PLANNED,
        )
        db.add(plan_day)
        await db.flush()

        # Create meal rows
        for order, (meal_key, meal_type) in enumerate([
            ("breakfast", PlanMealType.BREAKFAST),
            ("lunch",     PlanMealType.LUNCH),
            ("dinner",    PlanMealType.DINNER),
            ("snack",     PlanMealType.SNACK),
        ]):
            recipe_id_raw = form.get(f"day_{day_num}_{meal_key}_recipe_id", "").strip()
            recipe_id = int(recipe_id_raw) if recipe_id_raw.isdigit() else None

            if recipe_id:
                recipe_ids_needed.append(recipe_id)

            meal = WeeklyPlanMeal(
                plan_day_id=plan_day.id,
                recipe_id=recipe_id,
                meal_type=meal_type,
                sort_order=order,
                status=PlanMealStatus.PLANNED,
            )
            db.add(meal)
            meals_planned += 1

    plan.workouts_planned = workouts_planned
    plan.meals_planned = meals_planned

    await db.commit()

    # Generate shopping list
    await _generate_shopping_list(db, plan, recipe_ids_needed)

    return RedirectResponse(url=f"/plan/{plan.id}", status_code=303)


# ── Plan view ──────────────────────────────────────────────────────────────────

@router.get("/{plan_id}", response_class=HTMLResponse)
async def plan_view(
    plan_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    plan = await db.get(WeeklyPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    today = datetime.date.today()

    return templates.TemplateResponse("weekly_plan_view.html", {
        "request": request,
        "active_module": "plan",
        "plan": plan,
        "today": today,
    })


# ── Day override ───────────────────────────────────────────────────────────────

@router.patch("/{plan_id}/day/{date_str}", response_class=HTMLResponse)
async def override_day(
    plan_id: int,
    date_str: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Override a single day. Used for:
    - Marking a workout day as rest (knee pain, travel, etc.)
    - Adding a note to the day
    - Skipping a day entirely
    """
    form = await request.form()
    try:
        plan_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid date")

    result = await db.execute(
        select(WeeklyPlanDay)
        .where(WeeklyPlanDay.weekly_plan_id == plan_id)
        .where(WeeklyPlanDay.plan_date == plan_date)
    )
    day = result.scalar_one_or_none()
    if not day:
        raise HTTPException(status_code=404, detail="Plan day not found")

    override_type = str(form.get("override_type", "")).strip()

    if override_type == "rest":
        day.is_rest_day = True
        day.override_reason = str(form.get("reason", "")).strip() or "Marked as rest day"
    elif override_type == "skip":
        day.day_status = PlanDayStatus.SKIPPED
        day.override_reason = str(form.get("reason", "")).strip() or "Skipped"
    elif override_type == "note":
        day.notes = str(form.get("notes", "")).strip()

    await db.commit()
    await db.refresh(day)

    return templates.TemplateResponse("partials/plan_day_card.html", {
        "request": request,
        "day": day,
        "today": datetime.date.today(),
    })


# ── Meal status update ─────────────────────────────────────────────────────────

@router.patch("/meal/{meal_id}", response_class=HTMLResponse)
async def update_meal_status(
    meal_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Mark a meal as eaten, swapped, off-plan, or skipped.
    For swapped: accepts a swap_recipe_id.
    For off-plan: accepts a note (e.g. "work lunch — sushi").
    """
    form = await request.form()
    meal = await db.get(WeeklyPlanMeal, meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    try:
        new_status = PlanMealStatus(str(form.get("status", "eaten")))
    except ValueError:
        new_status = PlanMealStatus.EATEN

    meal.status = new_status

    if new_status == PlanMealStatus.SWAPPED:
        swap_id = str(form.get("swap_recipe_id", "")).strip()
        meal.swap_recipe_id = int(swap_id) if swap_id.isdigit() else None

    if new_status == PlanMealStatus.OFF_PLAN:
        meal.off_plan_note = str(form.get("off_plan_note", "")).strip() or None

    # Update plan adherence counters
    plan_day = await db.get(WeeklyPlanDay, meal.plan_day_id)
    if plan_day:
        plan = await db.get(WeeklyPlan, plan_day.weekly_plan_id)
        if plan and new_status == PlanMealStatus.EATEN:
            plan.meals_followed = min(plan.meals_followed + 1, plan.meals_planned)

    await db.commit()
    await db.refresh(meal)

    return templates.TemplateResponse("partials/plan_meal_row.html", {
        "request": request,
        "meal": meal,
    })


async def _sync_plan_status(db: AsyncSession) -> None:
    """
    Called by the dashboard loader. Advances plan status:
    - confirmed → active  (if today is within the week)
    - active → completed  (if week has ended)
    Also marks individual days as active/completed based on date.
    """
    today = datetime.date.today()

    plans_result = await db.execute(
        select(WeeklyPlan)
        .where(WeeklyPlan.status.in_([
            WeeklyPlanStatus.CONFIRMED,
            WeeklyPlanStatus.ACTIVE,
        ]))
    )
    for plan in plans_result.scalars().all():
        if plan.week_start_date <= today <= plan.week_end_date:
            plan.status = WeeklyPlanStatus.ACTIVE
        elif today > plan.week_end_date:
            plan.status = WeeklyPlanStatus.COMPLETED

        for day in plan.days:
            if day.plan_date < today and day.day_status == PlanDayStatus.PLANNED:
                day.day_status = PlanDayStatus.COMPLETED
            elif day.plan_date == today and day.day_status == PlanDayStatus.PLANNED:
                day.day_status = PlanDayStatus.ACTIVE

    await db.commit()
