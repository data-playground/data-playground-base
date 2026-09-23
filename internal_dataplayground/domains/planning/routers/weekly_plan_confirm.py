# domains/planning/routers/weekly_plan_confirm.py
"""
Weekly Planning System — Plan confirmation.

Endpoints:
  POST  /plan/confirm  → Confirm draft → generates shopping list

Split out of weekly_plan.py by Work Order #30 (contingency split): direct
verification found weekly_plan.py still at 413 lines after WO#10's own
three-way split — over the 300-line router ceiling (CONTRIBUTING.md /
GOVERNANCE.md §1.2). `confirm_plan` was the single largest handler
(builds WeeklyPlanDay / WeeklyPlanMeal / WorkoutSession rows across a
7-day loop), so it moves out here, mirroring the same
CRUD-router / single-large-handler split pattern used elsewhere in this
program (e.g. `workout_plans.py` → `workout_plans_crud.py` +
`workout_plan_ai_generator.py`).

Shares the `/plan` prefix with `weekly_plan.py`, `weekly_plan_generator.py`,
and `weekly_plan_shopping.py`. This file registers only `POST /plan/confirm`,
which does not collide with any route the sibling routers register.

`_generate_shopping_list()` (defined in `weekly_plan_shopping.py`) is
imported directly, exactly as `weekly_plan.py` imported it before this
split — not duplicated.
"""
import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.planning.models import (
    PlanDayStatus, PlanMealStatus, PlanMealType,
    UserIntent,
    WeeklyPlan, WeeklyPlanDay, WeeklyPlanMeal, WeeklyPlanStatus,
)
from domains.workout.models import WorkoutPlan, WorkoutSession, WeightUnit
from domains.planning.routers.weekly_plan_shopping import _generate_shopping_list

router = APIRouter(prefix="/plan", tags=["Weekly Plan"])


# ── Confirm ────────────────────────────────────────────────────────────────────

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