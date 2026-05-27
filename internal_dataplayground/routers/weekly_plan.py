# routers/weekly_plan.py
"""
Weekly Planning System

Endpoints:
  GET   /plan                    → Plan hub (current week + history)
  GET   /plan/new                → Plan generation page (pre-flight settings)
  POST  /plan/generate           → AI generation → returns draft review page
  POST  /plan/{id}/confirm       → Confirm draft → generates shopping list
  GET   /plan/{id}               → Full week view
  GET   /plan/{id}/day/{date}    → Single day detail
  PATCH /plan/{id}/day/{date}    → Override a day (rest, skip, note)
  PATCH /plan/meal/{meal_id}     → Update meal status (eaten/swapped/off-plan)
  GET   /plan/{id}/shopping      → Shopping list view
  POST  /plan/{id}/shopping/regenerate → Regenerate shopping list
"""
import datetime
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import (
    FitnessGoal, PlanDayStatus, PlanMealStatus, PlanMealType,
    Recipe, RecipeMealType, ShoppingList, UserIntent,
    WeeklyPlan, WeeklyPlanDay, WeeklyPlanMeal, WeeklyPlanStatus,
    WorkoutPlan, WorkoutPlanDay, WorkoutSession, WeightUnit,
    Ingredient, PantryItem,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/plan", tags=["Weekly Plan"])
templates = Jinja2Templates(directory="templates")


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


# ── Generate ───────────────────────────────────────────────────────────────────

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


@router.post("/generate", response_class=HTMLResponse)
async def generate_plan(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Runs both AI generators and returns a draft review page.
    Does not save anything — user must confirm the draft.
    """
    form = await request.form()

    week_start_raw = str(form.get("week_start_date", "")).strip()
    try:
        week_start = datetime.date.fromisoformat(week_start_raw)
    except ValueError:
        week_start = _get_monday(datetime.date.today())

    week_end = week_start + datetime.timedelta(days=6)

    # Collect unavailable dates from checkboxes
    unavailable_dates = [
        (week_start + datetime.timedelta(days=i)).isoformat()
        for i in range(7)
        if form.get(f"unavailable_{i}")
    ]

    # Load intent
    intent_result = await db.execute(select(UserIntent).limit(1))
    intent = intent_result.scalar_one_or_none()
    if not intent:
        intent = UserIntent()

    intent_context = intent.to_ai_context()

    # Load active workout plan days
    active_plan_result = await db.execute(
        select(WorkoutPlan).where(WorkoutPlan.is_active == True).limit(1)
    )
    active_plan = active_plan_result.scalar_one_or_none()
    plan_days_data = []
    if active_plan:
        plan_days_data = [
            {
                "id": d.id,
                "day_name": d.day_name or f"Day {d.day_number}",
                "exercises": [pe.exercise.name for pe in d.exercises],
            }
            for d in active_plan.days
        ]

    # Load recipe library (titles + meal types + tags)
    recipes_result = await db.execute(
        select(Recipe).where(Recipe.is_archived == False)
    )
    recipes = recipes_result.scalars().all()
    recipe_data = [
        {
            "id": r.id,
            "title": r.title,
            "meal_type": r.meal_type.value if r.meal_type else None,
            "tags": r.tag_names,
            "times_cooked": r.times_cooked,
        }
        for r in recipes
    ]

    # Run generators
    from airflow.agents.weekly_agents import agent_plan_meals, agent_schedule_workouts

    try:
        workout_schedule = agent_schedule_workouts(
            intent_context=intent_context,
            active_plan_days=plan_days_data,
            target_workout_days=intent.weekly_workout_days,
            unavailable_dates=unavailable_dates,
            week_start=week_start,
        )
    except Exception as exc:
        log.error("Workout scheduler failed: %s", exc)
        workout_schedule = []

    workout_day_numbers = [
        d["day_number"] for d in workout_schedule if d.get("is_workout")
    ]
    rest_day_numbers = [
        d["day_number"] for d in workout_schedule if not d.get("is_workout")
    ]

    try:
        meal_plan = agent_plan_meals(
            intent_context=intent_context,
            available_recipes=recipe_data,
            week_start=week_start,
            workout_days=workout_day_numbers,
            rest_days=rest_day_numbers,
        )
    except Exception as exc:
        log.error("Meal planner failed: %s", exc)
        meal_plan = []

    # Merge into unified day structure for the review page
    merged_days = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i in range(7):
        day_num = i + 1
        plan_date = week_start + datetime.timedelta(days=i)
        workout = next((d for d in workout_schedule if d["day_number"] == day_num), {})
        meals = next((d for d in meal_plan if d["day_number"] == day_num), {})
        merged_days.append({
            "day_number":  day_num,
            "day_name":    day_names[i],
            "date":        plan_date,
            "date_str":    plan_date.isoformat(),
            "is_workout":  workout.get("is_workout", False),
            "plan_day_id": workout.get("plan_day_id"),
            "day_label":   workout.get("day_label", "Rest"),
            "workout_notes": workout.get("notes", ""),
            "breakfast":   meals.get("breakfast", ""),
            "lunch":       meals.get("lunch", ""),
            "dinner":      meals.get("dinner", ""),
            "snack":       meals.get("snack", ""),
            "meal_notes":  meals.get("notes", ""),
        })

    return templates.TemplateResponse("weekly_plan_review.html", {
        "request": request,
        "active_module": "plan",
        "week_start": week_start,
        "week_end": week_end,
        "merged_days": merged_days,
        "intent": intent,
        "merged_days_json": json.dumps(
            [{**d, "date": d["date"].isoformat()} for d in merged_days]
        ),
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


# ── Shopping list ──────────────────────────────────────────────────────────────

async def _generate_shopping_list(
    db: AsyncSession,
    plan: WeeklyPlan,
    recipe_ids: list[int],
) -> ShoppingList:
    """
    Aggregates ingredients from all planned recipes, deducts pantry items,
    and groups by category. Saves as a ShoppingList row.
    """
    from sqlalchemy import func as sqlfunc
    from models import RecipeIngredient

    if not recipe_ids:
        sl = ShoppingList(weekly_plan_id=plan.id, items=[])
        db.add(sl)
        await db.commit()
        return sl

    # Fetch all recipe ingredients for planned recipes
    ri_result = await db.execute(
        select(RecipeIngredient)
        .where(RecipeIngredient.recipe_id.in_(set(recipe_ids)))
    )
    all_ris = ri_result.scalars().all()

    # Fetch pantry
    pantry_result = await db.execute(
        select(Ingredient.name)
        .join(PantryItem, PantryItem.ingredient_id == Ingredient.id)
    )
    pantry_names = {row[0].lower() for row in pantry_result.all()}

    # Aggregate by ingredient
    aggregated: dict[str, dict] = {}
    for ri in all_ris:
        if not ri.ingredient:
            continue
        key = ri.ingredient.name.lower()
        if key not in aggregated:
            aggregated[key] = {
                "ingredient": ri.ingredient.name,
                "category":   ri.ingredient.category.value,
                "quantity":   float(ri.quantity) if ri.quantity else None,
                "unit":       ri.unit.value if ri.unit else None,
                "in_pantry":  key in pantry_names,
            }
        elif ri.quantity and aggregated[key]["quantity"]:
            aggregated[key]["quantity"] += float(ri.quantity)

    # Sort by category then name
    items = sorted(
        aggregated.values(),
        key=lambda x: (x["in_pantry"], x["category"], x["ingredient"])
    )

    # Delete existing shopping list if regenerating
    existing_result = await db.execute(
        select(ShoppingList).where(ShoppingList.weekly_plan_id == plan.id)
    )
    existing_sl = existing_result.scalar_one_or_none()
    if existing_sl:
        await db.delete(existing_sl)
        await db.flush()

    sl = ShoppingList(weekly_plan_id=plan.id, items=items)
    db.add(sl)
    await db.commit()
    return sl


@router.get("/{plan_id}/shopping", response_class=HTMLResponse)
async def shopping_list_view(
    plan_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    plan = await db.get(WeeklyPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404)

    sl_result = await db.execute(
        select(ShoppingList).where(ShoppingList.weekly_plan_id == plan_id)
    )
    shopping_list = sl_result.scalar_one_or_none()

    # Group items by category for display
    grouped: dict[str, list] = {}
    if shopping_list:
        for item in shopping_list.items:
            cat = item.get("category", "other")
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(item)

    need_items   = []
    pantry_items = []
    need_grouped = {}  # {category: [item, ...]}

    if shopping_list:
        for item in shopping_list.items:
            if item.get("in_pantry"):
                pantry_items.append(item)
            else:
                cat = item.get("category", "other")
                if cat not in need_grouped:
                    need_grouped[cat] = []
                need_grouped[cat].append(item)
        # Sort categories alphabetically
        need_grouped = dict(sorted(need_grouped.items()))

    return templates.TemplateResponse("shopping_list.html", {
        "request":      request,
        "plan":         plan,
        "shopping_list": shopping_list,
        "need_grouped": need_grouped,   # {category: [items to buy]}
        "pantry_items": pantry_items,   # [items already in pantry]
        "total":        len(shopping_list.items) if shopping_list else 0,
        "to_buy":       len(need_items) + sum(len(v) for v in need_grouped.values()),
        "in_pantry":    len(pantry_items),
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