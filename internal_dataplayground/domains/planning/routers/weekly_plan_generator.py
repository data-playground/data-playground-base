# domains/planning/routers/weekly_plan_generator.py
"""
Weekly Planning System — AI generation endpoint.

Endpoints:
  POST  /plan/generate  → AI generation → returns draft review page

Split out of weekly_plan.py in WO#10 because this is the one handler in
the planning domain that calls out to `airflow.agents.weekly_agents`
(Gemini-based meal/workout generation) — kept separate so the CRUD/view
router stays free of AI-generation concerns, mirroring the CRUD/AI-
generator split already applied to `workout_plans.py` in WO#8.

Shares the `/plan` prefix with `weekly_plan.py`'s router. This is safe:
this file registers only `POST /plan/generate`, which does not collide
with any route the sibling router registers (verified during this
migration — see the WO#10 report's router-split verification item).

`airflow/agents/weekly_agents.py` itself is explicitly out of scope for
this migration (see the work order's HARD BOUNDARIES) and is imported
here exactly as it was in the pre-split file — as a local import inside
the handler, untouched.
"""
import datetime
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.planning.models import UserIntent
from domains.workout.models import WorkoutPlan
from domains.recipes.models import Recipe

log = logging.getLogger(__name__)
router = APIRouter(prefix="/plan", tags=["Weekly Plan"])

from core.templating import templates


def _get_monday(for_date: datetime.date) -> datetime.date:
    """
    Returns the Monday of the week containing for_date.

    Duplicated (not imported) from weekly_plan.py's identical private
    helper — both copies are two lines, and importing a leading-
    underscore helper across sibling router modules for something this
    small isn't worth the coupling. Flagged here rather than silently
    left unexplained.
    """
    return for_date - datetime.timedelta(days=for_date.weekday())


# ── Generate ───────────────────────────────────────────────────────────────────

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
