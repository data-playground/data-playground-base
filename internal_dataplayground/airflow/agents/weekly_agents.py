# airflow/agents/weekly_agents.py
"""
Weekly planning agents.

agent_plan_meals:    Generates a full week of meals from intent + recipe library.
agent_plan_workouts: Distributes workout days across a week from intent + active plan.
"""

import json
import logging
from datetime import date, timedelta

from services.ai import MODEL_FLASH, call_gemini_json

log = logging.getLogger(__name__)


# ── MEAL PLANNER ──────────────────────────────────────────────────────────────

_MEAL_PLAN_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "day_number":  {"type": "INTEGER"},   # 1=Mon … 7=Sun
            "breakfast":   {"type": "STRING"},     # recipe title or description
            "lunch":       {"type": "STRING"},
            "dinner":      {"type": "STRING"},
            "snack":       {"type": "STRING"},
            "notes":       {"type": "STRING"},     # e.g. "high protein day", "lighter — rest day"
        },
        "required": ["day_number", "breakfast", "lunch", "dinner"],
    },
}


def agent_plan_meals(
    intent_context: str,
    available_recipes: list[dict],   # [{"id": int, "title": str, "meal_type": str, ...}]
    week_start: date,
    workout_days: list[int],          # day numbers that are workout days (1=Mon)
    rest_days: list[int],
) -> list[dict]:
    """
    Generates a 7-day meal plan aligned to the user's intent.

    Prioritises recipes from the library where possible.
    If the library is sparse, suggests new recipe ideas by title
    that can be extracted and saved separately.

    Args:
        intent_context:    Output of UserIntent.to_ai_context()
        available_recipes: Recipes in the DB, with title + meal_type + tags
        week_start:        Monday of the target week
        workout_days:      Day numbers with workouts (slightly higher calories)
        rest_days:         Day numbers without workouts (slightly lighter)

    Returns:
        List of 7 dicts, one per day, with meal suggestions.
    """
    # Build a compact recipe catalog for the prompt
    recipe_catalog = "\n".join(
        f"- [{r['id']}] {r['title']} ({r.get('meal_type', '?')})"
        + (f" | Tags: {', '.join(r.get('tags', []))}" if r.get('tags') else "")
        for r in available_recipes[:60]  # keep prompt manageable
    )

    workout_day_names = [
        (week_start + timedelta(days=d - 1)).strftime("%A")
        for d in workout_days
    ]
    rest_day_names = [
        (week_start + timedelta(days=d - 1)).strftime("%A")
        for d in rest_days
    ]

    system = """
You are a nutrition-aware meal planner building a full week of meals for one person.

Your meal plan must:
  - Align to the user's fitness goal and macro preference
  - Vary meals across the week — no repeated dinners, limited repeated lunches
  - On workout days, lean slightly higher in protein and total calories
  - On rest days, keep meals lighter
  - Account for the user's food preferences and dislikes explicitly
  - Use recipes from the provided library where they fit the goal
  - Where the library lacks a good option, suggest a short recipe description
    the user can add later (prefix with "NEW: ")
  - Batch-cooking is encouraged — e.g. suggest making extra portions of
    Monday's dinner for Tuesday's lunch

For each day, return:
  breakfast: recipe title or "NEW: <brief description>"
  lunch:     recipe title or "NEW: <brief description>"
  dinner:    recipe title or "NEW: <brief description>"
  snack:     recipe title, "NEW: <brief description>", or "none"
  notes:     1-sentence context note for the day
"""

    prompt = f"""
Plan meals for the week of {week_start.strftime('%B %-d, %Y')} (Mon–Sun).

USER INTENT:
{intent_context}

WORKOUT DAYS (higher protein/calories): {', '.join(workout_day_names) or 'none'}
REST DAYS (lighter meals): {', '.join(rest_day_names) or 'none'}

AVAILABLE RECIPES IN LIBRARY:
{recipe_catalog or 'No recipes yet — suggest new recipes for all meals.'}

Generate 7 days (day_number 1=Monday through 7=Sunday).
Prioritise library recipes. Use "NEW: " prefix for suggestions outside the library.
"""

    raw = call_gemini_json(prompt, schema=_MEAL_PLAN_SCHEMA, system=system, model=MODEL_FLASH)
    return json.loads(raw)


# ── WORKOUT SCHEDULER ─────────────────────────────────────────────────────────

def agent_schedule_workouts(
    intent_context: str,
    active_plan_days: list[dict],   # [{"id": int, "day_name": str, "exercises": [...]}]
    target_workout_days: int,
    unavailable_dates: list[str],   # ISO date strings the user pre-flagged
    week_start: date,
) -> list[dict]:
    """
    Distributes workout sessions across the week.

    Returns a list of 7 dicts (one per day) indicating whether it's a
    workout day and which plan day to follow.

    Args:
        intent_context:       Output of UserIntent.to_ai_context()
        active_plan_days:     The days defined in the active WorkoutPlan
        target_workout_days:  From intent (e.g. 4)
        unavailable_dates:    Dates the user pre-flagged as unavailable
        week_start:            Monday of the target week

    Returns:
        List of 7 dicts:
        [{"day_number": 1, "is_workout": bool, "plan_day_id": int|None, "notes": str}]
    """
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "day_number":   {"type": "INTEGER"},
                "is_workout":   {"type": "BOOLEAN"},
                "plan_day_id":  {"type": "INTEGER"},   # null for rest days
                "day_label":    {"type": "STRING"},     # e.g. "Pull Day", "Rest"
                "notes":        {"type": "STRING"},
            },
            "required": ["day_number", "is_workout", "day_label"],
        },
    }

    plan_catalog = "\n".join(
        f"  Plan Day ID {d['id']}: {d['day_name']}"
        + (f" ({len(d.get('exercises', []))} exercises)" if d.get('exercises') else "")
        for d in active_plan_days
    ) if active_plan_days else "  No active plan — schedule free-form sessions."

    unavailable = [
        date.fromisoformat(ds).strftime("%A")
        for ds in unavailable_dates
    ] if unavailable_dates else []

    system = """
You are scheduling a week of workouts for someone working towards a fitness goal.

Rules:
  - Do not schedule workouts on unavailable days
  - Respect the target number of workout days
  - Avoid scheduling hard workouts (legs, full body) on back-to-back days
  - Distribute workout days sensibly — not all bunched at the start of the week
  - Assign plan days in rotation from the available plan day list
  - If no active plan exists, mark workout days with is_workout=true and plan_day_id=null
  - Rest days should have is_workout=false and day_label="Rest" or "Active Recovery"
"""

    prompt = f"""
Schedule workouts for the week of {week_start.strftime('%B %-d, %Y')}.

USER INTENT:
{intent_context}

TARGET WORKOUT DAYS THIS WEEK: {target_workout_days}
UNAVAILABLE DAYS: {', '.join(unavailable) or 'None'}

WORKOUT PLAN DAYS AVAILABLE (assign in rotation):
{plan_catalog}

Return 7 entries (day_number 1=Monday through 7=Sunday).
"""

    raw = call_gemini_json(prompt, schema=schema, system=system, model=MODEL_FLASH)
    return json.loads(raw)
