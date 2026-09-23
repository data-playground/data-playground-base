# domains/workout/routers/workout_plan_prompts.py
"""
Workout Tracker — AI Plan Generator prompt templates

Split out of workout_plan_ai_generator.py as an explicitly-authorized
follow-up to Work Order #8, per GOVERNANCE.md §1.2's 300-line router
ceiling (WO#29 Part B) — workout_plan_ai_generator.py had grown to 375
lines. This file owns exercise-history context building and the
system-prompt / prompt-template string construction that used to be
inline in generate_plan(). generate_plan(), save_generated_plan(),
_fuzzy_match_exercise(), and the thin _call_gemini_for_plan() wrapper
stayed behind in workout_plan_ai_generator.py.

Not a router module — no APIRouter here, nothing registered in main.py.
"""

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def _build_exercise_history_context(db: AsyncSession) -> str:
    """
    Builds the exercise history block for the AI prompt.
    Covers the last 30 days — top exercises by frequency with trend.
    """
    cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = cutoff - timedelta(days=30)

    freq_result = await db.execute(
        text("""
            SELECT ws.exercise_id, e.name,
                   COUNT(DISTINCT ws.session_id) AS session_count,
                   MAX(ws.weight_used) AS last_max_weight,
                   ws.weight_unit
            FROM workout_sets ws
            JOIN exercises e ON ws.exercise_id = e.id
            JOIN workout_sessions s ON ws.session_id = s.id
            WHERE s.session_date >= :cutoff
              AND ws.is_warmup = FALSE
              AND ws.weight_used IS NOT NULL
              AND s.ended_at IS NOT NULL
            GROUP BY ws.exercise_id, e.name, ws.weight_unit
            ORDER BY session_count DESC
            LIMIT 8
        """),
        {"cutoff": cutoff.date()}
    )
    freq_rows = freq_result.mappings().all()

    if not freq_rows:
        return "No exercise history in the last 30 days."

    lines = []
    for row in freq_rows:
        unit = row["weight_unit"] or "lb"
        lines.append(
            f"- {row['name']}: {row['session_count']} sessions · "
            f"Recent max: {row['last_max_weight']} {unit}"
        )

    muscle_result = await db.execute(
        text("""
            SELECT e.primary_muscle_group, COUNT(DISTINCT ws.session_id) AS cnt
            FROM workout_sets ws
            JOIN exercises e ON ws.exercise_id = e.id
            JOIN workout_sessions s ON ws.session_id = s.id
            WHERE s.session_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
              AND s.ended_at IS NOT NULL
            GROUP BY e.primary_muscle_group
        """)
    )
    muscle_rows = muscle_result.mappings().all()
    muscle_summary = ", ".join(
        f"{r['primary_muscle_group']} ({r['cnt']}x)" for r in muscle_rows
    ) if muscle_rows else "None"

    return "\n".join(lines) + f"\n\nMuscle groups trained (last 7 days): {muscle_summary}"


# ── Prompt templates ─────────────────────────────────────────────────────────
# Moved verbatim out of generate_plan()'s body — string content unchanged.

PLAN_SYSTEM_INSTRUCTION = """
You are an expert personal trainer building a structured workout plan.
Generate a JSON response only — no preamble, no markdown, no explanation.

The plan must:
  - Only use exercises from the provided catalog (exact name match required)
  - Be appropriate for the stated goal and available equipment
  - Distribute muscle groups intelligently across the specified number of days
  - Include compound movements as the foundation, with isolation work as accessories
  - Balance push/pull/legs/core across the week
  - Suggest realistic starting weights in lb for each exercise

Return this exact JSON structure:
{
  "plan_name": "string",
  "description": "string — 1-2 sentences about the plan approach",
  "days": [
    {
      "day_number": 1,
      "day_name": "string — e.g. 'Push Day' or 'Chest & Triceps'",
      "exercises": [
        {
          "exercise_name": "exact name from catalog",
          "target_sets": 3,
          "target_reps_min": 8,
          "target_reps_max": 12,
          "target_weight_lb": 135.0,
          "order_in_day": 1,
          "notes": "optional coaching note"
        }
      ]
    }
  ]
}
"""


def build_plan_prompt(
    target_days: int,
    goal,
    location: Optional[object],
    equipment_context: str,
    exercise_history: str,
    additional_notes: str,
    catalog_text: str,
) -> str:
    """
    Builds the user-turn prompt for the plan generator.

    Args:
        target_days:        Target training days per week (1-7).
        goal:                WorkoutGoal enum member — uses goal.label.
        location:            Optional WorkoutLocation ORM object — uses
                              location.name if present, else "Not specified".
        equipment_context:   Pre-built equipment summary string.
        exercise_history:    Output of _build_exercise_history_context().
        additional_notes:    Raw user-provided notes string (may be empty).
        catalog_text:        Pre-built exercise catalog string.

    Returns:
        The exact prompt string generate_plan() used to build inline —
        content and formatting unchanged.
    """
    return f"""Build a {target_days}-day/week workout plan.

Goal: {goal.label}
Days per week: {target_days}
Location: {location.name if location else "Not specified"}
Available equipment: {equipment_context}

Exercise history (last 30 days):
{exercise_history}

{f"Additional notes from user: {additional_notes}" if additional_notes else ""}

Exercise catalog (use ONLY these names, exact spelling):
{catalog_text}

Generate {target_days} days. Each day should have 4-7 exercises.
Do not repeat the same exercise across days in the same week.
Prioritize exercises the user is already doing (from history) but ensure full coverage."""
