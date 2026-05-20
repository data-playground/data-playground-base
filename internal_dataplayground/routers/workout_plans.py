# routers/workout_plans.py
"""
Workout Tracker — Plan Management + AI Plan Generator

Endpoints:
  GET   /workout/plans                  → List all plans
  POST  /workout/plans                  → Create plan manually
  PATCH /workout/plans/{id}/activate   → Activate a plan (deactivates others)
  POST  /workout/plans/generate         → AI plan generator (returns preview)
  POST  /workout/plans/{id}/save        → Save AI-previewed plan
  DELETE /workout/plans/{id}            → Delete a plan
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc, func, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, get_key
from models import (
    Exercise, ExerciseEquipmentType, MuscleGroup, PlanOrigin, WeightUnit,
    WorkoutGoal, WorkoutLocation, WorkoutPlan, WorkoutPlanDay,
    WorkoutPlanExercise, WorkoutSession, WorkoutSet,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout/plans", tags=["Workout"])
templates = Jinja2Templates(directory="templates")


# ── Plan list ──────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def list_plans(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WorkoutPlan).order_by(desc(WorkoutPlan.is_active), desc(WorkoutPlan.created_at))
    )
    plans = result.scalars().all()

    locations_result = await db.execute(
        select(WorkoutLocation).where(WorkoutLocation.is_active == True)
        .order_by(WorkoutLocation.name)
    )
    locations = locations_result.scalars().all()

    return templates.TemplateResponse("workout_plans.html", {
        "request": request,
        "active_module": "workout",
        "plans": plans,
        "locations": locations,
        "goals": list(WorkoutGoal),
    })


# ── Create plan manually ────────────────────────────────────────────────────────

@router.post("", response_class=HTMLResponse)
async def create_plan(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="Plan name is required")

    goal_raw = str(form.get("goal", "general_fitness")).strip()
    try:
        goal = WorkoutGoal(goal_raw)
    except ValueError:
        goal = WorkoutGoal.GENERAL_FITNESS

    days_raw = str(form.get("target_days_per_week", "3")).strip()
    target_days = int(days_raw) if days_raw.isdigit() else 3
    target_days = max(1, min(7, target_days))

    location_id_raw = form.get("location_id", "")
    location_id = int(location_id_raw) if location_id_raw else None

    description = str(form.get("description", "")).strip() or None
    notes = str(form.get("notes", "")).strip() or None

    plan = WorkoutPlan(
        name=name,
        description=description,
        generated_by=PlanOrigin.USER,
        location_id=location_id,
        target_days_per_week=target_days,
        goal=goal,
        notes=notes,
        is_active=False,
    )
    db.add(plan)
    await db.flush()  # Get plan.id before creating days

    # Create the named days
    day_names_raw = str(form.get("day_names", "")).strip()
    if day_names_raw:
        day_names = [d.strip() for d in day_names_raw.split(",") if d.strip()]
    else:
        day_names = [f"Day {i+1}" for i in range(target_days)]

    for i, day_name in enumerate(day_names[:target_days]):
        plan_day = WorkoutPlanDay(
            plan_id=plan.id,
            day_number=i + 1,
            day_name=day_name,
        )
        db.add(plan_day)

    await db.commit()
    await db.refresh(plan)

    result = await db.execute(
        select(WorkoutPlan).order_by(desc(WorkoutPlan.is_active), desc(WorkoutPlan.created_at))
    )
    plans = result.scalars().all()

    return templates.TemplateResponse("partials/workout/plan_list.html", {
        "request": request,
        "plans": plans,
        "toast": f"Plan '{name}' created.",
    })


# ── Activate a plan ────────────────────────────────────────────────────────────

@router.patch("/{plan_id}/activate", response_class=HTMLResponse)
async def activate_plan(
    plan_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Deactivates all plans, then activates the specified one."""
    all_plans_result = await db.execute(select(WorkoutPlan))
    for p in all_plans_result.scalars().all():
        p.is_active = False

    plan = await db.get(WorkoutPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    plan.is_active = True
    await db.commit()

    result = await db.execute(
        select(WorkoutPlan).order_by(desc(WorkoutPlan.is_active), desc(WorkoutPlan.created_at))
    )
    plans = result.scalars().all()

    return templates.TemplateResponse("partials/workout/plan_list.html", {
        "request": request,
        "plans": plans,
        "toast": f"'{plan.name}' is now your active plan.",
    })


# ── Delete a plan ──────────────────────────────────────────────────────────────

@router.delete("/{plan_id}", response_class=HTMLResponse)
async def delete_plan(
    plan_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    plan = await db.get(WorkoutPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    await db.delete(plan)
    await db.commit()

    result = await db.execute(
        select(WorkoutPlan).order_by(desc(WorkoutPlan.is_active), desc(WorkoutPlan.created_at))
    )
    plans = result.scalars().all()

    return templates.TemplateResponse("partials/workout/plan_list.html", {
        "request": request,
        "plans": plans,
        "toast": "Plan deleted.",
    })


# ── AI Plan Generator ──────────────────────────────────────────────────────────

async def _build_exercise_history_context(db: AsyncSession) -> str:
    """
    Builds the exercise history block for the AI prompt.
    Covers the last 30 days — top exercises by frequency with trend.
    """
    cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    cutoff = cutoff - timedelta(days=30)

    # Frequency: exercises logged most often in the last 30 days
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

    # Muscle groups trained in the last 7 days
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


def _call_gemini_for_plan(
    prompt: str,
    system: str,
) -> str:
    """Calls Gemini 2.5 Flash for plan generation. Sync wrapper for use in async context."""
    import requests as req
    api_key = get_key("Gemini-API")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }
    resp = req.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _fuzzy_match_exercise(db: AsyncSession, name: str) -> Optional[Exercise]:
    """
    Tries to find an exercise by exact name, then by LIKE match.
    Used to validate AI-suggested exercise names against the DB.
    """
    # Exact match first
    result = await db.execute(
        select(Exercise).where(Exercise.name == name)
    )
    ex = result.scalar_one_or_none()
    if ex:
        return ex

    # Case-insensitive LIKE match
    result = await db.execute(
        select(Exercise).where(Exercise.name.ilike(f"%{name}%")).limit(1)
    )
    return result.scalar_one_or_none()


@router.post("/generate", response_class=HTMLResponse)
async def generate_plan(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    AI plan generator. Sends structured numeric context to Gemini,
    returns a full-page preview for user review before saving.
    """
    form = await request.form()
    goal_raw = str(form.get("goal", "general_fitness")).strip()
    try:
        goal = WorkoutGoal(goal_raw)
    except ValueError:
        goal = WorkoutGoal.GENERAL_FITNESS

    days_raw = str(form.get("target_days_per_week", "3")).strip()
    target_days = int(days_raw) if days_raw.isdigit() else 3
    target_days = max(1, min(7, target_days))

    location_id_raw = form.get("location_id", "")
    location_id = int(location_id_raw) if location_id_raw else None

    additional_notes = str(form.get("additional_notes", "")).strip()

    # ── Build context for AI ───────────────────────────────────────────────
    location: Optional[WorkoutLocation] = None
    equipment_context = "No specific equipment logged."
    if location_id:
        location = await db.get(WorkoutLocation, location_id)
        if location:
            equipment_items = [
                f"{e.name} ({e.equipment_type.value})"
                for e in location.active_equipment
            ]
            equipment_context = (
                ", ".join(equipment_items) if equipment_items
                else "No equipment logged for this location — assume bodyweight only."
            )

    exercise_history = await _build_exercise_history_context(db)

    # Pull all exercise names from DB for the AI to choose from
    exercises_result = await db.execute(
        select(Exercise.name, Exercise.primary_muscle_group, Exercise.equipment_type, Exercise.is_compound)
        .order_by(Exercise.primary_muscle_group, Exercise.name)
    )
    exercise_catalog = exercises_result.all()
    catalog_text = "\n".join(
        f"- {r[0]} ({r[1]}, {r[2]}{', compound' if r[3] else ''})"
        for r in exercise_catalog
    )

    system = """
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

    prompt = f"""Build a {target_days}-day/week workout plan.

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

    try:
        raw_json = _call_gemini_for_plan(prompt, system)
        plan_data = json.loads(raw_json)
    except Exception as exc:
        log.error("AI plan generation failed: %s", exc)
        return templates.TemplateResponse("partials/workout/plan_generate_error.html", {
            "request": request,
            "error": f"Plan generation failed: {exc}. Try again or create a plan manually.",
        })

    # ── Validate AI exercise names against DB ──────────────────────────────
    unmatched = []
    for day in plan_data.get("days", []):
        for ex_data in day.get("exercises", []):
            matched = await _fuzzy_match_exercise(db, ex_data["exercise_name"])
            if matched:
                ex_data["exercise_id"] = matched.id
                ex_data["exercise_name"] = matched.name  # Normalize to DB name
            else:
                unmatched.append(ex_data["exercise_name"])
                ex_data["exercise_id"] = None

    return templates.TemplateResponse("workout_plan_preview.html", {
        "request": request,
        "active_module": "workout",
        "plan_data": plan_data,
        "goal": goal,
        "target_days": target_days,
        "location": location,
        "unmatched_exercises": unmatched,
        "plan_data_json": json.dumps(plan_data),
    })


@router.post("/{plan_id}/save", response_class=HTMLResponse)
async def save_generated_plan(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Saves a previewed AI plan to the DB after user approval.
    Accepts the plan JSON from the preview form and creates all DB rows.
    """
    form = await request.form()
    plan_json_raw = str(form.get("plan_data_json", "")).strip()
    goal_raw = str(form.get("goal", "general_fitness")).strip()
    location_id_raw = form.get("location_id", "")
    activate_raw = form.get("activate", "false")

    try:
        plan_data = json.loads(plan_json_raw)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=422, detail="Invalid plan data")

    try:
        goal = WorkoutGoal(goal_raw)
    except ValueError:
        goal = WorkoutGoal.GENERAL_FITNESS

    location_id = int(location_id_raw) if location_id_raw else None
    should_activate = activate_raw.lower() in ("true", "1", "on", "yes")

    # If activating, deactivate all existing plans
    if should_activate:
        all_plans_result = await db.execute(select(WorkoutPlan))
        for p in all_plans_result.scalars().all():
            p.is_active = False

    plan = WorkoutPlan(
        name=plan_data.get("plan_name", "AI Generated Plan"),
        description=plan_data.get("description"),
        generated_by=PlanOrigin.AI,
        location_id=location_id,
        target_days_per_week=len(plan_data.get("days", [])),
        goal=goal,
        is_active=should_activate,
    )
    db.add(plan)
    await db.flush()

    for day_data in plan_data.get("days", []):
        plan_day = WorkoutPlanDay(
            plan_id=plan.id,
            day_number=day_data["day_number"],
            day_name=day_data.get("day_name"),
        )
        db.add(plan_day)
        await db.flush()

        for order, ex_data in enumerate(day_data.get("exercises", []), 1):
            ex_id = ex_data.get("exercise_id")
            if not ex_id:
                continue  # Skip unmatched exercises

            plan_ex = WorkoutPlanExercise(
                plan_id=plan.id,
                plan_day_id=plan_day.id,
                exercise_id=ex_id,
                target_sets=ex_data.get("target_sets", 3),
                target_reps_min=ex_data.get("target_reps_min", 8),
                target_reps_max=ex_data.get("target_reps_max", 12),
                target_weight=ex_data.get("target_weight_lb"),
                order_in_day=order,
                notes=ex_data.get("notes"),
            )
            db.add(plan_ex)

    await db.commit()
    await db.refresh(plan)

    return templates.TemplateResponse("partials/workout/plan_saved.html", {
        "request": request,
        "plan": plan,
        "activated": should_activate,
    })
