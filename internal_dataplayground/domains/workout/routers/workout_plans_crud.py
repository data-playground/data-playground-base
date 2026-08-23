# domains/workout/routers/workout_plans_crud.py
"""
Workout Tracker — Plan Management (CRUD)

Split out of workout_plans.py as an explicitly-authorized follow-up to
Work Order #8, per GOVERNANCE.md §1.2's 300-line router ceiling —
workout_plans.py had grown to 523 lines (GOVERNANCE.md already named it
as pre-existing debt needing exactly this split). This file owns
list/create/activate/delete; the AI generator (context-building, the
Gemini call, fuzzy match, generate/save) moved to
workout_plan_ai_generator.py. Both routers share the "/workout/plans"
prefix and are included separately in main.py — the same two-routers-
sharing-a-prefix pattern already used for workout_log.py's
router/body_metrics_router split.

Endpoints:
  GET   /workout/plans                  → List all plans
  POST  /workout/plans                  → Create plan manually
  PATCH /workout/plans/{id}/activate   → Activate a plan (deactivates others)
  DELETE /workout/plans/{id}            → Delete a plan
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.workout.models import (
    PlanOrigin, WorkoutGoal, WorkoutLocation, WorkoutPlan, WorkoutPlanDay,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout/plans", tags=["Workout"])


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
