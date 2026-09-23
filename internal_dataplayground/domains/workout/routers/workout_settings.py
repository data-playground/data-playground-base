# domains/workout/routers/workout_settings.py
"""
Workout Tracker — Settings

Split from its original form as an explicitly-authorized follow-up to
Work Order #8, per GOVERNANCE.md §1.2's 300-line router ceiling (WO#29
Part A) — this file had grown to 378 lines. The Locations & Equipment
CRUD block moved to workout_locations.py; the Settings page itself and
Custom Exercises CRUD stayed here. Both routers share the "/workout"
prefix and are included separately in main.py.

The settings page (workout_settings()) still needs the active-locations
list for its template context, so it imports _fetch_locations() from
workout_locations.py rather than duplicating that query here.

Endpoints:
  GET    /workout/settings                           → Settings page
  POST   /workout/exercises                          → Add custom exercise
  GET    /workout/exercises                          → Exercise list as JSON (search)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.workout.models import (
    EquipmentType, Exercise, ExerciseEquipmentType,
    LocationType, MuscleGroup,
)
from domains.workout.routers.workout_locations import _fetch_locations

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout", tags=["Workout"])


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

@router.get("/settings", response_class=HTMLResponse)
async def workout_settings(request: Request, db: AsyncSession = Depends(get_db)):
    locations = await _fetch_locations(db)

    custom_exercises_result = await db.execute(
        select(Exercise)
        .where(Exercise.is_custom == True)
        .order_by(Exercise.name)
    )
    custom_exercises = custom_exercises_result.scalars().all()

    return templates.TemplateResponse("workout_settings.html", {
        "request": request,
        "active_module": "workout",
        "locations": locations,
        "custom_exercises": custom_exercises,
        "location_types": list(LocationType),
        "equipment_types": list(EquipmentType),
        "muscle_groups": list(MuscleGroup),
        "exercise_equipment_types": list(ExerciseEquipmentType),
    })


# ---------------------------------------------------------------------------
# Custom exercises
# ---------------------------------------------------------------------------

@router.post("/exercises", response_class=HTMLResponse)
async def create_custom_exercise(request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="Exercise name is required")

    existing_result = await db.execute(select(Exercise).where(Exercise.name.ilike(name)))
    if existing_result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"'{name}' already exists")

    muscle_raw = str(form.get("primary_muscle_group", "full_body")).strip()
    try:
        muscle = MuscleGroup(muscle_raw)
    except ValueError:
        muscle = MuscleGroup.FULL_BODY

    equip_raw = str(form.get("equipment_type", "other")).strip()
    try:
        equip_type = ExerciseEquipmentType(equip_raw)
    except ValueError:
        equip_type = ExerciseEquipmentType.OTHER

    is_compound = form.get("is_compound", "").lower() in ("true", "1", "on")
    notes = str(form.get("notes", "")).strip() or None

    exercise = Exercise(
        name=name, primary_muscle_group=muscle, secondary_muscle_groups=[],
        equipment_type=equip_type, is_compound=is_compound, is_custom=True, notes=notes,
    )
    db.add(exercise)
    await db.commit()

    custom_result = await db.execute(
        select(Exercise).where(Exercise.is_custom == True).order_by(Exercise.name)
    )
    return templates.TemplateResponse("partials/workout/custom_exercise_list.html", {
        "request": request,
        "custom_exercises": custom_result.scalars().all(),
        "toast": f"'{name}' added to your exercise library.",
    })


@router.get("/exercises", response_class=JSONResponse)
async def search_exercises(q: str = "", db: AsyncSession = Depends(get_db)):
    """Exercise autocomplete for the session log panel.

    BUGFIX (post-migration, explicitly authorized by project owner —
    see Work Order #8 report/follow-up, not part of the original migration
    diff): this endpoint previously raised `NameError: name 'rows' is not
    defined` — the query result was stored in `result` but the return
    comprehension iterated an undefined `rows`. Fixed by materializing
    `rows = result.all()`.

    A second, previously-masked bug was found while fixing the first:
    `primary_muscle_group` and `equipment_type` are plain Python
    `enum.Enum` members (not string-enums), and Starlette's `JSONResponse`
    calls raw `json.dumps()` with no enum handler — so once the NameError
    was fixed, this endpoint would have immediately failed instead with
    `TypeError: Object of type MuscleGroup is not JSON serializable`.
    Fixed by serializing `.value` for both enum fields.
    """
    stmt = select(
        Exercise.id, Exercise.name, Exercise.primary_muscle_group,
        Exercise.equipment_type, Exercise.is_compound, Exercise.is_custom,
    ).order_by(Exercise.primary_muscle_group, Exercise.name)

    if q.strip():
        stmt = stmt.where(Exercise.name.ilike(f"%{q.strip()}%"))

    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "id": r[0],
            "name": r[1],
            "muscle": r[2].value,
            "equipment": r[3].value,
            "compound": r[4],
            "custom": r[5],
        }
        for r in rows
    ]
