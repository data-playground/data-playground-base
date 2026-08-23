# domains/workout/routers/workout_settings.py
"""
Workout Tracker — Settings

Endpoints:
  GET    /workout/settings                           → Settings page
  POST   /workout/locations                          → Create location
  PATCH  /workout/locations/{id}                     → Update location
  DELETE /workout/locations/{id}                     → Soft delete location
  PATCH  /workout/locations/{id}/default             → Set as default location
  POST   /workout/locations/{id}/equipment           → Add equipment to location
  PATCH  /workout/equipment/{id}                     → Update equipment
  DELETE /workout/equipment/{id}                     → Soft delete equipment
  POST   /workout/exercises                          → Add custom exercise
  GET    /workout/exercises                          → Exercise list as JSON (search)
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.workout.models import (
    Equipment, EquipmentType, Exercise, ExerciseEquipmentType,
    LocationType, MuscleGroup, WeightUnit, WorkoutLocation,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/workout", tags=["Workout"])


# ---------------------------------------------------------------------------
# Shared helpers — always use these to avoid missing selectinload anywhere
# ---------------------------------------------------------------------------

async def _fetch_locations(db: AsyncSession) -> list:
    """Active locations with equipment eagerly loaded (avoids lazy-load errors)."""
    result = await db.execute(
        select(WorkoutLocation)
        .where(WorkoutLocation.is_active == True)
        .order_by(WorkoutLocation.is_default.desc(), WorkoutLocation.name)
        .options(selectinload(WorkoutLocation.equipment))
    )
    return result.scalars().all()


def _location_list_ctx(request: Request, locations: list, toast: str = None) -> dict:
    return {
        "request": request,
        "locations": locations,
        "equipment_types": list(EquipmentType),
        "toast": toast,
    }


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
# Locations
# ---------------------------------------------------------------------------

@router.post("/locations", response_class=HTMLResponse)
async def create_location(request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="Location name is required")

    loc_type_raw = str(form.get("location_type", "gym")).strip()
    try:
        loc_type = LocationType(loc_type_raw)
    except ValueError:
        loc_type = LocationType.GYM

    address = str(form.get("address", "")).strip() or None
    notes = str(form.get("notes", "")).strip() or None
    is_default = form.get("is_default", "").lower() in ("true", "1", "on")

    if is_default:
        existing_result = await db.execute(select(WorkoutLocation))
        for loc in existing_result.scalars().all():
            loc.is_default = False

    location = WorkoutLocation(
        name=name, location_type=loc_type, address=address,
        notes=notes, is_active=True, is_default=is_default,
    )
    db.add(location)
    await db.commit()

    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, f"'{name}' added."),
    )


@router.patch("/locations/{location_id}", response_class=HTMLResponse)
async def update_location(
    location_id: int, request: Request, db: AsyncSession = Depends(get_db),
):
    location = await db.get(WorkoutLocation, location_id)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    form = await request.form()
    if "name" in form and form.get("name", "").strip():
        location.name = str(form.get("name")).strip()
    if "address" in form:
        location.address = str(form.get("address", "")).strip() or None
    if "notes" in form:
        location.notes = str(form.get("notes", "")).strip() or None
    if "location_type" in form:
        try:
            location.location_type = LocationType(str(form.get("location_type")))
        except ValueError:
            pass

    await db.commit()

    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, "Location updated."),
    )


@router.delete("/locations/{location_id}", response_class=HTMLResponse)
async def delete_location(
    location_id: int, request: Request, db: AsyncSession = Depends(get_db),
):
    location = await db.get(WorkoutLocation, location_id)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    location.is_active = False
    location.is_default = False
    await db.commit()

    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, "Location removed."),
    )


@router.patch("/locations/{location_id}/default", response_class=HTMLResponse)
async def set_default_location(
    location_id: int, request: Request, db: AsyncSession = Depends(get_db),
):
    all_result = await db.execute(select(WorkoutLocation))
    for loc in all_result.scalars().all():
        loc.is_default = (loc.id == location_id)
    await db.commit()

    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, "Default location updated."),
    )


# ---------------------------------------------------------------------------
# Equipment
# ---------------------------------------------------------------------------

@router.post("/locations/{location_id}/equipment", response_class=HTMLResponse)
async def add_equipment(
    location_id: int, request: Request, db: AsyncSession = Depends(get_db),
):
    location = await db.get(WorkoutLocation, location_id)
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")

    form = await request.form()
    name = str(form.get("name", "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail="Equipment name is required")

    equip_type_raw = str(form.get("equipment_type", "other")).strip()
    try:
        equip_type = EquipmentType(equip_type_raw)
    except ValueError:
        equip_type = EquipmentType.OTHER

    max_weight_raw = str(form.get("max_weight", "")).strip()
    max_weight = float(max_weight_raw) if max_weight_raw else None

    weight_unit_raw = str(form.get("weight_unit", "lb")).strip()
    weight_unit = WeightUnit.KG if weight_unit_raw == "kg" else WeightUnit.LB

    notes = str(form.get("notes", "")).strip() or None

    equip = Equipment(
        location_id=location_id, name=name, equipment_type=equip_type,
        max_weight=max_weight, weight_unit=weight_unit, notes=notes, is_active=True,
    )
    db.add(equip)
    await db.commit()
    db.expire_all()
    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, f"'{name}' added."),
    )


@router.patch("/equipment/{equipment_id}", response_class=HTMLResponse)
async def update_equipment(
    equipment_id: int, request: Request, db: AsyncSession = Depends(get_db),
):
    equip = await db.get(Equipment, equipment_id)
    if not equip:
        raise HTTPException(status_code=404, detail="Equipment not found")

    form = await request.form()
    if "name" in form and form.get("name", "").strip():
        equip.name = str(form.get("name")).strip()
    if "equipment_type" in form:
        try:
            equip.equipment_type = EquipmentType(str(form.get("equipment_type")))
        except ValueError:
            pass
    if "max_weight" in form:
        raw = str(form.get("max_weight", "")).strip()
        equip.max_weight = float(raw) if raw else None
    if "notes" in form:
        equip.notes = str(form.get("notes", "")).strip() or None

    await db.commit()
    db.expire_all()

    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, "Equipment updated."),
    )


@router.delete("/equipment/{equipment_id}", response_class=HTMLResponse)
async def delete_equipment(
    equipment_id: int, request: Request, db: AsyncSession = Depends(get_db),
):
    equip = await db.get(Equipment, equipment_id)
    if not equip:
        raise HTTPException(status_code=404, detail="Equipment not found")

    equip.is_active = False
    await db.commit()
    db.expire_all()

    locations = await _fetch_locations(db)
    return templates.TemplateResponse(
        "partials/workout/location_list.html",
        _location_list_ctx(request, locations, "Equipment removed."),
    )


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
