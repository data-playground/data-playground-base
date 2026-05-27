# routers/intent.py
"""
User Intent Profile — the shared context for all AI generators.

Endpoints:
  GET   /intent          → Settings page
  POST  /intent          → Create or update intent (upsert)
  GET   /intent/context  → JSON — intent as AI-ready context string
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import datetime

from database import get_db
from models import FitnessGoal, UserIntent

router = APIRouter(prefix="/intent", tags=["Intent"])
templates = Jinja2Templates(directory="templates")


async def get_or_create_intent(db: AsyncSession) -> UserIntent:
    """Always returns the single UserIntent row, creating it if absent."""
    result = await db.execute(select(UserIntent).limit(1))
    intent = result.scalar_one_or_none()
    if not intent:
        intent = UserIntent()
        db.add(intent)
        await db.commit()
        await db.refresh(intent)
    return intent


@router.get("", response_class=HTMLResponse)
async def intent_page(request: Request, db: AsyncSession = Depends(get_db)):
    intent = await get_or_create_intent(db)
    return templates.TemplateResponse("intent.html", {
        "request": request,
        "active_module": "intent",
        "intent": intent,
        "fitness_goals": list(FitnessGoal),
        "macro_options": [
            ("high_protein", "High Protein"),
            ("balanced",     "Balanced"),
            ("low_carb",     "Low Carb"),
            ("flexible",     "Flexible"),
        ],
        "cooking_time_options": [
            ("minimal",  "Minimal (<20 min)"),
            ("moderate", "Moderate (20–45 min)"),
            ("generous", "Generous (45+ min)"),
        ],
    })


@router.post("", response_class=HTMLResponse)
async def save_intent(request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    intent = await get_or_create_intent(db)

    # Fitness goal
    try:
        intent.fitness_goal = FitnessGoal(str(form.get("fitness_goal", "weight_loss")))
    except ValueError:
        intent.fitness_goal = FitnessGoal.WEIGHT_LOSS

    # Numeric fields
    days_raw = str(form.get("weekly_workout_days", "4"))
    intent.weekly_workout_days = int(days_raw) if days_raw.isdigit() else 4
    intent.weekly_workout_days = max(1, min(7, intent.weekly_workout_days))

    cal_raw = str(form.get("target_calories", "")).strip()
    intent.target_calories = int(cal_raw) if cal_raw.isdigit() else None

    # String preferences
    intent.macro_preference = str(form.get("macro_preference", "high_protein"))
    intent.cooking_time_preference = str(form.get("cooking_time_preference", "moderate"))

    # Free-text fields
    intent.dietary_restrictions = str(form.get("dietary_restrictions", "")).strip() or None
    intent.food_preferences     = str(form.get("food_preferences", "")).strip() or None
    intent.food_dislikes        = str(form.get("food_dislikes", "")).strip() or None
    intent.health_notes         = str(form.get("health_notes", "")).strip() or None
    intent.updated_at           = datetime.datetime.utcnow()

    await db.commit()
    await db.refresh(intent)

    return templates.TemplateResponse("partials/intent_saved.html", {
        "request": request,
        "intent": intent,
    })


@router.get("/context", response_class=JSONResponse)
async def intent_context(db: AsyncSession = Depends(get_db)):
    """Returns intent as a plain-English AI context string."""
    intent = await get_or_create_intent(db)
    return {"context": intent.to_ai_context()}