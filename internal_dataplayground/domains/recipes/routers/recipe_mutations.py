# routers/recipe_mutations.py
"""
Recipe Mutations — small single-purpose endpoints

Split out of recipes.py (WO#26 Part B). Each endpoint does one thing and
returns a tiny HTMX partial.

Endpoints:
  PATCH  /recipes/{id}/rate              → Set user_rating (1-5)
  PATCH  /recipes/{id}/favorite          → Toggle is_favorite
  POST   /recipes/{id}/cook              → Log a cook: increment counter + update date
"""

import logging
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.recipes.models import Recipe
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes", tags=["Recipes"])


# ── Rate ───────────────────────────────────────────────────────────────────────

@router.patch("/{recipe_id}/rate", response_class=HTMLResponse)
async def rate_recipe(
    recipe_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    rating = int(form.get("rating", 0))
    if not 1 <= rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be 1-5")

    recipe = await db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(status_code=404)

    recipe.user_rating = rating
    recipe.updated_at = datetime.utcnow()
    await db.commit()

    # Return updated rating stars partial
    return templates.TemplateResponse(
        "partials/recipe_rating.html",
        {"request": request, "recipe": recipe},
    )


# ── Favorite toggle ────────────────────────────────────────────────────────────

@router.patch("/{recipe_id}/favorite", response_class=HTMLResponse)
async def toggle_favorite(
    recipe_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    recipe = await db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(status_code=404)

    recipe.is_favorite = not recipe.is_favorite
    recipe.updated_at = datetime.utcnow()
    await db.commit()

    return templates.TemplateResponse(
        "partials/recipe_favorite.html",
        {"request": request, "recipe": recipe},
    )


# ── Cook logger ────────────────────────────────────────────────────────────────

@router.post("/{recipe_id}/cook", response_class=HTMLResponse)
async def log_cook(
    recipe_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Increments times_cooked and sets last_cooked_at to today.
    No per-cook log table — simple counter model.
    """
    recipe = await db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(status_code=404)

    recipe.times_cooked = (recipe.times_cooked or 0) + 1
    recipe.last_cooked_at = date.today()
    recipe.updated_at = datetime.utcnow()
    await db.commit()

    return templates.TemplateResponse(
        "partials/recipe_cook_count.html",
        {"request": request, "recipe": recipe},
    )
