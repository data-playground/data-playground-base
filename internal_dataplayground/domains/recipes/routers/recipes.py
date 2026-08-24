# routers/recipes.py
"""
Recipe Library — Core CRUD

Endpoints:
  GET    /recipes                        → Recipe library (card grid with filters)
  GET    /recipes/{id}                   → Full recipe detail view
  POST   /recipes                        → Create recipe manually (form data)
  PATCH  /recipes/{id}                   → Update recipe metadata
  DELETE /recipes/{id}                   → Soft-delete (archive)
  PATCH  /recipes/{id}/rate              → Set user_rating (1-5)
  PATCH  /recipes/{id}/favorite          → Toggle is_favorite
  POST   /recipes/{id}/cook              → Log a cook: increment counter + update date
  GET    /recipes/tags                   → All tags as JSON (for autocomplete)
  GET    /recipes/ingredients/suggest    → Ingredient name autocomplete
"""

import logging
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.recipes.models import (
    Ingredient, IngredientCategory, Recipe, RecipeDifficulty,
    RecipeMealType, RecipeSourceType, RecipeTag,
)
from services.recipe_service import run_normalization_pipeline
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes", tags=["Recipes"])


# ── Library view ───────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def recipe_library(
    request: Request,
    db: AsyncSession = Depends(get_db),
    q: Optional[str] = Query(default=None),
    meal_type: Optional[str] = Query(default=None),
    cuisine: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    rating: Optional[int] = Query(default=None),
    favorites_only: bool = Query(default=False),
):
    stmt = (
        select(Recipe)
        .where(Recipe.is_archived == False)
        .order_by(desc(Recipe.updated_at))
    )

    if q:
        stmt = stmt.where(Recipe.title.ilike(f"%{q}%"))
    if meal_type:
        try:
            stmt = stmt.where(Recipe.meal_type == RecipeMealType(meal_type))
        except ValueError:
            pass
    if cuisine:
        stmt = stmt.where(Recipe.cuisine.ilike(f"%{cuisine}%"))
    if rating:
        stmt = stmt.where(Recipe.user_rating >= rating)
    if favorites_only:
        stmt = stmt.where(Recipe.is_favorite == True)
    if tag:
        stmt = stmt.join(Recipe.tags).where(func.lower(RecipeTag.name) == tag.lower())

    result = await db.execute(stmt)
    recipes = result.scalars().all()

    # Distinct cuisines for filter dropdown
    cuisines_result = await db.execute(
        select(Recipe.cuisine).where(Recipe.cuisine != None, Recipe.is_archived == False).distinct()
    )
    cuisines = sorted([r[0] for r in cuisines_result.all() if r[0]])

    return templates.TemplateResponse("recipes.html", {
        "request": request,
        "recipes": recipes,
        "active_module": "recipes",
        "q": q or "",
        "sel_meal_type": meal_type or "",
        "sel_cuisine": cuisine or "",
        "sel_tag": tag or "",
        "sel_rating": rating or 0,
        "favorites_only": favorites_only,
        "cuisines": cuisines,
        "meal_types": [e.value for e in RecipeMealType],
    })


# ── Recipe detail ──────────────────────────────────────────────────────────────

@router.get("/tags", response_class=JSONResponse)
async def list_tags(db: AsyncSession = Depends(get_db)):
    """All tags as JSON for autocomplete. Must be before /{id} to avoid conflict."""
    result = await db.execute(select(RecipeTag).order_by(RecipeTag.name))
    tags = result.scalars().all()
    return [{"id": t.id, "name": t.name} for t in tags]


@router.get("/ingredients/suggest", response_class=JSONResponse)
async def suggest_ingredients(
    q: str = Query(default=""),
    db: AsyncSession = Depends(get_db),
):
    """Ingredient name autocomplete for the pantry add form."""
    if len(q) < 2:
        return []
    result = await db.execute(
        select(Ingredient)
        .where(Ingredient.name.ilike(f"%{q}%"))
        .order_by(Ingredient.name)
        .limit(10)
    )
    ingredients = result.scalars().all()
    return [{"id": i.id, "name": i.name, "category": i.category.value} for i in ingredients]


@router.get("/{recipe_id}", response_class=HTMLResponse)
async def recipe_detail(
    recipe_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    recipe = await db.get(Recipe, recipe_id)
    if not recipe or recipe.is_archived:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return templates.TemplateResponse("recipe_detail.html", {
        "request": request,
        "recipe": recipe,
        "active_module": "recipes",
    })


# ── Create recipe (manual) ─────────────────────────────────────────────────────

@router.post("", response_class=HTMLResponse)
async def create_recipe(
    request: Request,
    db: AsyncSession = Depends(get_db),
    title: str = Form(...),
    source_url: Optional[str] = Form(None),
    cuisine: Optional[str] = Form(None),
    meal_type: Optional[str] = Form(None),
    prep_time_minutes: Optional[int] = Form(None),
    cook_time_minutes: Optional[int] = Form(None),
    servings: Optional[int] = Form(None),
    difficulty: Optional[str] = Form(None),
    instructions: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    raw_ingredients: Optional[str] = Form(None),  # Newline-separated raw ingredient lines
    tags: Optional[str] = Form(None),             # Comma-separated tag names
):
    meal_type_enum = None
    if meal_type:
        try:
            meal_type_enum = RecipeMealType(meal_type)
        except ValueError:
            pass

    difficulty_enum = None
    if difficulty:
        try:
            difficulty_enum = RecipeDifficulty(difficulty)
        except ValueError:
            pass

    recipe = Recipe(
        title=title.strip(),
        source_url=source_url or None,
        source_type=RecipeSourceType.MANUAL,
        cuisine=cuisine or None,
        meal_type=meal_type_enum,
        prep_time_minutes=prep_time_minutes,
        cook_time_minutes=cook_time_minutes,
        servings=servings,
        difficulty=difficulty_enum,
        instructions=instructions or None,
        notes=notes or None,
        image_url=image_url or None,
    )
    db.add(recipe)
    await db.flush()  # Get recipe.id for normalization pipeline

    # Parse raw ingredient lines and tag names
    ingredient_lines = [
        line.strip()
        for line in (raw_ingredients or "").split("\n")
        if line.strip()
    ]
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    await run_normalization_pipeline(db, recipe, ingredient_lines, tag_list)
    await db.commit()
    await db.refresh(recipe)

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/recipes/{recipe.id}", status_code=303)


# ── Update recipe metadata ─────────────────────────────────────────────────────

@router.patch("/{recipe_id}", response_class=HTMLResponse)
async def update_recipe(
    recipe_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Updates recipe metadata fields.
    Does NOT update ingredients — ingredient lists are replaced by deleting
    existing RecipeIngredient rows and re-running the normalization pipeline.
    The form must include a 'replace_ingredients' checkbox to trigger this.
    """
    form = await request.form()
    recipe = await db.get(Recipe, recipe_id)
    if not recipe or recipe.is_archived:
        raise HTTPException(status_code=404)

    if form.get("title"):
        recipe.title = str(form["title"]).strip()
    if "cuisine" in form:
        recipe.cuisine = str(form["cuisine"]).strip() or None
    if "meal_type" in form:
        try:
            recipe.meal_type = RecipeMealType(str(form["meal_type"]))
        except ValueError:
            recipe.meal_type = None
    if "prep_time_minutes" in form:
        recipe.prep_time_minutes = int(form["prep_time_minutes"]) if form["prep_time_minutes"] else None
    if "cook_time_minutes" in form:
        recipe.cook_time_minutes = int(form["cook_time_minutes"]) if form["cook_time_minutes"] else None
    if "servings" in form:
        recipe.servings = int(form["servings"]) if form["servings"] else None
    if "instructions" in form:
        recipe.instructions = str(form["instructions"]) or None
    if "notes" in form:
        recipe.notes = str(form["notes"]) or None
    if "image_url" in form:
        recipe.image_url = str(form["image_url"]).strip() or None
    if "difficulty" in form:
        try:
            recipe.difficulty = RecipeDifficulty(str(form["difficulty"]))
        except ValueError:
            recipe.difficulty = None

    # Optionally replace ingredients
    if form.get("replace_ingredients") and form.get("raw_ingredients"):
        from domains.recipes.models import RecipeIngredient
        await db.execute(
            __import__("sqlalchemy", fromlist=["delete"]).delete(RecipeIngredient)
            .where(RecipeIngredient.recipe_id == recipe_id)
        )
        ingredient_lines = [
            line.strip()
            for line in str(form["raw_ingredients"]).split("\n")
            if line.strip()
        ]
        tag_list = [t.strip() for t in str(form.get("tags", "")).split(",") if t.strip()]
        recipe.tags = []  # Clear tags before re-running pipeline
        await db.flush()
        await run_normalization_pipeline(db, recipe, ingredient_lines, tag_list)

    recipe.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(recipe)

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/recipes/{recipe.id}", status_code=303)


# ── Soft delete ────────────────────────────────────────────────────────────────

@router.delete("/{recipe_id}", response_class=HTMLResponse)
async def delete_recipe(
    recipe_id: int,
    db: AsyncSession = Depends(get_db),
):
    recipe = await db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(status_code=404)
    recipe.is_archived = True
    recipe.updated_at = datetime.utcnow()
    await db.commit()
    return HTMLResponse("")  # HTMX removes the card


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
