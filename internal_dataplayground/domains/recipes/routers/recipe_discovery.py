# routers/recipe_discovery.py
"""
Recipe Discovery

Endpoints:
  GET  /recipes/discover               → Discovery page (two-mode tab interface)
  POST /recipes/discover/pantry        → Pantry-based suggestions
  POST /recipes/discover/open          → Open discovery (mood + preferences)
  POST /recipes/discover/save          → Save an AI suggestion to the library
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.recipes.models import (
    Ingredient, PantryItem, Recipe, RecipeDifficulty,
    RecipeMealType, RecipeSourceType,
)
from services.recipe_service import run_normalization_pipeline
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes/discover", tags=["Recipe Discovery"])


@router.get("", response_class=HTMLResponse)
async def discovery_page(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Discovery landing page.
    Shows pantry item count so the user knows if pantry mode is meaningful.
    """
    pantry_count_result = await db.execute(select(func.count(PantryItem.id)))
    pantry_count = pantry_count_result.scalar() or 0

    return templates.TemplateResponse("recipe_discover.html", {
        "request": request,
        "active_module": "recipes",
        "pantry_count": pantry_count,
        "meal_types": [e.value for e in RecipeMealType],
    })


@router.post("/pantry", response_class=HTMLResponse)
async def discover_from_pantry(
    request: Request,
    db: AsyncSession = Depends(get_db),
    mood: str = Form(default=""),
    meal_type: str = Form(default=""),
    servings: int = Form(default=0),
    occasion: str = Form(default=""),
    match_threshold: str = Form(default="relaxed"),  # "exact" or "relaxed"
):
    """
    Pantry-based discovery.

    Step 1: Query the DB for recipes where the user already has most/all ingredients.
    Step 2: Call Gemini to suggest additional recipes using the pantry list.

    match_threshold controls the DB query:
      "exact"   → only show recipes where 100% of ingredients are in pantry
      "relaxed" → show recipes where ≥80% of ingredients are in pantry
    """
    # Get pantry ingredient names
    pantry_result = await db.execute(
        select(Ingredient.name)
        .join(PantryItem, PantryItem.ingredient_id == Ingredient.id)
        .order_by(Ingredient.name)
    )
    pantry_names = [row[0] for row in pantry_result.all()]

    if not pantry_names:
        return templates.TemplateResponse(
            "partials/discovery_results.html",
            {
                "request": request,
                "db_matches": [],
                "ai_suggestions": [],
                "error": "Your pantry is empty. Add some ingredients in the Pantry section first.",
                "mode": "pantry",
            },
        )

    # ── DB matching ───────────────────────────────────────────────────────────
    # For each non-archived recipe, compute what fraction of its ingredients
    # are in the pantry. This raw SQL is cleaner than trying to express it
    # in SQLAlchemy ORM for this specific aggregation query.
    from sqlalchemy import text
    pantry_lower = [name.lower() for name in pantry_names]

    # Build the pantry set as a SQL IN clause
    placeholders = ", ".join(f":p{i}" for i in range(len(pantry_lower)))
    params = {f"p{i}": name for i, name in enumerate(pantry_lower)}
    params["threshold"] = 0.80 if match_threshold == "relaxed" else 1.0

    match_sql = text(f"""
        SELECT
            r.id,
            COUNT(ri.id)                                    AS total_ingredients,
            SUM(CASE
                WHEN LOWER(ing.name) IN ({placeholders}) THEN 1
                ELSE 0
            END)                                            AS matched_ingredients,
            SUM(CASE
                WHEN LOWER(ing.name) IN ({placeholders}) THEN 1
                ELSE 0
            END) / COUNT(ri.id)                             AS match_ratio
        FROM recipes r
        JOIN recipe_ingredients ri ON ri.recipe_id = r.id
        JOIN ingredients ing ON ing.id = ri.ingredient_id
        WHERE r.is_archived = 0
        GROUP BY r.id
        HAVING match_ratio >= :threshold
        ORDER BY match_ratio DESC
        LIMIT 10
    """)

    try:
        match_result = await db.execute(match_sql, params)
        match_rows = match_result.mappings().all()
        matched_recipe_ids = [row["id"] for row in match_rows]
        match_ratios = {row["id"]: float(row["match_ratio"]) for row in match_rows}
    except Exception as exc:
        log.warning("Pantry match query failed: %s", exc)
        matched_recipe_ids = []
        match_ratios = {}

    db_matches = []
    if matched_recipe_ids:
        recipes_result = await db.execute(
            select(Recipe).where(Recipe.id.in_(matched_recipe_ids))
        )
        db_matches_raw = recipes_result.scalars().all()
        # Sort by match ratio and attach it for template display
        db_matches = sorted(
            db_matches_raw,
            key=lambda r: match_ratios.get(r.id, 0),
            reverse=True,
        )
        for r in db_matches:
            r._match_pct = int(match_ratios.get(r.id, 0) * 100)

    # ── AI suggestions ────────────────────────────────────────────────────────
    from airflow.agents.recipe_agents import agent_discover_recipes_pantry
    try:
        ai_suggestions = agent_discover_recipes_pantry(
            pantry_ingredients=pantry_names,
            mood=mood,
            meal_type=meal_type,
            servings=servings,
            occasion=occasion,
        )
    except Exception as exc:
        log.error("Pantry discovery AI call failed: %s", exc)
        ai_suggestions = []

    return templates.TemplateResponse(
        "partials/discovery_results.html",
        {
            "request": request,
            "db_matches": db_matches,
            "ai_suggestions": ai_suggestions,
            "error": None,
            "mode": "pantry",
            "match_threshold": match_threshold,
        },
    )


@router.post("/open", response_class=HTMLResponse)
async def discover_open(
    request: Request,
    mood: str = Form(default=""),
    meal_type: str = Form(default=""),
    servings: int = Form(default=0),
    occasion: str = Form(default=""),
    dietary_restrictions: str = Form(default=""),
    cuisine_preference: str = Form(default=""),
):
    """Open discovery — no pantry required. Pure AI suggestions."""
    from airflow.agents.recipe_agents import agent_discover_recipes_open
    try:
        ai_suggestions = agent_discover_recipes_open(
            mood=mood,
            meal_type=meal_type,
            servings=servings,
            occasion=occasion,
            dietary_restrictions=dietary_restrictions,
            cuisine_preference=cuisine_preference,
        )
    except Exception as exc:
        log.error("Open discovery failed: %s", exc)
        ai_suggestions = []

    return templates.TemplateResponse(
        "partials/discovery_results.html",
        {
            "request": request,
            "db_matches": [],
            "ai_suggestions": ai_suggestions,
            "error": None,
            "mode": "open",
        },
    )


@router.post("/save", response_class=HTMLResponse)
async def save_discovery_suggestion(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Saves an AI-generated recipe suggestion to the library.
    The suggestion data arrives as form fields (mirroring the discovery
    result JSON serialized into individual form inputs by the template).
    Runs the full normalization pipeline before saving.
    """
    form = await request.form()

    def _get(key: str) -> str:
        return str(form.get(key, "")).strip()

    def _get_int(key: str) -> Optional[int]:
        v = _get(key)
        try:
            return int(v) if v else None
        except ValueError:
            return None

    meal_type_enum = None
    try:
        if _get("meal_type"):
            meal_type_enum = RecipeMealType(_get("meal_type"))
    except ValueError:
        pass

    difficulty_enum = None
    try:
        if _get("difficulty"):
            difficulty_enum = RecipeDifficulty(_get("difficulty"))
    except ValueError:
        pass

    recipe = Recipe(
        title=_get("title") or "AI Suggested Recipe",
        source_type=RecipeSourceType.AI_GENERATED,
        cuisine=_get("cuisine") or None,
        meal_type=meal_type_enum,
        prep_time_minutes=_get_int("prep_time_minutes"),
        cook_time_minutes=_get_int("cook_time_minutes"),
        servings=_get_int("servings"),
        difficulty=difficulty_enum,
        instructions=_get("instructions") or None,
        notes=_get("description") or None,
    )
    db.add(recipe)
    await db.flush()

    ingredient_lines = [
        line.strip()
        for line in _get("raw_ingredients").split("\n")
        if line.strip()
    ]
    await run_normalization_pipeline(db, recipe, ingredient_lines)
    await db.commit()

    return templates.TemplateResponse(
        "partials/discovery_save_result.html",
        {"request": request, "recipe": recipe},
    )
