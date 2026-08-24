# routers/pantry.py
"""
Pantry Management — minimal add/remove interface

Endpoints:
  GET    /pantry                         → Pantry view grouped by category
  POST   /pantry                         → Add ingredient to pantry
  DELETE /pantry/{ingredient_id}         → Remove ingredient from pantry
  GET    /pantry/suggest                 → Ingredient autocomplete (JSON)
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.recipes.models import Ingredient, IngredientCategory, PantryItem
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/pantry", tags=["Pantry"])


async def _get_pantry_grouped(db: AsyncSession) -> dict:
    """
    Returns pantry items grouped by ingredient category.
    Returns a dict: {category_value: [PantryItem, ...]}
    """
    result = await db.execute(
        select(PantryItem)
        .join(Ingredient, Ingredient.id == PantryItem.ingredient_id)
        .order_by(Ingredient.category, Ingredient.name)
    )
    items = result.scalars().all()

    grouped = {}
    for item in items:
        cat = item.ingredient.category.value
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(item)

    return grouped


@router.get("", response_class=HTMLResponse)
async def pantry_view(request: Request, db: AsyncSession = Depends(get_db)):
    grouped = await _get_pantry_grouped(db)
    categories = [e.value for e in IngredientCategory]
    return templates.TemplateResponse("pantry.html", {
        "request": request,
        "active_module": "pantry",
        "grouped": grouped,
        "categories": categories,
        "total_items": sum(len(v) for v in grouped.values()),
    })


@router.post("", response_class=HTMLResponse)
async def add_to_pantry(
    request: Request,
    db: AsyncSession = Depends(get_db),
    ingredient_name: str = Form(...),
):
    """
    Adds an ingredient to the pantry by name.
    If the ingredient doesn't exist in the ingredients table yet,
    creates it with category='other' (the user can update category later
    or the next recipe import will update it via normalization).
    If the ingredient is already in the pantry, this is a no-op.
    """
    canonical = ingredient_name.strip().lower()
    if not canonical:
        raise HTTPException(status_code=422, detail="Ingredient name required")

    # Find or create ingredient
    result = await db.execute(
        select(Ingredient).where(func.lower(Ingredient.name) == canonical)
    )
    ingredient = result.scalar_one_or_none()

    if not ingredient:
        ingredient = Ingredient(
            name=canonical,
            category=IngredientCategory.OTHER,
        )
        db.add(ingredient)
        await db.flush()

    # Check if already in pantry
    existing = await db.execute(
        select(PantryItem).where(PantryItem.ingredient_id == ingredient.id)
    )
    if not existing.scalar_one_or_none():
        pantry_item = PantryItem(ingredient_id=ingredient.id)
        db.add(pantry_item)
        await db.commit()

    grouped = await _get_pantry_grouped(db)
    categories = [e.value for e in IngredientCategory]
    return templates.TemplateResponse(
        "partials/pantry_list.html",
        {
            "request": request,
            "grouped": grouped,
            "categories": categories,
            "total_items": sum(len(v) for v in grouped.values()),
            "toast": f"Added {canonical} to pantry.",
        },
    )


@router.delete("/{ingredient_id}", response_class=HTMLResponse)
async def remove_from_pantry(
    ingredient_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PantryItem).where(PantryItem.ingredient_id == ingredient_id)
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404)

    await db.delete(item)
    await db.commit()

    grouped = await _get_pantry_grouped(db)
    categories = [e.value for e in IngredientCategory]
    return templates.TemplateResponse(
        "partials/pantry_list.html",
        {
            "request": request,
            "grouped": grouped,
            "categories": categories,
            "total_items": sum(len(v) for v in grouped.values()),
        },
    )


@router.get("/suggest", response_class=JSONResponse)
async def suggest_pantry_ingredient(
    q: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Autocomplete for the pantry add form — searches the ingredients table."""
    if len(q) < 2:
        return []
    result = await db.execute(
        select(Ingredient)
        .where(Ingredient.name.ilike(f"%{q.lower()}%"))
        .order_by(Ingredient.name)
        .limit(10)
    )
    ingredients = result.scalars().all()
    return [{"id": i.id, "name": i.name, "category": i.category.value} for i in ingredients]
