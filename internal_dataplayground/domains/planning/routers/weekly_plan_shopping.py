# domains/planning/routers/weekly_plan_shopping.py
"""
Weekly Planning System — Shopping list generation & view.

Endpoints:
  GET  /plan/{plan_id}/shopping  → Shopping list view

Split out of weekly_plan.py after the original migration, at the
project owner's request, for the same reason weekly_plan_generator.py
was split out: keeps weekly_plan.py's line count down and groups the
shopping-list responsibility (generation + display) in one place,
separate from day/meal CRUD and separate from AI generation.

`_generate_shopping_list()` is called by `confirm_plan()` in the
sibling `weekly_plan.py` module — imported directly, not duplicated.
Shares the `/plan` prefix with the other two planning routers; this is
safe since `GET /{plan_id}/shopping` (2 path segments) never collides
with any route those routers register (verified — see this domain's
work-order report for the full route enumeration).
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.planning.models import ShoppingList, WeeklyPlan
from domains.recipes.models import Ingredient, PantryItem

router = APIRouter(prefix="/plan", tags=["Weekly Plan"])

from core.templating import templates


# ── Shopping list ──────────────────────────────────────────────────────────────

async def _generate_shopping_list(
    db: AsyncSession,
    plan: WeeklyPlan,
    recipe_ids: list[int],
) -> ShoppingList:
    """
    Aggregates ingredients from all planned recipes, deducts pantry items,
    and groups by category. Saves as a ShoppingList row.
    """
    from domains.recipes.models import RecipeIngredient

    if not recipe_ids:
        sl = ShoppingList(weekly_plan_id=plan.id, items=[])
        db.add(sl)
        await db.commit()
        return sl

    # Fetch all recipe ingredients for planned recipes
    ri_result = await db.execute(
        select(RecipeIngredient)
        .where(RecipeIngredient.recipe_id.in_(set(recipe_ids)))
    )
    all_ris = ri_result.scalars().all()

    # Fetch pantry
    pantry_result = await db.execute(
        select(Ingredient.name)
        .join(PantryItem, PantryItem.ingredient_id == Ingredient.id)
    )
    pantry_names = {row[0].lower() for row in pantry_result.all()}

    # Aggregate by ingredient
    aggregated: dict[str, dict] = {}
    for ri in all_ris:
        if not ri.ingredient:
            continue
        key = ri.ingredient.name.lower()
        if key not in aggregated:
            aggregated[key] = {
                "ingredient": ri.ingredient.name,
                "category":   ri.ingredient.category.value,
                "quantity":   float(ri.quantity) if ri.quantity else None,
                "unit":       ri.unit.value if ri.unit else None,
                "in_pantry":  key in pantry_names,
            }
        elif ri.quantity and aggregated[key]["quantity"]:
            aggregated[key]["quantity"] += float(ri.quantity)

    # Sort by category then name
    items = sorted(
        aggregated.values(),
        key=lambda x: (x["in_pantry"], x["category"], x["ingredient"])
    )

    # Delete existing shopping list if regenerating
    existing_result = await db.execute(
        select(ShoppingList).where(ShoppingList.weekly_plan_id == plan.id)
    )
    existing_sl = existing_result.scalar_one_or_none()
    if existing_sl:
        await db.delete(existing_sl)
        await db.flush()

    sl = ShoppingList(weekly_plan_id=plan.id, items=items)
    db.add(sl)
    await db.commit()
    return sl


@router.get("/{plan_id}/shopping", response_class=HTMLResponse)
async def shopping_list_view(
    plan_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    plan = await db.get(WeeklyPlan, plan_id)
    if not plan:
        raise HTTPException(status_code=404)

    sl_result = await db.execute(
        select(ShoppingList).where(ShoppingList.weekly_plan_id == plan_id)
    )
    shopping_list = sl_result.scalar_one_or_none()

    # Group items by category for display
    grouped: dict[str, list] = {}
    if shopping_list:
        for item in shopping_list.items:
            cat = item.get("category", "other")
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(item)

    need_items   = []
    pantry_items = []
    need_grouped = {}  # {category: [item, ...]}

    if shopping_list:
        for item in shopping_list.items:
            if item.get("in_pantry"):
                pantry_items.append(item)
            else:
                cat = item.get("category", "other")
                if cat not in need_grouped:
                    need_grouped[cat] = []
                need_grouped[cat].append(item)
        # Sort categories alphabetically
        need_grouped = dict(sorted(need_grouped.items()))

    return templates.TemplateResponse("shopping_list.html", {
        "request":      request,
        "plan":         plan,
        "shopping_list": shopping_list,
        "need_grouped": need_grouped,   # {category: [items to buy]}
        "pantry_items": pantry_items,   # [items already in pantry]
        "total":        len(shopping_list.items) if shopping_list else 0,
        "to_buy":       len(need_items) + sum(len(v) for v in need_grouped.values()),
        "in_pantry":    len(pantry_items),
    })
