# services/recipe_service.py
"""
Shared recipe normalization pipeline.

This service is the single place where raw extracted recipe data
(from URL, PDF, image, manual form, or AI discovery) gets converted
into proper ORM objects and saved to the database.

All recipe routers call _run_normalization_pipeline() — never duplicate
the ingredient normalization logic across routers.

Pipeline steps:
  1. Call agent_normalize_ingredients() on raw_ingredient_lines.
  2. For each normalized ingredient:
       a. Look up ingredients table by canonical_name (case-insensitive).
       b. If missing, insert a new Ingredient row with the category.
  3. Build RecipeIngredient rows with ingredient_id, quantity, unit,
     preparation_note, is_optional, sort_order.
  4. Handle tags: find or create RecipeTag rows, wire up the junction.
  5. Return the saved Recipe ORM object.
"""

import logging
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


async def run_normalization_pipeline(
    db: AsyncSession,
    recipe,            # Recipe ORM object (already added to session, not committed)
    raw_ingredient_lines: list[str],
    tag_names: list[str] | None = None,
) -> None:
    """
    Normalizes raw ingredient strings and saves RecipeIngredient rows.
    Mutates the database session — caller must commit after this returns.

    Args:
        db:                    Async SQLAlchemy session.
        recipe:                Recipe ORM object already added to the session.
                               Must have recipe.id available (flush before calling).
        raw_ingredient_lines:  Raw ingredient strings from extraction or manual input.
        tag_names:             Optional list of tag name strings to associate.
    """
    # Deferred import to avoid circular deps and keep agents testable standalone
    from airflow.agents.recipe_agents import agent_normalize_ingredients
    from domains.recipes.models import Ingredient, IngredientCategory, RecipeIngredient, IngredientUnit, RecipeTag

    # ── Step 1: normalize ─────────────────────────────────────────────────────
    if not raw_ingredient_lines:
        log.info("No ingredient lines to normalize for recipe %s", recipe.id)
        return

    normalized = agent_normalize_ingredients(raw_ingredient_lines)
    log.info("Normalized %d ingredients for recipe %s", len(normalized), recipe.id)

    # ── Step 2 & 3: upsert ingredients + build RecipeIngredient rows ──────────
    for sort_idx, norm in enumerate(normalized):
        canonical = norm["canonical_name"].strip().lower()
        if not canonical:
            continue

        # Look up existing ingredient (case-insensitive match)
        result = await db.execute(
            select(Ingredient).where(func.lower(Ingredient.name) == canonical)
        )
        ingredient = result.scalar_one_or_none()

        if not ingredient:
            # New ingredient — insert with the category from normalization
            category_str = norm.get("category", "other")
            try:
                category = IngredientCategory(category_str)
            except ValueError:
                category = IngredientCategory.OTHER

            ingredient = Ingredient(
                name=canonical,  # Store in lowercase for consistency
                category=category,
            )
            db.add(ingredient)
            await db.flush()  # Get ingredient.id without committing
            log.debug("New ingredient inserted: '%s' (%s)", canonical, category_str)

        # Resolve unit enum
        unit = None
        unit_str = norm.get("unit")
        if unit_str:
            try:
                unit = IngredientUnit(unit_str)
            except ValueError:
                unit = None

        # Build the RecipeIngredient row
        qty_raw = norm.get("quantity")
        qty = float(qty_raw) if qty_raw is not None else None

        ri = RecipeIngredient(
            recipe_id=recipe.id,
            ingredient_id=ingredient.id,
            quantity=qty,
            unit=unit,
            preparation_note=norm.get("preparation_note"),
            is_optional=bool(norm.get("is_optional", False)),
            sort_order=sort_idx,
        )
        db.add(ri)

    # ── Step 4: tags ──────────────────────────────────────────────────────────
    if tag_names:
        for tag_name in tag_names:
            tag_name = tag_name.strip().lower()
            if not tag_name:
                continue
            result = await db.execute(
                select(RecipeTag).where(func.lower(RecipeTag.name) == tag_name)
            )
            tag = result.scalar_one_or_none()
            if not tag:
                tag = RecipeTag(name=tag_name)
                db.add(tag)
                await db.flush()
            if tag not in recipe.tags:
                recipe.tags.append(tag)

    log.info(
        "Normalization pipeline complete for recipe %s: %d ingredients, %d tags",
        recipe.id,
        len(normalized),
        len(tag_names or []),
    )
