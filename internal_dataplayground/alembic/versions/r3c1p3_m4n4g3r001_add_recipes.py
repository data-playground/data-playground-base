"""add recipe manager tables

Revision ID: r3c1p3_m4n4g3r001
Revises: 0ecac75145ac
Create Date: 2026-05-15

Creates:
  - ingredients          (normalized ingredient reference table)
  - recipe_tags          (tag reference table)
  - recipes              (core recipe table)
  - recipe_tags_junction (many-to-many: recipes ↔ tags)
  - recipe_ingredients   (structured ingredient list per recipe)
  - pantry_items         (minimal pantry: just which ingredients you have)

Design decisions recorded here:
  - recipe_cook_log was explicitly excluded — cook history is tracked via
    recipes.times_cooked (INT) and recipes.last_cooked_at (DATE) only.
    A full log table can be added in a later migration if needed.
  - pantry_items is minimal by design: no quantity, no unit, no expiry.
    The ingredient_id UNIQUE constraint means one row per ingredient.
    Quantities and expiry can be added via ALTER TABLE in a future migration.
  - images are stored as source_url VARCHAR only — no local file storage.
  - ingredient normalization uses a single batched Gemini call (not two
    separate calls) — the agent functions are defined separately in
    airflow/agents/recipe_agents.py to allow future splitting.

TODO (Playwright): URL extraction currently uses requests + Schema.org
  JSON-LD parsing with a Gemini fallback. When headless browser support
  is added, the integration point is routers/recipe_extract.py in the
  _fetch_url_content() helper function. Add Playwright as a secondary
  strategy after the requests attempt fails on JS-heavy sites.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'r3c1p3_m4n4g3r001'
down_revision: Union[str, None] = '0ecac75145ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── ingredients — normalized reference table ───────────────────────────
    # Every unique ingredient lives here exactly once.
    # "garlic", "minced garlic", "garlic cloves" all resolve to "garlic" here.
    # New rows are added automatically during recipe import via the
    # normalization agent — never inserted manually.
    op.create_table(
        'ingredients',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(150), nullable=False, unique=True),
        sa.Column(
            'category',
            sa.Enum(
                'produce', 'protein', 'dairy', 'grain', 'pantry',
                'spice', 'condiment', 'beverage', 'frozen', 'other',
                name='ingredientcategory',
            ),
            nullable=False,
            server_default='other',
        ),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_ingredients_name', 'ingredients', ['name'])

    # ── recipe_tags — tag reference table ─────────────────────────────────
    op.create_table(
        'recipe_tags',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(50), nullable=False, unique=True),
    )

    # ── recipes — core recipe table ───────────────────────────────────────
    op.create_table(
        'recipes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('source_url', sa.String(1000), nullable=True),
        sa.Column(
            'source_type',
            sa.Enum('manual', 'url', 'pdf', 'image', 'ai_generated', name='recipesourcetype'),
            nullable=False,
            server_default='manual',
        ),
        sa.Column('cuisine', sa.String(100), nullable=True),
        sa.Column(
            'meal_type',
            sa.Enum(
                'breakfast', 'lunch', 'dinner', 'snack',
                'dessert', 'side', 'drink', 'other',
                name='recipemealtype',
            ),
            nullable=True,
        ),
        sa.Column('prep_time_minutes', sa.Integer(), nullable=True),
        sa.Column('cook_time_minutes', sa.Integer(), nullable=True),
        sa.Column('total_time_minutes', sa.Integer(), nullable=True),
        sa.Column('servings', sa.SmallInteger(), nullable=True),
        sa.Column(
            'difficulty',
            sa.Enum('easy', 'medium', 'hard', name='recipedifficulty'),
            nullable=True,
        ),
        # Markdown-formatted instructions — numbered steps preferred.
        sa.Column('instructions', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        # External image URL only — no local storage.
        sa.Column('image_url', sa.String(1000), nullable=True),
        # User's personal 1-5 rating — separate from any per-cook rating.
        sa.Column('user_rating', sa.SmallInteger(), nullable=True),
        # Cook tracking — intentionally simple (no log table).
        sa.Column('times_cooked', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_cooked_at', sa.Date(), nullable=True),
        sa.Column('is_favorite', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index('ix_recipes_meal_type', 'recipes', ['meal_type'])
    op.create_index('ix_recipes_cuisine', 'recipes', ['cuisine'])
    op.create_index('ix_recipes_user_rating', 'recipes', ['user_rating'])
    op.create_index('ix_recipes_is_favorite', 'recipes', ['is_favorite'])
    op.create_index('ix_recipes_is_archived', 'recipes', ['is_archived'])

    # ── recipe_tags_junction — many-to-many: recipes ↔ tags ───────────────
    op.create_table(
        'recipe_tags_junction',
        sa.Column(
            'recipe_id',
            sa.Integer(),
            sa.ForeignKey('recipes.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'tag_id',
            sa.Integer(),
            sa.ForeignKey('recipe_tags.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('recipe_id', 'tag_id'),
    )

    # ── recipe_ingredients — structured ingredient rows per recipe ─────────
    # quantity is DECIMAL NULL — NULL means "to taste" or "as needed".
    # preparation_note holds the prep method separate from the ingredient name,
    # e.g. "finely diced", "at room temperature", "roughly chopped".
    op.create_table(
        'recipe_ingredients',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            'recipe_id',
            sa.Integer(),
            sa.ForeignKey('recipes.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
        ),
        sa.Column(
            'ingredient_id',
            sa.Integer(),
            sa.ForeignKey('ingredients.id'),
            nullable=False,
            index=True,
        ),
        sa.Column('quantity', sa.Numeric(8, 3), nullable=True),
        sa.Column(
            'unit',
            sa.Enum(
                'cup', 'tbsp', 'tsp', 'ml', 'l', 'g', 'kg',
                'oz', 'lb', 'piece', 'clove', 'bunch', 'slice',
                'can', 'package', 'to_taste', 'as_needed', 'pinch', 'handful',
                name='ingredientunit',
            ),
            nullable=True,
        ),
        sa.Column('preparation_note', sa.String(150), nullable=True),
        sa.Column('is_optional', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('sort_order', sa.SmallInteger(), nullable=False, server_default='0'),
    )
    op.create_index(
        'ix_recipe_ingredients_recipe', 'recipe_ingredients', ['recipe_id']
    )
    op.create_index(
        'ix_recipe_ingredients_ingredient', 'recipe_ingredients', ['ingredient_id']
    )

    # ── pantry_items — minimal pantry: just which ingredients you have ─────
    # Intentionally minimal by design: no quantity, no unit, no expiry.
    # The UNIQUE constraint on ingredient_id means one row per ingredient.
    # Future migration can ADD COLUMN quantity, unit, expires_at if needed.
    op.create_table(
        'pantry_items',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            'ingredient_id',
            sa.Integer(),
            sa.ForeignKey('ingredients.id', ondelete='CASCADE'),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )
    op.create_index('ix_pantry_ingredient', 'pantry_items', ['ingredient_id'])


def downgrade() -> None:
    op.drop_index('ix_pantry_ingredient', table_name='pantry_items')
    op.drop_table('pantry_items')

    op.drop_index('ix_recipe_ingredients_ingredient', table_name='recipe_ingredients')
    op.drop_index('ix_recipe_ingredients_recipe', table_name='recipe_ingredients')
    op.drop_table('recipe_ingredients')

    op.drop_table('recipe_tags_junction')

    op.drop_index('ix_recipes_is_archived', table_name='recipes')
    op.drop_index('ix_recipes_is_favorite', table_name='recipes')
    op.drop_index('ix_recipes_user_rating', table_name='recipes')
    op.drop_index('ix_recipes_cuisine', table_name='recipes')
    op.drop_index('ix_recipes_meal_type', table_name='recipes')
    op.drop_table('recipes')

    op.drop_table('recipe_tags')

    op.drop_index('ix_ingredients_name', table_name='ingredients')
    op.drop_table('ingredients')

    # Drop custom ENUM types (MariaDB ignores these, but Postgres needs them)
    op.execute("DROP TYPE IF EXISTS ingredientunit")
    op.execute("DROP TYPE IF EXISTS recipedifficulty")
    op.execute("DROP TYPE IF EXISTS recipemealtype")
    op.execute("DROP TYPE IF EXISTS recipesourcetype")
    op.execute("DROP TYPE IF EXISTS ingredientcategory")
