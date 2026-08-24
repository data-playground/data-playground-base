import datetime
import enum
from decimal import Decimal
from typing import Optional

from core.base_model import Base
from pydantic import BaseModel
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Table,
    Text,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

"""
RECIPE MANAGER MODULE — moved to domains/recipes/models.py as part of the
domain-folder migration (Work Order #7). Re-exported from root models.py
so any other file still doing `from models import Recipe` (etc.) keeps
working unchanged.

Design decisions:
  - recipe_cook_log excluded — cook history lives on recipes.times_cooked
    and recipes.last_cooked_at only. Simple and sufficient.
  - PantryItem is intentionally minimal (ingredient_id only).
    No quantity, unit, or expiry — those can be added later without
    touching any existing code.
  - RecipeIngredient.quantity is Decimal NULL — NULL means "to taste".
  - Images stored as source URL strings only — no local file handling.
  - Normalization pipeline (agent functions) lives in recipe_agents.py,
    not in models — models stay pure data definitions.
"""


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class IngredientCategory(enum.Enum):
    PRODUCE    = "produce"
    PROTEIN    = "protein"
    DAIRY      = "dairy"
    GRAIN      = "grain"
    PANTRY     = "pantry"
    SPICE      = "spice"
    CONDIMENT  = "condiment"
    BEVERAGE   = "beverage"
    FROZEN     = "frozen"
    OTHER      = "other"


class RecipeSourceType(enum.Enum):
    MANUAL       = "manual"
    URL          = "url"
    PDF          = "pdf"
    IMAGE        = "image"
    AI_GENERATED = "ai_generated"


class RecipeMealType(enum.Enum):
    BREAKFAST = "breakfast"
    LUNCH     = "lunch"
    DINNER    = "dinner"
    SNACK     = "snack"
    DESSERT   = "dessert"
    SIDE      = "side"
    DRINK     = "drink"
    OTHER     = "other"


class RecipeDifficulty(enum.Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class IngredientUnit(enum.Enum):
    CUP       = "cup"
    TBSP      = "tbsp"
    TSP       = "tsp"
    ML        = "ml"
    L         = "l"
    G         = "g"
    KG        = "kg"
    OZ        = "oz"
    LB        = "lb"
    PIECE     = "piece"
    CLOVE     = "clove"
    BUNCH     = "bunch"
    SLICE     = "slice"
    CAN       = "can"
    PACKAGE   = "package"
    TO_TASTE  = "to_taste"
    AS_NEEDED = "as_needed"
    PINCH     = "pinch"
    HANDFUL   = "handful"


# ── ORM MODELS ────────────────────────────────────────────────────────────────

class Ingredient(Base):
    """
    Normalized ingredient reference table.
    Every unique ingredient appears here exactly once, by canonical name.
    "garlic", "minced garlic", and "2 garlic cloves" all resolve to
    one row with name="garlic" and category=PRODUCE.

    New rows are created automatically during recipe import by the
    normalization pipeline in recipe_agents.py — never inserted manually.
    """
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    category: Mapped[IngredientCategory] = mapped_column(
        Enum(IngredientCategory, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=IngredientCategory.OTHER,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    recipe_ingredients: Mapped[list["RecipeIngredient"]] = relationship(
        "RecipeIngredient", back_populates="ingredient"
    )
    pantry_item: Mapped[Optional["PantryItem"]] = relationship(
        "PantryItem", back_populates="ingredient", uselist=False
    )

    @property
    def in_pantry(self) -> bool:
        """True if this ingredient exists in the user's pantry."""
        return self.pantry_item is not None


class RecipeTag(Base):
    """
    Tag reference table. Tags are user-defined strings like
    "quick", "vegetarian", "meal-prep", "spicy", etc.
    """
    __tablename__ = "recipe_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)

    # Relationships
    recipes: Mapped[list["Recipe"]] = relationship(
        "Recipe", secondary="recipe_tags_junction", back_populates="tags"
    )


# Association table for the Recipe ↔ RecipeTag many-to-many.
# Defined as a plain Table (not a mapped class) because it carries
# no extra columns — just the two foreign keys.

recipe_tags_junction = Table(
    "recipe_tags_junction",
    Base.metadata,
    Column(
        "recipe_id",
        Integer,
        ForeignKey("recipes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "tag_id",
        Integer,
        ForeignKey("recipe_tags.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Recipe(Base):
    """
    Core recipe table.

    Cook tracking is intentionally simple: times_cooked (INT) and
    last_cooked_at (DATE) are updated by the POST /recipes/{id}/cook
    endpoint. No per-cook log table — the simpler model is sufficient.

    Instructions are stored as Markdown — the template renders them
    with a JS markdown parser (marked.js, already loaded in base.html).

    image_url stores the source URL of an external image only.
    No local file storage — if the source URL breaks, the image disappears,
    which is acceptable for a personal tool.
    """
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    source_type: Mapped[RecipeSourceType] = mapped_column(
        Enum(RecipeSourceType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=RecipeSourceType.MANUAL,
    )
    cuisine: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    meal_type: Mapped[Optional[RecipeMealType]] = mapped_column(
        Enum(RecipeMealType, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    prep_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cook_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    servings: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    difficulty: Mapped[Optional[RecipeDifficulty]] = mapped_column(
        Enum(RecipeDifficulty, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    # Markdown formatted — numbered steps preferred for the step-mode reader.
    instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # External image URL only — no local storage.
    image_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    # 1–5 personal rating. NULL = not yet rated.
    user_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    # Cook tracking (simple — no log table).
    times_cooked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_cooked_at: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    is_favorite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Soft delete — archived recipes are excluded from library queries by default.
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    ingredients: Mapped[list["RecipeIngredient"]] = relationship(
        "RecipeIngredient",
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.sort_order",
        lazy="selectin",
    )
    tags: Mapped[list["RecipeTag"]] = relationship(
        "RecipeTag",
        secondary=recipe_tags_junction,
        back_populates="recipes",
        lazy="selectin",
    )

    @property
    def total_time(self) -> Optional[int]:
        """
        Returns total_time_minutes if set, otherwise computes from
        prep + cook if both are available.
        """
        if self.total_time_minutes is not None:
            return self.total_time_minutes
        if self.prep_time_minutes is not None and self.cook_time_minutes is not None:
            return self.prep_time_minutes + self.cook_time_minutes
        return None

    @property
    def time_display(self) -> str:
        """Human-readable total time for template display, e.g. '1h 20min'."""
        t = self.total_time
        if t is None:
            return "—"
        if t < 60:
            return f"{t}min"
        hours = t // 60
        mins = t % 60
        if mins == 0:
            return f"{hours}h"
        return f"{hours}h {mins}min"

    @property
    def rating_display(self) -> str:
        """Returns filled/empty star string for template rendering."""
        if not self.user_rating:
            return "☆☆☆☆☆"
        return "★" * self.user_rating + "☆" * (5 - self.user_rating)

    @property
    def tag_names(self) -> list[str]:
        return [t.name for t in self.tags]


class RecipeIngredient(Base):
    """
    One row per ingredient per recipe.
    Quantity is Decimal NULL — NULL means "to taste" or "as needed".
    preparation_note holds the prep method ("finely diced", "room temperature")
    separate from the canonical ingredient name — this is what the
    normalization agent strips out and stores here instead of polluting
    the ingredients table with "minced garlic" vs "garlic".
    sort_order controls display sequence in the recipe view.
    """
    __tablename__ = "recipe_ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recipe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recipes.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    ingredient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ingredients.id"),
        nullable=False, index=True,
    )
    # NULL = "to taste" / "as needed" — rendered differently in templates.
    quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 3), nullable=True)
    unit: Mapped[Optional[IngredientUnit]] = mapped_column(
        Enum(IngredientUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    # The preparation method, separated from the ingredient name.
    # e.g. "finely diced", "at room temperature", "roughly chopped"
    preparation_note: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    is_optional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="ingredients")
    ingredient: Mapped["Ingredient"] = relationship(
        "Ingredient", back_populates="recipe_ingredients", lazy="selectin"
    )


    def quantity_scaled(self, scale_factor: float) -> Optional[str]:
        """
        Returns a human-readable scaled quantity string.
        Used by the JavaScript servings scaler — this Python version is
        available for server-side rendering if needed.
        e.g. scale_factor=2.0 → "2 cups" becomes "4 cups"
        """
        if self.quantity is None:
            return None
        scaled = float(self.quantity) * scale_factor
        # Clean up trailing zeros: 2.0 → "2", 2.5 → "2.5"
        if scaled == int(scaled):
            return str(int(scaled))
        return f"{scaled:.2f}".rstrip("0")

    @property
    def display_quantity(self) -> str:
        """
        Human-readable quantity for the ingredient line.
        Returns "to taste" for NULL quantities, otherwise formats the decimal.
        """
        if self.quantity is None:
            return "to taste"
        q = float(self.quantity)
        if q == int(q):
            return str(int(q))
        return f"{q:.2f}".rstrip("0")

    @property
    def full_ingredient_line(self) -> str:
        """
        Renders the complete ingredient line as a string for display.
        e.g. "2 cups flour, sifted" or "to taste salt"
        """
        parts = []
        if self.quantity is not None:
            parts.append(self.display_quantity)
        if self.unit:
            parts.append(self.unit.value)
        parts.append(self.ingredient.name)
        if self.preparation_note:
            parts.append(f", {self.preparation_note}")
        if self.is_optional:
            parts.append(" (optional)")
        return " ".join(parts)


class PantryItem(Base):
    """
    Minimal pantry — just which ingredients the user currently has.
    One row per ingredient (enforced by UNIQUE on ingredient_id).
    No quantity, unit, or expiry by design — keeps the UX lightweight.
    A future migration can add those columns without touching this model.
    """
    __tablename__ = "pantry_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ingredient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ingredients.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    ingredient: Mapped["Ingredient"] = relationship(
        "Ingredient", back_populates="pantry_item"
    )


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

class IngredientResponse(BaseModel):
    id: int
    name: str
    category: IngredientCategory
    in_pantry: bool

    class Config:
        from_attributes = True


class RecipeIngredientResponse(BaseModel):
    id: int
    ingredient_id: int
    ingredient_name: str
    quantity: Optional[Decimal]
    unit: Optional[IngredientUnit]
    preparation_note: Optional[str]
    is_optional: bool
    sort_order: int
    display_quantity: str
    full_ingredient_line: str

    class Config:
        from_attributes = True


class RecipeTagResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class RecipeResponse(BaseModel):
    id: int
    title: str
    source_url: Optional[str]
    source_type: RecipeSourceType
    cuisine: Optional[str]
    meal_type: Optional[RecipeMealType]
    prep_time_minutes: Optional[int]
    cook_time_minutes: Optional[int]
    total_time_minutes: Optional[int]
    servings: Optional[int]
    difficulty: Optional[RecipeDifficulty]
    user_rating: Optional[int]
    times_cooked: int
    last_cooked_at: Optional[datetime.date]
    is_favorite: bool
    image_url: Optional[str]
    time_display: str
    rating_display: str
    tag_names: list[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class RecipeCreate(BaseModel):
    """Used when creating a recipe manually from form data."""
    title: str
    source_url: Optional[str] = None
    source_type: RecipeSourceType = RecipeSourceType.MANUAL
    cuisine: Optional[str] = None
    meal_type: Optional[RecipeMealType] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    servings: Optional[int] = None
    difficulty: Optional[RecipeDifficulty] = None
    instructions: Optional[str] = None
    notes: Optional[str] = None
    image_url: Optional[str] = None


class PantryItemResponse(BaseModel):
    id: int
    ingredient_id: int
    ingredient_name: str
    ingredient_category: IngredientCategory

    class Config:
        from_attributes = True
