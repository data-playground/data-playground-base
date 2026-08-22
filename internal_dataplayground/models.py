import datetime
import enum
import math
from decimal import Decimal
from typing import Optional

# The shared Base class for all tables now lives in core/base_model.py.
# Re-exported here (temporary — see domains/habits pilot roadmap) so every
# other file still doing `from models import Base` keeps working unchanged.
from core.base_model import Base
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    BigInteger,
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
    UniqueConstraint,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

"""
JOBS MODULE — moved to domains/jobs/models.py as part of the domain-folder
migration (see domains/jobs/routers/*.py for usage). Re-exported here so
any other file still doing `from models import Job` (etc.) keeps working
unchanged.
"""
# TODO: remove after all cross-references are updated
from domains.jobs.models import (
    ApplicationStatus,
    Job,
    ApplicationLog,
    JobSearchKeyword,
    WatchedCompany,
    JobScoutRunLog,
    JobResponse,
    ApplicationLogCreate,
    ApplicationLogResponse,
    StagingJobStatus,
    StagingJob,
    StagingJobCreate,
    StagingJobResponse,
)

        
"""
FINANCE MODULE — moved to domains/finance/models.py as part of the
domain-folder migration (Work Order #5). Re-exported here so any other
file still doing `from models import Account` (etc.) keeps working
unchanged.
"""
from domains.finance.models import (
    AccountType,
    Category,
    Account,
    Transaction,
    AccountCreate,
    AccountResponse,
    CategoryCreate,
    CategoryResponse,
    TransactionResponse,
)


"""
BLOG MODULE — moved to domains/blog/models.py as part of the domain-folder
migration (see routers/blog.py for usage). Re-exported here so any other
file still doing `from models import BlogIdea` (etc.) keeps working
unchanged.
"""
# TODO: remove after all cross-references are updated
from domains.blog.models import (
    BlogProjectType,
    BlogIdeaStatus,
    DIFFICULTY_LEVELS,
    BlogIdea,
    BlogIdeaCreate,
    BlogIdeaResponse,
)


"""
CODE INTELLIGENCE MODULE — moved to domains/code_intel/models.py as part of
the domain-folder migration (see routers/ci_projects.py, routers/ci_files.py,
routers/ci_readme.py for usage). Re-exported here so any other file still
doing `from models import CodeProject` (etc.) keeps working unchanged.
"""
# TODO: remove after all cross-references are updated
from domains.code_intel.models import (
    ReadmeStatus,
    FolderReadmeStatus,
    CommentedStatus,
    ImprovementStatus,
    CodeProject,
    CodeFile,
    FolderReadme,
    CodeProjectCreate,
    CodeProjectResponse,
    CodeFileResponse,
    FolderReadmeResponse,
    FolderReadmeCreate,
)

# ── HABIT TRACKER MODULE ─────────────────────────────────────────────────────
# Moved to domains/habits/models.py as part of the domains-folder pilot
# migration. Re-exported here so any other file still doing
# `from models import Habit` (etc.) keeps working unchanged.
# TODO: remove after all cross-references are updated

from domains.habits.models import (
    Habit,
    HabitCreate,
    HabitLog,
    HabitLogResponse,
    HabitResponse,
    HabitSettings,
    HabitUpdate,
)



# ── JOURNAL MODULE ────────────────────────────────────────────────────────────
# Moved to domains/journal/models.py as part of the domain-folder migration
# (Work Order #6). Re-exported here so any other file still doing
# `from models import JournalEntry` (etc.) keeps working unchanged.
#
# PRIVACY ARCHITECTURE — HARD CONSTRAINT (see domains/journal/models.py):
#   content, gratitude, and challenges fields are NEVER sent to external AI.
#   Weekly synthesis is generated from mood_score and energy_score ONLY.
#   Violating this constraint is a critical privacy bug.
# TODO: remove after all cross-references are updated

from domains.journal.models import (
    JournalEntry,
    WeeklySynthesis,
)


# ── RECIPE MANAGER MODULE ────────────────────────────────────────────────────
# Append everything below to the bottom of models.py.
#
# Design decisions:
#   - recipe_cook_log excluded — cook history lives on recipes.times_cooked
#     and recipes.last_cooked_at only. Simple and sufficient.
#   - PantryItem is intentionally minimal (ingredient_id only).
#     No quantity, unit, or expiry — those can be added later without
#     touching any existing code.
#   - RecipeIngredient.quantity is Decimal NULL — NULL means "to taste".
#   - Images stored as source URL strings only — no local file handling.
#   - Normalization pipeline (agent functions) lives in recipe_agents.py,
#     not in models — models stay pure data definitions.




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

"""
WORKOUT TRACKER MODULE — append these classes to the bottom of models.py

Imports to add at the top of models.py if not already present:
  from sqlalchemy import SmallInteger, Numeric, JSON  (Numeric + JSON likely already imported)
"""


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class LocationType(enum.Enum):
    HOME    = "home"
    GYM     = "gym"
    OUTDOOR = "outdoor"
    OTHER   = "other"


class EquipmentType(enum.Enum):
    BARBELL         = "barbell"
    DUMBBELL        = "dumbbell"
    MACHINE         = "machine"
    CABLE           = "cable"
    BODYWEIGHT      = "bodyweight"
    CARDIO          = "cardio"
    RESISTANCE_BAND = "resistance_band"
    KETTLEBELL      = "kettlebell"
    OTHER           = "other"


class MuscleGroup(enum.Enum):
    CHEST      = "chest"
    BACK       = "back"
    SHOULDERS  = "shoulders"
    BICEPS     = "biceps"
    TRICEPS    = "triceps"
    FOREARMS   = "forearms"
    QUADS      = "quads"
    HAMSTRINGS = "hamstrings"
    GLUTES     = "glutes"
    CALVES     = "calves"
    CORE       = "core"
    FULL_BODY  = "full_body"
    CARDIO     = "cardio"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class ExerciseEquipmentType(enum.Enum):
    BARBELL         = "barbell"
    DUMBBELL        = "dumbbell"
    MACHINE         = "machine"
    CABLE           = "cable"
    BODYWEIGHT      = "bodyweight"
    RESISTANCE_BAND = "resistance_band"
    KETTLEBELL      = "kettlebell"
    CARDIO          = "cardio"
    OTHER           = "other"
    ANY             = "any"


class PlanOrigin(enum.Enum):
    USER = "user"
    AI   = "ai"


class WorkoutGoal(enum.Enum):
    STRENGTH        = "strength"
    HYPERTROPHY     = "hypertrophy"
    ENDURANCE       = "endurance"
    GENERAL_FITNESS = "general_fitness"
    WEIGHT_LOSS     = "weight_loss"

    @property
    def label(self) -> str:
        return {
            "strength":        "Strength",
            "hypertrophy":     "Hypertrophy (Muscle Size)",
            "endurance":       "Endurance",
            "general_fitness": "General Fitness",
            "weight_loss":     "Weight Loss",
        }[self.value]


class WeightUnit(enum.Enum):
    KG = "kg"
    LB = "lb"


# ── MODELS ────────────────────────────────────────────────────────────────────


# Base is imported from wherever it's defined in models.py


class WorkoutLocation(Base):
    """
    Physical location where the user trains.
    Equipment is scoped to a location — the AI plan generator uses this
    to know what's available at each location.
    is_default=True means this location is pre-selected when starting a session.
    Only one location should have is_default=True (enforced at application layer).
    """
    __tablename__ = "workout_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    location_type: Mapped[LocationType] = mapped_column(
        Enum(LocationType, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=LocationType.GYM,
    )
    address: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    equipment: Mapped[list["Equipment"]] = relationship(
        "Equipment", back_populates="location",
        cascade="all, delete-orphan", lazy="selectin",
    )
    sessions: Mapped[list["WorkoutSession"]] = relationship(
        "WorkoutSession", back_populates="location",
    )
    plans: Mapped[list["WorkoutPlan"]] = relationship(
        "WorkoutPlan", back_populates="location",
    )

    @property
    def active_equipment(self) -> list["Equipment"]:
        return [e for e in self.equipment if e.is_active]

    @property
    def equipment_summary(self) -> str:
        """Short description of available equipment for AI prompt construction."""
        types = list({e.equipment_type.value for e in self.active_equipment})
        return ", ".join(sorted(types)) if types else "No equipment logged"


class Equipment(Base):
    """
    Individual piece of equipment available at a workout location.
    Used by the AI plan generator to scope exercise recommendations.
    """
    __tablename__ = "equipment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    location_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_locations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    equipment_type: Mapped[EquipmentType] = mapped_column(
        Enum(EquipmentType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    max_weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), nullable=True)
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    location: Mapped["WorkoutLocation"] = relationship(
        "WorkoutLocation", back_populates="equipment"
    )


class Exercise(Base):
    """
    Reference exercise library. Seeded with 75 common exercises.
    is_custom=True marks user-added exercises that are not in the seed data.
    The AI plan generator references exercise names — fuzzy matching happens
    in the router when AI suggestions don't exactly match DB entries.
    """
    __tablename__ = "exercises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    primary_muscle_group: Mapped[MuscleGroup] = mapped_column(
        Enum(MuscleGroup, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    # JSON array of secondary muscle group strings
    secondary_muscle_groups: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    equipment_type: Mapped[ExerciseEquipmentType] = mapped_column(
        Enum(ExerciseEquipmentType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    is_compound: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_custom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    plan_exercises: Mapped[list["WorkoutPlanExercise"]] = relationship(
        "WorkoutPlanExercise", back_populates="exercise",
    )
    sets: Mapped[list["WorkoutSet"]] = relationship(
        "WorkoutSet", back_populates="exercise",
    )


class WorkoutPlan(Base):
    """
    A structured training program — either user-created or AI-generated.
    Only one plan can be active at a time (enforced in the router via
    SET is_active=FALSE on all plans before activating the new one).
    target_days_per_week drives how many WorkoutPlanDay rows to generate.
    """
    __tablename__ = "workout_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_by: Mapped[PlanOrigin] = mapped_column(
        Enum(PlanOrigin, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=PlanOrigin.USER,
    )
    location_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_locations.id", ondelete="SET NULL"), nullable=True
    )
    target_days_per_week: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)
    goal: Mapped[WorkoutGoal] = mapped_column(
        Enum(WorkoutGoal, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WorkoutGoal.GENERAL_FITNESS,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    location: Mapped[Optional["WorkoutLocation"]] = relationship(
        "WorkoutLocation", back_populates="plans"
    )
    days: Mapped[list["WorkoutPlanDay"]] = relationship(
        "WorkoutPlanDay", back_populates="plan",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutPlanDay.day_number",
    )
    sessions: Mapped[list["WorkoutSession"]] = relationship(
        "WorkoutSession", back_populates="plan",
    )

    @property
    def total_exercises(self) -> int:
        return sum(len(d.exercises) for d in self.days)


class WorkoutPlanDay(Base):
    """
    A named day within a workout plan (e.g. "Day 1 — Chest & Triceps").
    day_number is 1-indexed and determines rotation order during logging.
    The user can follow any day they choose when logging — plans are guides,
    not strict schedules.
    """
    __tablename__ = "workout_plan_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    day_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # Human-readable label, e.g. "Chest & Triceps", "Pull Day", "Leg Day"
    day_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan: Mapped["WorkoutPlan"] = relationship("WorkoutPlan", back_populates="days")
    exercises: Mapped[list["WorkoutPlanExercise"]] = relationship(
        "WorkoutPlanExercise", back_populates="plan_day",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutPlanExercise.order_in_day",
    )


class WorkoutPlanExercise(Base):
    """
    An exercise prescribed within a specific day of a workout plan.
    target_weight is in lb (user's default unit) — a starting suggestion only.
    """
    __tablename__ = "workout_plan_exercises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    plan_day_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plan_days.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    exercise_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("exercises.id"), nullable=False
    )
    target_sets: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)
    target_reps_min: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=8)
    target_reps_max: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=12)
    target_weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), nullable=True)
    order_in_day: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan_day: Mapped["WorkoutPlanDay"] = relationship(
        "WorkoutPlanDay", back_populates="exercises"
    )
    exercise: Mapped["Exercise"] = relationship(
        "Exercise", back_populates="plan_exercises", lazy="selectin"
    )

    @property
    def rep_range(self) -> str:
        if self.target_reps_min == self.target_reps_max:
            return str(self.target_reps_min)
        return f"{self.target_reps_min}–{self.target_reps_max}"


class WorkoutSession(Base):
    """
    A single training session. Can be linked to a plan day (structured)
    or standalone (free-form logging).
    weight_unit is the per-session toggle — defaults to lb but user can
    switch to kg for a session without affecting the global default.
    duration_minutes is calculated from started_at/ended_at when ending
    the session, or can be entered manually.
    """
    __tablename__ = "workout_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    plan_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="SET NULL"), nullable=True
    )
    plan_day_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_plan_days.id", ondelete="SET NULL"), nullable=True
    )
    location_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_locations.id", ondelete="SET NULL"), nullable=True
    )
    session_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fatigue_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)  # 1-5
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    plan: Mapped[Optional["WorkoutPlan"]] = relationship(
        "WorkoutPlan", back_populates="sessions"
    )
    plan_day: Mapped[Optional["WorkoutPlanDay"]] = relationship(
        "WorkoutPlanDay", lazy="selectin"
    )
    location: Mapped[Optional["WorkoutLocation"]] = relationship(
        "WorkoutLocation", back_populates="sessions"
    )
    sets: Mapped[list["WorkoutSet"]] = relationship(
        "WorkoutSet", back_populates="session",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutSet.created_at",
    )

    @property
    def is_active(self) -> bool:
        """True if the session has been started but not ended."""
        return self.started_at is not None and self.ended_at is None

    @property
    def working_sets(self) -> list["WorkoutSet"]:
        return [s for s in self.sets if not s.is_warmup]

    @property
    def exercise_count(self) -> int:
        return len({s.exercise_id for s in self.working_sets})

    @property
    def total_volume_lb(self) -> Decimal:
        """Sum of weight × reps across all working sets."""
        total = Decimal("0")
        for s in self.working_sets:
            if s.weight_used:
                w = s.weight_used
                if s.weight_unit == WeightUnit.KG:
                    w = w * Decimal("2.20462")
                total += w * s.reps_completed
        return total

    @property
    def duration_display(self) -> str:
        if self.duration_minutes:
            h, m = divmod(self.duration_minutes, 60)
            return f"{h}h {m}m" if h else f"{m}m"
        if self.started_at and self.ended_at:
            mins = int((self.ended_at - self.started_at).total_seconds() / 60)
            h, m = divmod(mins, 60)
            return f"{h}h {m}m" if h else f"{m}m"
        return "—"

    @property
    def fatigue_label(self) -> str:
        return {1: "Easy", 2: "Light", 3: "Moderate", 4: "Hard", 5: "Brutal"}.get(
            self.fatigue_rating or 0, "—"
        )


class WorkoutSet(Base):
    """
    An individual set logged during a session.
    is_warmup=True sets are tracked separately — they don't count toward
    working volume calculations but are shown in the session log.
    rpe (Rate of Perceived Exertion) is 1-10 — optional quality signal.
    weight_unit inherits from the session but is stored per set for
    historical accuracy if the user toggles units mid-session.
    """
    __tablename__ = "workout_sets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("workout_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    exercise_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("exercises.id"), nullable=False
    )
    set_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    reps_completed: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    weight_used: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), nullable=True)
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    rpe: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    is_warmup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    session: Mapped["WorkoutSession"] = relationship("WorkoutSession", back_populates="sets")
    exercise: Mapped["Exercise"] = relationship(
        "Exercise", back_populates="sets", lazy="selectin"
    )

    @property
    def weight_display(self) -> str:
        if self.weight_used is None:
            return "BW"
        return f"{self.weight_used:g} {self.weight_unit.value}"

    @property
    def volume(self) -> Decimal:
        """Weight × reps for this set."""
        if self.weight_used is None:
            return Decimal("0")
        return self.weight_used * self.reps_completed


class BodyMetric(Base):
    """
    Daily body weight and body fat percentage tracking.
    Unique constraint on metric_date — one entry per day, upserted by the router.
    Default unit is lb (user preference).
    """
    __tablename__ = "body_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)
    weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2), nullable=True)
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    body_fat_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    @property
    def weight_display(self) -> str:
        if self.weight is None:
            return "—"
        return f"{self.weight:g} {self.weight_unit.value}"


# models_media.py
"""
Media Tracker ORM models — append to models.py.

IMPORTS TO ADD at top of models.py if not already present:
  from sqlalchemy import SmallInteger, JSON  (likely already imported)

PRIVACY NOTE: No privacy constraints here — media tracking is not sensitive.
All fields can be used freely in AI recommendation prompts.

RATING SYSTEM:
  user_rating stores 1-10. UI displays as half-stars:
  1=½★  2=★  3=1½★  4=★★  5=2½★  6=★★★  7=3½★  8=★★★★  9=4½★  10=★★★★★
  Even numbers = whole stars. Odd numbers = half-star.

PREDEFINED MOOD TAGS:
  light, cerebral, emotional, funny, tense, dark, inspiring,
  relaxing, thrilling, romantic, nostalgic, thought-provoking

  Custom tags can be added alongside predefined ones — all stored as
  a JSON string array in mood_tags.
"""

# Base imported from wherever it's defined in models.py


# ── PREDEFINED MOOD TAGS ───────────────────────────────────────────────────────
# Used by both the UI tag selector and the ML recommendation query builder.
# Custom tags are allowed alongside these — the full list is the union of
# PREDEFINED_MOOD_TAGS and any user-defined strings in mood_tags JSON.

PREDEFINED_MOOD_TAGS = [
    "light", "cerebral", "emotional", "funny", "tense",
    "dark", "inspiring", "relaxing", "thrilling", "romantic",
    "nostalgic", "thought-provoking",
]


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class MediaExternalSource(enum.Enum):
    TMDB_MOVIE  = "tmdb_movie"
    TMDB_TV     = "tmdb_tv"
    OPENLIBRARY = "openlibrary"
    MANUAL      = "manual"


class MediaType(enum.Enum):
    MOVIE   = "movie"
    TV_SHOW = "tv_show"
    BOOK    = "book"

    @property
    def label(self) -> str:
        return {"movie": "Movie", "tv_show": "TV Show", "book": "Book"}[self.value]

    @property
    def icon(self) -> str:
        return {"movie": "🎬", "tv_show": "📺", "book": "📚"}[self.value]


class UserMediaStatus(enum.Enum):
    WANT_TO    = "want_to"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    ABANDONED  = "abandoned"

    @property
    def label(self) -> str:
        return {
            "want_to":     "Want To",
            "in_progress": "In Progress",
            "completed":   "Completed",
            "abandoned":   "Abandoned",
        }[self.value]

    @property
    def color(self) -> str:
        return {
            "want_to":     "var(--accent)",
            "in_progress": "var(--yellow)",
            "completed":   "var(--green)",
            "abandoned":   "var(--text-muted)",
        }[self.value]


class RecommendationMediaType(enum.Enum):
    MOVIE   = "movie"
    TV_SHOW = "tv_show"
    BOOK    = "book"
    ANY     = "any"


# ── STREAMING SERVICES ────────────────────────────────────────────────────────

class StreamingService(Base):
    """
    Reference table of streaming services. Seeded on migration.
    is_subscribed=True means the user actively subscribes to this service
    and it will be preferred in recommendations.
    tmdb_provider_id is used to match TMDB watch provider API responses.
    """
    __tablename__ = "streaming_services"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    tmdb_provider_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tmdb_provider_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_subscribed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )


# ── MEDIA ITEMS ───────────────────────────────────────────────────────────────

class MediaItem(Base):
    """
    The catalog table — one row per unique movie/TV show/book.
    The UNIQUE constraint on (external_id, external_source) prevents duplicates
    when the same item is searched multiple times.

    embedding is a 384-dimensional float vector from all-MiniLM-L6-v2,
    stored as a JSON array. The ML service generates these; the recommendation
    router computes cosine similarity in Python.

    streaming_provider_ids is a JSON array of TMDB provider IDs (integers)
    that have this item available for streaming in the US.
    """
    __tablename__ = "media_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── External source ────────────────────────────────────────────────────────
    external_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    external_source: Mapped[MediaExternalSource] = mapped_column(
        Enum(MediaExternalSource, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=MediaExternalSource.MANUAL,
    )

    # ── Core metadata ──────────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[MediaType] = mapped_column(
        Enum(MediaType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    release_year: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    poster_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    external_rating: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 1), nullable=True)

    # ── Movie-specific ─────────────────────────────────────────────────────────
    runtime_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ── TV-specific ────────────────────────────────────────────────────────────
    total_seasons: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    total_episodes: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # ── Book-specific ──────────────────────────────────────────────────────────
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # ── Streaming availability ─────────────────────────────────────────────────
    # JSON array of TMDB provider IDs, e.g. [8, 119] = Netflix + Prime
    streaming_provider_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    streaming_fetched_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )
    seasons_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ── ML embedding ──────────────────────────────────────────────────────────
    # 384-dim float vector. None means the embedding job hasn't run yet.
    embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    embedding_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    user_media: Mapped[Optional["UserMedia"]] = relationship(
        "UserMedia", back_populates="media_item", uselist=False
    )

    # ── Computed properties ────────────────────────────────────────────────────

    @property
    def is_tracked(self) -> bool:
        """True if the user has added this item to their list."""
        return self.user_media is not None

    @property
    def genre_list(self) -> list[str]:
        return self.genres or []

    @property
    def runtime_display(self) -> str:
        """Human-readable runtime, e.g. '2h 15m' or '45 min'."""
        if not self.runtime_minutes:
            return "—"
        if self.runtime_minutes < 60:
            return f"{self.runtime_minutes} min"
        h, m = divmod(self.runtime_minutes, 60)
        return f"{h}h {m}m" if m else f"{h}h"

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0

    @property
    def streaming_available_on(self) -> list[int]:
        """Returns list of TMDB provider IDs where this item streams."""
        return self.streaming_provider_ids or []


class UserMedia(Base):
    """
    The user's personal tracking record for a media item.
    One row per media_item — UNIQUE constraint prevents double-tracking.

    Rating system: 1-10 stored, displayed as half-stars.
    Odd = half-star, even = whole star:
      1=½★  2=★  3=1½★  4=★★  5=2½★  6=★★★  7=3½★  8=★★★★  9=4½★  10=★★★★★

    mood_tags is a JSON string array mixing predefined and custom tags:
      ["light", "funny", "my-custom-tag"]
    """
    __tablename__ = "user_media"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    media_item_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("media_items.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    status: Mapped[UserMediaStatus] = mapped_column(
        Enum(UserMediaStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=UserMediaStatus.WANT_TO,
    )

    # 1-10; None = not yet rated
    user_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    started_at: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    completed_at: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rewatch_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    # JSON string array: predefined + custom mood tags
    mood_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    media_item: Mapped["MediaItem"] = relationship(
        "MediaItem", back_populates="user_media", lazy="selectin"
    )
    season_progress: Mapped[list["TVSeasonProgress"]] = relationship(
        "TVSeasonProgress", back_populates="user_media",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="TVSeasonProgress.season_number",
    )

    # ── Rating display helpers ─────────────────────────────────────────────────

    @property
    def rating_stars(self) -> str:
        """
        Returns a star string for template display.
        Odd rating = half star. Even rating = whole star.
        1→½★  2→★  3→1½★  4→★★  5→2½★  6→★★★  7→3½★  8→★★★★  9→4½★  10→★★★★★
        """
        if not self.user_rating:
            return "☆☆☆☆☆"
        r = self.user_rating
        full = r // 2
        half = r % 2
        empty = 5 - full - half
        return "★" * full + ("½" if half else "") + "☆" * empty

    @property
    def rating_numeric(self) -> str:
        """e.g. '8/10' or '—'"""
        if not self.user_rating:
            return "—"
        return f"{self.user_rating}/10"

    @property
    def tag_list(self) -> list[str]:
        return self.mood_tags or []

    @property
    def is_rated(self) -> bool:
        return self.user_rating is not None

    @property
    def total_episodes_watched(self) -> int:
        """Total episodes watched across all seasons (TV shows only)."""
        return sum(sp.episodes_watched for sp in self.season_progress)


class TVSeasonProgress(Base):
    """
    Per-season episode progress for TV shows.
    Sparse — only seasons the user has started appear here.
    """
    __tablename__ = "tv_season_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_media_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user_media.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    season_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    episodes_watched: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    total_episodes: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    user_media: Mapped["UserMedia"] = relationship(
        "UserMedia", back_populates="season_progress"
    )

    @property
    def completion_pct(self) -> int:
        """Percentage of episodes watched in this season (0-100)."""
        if not self.total_episodes or self.total_episodes == 0:
            return 0
        return min(100, round((self.episodes_watched / self.total_episodes) * 100))

    @property
    def is_complete(self) -> bool:
        return (
            self.total_episodes is not None
            and self.episodes_watched >= self.total_episodes
        )


class MediaRecommendation(Base):
    """
    Cached recommendation sessions.
    recommendations JSON schema:
    [
      {
        "media_item_id": int,
        "title": str,
        "score": float,           # cosine similarity from ML layer
        "reasoning": str | null   # Gemini explanation (null if Gemini skipped)
      }
    ]
    """
    __tablename__ = "media_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    input_mood: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input_context: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    media_type_filter: Mapped[RecommendationMediaType] = mapped_column(
        Enum(RecommendationMediaType, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=RecommendationMediaType.ANY,
    )
    include_unsubscribed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    used_gemini: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    recommendations: Mapped[list] = mapped_column(JSON, nullable=False)
    ml_model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    @property
    def result_count(self) -> int:
        return len(self.recommendations or [])

    @property
    def age_minutes(self) -> float:
        delta = datetime.datetime.utcnow() - self.generated_at
        return delta.total_seconds() / 60


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

class MediaItemResponse(BaseModel):
    id: int
    title: str
    media_type: MediaType
    external_source: MediaExternalSource
    genres: Optional[list]
    release_year: Optional[int]
    description: Optional[str]
    poster_url: Optional[str]
    external_rating: Optional[Decimal]
    runtime_minutes: Optional[int]
    total_seasons: Optional[int]
    author: Optional[str]
    has_embedding: bool
    streaming_available_on: list

    class Config:
        from_attributes = True


class UserMediaCreate(BaseModel):
    media_item_id: int
    status: UserMediaStatus = UserMediaStatus.WANT_TO


class UserMediaUpdate(BaseModel):
    status: Optional[UserMediaStatus] = None
    user_rating: Optional[int] = None
    mood_tags: Optional[list] = None
    notes: Optional[str] = None
    started_at: Optional[datetime.date] = None
    completed_at: Optional[datetime.date] = None


class UserMediaResponse(BaseModel):
    id: int
    media_item_id: int
    status: UserMediaStatus
    user_rating: Optional[int]
    rating_stars: str
    rating_numeric: str
    tag_list: list
    notes: Optional[str]
    started_at: Optional[datetime.date]
    completed_at: Optional[datetime.date]
    rewatch_count: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class StreamingServiceResponse(BaseModel):
    id: int
    name: str
    tmdb_provider_id: Optional[int]
    logo_url: Optional[str]
    is_subscribed: bool
    sort_order: int

    class Config:
        from_attributes = True

# ── WEEKLY PLANNING MODULE ────────────────────────────────────────────────────

class FitnessGoal(enum.Enum):
    WEIGHT_LOSS    = "weight_loss"
    MUSCLE_GAIN    = "muscle_gain"
    MAINTENANCE    = "maintenance"
    ENDURANCE      = "endurance"
    GENERAL_HEALTH = "general_health"

    @property
    def label(self) -> str:
        return {
            "weight_loss":    "Weight Loss",
            "muscle_gain":    "Muscle Gain",
            "maintenance":    "Maintenance",
            "endurance":      "Endurance",
            "general_health": "General Health",
        }[self.value]


class WeeklyPlanStatus(enum.Enum):
    DRAFT     = "draft"
    CONFIRMED = "confirmed"
    ACTIVE    = "active"
    COMPLETED = "completed"


class PlanDayStatus(enum.Enum):
    PLANNED   = "planned"
    ACTIVE    = "active"
    COMPLETED = "completed"
    SKIPPED   = "skipped"


class PlanMealType(enum.Enum):
    BREAKFAST = "breakfast"
    LUNCH     = "lunch"
    DINNER    = "dinner"
    SNACK     = "snack"


class PlanMealStatus(enum.Enum):
    PLANNED  = "planned"
    EATEN    = "eaten"
    SWAPPED  = "swapped"
    OFF_PLAN = "off_plan"
    SKIPPED  = "skipped"

    @property
    def label(self) -> str:
        return {
            "planned":  "Planned",
            "eaten":    "✓ Eaten",
            "swapped":  "↔ Swapped",
            "off_plan": "↗ Off-plan",
            "skipped":  "— Skipped",
        }[self.value]


class UserIntent(Base):
    """
    Single-row table. Always query with LIMIT 1.
    Stores the user's current fitness goal and AI-generator preferences.
    All weekly plan generators read from this before producing suggestions.
    """
    __tablename__ = "user_intent"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    fitness_goal: Mapped[FitnessGoal] = mapped_column(
        Enum(FitnessGoal, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=FitnessGoal.WEIGHT_LOSS,
    )
    weekly_workout_days: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=4
    )
    target_calories: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    macro_preference: Mapped[str] = mapped_column(
        String(30), nullable=False, default="high_protein"
    )
    cooking_time_preference: Mapped[str] = mapped_column(
        String(20), nullable=False, default="moderate"
    )
    dietary_restrictions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    food_preferences: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    food_dislikes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    health_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    @property
    def cooking_time_label(self) -> str:
        return {
            "minimal":  "Minimal (<20 min)",
            "moderate": "Moderate (20–45 min)",
            "generous": "Generous (45+ min)",
        }.get(self.cooking_time_preference, self.cooking_time_preference)

    def to_ai_context(self) -> str:
        """
        Renders the intent as a plain-English block for injection into
        AI generator prompts. Called by both the workout planner and
        the meal planner agents.
        """
        lines = [
            f"Primary goal: {self.fitness_goal.label}",
            f"Target workout days per week: {self.weekly_workout_days}",
            f"Macro preference: {self.macro_preference}",
            f"Cooking time available: {self.cooking_time_label}",
        ]
        if self.target_calories:
            lines.append(f"Daily calorie target: ~{self.target_calories} kcal")
        if self.dietary_restrictions:
            lines.append(f"Dietary restrictions: {self.dietary_restrictions}")
        if self.food_preferences:
            lines.append(f"Food preferences: {self.food_preferences}")
        if self.food_dislikes:
            lines.append(f"Food dislikes: {self.food_dislikes}")
        if self.health_notes:
            lines.append(f"Health/injury notes: {self.health_notes}")
        return "\n".join(lines)


class WeeklyPlan(Base):
    """
    One row per planned week. week_start_date is always a Monday.
    Contains 7 WeeklyPlanDay children covering Mon–Sun.
    """
    __tablename__ = "weekly_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_start_date: Mapped[datetime.date] = mapped_column(
        Date, nullable=False, unique=True
    )
    week_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    intent_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[WeeklyPlanStatus] = mapped_column(
        Enum(WeeklyPlanStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeeklyPlanStatus.DRAFT,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    meals_planned: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    meals_followed: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    workouts_planned: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    workouts_completed: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    days: Mapped[list["WeeklyPlanDay"]] = relationship(
        "WeeklyPlanDay", back_populates="weekly_plan",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WeeklyPlanDay.day_number",
    )
    shopping_list: Mapped[Optional["ShoppingList"]] = relationship(
        "ShoppingList", back_populates="weekly_plan", uselist=False
    )

    @property
    def week_label(self) -> str:
        return (
            f"{self.week_start_date.strftime('%b %-d')} – "
            f"{self.week_end_date.strftime('%-d, %Y')}"
        )

    @property
    def meal_adherence_pct(self) -> int:
        if not self.meals_planned:
            return 0
        return min(100, round((self.meals_followed / self.meals_planned) * 100))

    @property
    def workout_adherence_pct(self) -> int:
        if not self.workouts_planned:
            return 0
        return min(100, round((self.workouts_completed / self.workouts_planned) * 100))


class WeeklyPlanDay(Base):
    """One calendar day within a weekly plan."""
    __tablename__ = "weekly_plan_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    weekly_plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weekly_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    plan_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    day_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 1=Mon

    workout_session_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("workout_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_rest_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    day_status: Mapped[PlanDayStatus] = mapped_column(
        Enum(PlanDayStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=PlanDayStatus.PLANNED,
    )
    override_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    journal_entry_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("journal_entries.id", ondelete="SET NULL"), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    weekly_plan: Mapped["WeeklyPlan"] = relationship(
        "WeeklyPlan", back_populates="days"
    )
    meals: Mapped[list["WeeklyPlanMeal"]] = relationship(
        "WeeklyPlanMeal", back_populates="plan_day",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WeeklyPlanMeal.sort_order",
    )
    workout_session: Mapped[Optional["WorkoutSession"]] = relationship(
        "WorkoutSession", foreign_keys=[workout_session_id], lazy="selectin"
    )

    @property
    def day_name(self) -> str:
        return self.plan_date.strftime("%A")  # "Monday", "Tuesday", etc.

    @property
    def is_today(self) -> bool:
        return self.plan_date == datetime.date.today()

    @property
    def dinner(self) -> Optional["WeeklyPlanMeal"]:
        for m in self.meals:
            if m.meal_type == PlanMealType.DINNER:
                return m
        return None

    @property
    def workout_is_done(self) -> bool:
        if self.is_rest_day:
            return True
        if not self.workout_session:
            return False
        return self.workout_session.ended_at is not None


class WeeklyPlanMeal(Base):
    """One planned meal within a single day of a weekly plan."""
    __tablename__ = "weekly_plan_meals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_day_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weekly_plan_days.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    recipe_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True
    )
    meal_type: Mapped[PlanMealType] = mapped_column(
        Enum(PlanMealType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    status: Mapped[PlanMealStatus] = mapped_column(
        Enum(PlanMealStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=PlanMealStatus.PLANNED,
    )
    swap_recipe_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True
    )
    off_plan_note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan_day: Mapped["WeeklyPlanDay"] = relationship(
        "WeeklyPlanDay", back_populates="meals"
    )
    recipe: Mapped[Optional["Recipe"]] = relationship(
        "Recipe", foreign_keys=[recipe_id], lazy="selectin"
    )
    swap_recipe: Mapped[Optional["Recipe"]] = relationship(
        "Recipe", foreign_keys=[swap_recipe_id], lazy="selectin"
    )

    @property
    def active_recipe(self) -> Optional["Recipe"]:
        """Returns the swap recipe if swapped, otherwise the original."""
        if self.status == PlanMealStatus.SWAPPED and self.swap_recipe:
            return self.swap_recipe
        return self.recipe


class ShoppingList(Base):
    """Aggregated shopping list derived from a confirmed weekly plan."""
    __tablename__ = "shopping_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    weekly_plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weekly_plans.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    items: Mapped[list] = mapped_column(JSON, nullable=False)
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    weekly_plan: Mapped["WeeklyPlan"] = relationship(
        "WeeklyPlan", back_populates="shopping_list"
    )
