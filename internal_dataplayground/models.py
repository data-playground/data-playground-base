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

# ── WORKOUT TRACKER MODULE ────────────────────────────────────────────────────
# Moved to domains/workout/models.py as part of the domain-folder migration
# (Work Order #8 — see domains/workout/routers/*.py for usage). Re-exported
# here so any other file still doing `from models import WorkoutSession`
# (etc.) keeps working unchanged.
# TODO: remove after all cross-references are updated
from domains.workout.models import (
    LocationType,
    EquipmentType,
    MuscleGroup,
    ExerciseEquipmentType,
    PlanOrigin,
    WorkoutGoal,
    WeightUnit,
    WorkoutLocation,
    Equipment,
    Exercise,
    WorkoutPlan,
    WorkoutPlanDay,
    WorkoutPlanExercise,
    WorkoutSession,
    WorkoutSet,
    BodyMetric,
)

"""
MEDIA MODULE — moved to domains/media/models.py as part of the
domain-folder migration (Work Order #9). Re-exported here so any other
file still doing `from models import MediaItem` (etc.) keeps working
unchanged. Unlike every prior domain, no other file in the codebase
currently consumes this shim (confirmed during WO#9) — it exists purely
for forward-compatibility/consistency with the other domains' shims and
can likely be removed in the same pass as the others once the shim-removal
cleanup work order runs.
"""
# TODO: remove after all cross-references are updated
from domains.media.models import (
    PREDEFINED_MOOD_TAGS,
    MediaExternalSource,
    MediaType,
    UserMediaStatus,
    RecommendationMediaType,
    StreamingService,
    MediaItem,
    UserMedia,
    TVSeasonProgress,
    MediaRecommendation,
    MediaItemResponse,
    UserMediaCreate,
    UserMediaUpdate,
    UserMediaResponse,
    StreamingServiceResponse,
)

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
