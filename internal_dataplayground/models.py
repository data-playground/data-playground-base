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
# Moved to domains/recipes/models.py as part of the domain-folder migration
# (Work Order #7). Re-exported here so any other file still doing
# `from models import Recipe` (etc.) keeps working unchanged.
# TODO: remove after all cross-references are updated

from domains.recipes.models import (
    IngredientCategory,
    RecipeSourceType,
    RecipeMealType,
    RecipeDifficulty,
    IngredientUnit,
    Ingredient,
    RecipeTag,
    recipe_tags_junction,
    Recipe,
    RecipeIngredient,
    PantryItem,
    IngredientResponse,
    RecipeIngredientResponse,
    RecipeTagResponse,
    RecipeResponse,
    RecipeCreate,
    PantryItemResponse,
)


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
