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

"""
WEEKLY PLANNING MODULE — moved to domains/planning/models.py as part of
the domain-folder migration (Work Order #10 — see
domains/planning/routers/*.py for usage). Re-exported here so any other
file still doing `from models import WeeklyPlan` (etc.) keeps working
unchanged.
"""
# TODO: remove after all cross-references are updated
from domains.planning.models import (
    FitnessGoal,
    WeeklyPlanStatus,
    PlanDayStatus,
    PlanMealType,
    PlanMealStatus,
    UserIntent,
    WeeklyPlan,
    WeeklyPlanDay,
    WeeklyPlanMeal,
    ShoppingList,
)
