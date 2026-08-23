"""
domains/workout/models.py

Workout Tracker domain — ORM models.
Moved from root models.py as part of Work Order #8 (see migration_docs/GOVERNANCE.md).
Imports Base from core.base_model directly (per §2.1) rather than the
root models.py re-export.
"""
import datetime
import enum
from decimal import Decimal
from typing import Optional

from core.base_model import Base
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship


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


class WorkoutLocation(Base):
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
        types = list({e.equipment_type.value for e in self.active_equipment})
        return ", ".join(sorted(types)) if types else "No equipment logged"


class Equipment(Base):
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
    __tablename__ = "exercises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    primary_muscle_group: Mapped[MuscleGroup] = mapped_column(
        Enum(MuscleGroup, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
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
    __tablename__ = "workout_plan_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    day_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    day_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan: Mapped["WorkoutPlan"] = relationship("WorkoutPlan", back_populates="days")
    exercises: Mapped[list["WorkoutPlanExercise"]] = relationship(
        "WorkoutPlanExercise", back_populates="plan_day",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutPlanExercise.order_in_day",
    )


class WorkoutPlanExercise(Base):
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
    fatigue_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
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
        return self.started_at is not None and self.ended_at is None

    @property
    def working_sets(self) -> list["WorkoutSet"]:
        return [s for s in self.sets if not s.is_warmup]

    @property
    def exercise_count(self) -> int:
        return len({s.exercise_id for s in self.working_sets})

    @property
    def total_volume_lb(self) -> Decimal:
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
        if self.weight_used is None:
            return Decimal("0")
        return self.weight_used * self.reps_completed


class BodyMetric(Base):
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

