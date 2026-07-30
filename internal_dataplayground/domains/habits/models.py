import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, ForeignKey, Integer, String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel

from core.base_model import Base

# ── HABIT TRACKER MODULE ─────────────────────────────────────────────────────
# Moved verbatim from models.py as part of the domains/habits pilot migration
# (no behavior/schema/table/column changes in that move). HabitLog's
# UniqueConstraint was added afterward, as a separate, explicitly-scoped fix
# — see that class's docstring for why and what it does/doesn't change.


class HabitSettings(Base):
    """
    Single-row global configuration for the Habit Tracker.
    Always query with LIMIT 1 — only one row ever exists.
    """
    __tablename__ = "habit_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Number of missed days still allowed within a streak window.
    # 0 = strict consecutive days. 1 = one skipped day is forgiven (default).
    # The streak calculation walks backwards from yesterday and skips up to
    # grace_period_days missed days before breaking the streak.
    grace_period_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )


class Habit(Base):
    """
    A single trackable behaviour the user wants to build.
    Supports icon (emoji) and color (hex) for visual identity on the check-in
    card. sort_order is user-controlled via drag-and-drop in the settings page.
    """
    __tablename__ = "habits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # emoji character, e.g. '💧' '🏃' '📚'
    icon: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # hex color string, e.g. '#7c6fff' — used for card accent and dot color
    color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # User-controlled display order. Lower = appears first (among incomplete habits).
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # One log per day per habit — cascade delete so removing a habit removes its history.
    logs: Mapped[list["HabitLog"]] = relationship(
        "HabitLog",
        back_populates="habit",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @property
    def color_or_default(self) -> str:
        """Returns the habit color or the CSS accent variable as fallback."""
        return self.color or "#7c6fff"


class HabitLog(Base):
    """
    One row per habit per calendar day.

    The UNIQUE constraint on (habit_id, logged_date) below makes that
    invariant explicit and enforced by SQLAlchemy/the DB, rather than an
    assumption living only in this comment. The router's INSERT path
    (`log_habit()` in domains/habits/routers/habits.py) relies on this
    constraint firing an IntegrityError on a duplicate same-day log, which
    it then treats as idempotent (already logged = success).

    NOTE: prior to this declaration, this constraint was assumed to exist
    "at the DB level" without being represented in the ORM model at all —
    i.e. `Base.metadata.create_all()` (used by tests / fresh dev DBs) never
    actually created it, silently allowing duplicate rows in any DB built
    from the models rather than from the real migration history. If your
    production DB was provisioned via Alembic and does NOT already have a
    matching unique index, a migration adding
    `uq_habit_logs_habit_date` needs to be generated and applied — this
    model change alone does not touch a live database.
    """
    __tablename__ = "habit_logs"
    __table_args__ = (
        UniqueConstraint("habit_id", "logged_date", name="uq_habit_logs_habit_date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    habit_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("habits.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The calendar date this log represents — not the insert timestamp.
    logged_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    # Optional quick note for that day, e.g. "ran 5k", "only 6 hours"
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    habit: Mapped["Habit"] = relationship("Habit", back_populates="logs")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
# NOTE: none of the four schemas below are currently referenced by
# domains/habits/routers/habits.py — every endpoint there parses raw
# `request.form()` / `request.json()` instead of using FastAPI's
# Pydantic-based request-body validation. They exist as a documented,
# type-checked description of each shape (useful for API docs, future
# refactors toward typed request bodies, or external consumers), not as
# active validation in the current request path.

class HabitCreate(BaseModel):
    """
    Shape of the fields accepted when creating a habit.

    Mirrors the form fields read manually in `create_habit()`
    (POST /habits/new) — `name` is the only required field there too
    (icon/color fall back to defaults, description to None). Not currently
    used to validate that endpoint's request body; see module-level note.
    """
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None


class HabitUpdate(BaseModel):
    """
    Shape of the fields accepted when editing a habit.

    Mirrors what `update_habit()` (PATCH /habits/{habit_id}) reads from
    form data, including `is_active` for the active/inactive toggle and
    `sort_order` for drag-and-drop reordering (though in practice reordering
    goes through PATCH /habits/reorder's raw JSON list, not this shape).
    Not currently used to validate that endpoint's request body; see
    module-level note.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None


class HabitResponse(BaseModel):
    """
    Serializable representation of a Habit row.

    `from_attributes = True` lets this be built directly from a `Habit`
    ORM instance (`HabitResponse.model_validate(habit)`), which is how
    it would be used if/when an endpoint here starts returning JSON
    instead of the HTML fragments every current handler returns.
    """
    id: int
    name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    is_active: bool
    sort_order: int

    class Config:
        """Pydantic v2 config: enables building this model from an ORM object's attributes rather than only from a dict."""
        from_attributes = True


class HabitLogResponse(BaseModel):
    """
    Serializable representation of a HabitLog row.

    Same `from_attributes` pattern as HabitResponse — would let a future
    JSON endpoint return log entries built directly from ORM instances.
    """
    id: int
    habit_id: int
    logged_date: datetime.date
    notes: Optional[str]

    class Config:
        """Pydantic v2 config: enables building this model from an ORM object's attributes rather than only from a dict."""
        from_attributes = True
