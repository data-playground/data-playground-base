import datetime
from decimal import Decimal
from typing import Optional

from core.base_model import Base
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column


# ── JOURNAL MODULE ────────────────────────────────────────────────────────────
# Append these classes to the bottom of models.py
#
# PRIVACY ARCHITECTURE — HARD CONSTRAINT:
#   content, gratitude, and challenges fields are NEVER sent to external AI.
#   Weekly synthesis is generated from mood_score and energy_score ONLY.
#   Violating this constraint is a critical privacy bug.
class JournalEntry(Base):
    """
    One row per calendar day. Entry text is private — never sent externally.
    Numeric scores (mood, energy) feed the weekly synthesis DAG.

    Locking: entries become read-only 24 hours after created_at.
    Enforcement: router checks is_locked before accepting edits;
    the nightly DAG sets is_locked=True as a cleanup pass.
    """
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # DB-level UNIQUE constraint ensures one entry per day
    entry_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)

    # 1–5 scales. NULL = not rated for the day (perfectly valid).
    mood_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    energy_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # ── PRIVATE TEXT — NEVER SENT TO EXTERNAL AI ──────────────────────────────
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gratitude: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    challenges: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # ──────────────────────────────────────────────────────────────────────────

    # Once True, the router rejects edits. Set by router check and nightly DAG.
    is_locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    @property
    def hours_until_lock(self) -> float:
        """Hours remaining before this entry locks. Negative means already past."""
        lock_at = self.created_at + datetime.timedelta(hours=24)
        delta = lock_at - datetime.datetime.utcnow()
        return delta.total_seconds() / 3600

    @property
    def should_be_locked(self) -> bool:
        """True if 24 hours have elapsed since creation."""
        return self.hours_until_lock <= 0

    @property
    def mood_label(self) -> str:
        return {1: "😞", 2: "😕", 3: "😐", 4: "🙂", 5: "😄"}.get(self.mood_score or 0, "—")

    @property
    def energy_label(self) -> str:
        return {1: "▁", 2: "▂", 3: "▄", 4: "▆", 5: "█"}.get(self.energy_score or 0, "—")

    @property
    def mood_color_class(self) -> str:
        """CSS class suffix for mood-dot coloring on the calendar."""
        if not self.mood_score:
            return "none"
        if self.mood_score <= 2:
            return "low"
        if self.mood_score == 3:
            return "mid"
        return "high"


class WeeklySynthesis(Base):
    """
    AI-generated weekly pattern summary. Built exclusively from numeric scores
    and habit/workout counts — no personal text from journal_entries.

    data_sources JSON records which Life OS modules contributed data,
    allowing honest attribution in the UI and future auditing.
    """
    __tablename__ = "weekly_syntheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Week boundaries — Monday to Sunday, enforced by the DAG
    week_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)
    week_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    # Aggregated from journal_entries for the week
    avg_mood: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 2), nullable=True)
    avg_energy: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 2), nullable=True)

    # From habit_logs (Phase 2) — None if habit module not yet built
    habits_completion_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2), nullable=True)

    # From workout_sessions (Phase 4) — None until that phase is built
    workout_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # The AI output — generated from numbers only, never from personal text
    synthesis_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Audit: which modules fed this synthesis, and which model produced it
    data_sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    @property
    def week_label(self) -> str:
        """Human-readable week range, e.g. 'May 4 – 10, 2026'."""
        if self.week_start_date.month == self.week_end_date.month:
            return (
                f"{self.week_start_date.strftime('%b %-d')} – "
                f"{self.week_end_date.strftime('%-d, %Y')}"
            )
        return (
            f"{self.week_start_date.strftime('%b %-d')} – "
            f"{self.week_end_date.strftime('%b %-d, %Y')}"
        )
