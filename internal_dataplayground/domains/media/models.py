"""
Media Tracker ORM models — domains/media/models.py

Moved here from the root models.py as part of the domain-folder migration
(Work Order #9). This file's only job is to define this domain's ORM +
Pydantic classes; see domains/media/routers/*.py for usage.

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

import datetime
import enum
from decimal import Decimal
from typing import Optional

from core.base_model import Base
from pydantic import BaseModel
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
