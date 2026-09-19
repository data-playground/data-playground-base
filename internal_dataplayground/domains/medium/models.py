# domains/medium/models.py
"""
Persistence for domains/medium — currently just the tracked-sources
list. Article storage (the table the daily RSS ingest will actually
write ParsedArticle rows into) is intentionally not modeled yet — per
WO#35, RSS automation itself is on hold pending confirmation that
rss_ingest.py works against real Medium feeds, so there's no consumer
for an articles table yet. Adding it now would be guessing at the
shape before we've seen real data.

`MediumFeedSource` is deliberately NOT named `FeedSource` — that name
is already used by the plain dataclass in rss_ingest.py that
`ingest_sources()` consumes. Two different things (a DB row vs. an
in-memory value object handed to the ingest function) sharing one name
across modules is exactly the kind of thing that causes a confusing
bug later, so the DB model gets its own name and routers/services
translate between the two explicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import Mapped, mapped_column

from core.base_model import Base


class MediumFeedSource(Base):
    __tablename__ = "medium_feed_sources"
    __table_args__ = (
        UniqueConstraint("source_type", "identifier", name="uq_medium_feed_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # Mirrors rss_ingest.FeedSourceType's values ("profile", "publication",
    # "topic", "custom_domain") but stored as a plain string rather than a
    # DB-level enum, so adding a new source type never needs a migration —
    # only rss_ingest.py's FeedSourceType and the settings form need updating.
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)

    identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class MediumArticle(Base):
    """
    One ingested article. `guid` is the dedup key the eventual
    dag_db.py upserts on — Medium's own post id via RSS <guid>, stable
    across re-fetches of the same feed.

    `source_label`/`source_identifier`/`source_type` are denormalized
    from MediumFeedSource at ingest time rather than joined at read
    time. Trade-off, stated explicitly rather than silently: renaming a
    FeedSource's label later won't retroactively relabel articles
    already stored — acceptable for a personal reading list, worth
    revisiting if that ever matters.

    No thumbnail column yet — rss_ingest.py doesn't extract one from
    the feed (no <media:thumbnail>/<enclosure> handling today), so the
    articles page renders a placeholder icon instead of a real image.
    Natural follow-on, not built speculatively ahead of a real need.
    """
    __tablename__ = "medium_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    guid: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)

    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    source_label: Mapped[str | None] = mapped_column(String(255), nullable=True)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    tags: Mapped[list] = mapped_column(JSON, default=list)
    # MEDIUMTEXT (16MB), not Text/TEXT (65,535 bytes) — a real article's
    # full body, or its full verbatim <item> XML, can and does exceed
    # MySQL's 64KB TEXT cap; see the bug report this widening fixes
    # (WO#35, 2026-09-18: DataError 1406 on raw_item, which rolled back
    # an entire day's ingest batch across all 4 sources, not just the
    # one oversized row — every article in ingest_sources() currently
    # lands in one execute_many() transaction).
    summary: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=False, default="")
    content_html: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=False, default="")

    # Verbatim <item> XML, straight from rss_ingest.ParsedArticle.raw_item.
    # Exists so a field nobody thought to extract today can be backfilled
    # from stored rows later, instead of needing a re-fetch that may no
    # longer be possible — Medium's live feed only shows the ~10-25 most
    # recent items per source.
    raw_item: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=False, default="")

    # First real <img> src found in content_html (rss_ingest.extract_thumbnail()),
    # skipping Medium's own 1x1 view-tracking pixel. Nullable, not a
    # placeholder default — some articles genuinely have no image at
    # all, and the template renders the SVG placeholder for those, not
    # an empty-string src.
    thumbnail_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
