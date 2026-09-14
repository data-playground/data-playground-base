"""add medium domain tables (feed sources, articles)

Revision ID: m3d1um_d0ma1n001
Revises: nb4_d0ma1n001
Create Date: 2026-09-13

⚠ down_revision assumes nb4_d0ma1n001 is your current single head. Before
applying this migration:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's nb4_d0ma1n001, you're good —
     apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before medium domain"
     first, then point down_revision at that new merge revision instead.

Creates the two tables backing WO#35's medium domain
(domains/medium/models.py):

  medium_feed_sources    — the tracked-sources list (profiles,
                            publications, topics, custom domains),
                            managed entirely through the /medium/settings
                            page. Not seeded — starts empty, same as
                            nba_players/nba_games above; there's no
                            static list to seed the way nba_teams has.

  medium_articles         — ingested article content. Not seeded either;
                            populated by the daily RSS ingest DAG once
                            that's wired up (rss_ingest.py exists;
                            dag_db.py doesn't yet — this migration only
                            creates the table it will write into).
                            `guid` is that DAG's dedup key on upsert.

No foreign key between the two tables — deliberate, not an oversight.
medium_articles denormalizes source_type/source_identifier/source_label
at ingest time rather than joining back to medium_feed_sources on every
read (see the MediumArticle docstring in models.py for the trade-off
this implies: renaming a source's label later won't relabel articles
already stored).

medium_feed_sources doesn't get a bare index on source_type alone: its
composite UNIQUE constraint (uq_medium_feed_source) already has
source_type as its leftmost column, and InnoDB can satisfy a
source_type-only lookup from that composite index's leftmost prefix —
same reasoning nb4_d0ma1n001 gave for skipping redundant game_id
indexes above. medium_articles.published_at gets its own explicit
index since it's what the articles page's ORDER BY actually uses, and
nothing else covers it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'm3d1um_d0ma1n001'
down_revision: Union[str, None] = 'nb4_d0ma1n001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── medium_feed_sources ──────────────────────────────────────────────
    op.create_table(
        'medium_feed_sources',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('source_type', sa.String(20), nullable=False),
        sa.Column('identifier', sa.String(255), nullable=False),
        sa.Column('label', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_unique_constraint(
        'uq_medium_feed_source', 'medium_feed_sources', ['source_type', 'identifier'],
    )

    # ── medium_articles ──────────────────────────────────────────────────
    op.create_table(
        'medium_articles',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('guid', sa.String(512), nullable=False, unique=True),
        sa.Column('source_type', sa.String(20), nullable=False),
        sa.Column('source_identifier', sa.String(255), nullable=False),
        sa.Column('source_label', sa.String(255), nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('url', sa.String(1000), nullable=False),
        sa.Column('author', sa.String(255), nullable=False, server_default=''),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('content_html', sa.Text(), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_medium_articles_published_at', 'medium_articles', ['published_at'])


def downgrade() -> None:
    op.drop_table('medium_articles')
    op.drop_table('medium_feed_sources')
