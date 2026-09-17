"""add thumbnail_url column to medium_articles

Revision ID: m3d1um_d0ma1n003
Revises: s0cc3r_l1n3ups001
Create Date: 2026-09-16

⚠ down_revision assumes s0cc3r_l1n3ups001 is your current single head —
per its own docstring, that chain has already drifted twice from
unrelated parallel work landing on other domains, so don't take this on
faith. Same check as always:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's s0cc3r_l1n3ups001, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before medium thumbnails"
     first, then point down_revision at that new merge revision instead.

Adds medium_articles.thumbnail_url — the first real <img> src found in
an article's content_html (rss_ingest.extract_thumbnail()), skipping
Medium's own 1x1 view-tracking pixel that appears at the end of every
item. Confirmed against a real feed (a Google Cloud publication) that
this isn't always the literal first element in the body — several
articles open with a paragraph, sometimes a heading and more than one
paragraph, before their lead image appears — so extraction scans the
whole body for the first <img> in document order rather than checking
only the first child.

nullable=True, no server_default — unlike raw_item/summary/content_html
on this same table, an empty string isn't the right "nothing here"
sentinel for a URL column. Some articles genuinely have no image at
all; the articles page renders the existing SVG placeholder for those
rows rather than an empty <img src="">.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'm3d1um_d0ma1n003'
down_revision: Union[str, None] = 's0cc3r_l1n3ups001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'medium_articles',
        sa.Column('thumbnail_url', sa.String(1000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('medium_articles', 'thumbnail_url')
