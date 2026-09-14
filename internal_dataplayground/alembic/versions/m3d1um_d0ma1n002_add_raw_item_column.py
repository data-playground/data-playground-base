"""add raw_item column to medium_articles

Revision ID: m3d1um_d0ma1n002
Revises: m3d1um_d0ma1n001
Create Date: 2026-09-13

⚠ down_revision assumes m3d1um_d0ma1n001 is your current single head.
Same verification as always — run `alembic heads` first. If you
haven't actually applied m3d1um_d0ma1n001 yet, that's fine too:
running both in sequence for the first time works exactly the same as
running this one afterward.

Adds medium_articles.raw_item — the original <item> element from the
RSS feed, serialized back to a string exactly as parse_feed() produced
it (see rss_ingest.py's ParsedArticle.raw_item). Exists so a field
nobody thought to extract today can be backfilled from already-stored
rows later, instead of needing a re-fetch that may no longer be
possible — Medium's live feed only ever shows the ~10-25 most recent
items per source.

server_default='' (not nullable=True) so this stays consistent with
summary/content_html on the same table — an empty string, not NULL, is
this table's "nothing captured" sentinel throughout.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'm3d1um_d0ma1n002'
down_revision: Union[str, None] = 'm3d1um_d0ma1n001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'medium_articles',
        sa.Column('raw_item', sa.Text(), nullable=False, server_default=''),
    )


def downgrade() -> None:
    op.drop_column('medium_articles', 'raw_item')
