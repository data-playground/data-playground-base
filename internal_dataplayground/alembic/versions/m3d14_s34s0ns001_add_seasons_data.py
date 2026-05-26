"""add seasons_data to media_items

Revision ID: m3d14_s34s0ns001
Revises: m3d14_tr4ck3r001
Create Date: 2026-05-26

Adds a JSON column to media_items to store per-season episode counts
fetched from TMDB. This avoids a separate seasons table and keeps the
data alongside the item it belongs to.

Schema: {"1": 22, "2": 24, "3": 25, ...}  (season_number string → episode_count int)
String keys because JSON object keys are always strings.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'm3d14_s34s0ns001'
down_revision: Union[str, None] = 'm3d14_tr4ck3r001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'media_items',
        sa.Column('seasons_data', sa.JSON(), nullable=True,
                  comment='Per-season episode counts: {"1": 22, "2": 24, ...}')
    )


def downgrade() -> None:
    op.drop_column('media_items', 'seasons_data')
