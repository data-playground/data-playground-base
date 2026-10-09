"""add team picture url columns to soccer_matches

Revision ID: s0cc3r_cr3sts001
Revises: nb4_t34ms_null4bl3001
Create Date: 2026-10-07

Stores FIFA's own PictureUrl template (HomeTeam/AwayTeam.PictureUrl on the
/live payload) so crest/flag images use the URL FIFA supplies rather than
one built from IdTeam. The templates keep their literal {format}/{size}
placeholders; they are filled in at render time.

down_revision was the single head when written. Before applying run
`alembic heads`; if it prints a different single hash, change
down_revision to it.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_cr3sts001'
down_revision: Union[str, None] = 'nb4_t34ms_null4bl3001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('soccer_matches', sa.Column('home_team_picture_url', sa.String(255), nullable=True))
    op.add_column('soccer_matches', sa.Column('away_team_picture_url', sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column('soccer_matches', 'away_team_picture_url')
    op.drop_column('soccer_matches', 'home_team_picture_url')
