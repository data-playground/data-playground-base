"""add fifa team id columns to soccer_matches

Revision ID: s0cc3r_t34m1ds001
Revises: s0cc3r_l1n3ups001
Create Date: 2026-09-13

⚠ Same recurring caveat as the last two soccer migrations — this repo's
alembic head has moved twice already from unrelated parallel domain
work landing while a soccer migration sat unapplied. This one chains
after s0cc3r_l1n3ups001 on the assumption that migration is applied (or
being applied) immediately before this one, as part of the same
deploy. If that's not true by the time you run this — check
`alembic heads` first, same as always.

Adds soccer_matches.fifa_home_team_id / fifa_away_team_id — FIFA's
IdTeam for each side. Confirmed present on BOTH endpoints we've parsed
(the calendar endpoint's Home.IdTeam/Away.IdTeam, and the /live
endpoint's HomeTeam.IdTeam/AwayTeam.IdTeam), so this gets populated by
parse_match_summary() during the normal daily ingest — it does not
require a match to be finished or a /live fetch to have happened.

The only reason this exists: rendering a team's real crest image
requires FIFA's picture endpoint
(https://api.fifa.com/api/v3/picture/teams-{format}-{size}/{IdTeam}),
and we had never stored IdTeam anywhere because nothing needed it until
now.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_t34m1ds001'
down_revision: Union[str, None] = 'm3d1um_d0ma1n003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('soccer_matches', sa.Column('fifa_home_team_id', sa.String(64), nullable=True))
    op.add_column('soccer_matches', sa.Column('fifa_away_team_id', sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column('soccer_matches', 'fifa_away_team_id')
    op.drop_column('soccer_matches', 'fifa_home_team_id')
