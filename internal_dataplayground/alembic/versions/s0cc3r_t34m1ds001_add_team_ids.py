"""add fifa team id columns to soccer_matches

Revision ID: s0cc3r_t34m1ds001
Revises: m3d1um_d0ma1n003
Create Date: 2026-09-13

⚠ Third time this exact thing has happened — this migration's
down_revision has now moved twice while it sat unapplied, both times
because unrelated parallel work landed on another domain first:
  1st: s0cc3r_l1n3ups001 (assumed head at the time)
  Current: m3d1um_d0ma1n003, which — per its own header — chains
  directly after s0cc3r_l1n3ups001, meaning this migration and
  m3d1um_d0ma1n003 were briefly siblings off the same parent until this
  rebase. Given the track record, assume this will need rechecking again
  before it's actually applied:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's m3d1um_d0ma1n003, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before soccer team ids"
     first, then point down_revision at that new merge revision instead.

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
