"""add penalty scores and event-derived stat columns to soccer_matches

Revision ID: s0cc3r_st4ts001
Revises: m3d1um_d0ma1n004
Create Date: 2026-09-18

⚠ This one's down_revision has already moved once before being applied
— it was originally written against s0cc3r_t34m1ds001, which
m3d1um_d0ma1n004 (an unrelated Medium-domain fix) then chained after,
per that migration's own header. Given this project's established
pattern of the head moving between sessions, don't take m3d1um_d0ma1n004
on faith either:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's m3d1um_d0ma1n004, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before soccer stats"
     first, then point down_revision at that new merge revision instead.

Adds:
  home_penalty_score, away_penalty_score — confirmed real fields
  (HomeTeamPenaltyScore/AwayTeamPenaltyScore on the /live endpoint),
  directly motivated by a real penalty-shootout match this conversation
  worked through (goals recorded with an unconfirmed Period code turned
  out to be shootout kicks, confirmed by the project owner having
  watched the match — not something inferable from the payload alone).

  shots_home/away, corners_home/away, fouls_home/away, offsides_home/away
  — derived from the /timelines event stream by counting clean, structured
  Type codes per team (12=Attempt at Goal, 16=Corner, 18=Foul,
  15=Offside) — see airflow/agents/soccer_agents.py::parse_match_event_stats()
  for the parser and its explicit note on what's deliberately NOT
  attempted (shots on target specifically, saves/blocks) because those
  would need Qualifier decoding or English-text matching rather than a
  bare Type code.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_st4ts001'
down_revision: Union[str, None] = 'm3d1um_d0ma1n004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('soccer_matches', sa.Column('home_penalty_score', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('away_penalty_score', sa.Integer(), nullable=True))

    op.add_column('soccer_matches', sa.Column('shots_home', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('shots_away', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('corners_home', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('corners_away', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('fouls_home', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('fouls_away', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('offsides_home', sa.Integer(), nullable=True))
    op.add_column('soccer_matches', sa.Column('offsides_away', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('soccer_matches', 'offsides_away')
    op.drop_column('soccer_matches', 'offsides_home')
    op.drop_column('soccer_matches', 'fouls_away')
    op.drop_column('soccer_matches', 'fouls_home')
    op.drop_column('soccer_matches', 'corners_away')
    op.drop_column('soccer_matches', 'corners_home')
    op.drop_column('soccer_matches', 'shots_away')
    op.drop_column('soccer_matches', 'shots_home')
    op.drop_column('soccer_matches', 'away_penalty_score')
    op.drop_column('soccer_matches', 'home_penalty_score')
