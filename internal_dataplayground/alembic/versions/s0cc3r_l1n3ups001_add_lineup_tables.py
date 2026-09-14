"""add soccer match lineup, goal, booking, substitution, coach tables

Revision ID: s0cc3r_l1n3ups001
Revises: m3d1um_d0ma1n002
Create Date: 2026-09-12

⚠ down_revision history for this one has moved twice already, both
times because of unrelated parallel work landing on other domains
while this migration sat unapplied:
  1st guess: j0b_sc0ut_upgr4de001 (wrong — nb4_d0ma1n001 actually chains
     after s0cc3r_d0ma1n001, not after the job scout migration)
  2nd guess: nb4_d0ma1n001 (superseded by the Medium domain's
     m3d1um_d0ma1n001 → m3d1um_d0ma1n002 landing afterward)
  Current: m3d1um_d0ma1n002, per the project owner confirming this is
  now the real head.

Given that track record, treat this as likely to drift again before
it's actually applied. Before running this migration:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's m3d1um_d0ma1n002, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before soccer lineups"
     first, then point down_revision at that new merge revision instead.

Adds the normalized tables proposed in the WO#34 conversation, built
against a real, fully-populated /live payload (Bournemouth 2-2 Brentford,
2026-09-12) rather than guessed field shapes — see
airflow/agents/soccer_agents.py::parse_match_lineup_data() for the
confirmed field mapping this schema is built from.

Creates:
  soccer_match_lineups  — one row per squad player per match (starter or
                          bench), with position code, captain flag, and
                          whether they were still on the field at the
                          final whistle.
  soccer_goals          — one row per goal, with scorer + optional
                          assist.
  soccer_bookings       — one row per card.
  soccer_substitutions  — one row per substitution event.
  soccer_coaches        — one row per coach/manager per team per match.

Alters:
  soccer_matches — adds home_formation, away_formation (e.g. "4-2-3-1",
  confirmed as a real per-team field — Tactics — not something we have
  to infer from player positions), possession_home, possession_away
  (confirmed real field: BallPossession.OverallHome/OverallAway), and
  attendance.

All five new tables are populated by a new idempotent DAG task
(life_os_soccer_ingest.py::parse_finished_match_details) that reads
already-stored raw payloads from soccer_raw_payloads — it does not call
the FIFA API. This is deliberate: every match that was already fetched
before this schema existed (regardless of when, or under what earlier
version of the ingest code) gets backfilled automatically the next time
the DAG runs, with no separate one-off script needed.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_l1n3ups001'
down_revision: Union[str, None] = 'm3d1um_d0ma1n002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── soccer_matches: new scalar columns ───────────────────────────────
    op.add_column('soccer_matches', sa.Column('home_formation', sa.String(20), nullable=True))
    op.add_column('soccer_matches', sa.Column('away_formation', sa.String(20), nullable=True))
    op.add_column('soccer_matches', sa.Column('possession_home', sa.Float(), nullable=True))
    op.add_column('soccer_matches', sa.Column('possession_away', sa.Float(), nullable=True))
    op.add_column('soccer_matches', sa.Column('attendance', sa.Integer(), nullable=True))

    # ── soccer_match_lineups ──────────────────────────────────────────────
    op.create_table(
        'soccer_match_lineups',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('match_id', sa.Integer(), sa.ForeignKey('soccer_matches.id'), nullable=False),
        sa.Column('team_side', sa.String(4), nullable=False),  # 'home' | 'away'
        sa.Column('fifa_player_id', sa.String(64), nullable=False),
        sa.Column('shirt_number', sa.Integer(), nullable=True),
        sa.Column('player_name', sa.String(255), nullable=True),
        # Confirmed real FIFA Position codes: 0=GK, 1=DEF, 2=AM/wide, 3=FWD,
        # 6=DM. Stored as the raw int rather than translated to a label —
        # translation happens at render time, so an unconfirmed code (this
        # match only exercised 0/1/2/3/6) doesn't need a schema change.
        sa.Column('position_code', sa.Integer(), nullable=True),
        sa.Column('is_starter', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('is_captain', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('on_field_at_finish', sa.Boolean(), nullable=False, server_default='0'),
    )
    op.create_index('ix_soccer_lineups_match', 'soccer_match_lineups', ['match_id'])
    op.create_unique_constraint(
        'uq_soccer_lineup_player', 'soccer_match_lineups', ['match_id', 'fifa_player_id']
    )

    # ── soccer_goals ──────────────────────────────────────────────────────
    op.create_table(
        'soccer_goals',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('match_id', sa.Integer(), sa.ForeignKey('soccer_matches.id'), nullable=False),
        sa.Column('team_side', sa.String(4), nullable=False),
        sa.Column('fifa_player_id', sa.String(64), nullable=True),
        sa.Column('fifa_assist_player_id', sa.String(64), nullable=True),
        sa.Column('minute_display', sa.String(10), nullable=True),  # e.g. "45'+4'", kept verbatim
        sa.Column('minute_numeric', sa.Integer(), nullable=True),   # e.g. 49 — for sorting/placement
        sa.Column('period', sa.Integer(), nullable=True),
    )
    op.create_index('ix_soccer_goals_match', 'soccer_goals', ['match_id'])

    # ── soccer_bookings ───────────────────────────────────────────────────
    op.create_table(
        'soccer_bookings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('match_id', sa.Integer(), sa.ForeignKey('soccer_matches.id'), nullable=False),
        sa.Column('team_side', sa.String(4), nullable=False),
        sa.Column('fifa_player_id', sa.String(64), nullable=True),
        # Confirmed: 1 = yellow. No red-card example seen yet — see
        # parse_match_lineup_data()'s docstring caveat.
        sa.Column('card_type_code', sa.Integer(), nullable=True),
        sa.Column('minute_display', sa.String(10), nullable=True),
        sa.Column('minute_numeric', sa.Integer(), nullable=True),
        sa.Column('period', sa.Integer(), nullable=True),
        sa.Column('reason', sa.String(100), nullable=True),
    )
    op.create_index('ix_soccer_bookings_match', 'soccer_bookings', ['match_id'])

    # ── soccer_substitutions ──────────────────────────────────────────────
    op.create_table(
        'soccer_substitutions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('match_id', sa.Integer(), sa.ForeignKey('soccer_matches.id'), nullable=False),
        sa.Column('team_side', sa.String(4), nullable=False),
        sa.Column('fifa_player_off_id', sa.String(64), nullable=True),
        sa.Column('fifa_player_on_id', sa.String(64), nullable=True),
        sa.Column('player_off_name', sa.String(255), nullable=True),
        sa.Column('player_on_name', sa.String(255), nullable=True),
        sa.Column('minute_display', sa.String(10), nullable=True),
        sa.Column('minute_numeric', sa.Integer(), nullable=True),
        sa.Column('period', sa.Integer(), nullable=True),
    )
    op.create_index('ix_soccer_substitutions_match', 'soccer_substitutions', ['match_id'])

    # ── soccer_coaches ────────────────────────────────────────────────────
    op.create_table(
        'soccer_coaches',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('match_id', sa.Integer(), sa.ForeignKey('soccer_matches.id'), nullable=False),
        sa.Column('team_side', sa.String(4), nullable=False),
        sa.Column('fifa_coach_id', sa.String(64), nullable=True),
        sa.Column('name', sa.String(255), nullable=True),
        # Confirmed pattern (both teams, independently): 0 = head coach,
        # 1 = assistant. Only two data points, so treated as a strong
        # signal, not certainty.
        sa.Column('role_code', sa.Integer(), nullable=True),
    )
    op.create_index('ix_soccer_coaches_match', 'soccer_coaches', ['match_id'])


def downgrade() -> None:
    op.drop_table('soccer_coaches')
    op.drop_table('soccer_substitutions')
    op.drop_table('soccer_bookings')
    op.drop_table('soccer_goals')
    op.drop_table('soccer_match_lineups')

    op.drop_column('soccer_matches', 'attendance')
    op.drop_column('soccer_matches', 'possession_away')
    op.drop_column('soccer_matches', 'possession_home')
    op.drop_column('soccer_matches', 'away_formation')
    op.drop_column('soccer_matches', 'home_formation')
