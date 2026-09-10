"""add soccer domain tables (competitions, matches, raw payloads, settings)

Revision ID: s0cc3r_d0ma1n001
Revises: j0b_sc0ut_upgr4de001
Create Date: 2026-09-09

⚠ down_revision assumes j0b_sc0ut_upgr4de001 is your current single head.
That file's own docstring flagged an earlier unmerged-heads situation
(w0rk0ut_tr4ck3r001 / m3d14_s34s0ns001 / w33kly_pl4nn3r001) — if that was
never actually resolved, j0b_sc0ut_upgr4de001 may not be the true head.
Before applying this migration:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's j0b_sc0ut_upgr4de001, you're
     good — apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before soccer domain"
     first, then point down_revision at that new merge revision instead.

Creates the four tables backing WO#34's soccer domain
(domains/soccer/models.py):

  soccer_competitions  — the watch list the daily ingest DAG
                         (airflow/dags/soccer/life_os_soccer_ingest.py)
                         iterates over. Seeded here with the same five
                         competitions that DAG's own _SEED_COMPETITIONS
                         seeds idempotently on every run — duplicated
                         intentionally, not imported, so this migration
                         has no dependency on application code (same
                         reasoning as SEED_KEYWORDS in
                         j0b_sc0ut_upgr4de001, which duplicates rather
                         than imports life_os_job_scout.py's
                         DEFAULT_SEARCHES). Both inserts are safe
                         together — the DAG's seed step checks for an
                         existing fifa_competition_id before inserting,
                         so it no-ops once this migration has already
                         created these rows.

  soccer_matches       — normalized fixtures/results, upserted daily.
                         fifa_competition_id/season/stage/match are
                         VARCHAR, not INTEGER — one real FIFA competition
                         ID (the men's Euros) is alphanumeric.

  soccer_raw_payloads  — verbatim FIFA API responses (JSON column). No
                         FK to soccer_competitions by design — this table
                         intentionally has the loosest possible shape so
                         a future new endpoint (e.g. standings) never
                         needs its own migration, just a new `endpoint`
                         string value.

  soccer_settings      — singleton row for the ingest window
                         (window_past_days / window_future_days), tuned
                         from the /soccer/settings page. Seeded here with
                         one default row so the DAG's first run doesn't
                         depend on the FastAPI app having started first.
"""
from typing import Sequence, Union
import datetime

from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_d0ma1n001'
down_revision: Union[str, None] = 'j0b_sc0ut_upgr4de001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Mirrors airflow/dags/soccer/life_os_soccer_ingest.py's _SEED_COMPETITIONS
# exactly. Duplicated here rather than imported — see module docstring.
SEED_COMPETITIONS = [
    {"fifa_competition_id": "17",         "name": "FIFA World Cup",                "backfill_from_date": datetime.date(2022, 1, 1)},
    {"fifa_competition_id": "2000001032", "name": "UEFA Champions League",         "backfill_from_date": datetime.date(2024, 7, 1)},
    {"fifa_competition_id": "2000000000", "name": "Barclays Premier League",       "backfill_from_date": datetime.date(2024, 7, 1)},
    {"fifa_competition_id": "2000000078", "name": "Campeonato Brasileiro Série A", "backfill_from_date": datetime.date(2024, 1, 1)},
    {"fifa_competition_id": "2000001035", "name": "Copa Libertadores",             "backfill_from_date": datetime.date(2024, 1, 1)},
]


def upgrade() -> None:
    # ── soccer_competitions ──────────────────────────────────────────────
    op.create_table(
        'soccer_competitions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('fifa_competition_id', sa.String(64), nullable=False, unique=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('backfill_from_date', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.bulk_insert(
        sa.table('soccer_competitions',
                 sa.column('fifa_competition_id', sa.String),
                 sa.column('name', sa.String),
                 sa.column('is_active', sa.Boolean),
                 sa.column('backfill_from_date', sa.Date)),
        [{**c, 'is_active': True} for c in SEED_COMPETITIONS],
    )

    # ── soccer_matches ────────────────────────────────────────────────────
    op.create_table(
        'soccer_matches',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('competition_id', sa.Integer(),
                  sa.ForeignKey('soccer_competitions.id'), nullable=False),
        sa.Column('fifa_competition_id', sa.String(64), nullable=False),
        sa.Column('fifa_season_id', sa.String(64), nullable=False),
        sa.Column('fifa_stage_id', sa.String(64), nullable=False),
        sa.Column('fifa_match_id', sa.String(64), nullable=False),
        sa.Column('home_team_name', sa.String(255), nullable=True),
        sa.Column('away_team_name', sa.String(255), nullable=True),
        sa.Column('home_team_score', sa.Integer(), nullable=True),
        sa.Column('away_team_score', sa.Integer(), nullable=True),
        sa.Column('kickoff_at', sa.DateTime(), nullable=True),
        sa.Column('fifa_match_status_code', sa.Integer(), nullable=True),
        sa.Column('status_label', sa.String(20), nullable=False, server_default='unknown'),
        sa.Column('venue_name', sa.String(255), nullable=True),
        sa.Column('details_fetched_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_unique_constraint(
        'uq_soccer_match_identity',
        'soccer_matches',
        ['fifa_competition_id', 'fifa_season_id', 'fifa_stage_id', 'fifa_match_id'],
    )
    op.create_index('ix_soccer_matches_kickoff', 'soccer_matches', ['kickoff_at'])

    # ── soccer_raw_payloads ───────────────────────────────────────────────
    op.create_table(
        'soccer_raw_payloads',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('endpoint', sa.String(50), nullable=False),
        sa.Column('fifa_competition_id', sa.String(64), nullable=True),
        sa.Column('fifa_match_id', sa.String(64), nullable=True),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('fetched_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        'ix_soccer_raw_payloads_lookup',
        'soccer_raw_payloads',
        ['endpoint', 'fifa_competition_id', 'fifa_match_id'],
    )

    # ── soccer_settings ───────────────────────────────────────────────────
    op.create_table(
        'soccer_settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('window_past_days', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('window_future_days', sa.Integer(), nullable=False, server_default='60'),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.bulk_insert(
        sa.table('soccer_settings',
                 sa.column('window_past_days', sa.Integer),
                 sa.column('window_future_days', sa.Integer)),
        [{'window_past_days': 3, 'window_future_days': 60}],
    )


def downgrade() -> None:
    op.drop_table('soccer_matches')
    op.drop_table('soccer_raw_payloads')
    op.drop_table('soccer_settings')
    op.drop_table('soccer_competitions')
