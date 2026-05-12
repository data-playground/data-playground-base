"""add journal tables

Revision ID: d41ly_j0urn4l001
Revises: bl0g_d1ff001
Create Date: 2026-05-10

Creates:
  - journal_entries   (daily mood/energy/content entries)
  - weekly_syntheses  (AI-generated weekly pattern summaries)

Privacy note: content, gratitude, and challenges fields are NEVER
sent to external AI APIs. Synthesis is generated from numeric scores only.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'd41ly_j0urn4l001'
down_revision: Union[str, None] = 'bl0g_d1ff001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── journal_entries ────────────────────────────────────────────────────────
    op.create_table(
        'journal_entries',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),

        # One entry per day — enforced at DB level with UNIQUE constraint
        sa.Column('entry_date', sa.Date(), nullable=False),

        # Numeric scores — the ONLY data ever sent to external AI
        sa.Column('mood_score', sa.SmallInteger(), nullable=True),    # 1-5
        sa.Column('energy_score', sa.SmallInteger(), nullable=True),  # 1-5

        # Private freeform text — NEVER leaves the server
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('gratitude', sa.Text(), nullable=True),
        sa.Column('challenges', sa.Text(), nullable=True),

        # Locked 24 hours after created_at — enforced by router and nightly DAG
        sa.Column('is_locked', sa.Boolean(), nullable=False, server_default='0'),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_unique_constraint(
        'uq_journal_entries_entry_date',
        'journal_entries',
        ['entry_date'],
    )
    op.create_index(
        'ix_journal_entries_entry_date',
        'journal_entries',
        ['entry_date'],
    )

    # ── weekly_syntheses ───────────────────────────────────────────────────────
    op.create_table(
        'weekly_syntheses',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),

        # Week boundaries — Monday to Sunday
        sa.Column('week_start_date', sa.Date(), nullable=False),
        sa.Column('week_end_date', sa.Date(), nullable=False),

        # Aggregated numeric metrics from the week
        sa.Column('avg_mood', sa.Numeric(3, 2), nullable=True),
        sa.Column('avg_energy', sa.Numeric(3, 2), nullable=True),
        sa.Column('habits_completion_rate', sa.Numeric(5, 2), nullable=True),
        sa.Column('workout_count', sa.Integer(), nullable=True),

        # AI-generated analysis — built from numbers only, not from personal text
        sa.Column('synthesis_text', sa.Text(), nullable=False),

        # Audit fields
        sa.Column('data_sources', sa.JSON(), nullable=True),
        sa.Column('generated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('model_used', sa.String(100), nullable=True),
    )
    op.create_unique_constraint(
        'uq_weekly_syntheses_week_start',
        'weekly_syntheses',
        ['week_start_date'],
    )
    op.create_index(
        'ix_weekly_syntheses_week_start',
        'weekly_syntheses',
        ['week_start_date'],
    )


def downgrade() -> None:
    op.drop_index('ix_weekly_syntheses_week_start', table_name='weekly_syntheses')
    op.drop_constraint('uq_weekly_syntheses_week_start', 'weekly_syntheses', type_='unique')
    op.drop_table('weekly_syntheses')

    op.drop_index('ix_journal_entries_entry_date', table_name='journal_entries')
    op.drop_constraint('uq_journal_entries_entry_date', 'journal_entries', type_='unique')
    op.drop_table('journal_entries')
