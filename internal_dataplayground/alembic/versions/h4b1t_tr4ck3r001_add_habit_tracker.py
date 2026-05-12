"""add habit tracker tables

Revision ID: h4b1t_tr4ck3r001
Revises: bl0g_d1ff001
Create Date: 2026-05-07

Creates:
  - habit_settings  (single-row global config, e.g. grace_period_days)
  - habits          (habit definitions with icon, color, sort order)
  - habit_logs      (one row per habit per day, unique constraint enforced)

Seeds 6 default habits on upgrade.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'h4b1t_tr4ck3r001'
down_revision: Union[str, None] = 'bl0g_d1ff001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SEED_HABITS = [
    ('💧', 'Hydration',   'Drink enough water today',              '#00bfff', 1),
    ('🏃', 'Exercise',    'Any physical activity counts',          '#00c980', 2),
    ('📚', 'Reading',     'Any reading — book, article, or paper', '#7c6fff', 3),
    ('🧘', 'Mindfulness', 'Meditation, breathing, or quiet time',  '#e8a020', 4),
    ('🥗', 'Nutrition',   'Ate well today',                        '#00e5a0', 5),
    ('😴', 'Sleep',       'Got 7+ hours last night',               '#ff8c42', 6),
]


def upgrade() -> None:
    # ── habit_settings — single-row global config ──────────────────────────
    op.create_table(
        'habit_settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        # Number of missed days that still count toward streak continuity.
        # 0 = strict consecutive days only. 1 = one missed day allowed (default).
        sa.Column('grace_period_days', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    # Seed the single config row
    op.execute(
        "INSERT INTO habit_settings (grace_period_days) VALUES (1)"
    )

    # ── habits ─────────────────────────────────────────────────────────────
    op.create_table(
        'habits',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.String(255), nullable=True),
        sa.Column('icon', sa.String(10), nullable=True),   # emoji
        sa.Column('color', sa.String(7), nullable=True),   # hex e.g. '#7c6fff'
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        # Custom ordering controlled by the user via drag-and-drop in settings.
        sa.Column('sort_order', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_habits_sort_order', 'habits', ['sort_order'])

    # Seed default habits
    op.bulk_insert(
        sa.table('habits',
                 sa.column('icon', sa.String),
                 sa.column('name', sa.String),
                 sa.column('description', sa.String),
                 sa.column('color', sa.String),
                 sa.column('sort_order', sa.Integer),
                 sa.column('is_active', sa.Boolean)),
        [
            {'icon': icon, 'name': name, 'description': desc,
             'color': color, 'sort_order': order, 'is_active': True}
            for icon, name, desc, color, order in SEED_HABITS
        ]
    )

    # ── habit_logs ─────────────────────────────────────────────────────────
    op.create_table(
        'habit_logs',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            'habit_id',
            sa.Integer(),
            sa.ForeignKey('habits.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
        ),
        # One log row per habit per calendar day — enforced by unique constraint.
        sa.Column('logged_date', sa.Date(), nullable=False),
        sa.Column('notes', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    # Prevents double-logging the same habit on the same day.
    op.create_unique_constraint(
        'uq_habit_logs_habit_date',
        'habit_logs',
        ['habit_id', 'logged_date'],
    )
    # The heatmap and streak queries filter by date range — this index is critical.
    op.create_index('ix_habit_logs_logged_date', 'habit_logs', ['logged_date'])


def downgrade() -> None:
    op.drop_index('ix_habit_logs_logged_date', table_name='habit_logs')
    op.drop_constraint('uq_habit_logs_habit_date', 'habit_logs', type_='unique')
    op.drop_table('habit_logs')

    op.drop_index('ix_habits_sort_order', table_name='habits')
    op.drop_table('habits')

    op.drop_table('habit_settings')
