"""add user intent and weekly planning tables

Revision ID: w33kly_pl4nn3r001
Revises: m3d14_tr4ck3r001
Create Date: 2026-06-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'w33kly_pl4nn3r001'
down_revision: Union[str, None] = 'm3d14_tr4ck3r001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── user_intent ────────────────────────────────────────────────────────────
    # Single-row table (always query with LIMIT 1).
    # Stores the user's current fitness goal and preferences.
    # All AI generators read from this before producing suggestions.
    op.create_table(
        'user_intent',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),

        # Primary fitness goal
        sa.Column('fitness_goal',
                  sa.Enum('weight_loss', 'muscle_gain', 'maintenance',
                          'endurance', 'general_health',
                          name='fitnessgoal'),
                  nullable=False, server_default='weight_loss'),

        # Weekly targets
        sa.Column('weekly_workout_days', sa.SmallInteger(),
                  nullable=False, server_default='4'),
        sa.Column('target_calories', sa.Integer(), nullable=True),
        # 'high_protein' | 'balanced' | 'low_carb' | 'flexible'
        sa.Column('macro_preference', sa.String(30),
                  nullable=False, server_default='high_protein'),

        # Cooking preferences
        # 'minimal' (<20 min) | 'moderate' (20-45 min) | 'generous' (45+ min)
        sa.Column('cooking_time_preference', sa.String(20),
                  nullable=False, server_default='moderate'),

        # Free-text fields — fed directly into AI prompts
        sa.Column('dietary_restrictions', sa.Text(), nullable=True),
        sa.Column('food_preferences', sa.Text(), nullable=True),
        # e.g. "big sweet tooth, love chocolate and fruit-based desserts"
        sa.Column('food_dislikes', sa.Text(), nullable=True),
        sa.Column('health_notes', sa.Text(), nullable=True),
        # e.g. "bad left knee, avoid high-impact on Tuesdays"

        sa.Column('updated_at', sa.DateTime(),
                  nullable=False, server_default=sa.func.now(),
                  onupdate=sa.func.now()),
    )

    # ── weekly_plans ───────────────────────────────────────────────────────────
    # One row per planned week. week_start_date is always a Monday.
    # status flow: draft → confirmed → active → completed
    op.create_table(
        'weekly_plans',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('week_start_date', sa.Date(), nullable=False, unique=True),
        sa.Column('week_end_date', sa.Date(), nullable=False),

        # Snapshot of intent at generation time
        # (intent may change; plan should reflect what was set when generated)
        sa.Column('intent_snapshot', sa.JSON(), nullable=True),

        sa.Column('status',
                  sa.Enum('draft', 'confirmed', 'active', 'completed',
                          name='weeklyplanstatus'),
                  nullable=False, server_default='draft'),

        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('ai_run_id', sa.String(255), nullable=True),

        # Adherence metrics (populated as the week progresses)
        sa.Column('meals_planned', sa.SmallInteger(),
                  nullable=False, server_default='0'),
        sa.Column('meals_followed', sa.SmallInteger(),
                  nullable=False, server_default='0'),
        sa.Column('workouts_planned', sa.SmallInteger(),
                  nullable=False, server_default='0'),
        sa.Column('workouts_completed', sa.SmallInteger(),
                  nullable=False, server_default='0'),

        sa.Column('created_at', sa.DateTime(),
                  nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(),
                  nullable=False, server_default=sa.func.now(),
                  onupdate=sa.func.now()),
    )
    op.create_index('ix_weekly_plans_week_start',
                    'weekly_plans', ['week_start_date'])
    op.create_index('ix_weekly_plans_status',
                    'weekly_plans', ['status'])

    # ── weekly_plan_days ───────────────────────────────────────────────────────
    # One row per calendar day within a weekly plan.
    # day_number: 1=Monday … 7=Sunday
    op.create_table(
        'weekly_plan_days',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('weekly_plan_id', sa.Integer(),
                  sa.ForeignKey('weekly_plans.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('plan_date', sa.Date(), nullable=False),
        sa.Column('day_number', sa.SmallInteger(), nullable=False),  # 1-7

        # Workout linkage
        # NULL = rest day. Points to a WorkoutSession stub created during generation.
        sa.Column('workout_session_id', sa.BigInteger(),
                  sa.ForeignKey('workout_sessions.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('is_rest_day', sa.Boolean(),
                  nullable=False, server_default='0'),

        # Day-level override info
        sa.Column('day_status',
                  sa.Enum('planned', 'active', 'completed', 'skipped',
                          name='plandaystatus'),
                  nullable=False, server_default='planned'),
        sa.Column('override_reason', sa.String(255), nullable=True),
        # e.g. "work lunch", "knee pain", "travel day"

        # Journal linkage (auto-linked when user logs that day)
        sa.Column('journal_entry_id', sa.Integer(),
                  sa.ForeignKey('journal_entries.id', ondelete='SET NULL'),
                  nullable=True),

        sa.Column('notes', sa.Text(), nullable=True),
    )
    op.create_unique_constraint(
        'uq_plan_day_date', 'weekly_plan_days',
        ['weekly_plan_id', 'plan_date']
    )

    # ── weekly_plan_meals ──────────────────────────────────────────────────────
    # One row per planned meal within a day.
    op.create_table(
        'weekly_plan_meals',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('plan_day_id', sa.Integer(),
                  sa.ForeignKey('weekly_plan_days.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('recipe_id', sa.Integer(),
                  sa.ForeignKey('recipes.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('meal_type',
                  sa.Enum('breakfast', 'lunch', 'dinner', 'snack',
                          name='planmealtype'),
                  nullable=False),
        sa.Column('sort_order', sa.SmallInteger(),
                  nullable=False, server_default='0'),

        # Execution status
        sa.Column('status',
                  sa.Enum('planned', 'eaten', 'swapped', 'off_plan', 'skipped',
                          name='planmealstatus'),
                  nullable=False, server_default='planned'),
        sa.Column('swap_recipe_id', sa.Integer(),
                  sa.ForeignKey('recipes.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('off_plan_note', sa.String(255), nullable=True),
        # e.g. "work lunch — sushi"
    )

    # ── shopping_lists ─────────────────────────────────────────────────────────
    # Generated from confirmed weekly_plan. One per week.
    op.create_table(
        'shopping_lists',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('weekly_plan_id', sa.Integer(),
                  sa.ForeignKey('weekly_plans.id', ondelete='CASCADE'),
                  nullable=False, unique=True),
        # JSON: [{ingredient, quantity, unit, category, have_in_pantry}]
        sa.Column('items', sa.JSON(), nullable=False),
        sa.Column('generated_at', sa.DateTime(),
                  nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('shopping_lists')
    op.drop_table('weekly_plan_meals')
    op.drop_table('weekly_plan_days')
    op.drop_table('weekly_plans')
    op.drop_table('user_intent')

    op.execute("DROP TYPE IF EXISTS planmealstatus")
    op.execute("DROP TYPE IF EXISTS plandaystatus")
    op.execute("DROP TYPE IF EXISTS weeklyplanstatus")
    op.execute("DROP TYPE IF EXISTS fitnessgoal")
    op.execute("DROP TYPE IF EXISTS planmealtype")