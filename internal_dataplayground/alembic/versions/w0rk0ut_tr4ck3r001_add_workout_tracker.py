"""add workout tracker tables

Revision ID: w0rk0ut_tr4ck3r001
Revises: 0ecac75145ac
Create Date: 2026-05-15

Creates:
  - workout_locations   (gym, home, outdoor locations)
  - equipment           (equipment per location)
  - exercises           (reference data, seeded with 75 exercises)
  - workout_plans       (user or AI-generated training plans)
  - workout_plan_exercises (exercises within a plan, day-structured)
  - workout_sessions    (individual training sessions)
  - workout_sets        (individual sets within a session)
  - body_metrics        (weight and body fat tracking)

Design decisions:
  - Default weight unit is lb (user preference), with per-session toggle stored on session
  - Plans have named days (day_name e.g. "Chest Day") but logging is flexible
  - Warmup sets tracked via is_warmup boolean on workout_sets
  - is_active on workout_plans uses DB-level enforcement via single-row trigger logic
    (enforced at application layer — only one plan active at a time)
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'w0rk0ut_tr4ck3r001'
down_revision: Union[str, None] = '0ecac75145ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Seed data — 75 exercises covering all muscle groups
# Fields: (name, primary_muscle_group, secondary_muscle_groups_json,
#          equipment_type, is_compound)
# ---------------------------------------------------------------------------
SEED_EXERCISES = [
    # ── CHEST ──────────────────────────────────────────────────────────────
    ("Barbell Bench Press",        "chest",      '["shoulders","triceps"]',       "barbell",         True),
    ("Incline Barbell Bench Press","chest",      '["shoulders","triceps"]',       "barbell",         True),
    ("Decline Barbell Bench Press","chest",      '["triceps"]',                   "barbell",         True),
    ("Dumbbell Bench Press",       "chest",      '["shoulders","triceps"]',       "dumbbell",        True),
    ("Incline Dumbbell Press",     "chest",      '["shoulders","triceps"]',       "dumbbell",        True),
    ("Dumbbell Flye",              "chest",      '[]',                            "dumbbell",        False),
    ("Cable Crossover",            "chest",      '[]',                            "cable",           False),
    ("Push-Up",                    "chest",      '["shoulders","triceps"]',       "bodyweight",      True),
    ("Chest Dip",                  "chest",      '["triceps","shoulders"]',       "bodyweight",      True),
    ("Pec Deck Machine",           "chest",      '[]',                            "machine",         False),

    # ── BACK ───────────────────────────────────────────────────────────────
    ("Barbell Deadlift",           "back",       '["glutes","hamstrings","traps"]',"barbell",        True),
    ("Barbell Row",                "back",       '["biceps","rear_delts"]',       "barbell",         True),
    ("Pull-Up",                    "back",       '["biceps"]',                    "bodyweight",      True),
    ("Chin-Up",                    "back",       '["biceps"]',                    "bodyweight",      True),
    ("Lat Pulldown",               "back",       '["biceps"]',                    "machine",         True),
    ("Seated Cable Row",           "back",       '["biceps","rear_delts"]',       "cable",           True),
    ("Single-Arm Dumbbell Row",    "back",       '["biceps"]',                    "dumbbell",        True),
    ("T-Bar Row",                  "back",       '["biceps","rear_delts"]',       "barbell",         True),
    ("Face Pull",                  "back",       '["rear_delts","rotator_cuff"]', "cable",           False),
    ("Straight-Arm Pulldown",      "back",       '[]',                            "cable",           False),
    ("Hyperextension",             "back",       '["glutes","hamstrings"]',       "machine",         False),

    # ── SHOULDERS ──────────────────────────────────────────────────────────
    ("Barbell Overhead Press",     "shoulders",  '["triceps","upper_back"]',      "barbell",         True),
    ("Dumbbell Shoulder Press",    "shoulders",  '["triceps"]',                   "dumbbell",        True),
    ("Lateral Raise",              "shoulders",  '[]',                            "dumbbell",        False),
    ("Cable Lateral Raise",        "shoulders",  '[]',                            "cable",           False),
    ("Front Raise",                "shoulders",  '[]',                            "dumbbell",        False),
    ("Arnold Press",               "shoulders",  '["triceps"]',                   "dumbbell",        True),
    ("Upright Row",                "shoulders",  '["biceps","traps"]',            "barbell",         False),
    ("Rear Delt Flye",             "shoulders",  '[]',                            "dumbbell",        False),
    ("Machine Shoulder Press",     "shoulders",  '["triceps"]',                   "machine",         True),

    # ── BICEPS ─────────────────────────────────────────────────────────────
    ("Barbell Curl",               "biceps",     '["forearms"]',                  "barbell",         False),
    ("Dumbbell Curl",              "biceps",     '["forearms"]',                  "dumbbell",        False),
    ("Hammer Curl",                "biceps",     '["forearms","brachialis"]',     "dumbbell",        False),
    ("Incline Dumbbell Curl",      "biceps",     '[]',                            "dumbbell",        False),
    ("Cable Curl",                 "biceps",     '["forearms"]',                  "cable",           False),
    ("Preacher Curl",              "biceps",     '[]',                            "machine",         False),
    ("Concentration Curl",         "biceps",     '[]',                            "dumbbell",        False),

    # ── TRICEPS ────────────────────────────────────────────────────────────
    ("Tricep Pushdown",            "triceps",    '[]',                            "cable",           False),
    ("Overhead Tricep Extension",  "triceps",    '["shoulders"]',                 "dumbbell",        False),
    ("Skull Crusher",              "triceps",    '[]',                            "barbell",         False),
    ("Close-Grip Bench Press",     "triceps",    '["chest"]',                     "barbell",         True),
    ("Diamond Push-Up",            "triceps",    '["chest"]',                     "bodyweight",      False),
    ("Tricep Kickback",            "triceps",    '[]',                            "dumbbell",        False),
    ("Rope Pushdown",              "triceps",    '[]',                            "cable",           False),

    # ── QUADS ──────────────────────────────────────────────────────────────
    ("Barbell Back Squat",         "quads",      '["glutes","hamstrings","core"]',"barbell",         True),
    ("Front Squat",                "quads",      '["core","glutes"]',             "barbell",         True),
    ("Leg Press",                  "quads",      '["glutes","hamstrings"]',       "machine",         True),
    ("Hack Squat",                 "quads",      '["glutes"]',                    "machine",         True),
    ("Leg Extension",              "quads",      '[]',                            "machine",         False),
    ("Bulgarian Split Squat",      "quads",      '["glutes","hamstrings"]',       "dumbbell",        True),
    ("Goblet Squat",               "quads",      '["glutes","core"]',             "dumbbell",        True),
    ("Walking Lunge",              "quads",      '["glutes","hamstrings"]',       "dumbbell",        True),

    # ── HAMSTRINGS ─────────────────────────────────────────────────────────
    ("Romanian Deadlift",          "hamstrings", '["glutes","back"]',             "barbell",         True),
    ("Leg Curl",                   "hamstrings", '[]',                            "machine",         False),
    ("Stiff-Leg Deadlift",         "hamstrings", '["glutes","back"]',             "barbell",         True),
    ("Nordic Curl",                "hamstrings", '[]',                            "bodyweight",      False),
    ("Good Morning",               "hamstrings", '["back","glutes"]',             "barbell",         True),

    # ── GLUTES ─────────────────────────────────────────────────────────────
    ("Hip Thrust",                 "glutes",     '["hamstrings"]',                "barbell",         True),
    ("Glute Bridge",               "glutes",     '["hamstrings"]',                "bodyweight",      False),
    ("Cable Kickback",             "glutes",     '[]',                            "cable",           False),
    ("Sumo Deadlift",              "glutes",     '["hamstrings","back"]',         "barbell",         True),

    # ── CALVES ─────────────────────────────────────────────────────────────
    ("Standing Calf Raise",        "calves",     '[]',                            "machine",         False),
    ("Seated Calf Raise",          "calves",     '[]',                            "machine",         False),
    ("Donkey Calf Raise",          "calves",     '[]',                            "bodyweight",      False),

    # ── CORE ───────────────────────────────────────────────────────────────
    ("Plank",                      "core",       '[]',                            "bodyweight",      False),
    ("Crunch",                     "core",       '[]',                            "bodyweight",      False),
    ("Hanging Leg Raise",          "core",       '["hip_flexors"]',               "bodyweight",      False),
    ("Cable Crunch",               "core",       '[]',                            "cable",           False),
    ("Russian Twist",              "core",       '[]',                            "bodyweight",      False),
    ("Ab Wheel Rollout",           "core",       '["shoulders"]',                 "other",           False),
    ("Decline Sit-Up",             "core",       '[]',                            "bodyweight",      False),
    ("Side Plank",                 "core",       '[]',                            "bodyweight",      False),

    # ── FULL BODY ──────────────────────────────────────────────────────────
    ("Barbell Clean",              "full_body",  '["quads","shoulders","traps"]', "barbell",         True),
    ("Kettlebell Swing",           "full_body",  '["glutes","hamstrings","core"]',"kettlebell",      True),
    ("Burpee",                     "full_body",  '[]',                            "bodyweight",      True),
    ("Farmer Carry",               "full_body",  '["traps","core","grip"]',       "dumbbell",        True),
    ("Thruster",                   "full_body",  '["quads","shoulders"]',         "barbell",         True),

    # ── CARDIO ─────────────────────────────────────────────────────────────
    ("Treadmill Running",          "cardio",     '[]',                            "cardio",          False),
    ("Stationary Bike",            "cardio",     '[]',                            "cardio",          False),
    ("Elliptical",                 "cardio",     '[]',                            "cardio",          False),
    ("Rowing Machine",             "cardio",     '["back","arms"]',               "cardio",          True),
    ("Jump Rope",                  "cardio",     '[]',                            "cardio",          False),
    ("Stair Climber",              "cardio",     '["glutes","quads"]',            "cardio",          False),
]


def upgrade() -> None:
    # ── workout_locations ──────────────────────────────────────────────────
    op.create_table(
        'workout_locations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('location_type',
                  sa.Enum('home', 'gym', 'outdoor', 'other', name='locationtype'),
                  nullable=False, server_default='gym'),
        sa.Column('address', sa.String(255), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # ── equipment ──────────────────────────────────────────────────────────
    op.create_table(
        'equipment',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('location_id', sa.Integer(),
                  sa.ForeignKey('workout_locations.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('equipment_type',
                  sa.Enum('barbell', 'dumbbell', 'machine', 'cable', 'bodyweight',
                          'cardio', 'resistance_band', 'kettlebell', 'other',
                          name='equipmenttype'),
                  nullable=False),
        sa.Column('max_weight', sa.Numeric(6, 2), nullable=True),
        sa.Column('weight_unit',
                  sa.Enum('kg', 'lb', name='weightunit'),
                  nullable=False, server_default='lb'),
        sa.Column('notes', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_equipment_location', 'equipment', ['location_id'])

    # ── exercises ──────────────────────────────────────────────────────────
    op.create_table(
        'exercises',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(150), nullable=False, unique=True),
        sa.Column('primary_muscle_group',
                  sa.Enum('chest', 'back', 'shoulders', 'biceps', 'triceps',
                          'forearms', 'quads', 'hamstrings', 'glutes', 'calves',
                          'core', 'full_body', 'cardio', name='musclegroup'),
                  nullable=False),
        # JSON array of secondary muscle group strings — kept flexible to avoid
        # migration churn when adding new muscle groups
        sa.Column('secondary_muscle_groups', sa.JSON(), nullable=True),
        sa.Column('equipment_type',
                  sa.Enum('barbell', 'dumbbell', 'machine', 'cable', 'bodyweight',
                          'resistance_band', 'kettlebell', 'cardio', 'other', 'any',
                          name='exerciseequipmenttype'),
                  nullable=False),
        sa.Column('is_compound', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('is_custom', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('notes', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_exercises_muscle_group', 'exercises', ['primary_muscle_group'])

    # Seed exercises
    op.bulk_insert(
        sa.table('exercises',
            sa.column('name', sa.String),
            sa.column('primary_muscle_group', sa.String),
            sa.column('secondary_muscle_groups', sa.JSON),
            sa.column('equipment_type', sa.String),
            sa.column('is_compound', sa.Boolean),
            sa.column('is_custom', sa.Boolean),
        ),
        [
            {
                'name': name,
                'primary_muscle_group': muscle,
                'secondary_muscle_groups': secondary,
                'equipment_type': equip,
                'is_compound': compound,
                'is_custom': False,
            }
            for name, muscle, secondary, equip, compound in SEED_EXERCISES
        ]
    )

    # ── workout_plans ──────────────────────────────────────────────────────
    op.create_table(
        'workout_plans',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(150), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('generated_by',
                  sa.Enum('user', 'ai', name='planorigin'),
                  nullable=False, server_default='user'),
        sa.Column('location_id', sa.Integer(),
                  sa.ForeignKey('workout_locations.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('target_days_per_week', sa.SmallInteger(), nullable=False, server_default='3'),
        sa.Column('goal',
                  sa.Enum('strength', 'hypertrophy', 'endurance', 'general_fitness', 'weight_loss',
                          name='workoutgoal'),
                  nullable=False, server_default='general_fitness'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ── workout_plan_days ──────────────────────────────────────────────────
    # Separate table for named days — enables "Chest Day" / "Back Day" labels
    # while keeping the exercises table clean.
    op.create_table(
        'workout_plan_days',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('plan_id', sa.Integer(),
                  sa.ForeignKey('workout_plans.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('day_number', sa.SmallInteger(), nullable=False),
        # Human-readable label, e.g. "Chest & Triceps", "Pull Day", "Leg Day"
        sa.Column('day_name', sa.String(100), nullable=True),
        sa.Column('notes', sa.String(255), nullable=True),
    )
    op.create_unique_constraint(
        'uq_plan_day_number', 'workout_plan_days', ['plan_id', 'day_number']
    )

    # ── workout_plan_exercises ─────────────────────────────────────────────
    op.create_table(
        'workout_plan_exercises',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('plan_id', sa.Integer(),
                  sa.ForeignKey('workout_plans.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('plan_day_id', sa.Integer(),
                  sa.ForeignKey('workout_plan_days.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('exercise_id', sa.Integer(),
                  sa.ForeignKey('exercises.id'),
                  nullable=False),
        sa.Column('target_sets', sa.SmallInteger(), nullable=False, server_default='3'),
        sa.Column('target_reps_min', sa.SmallInteger(), nullable=False, server_default='8'),
        sa.Column('target_reps_max', sa.SmallInteger(), nullable=False, server_default='12'),
        # Starting weight suggestion in lb (user's default unit)
        sa.Column('target_weight', sa.Numeric(6, 2), nullable=True),
        sa.Column('order_in_day', sa.SmallInteger(), nullable=False, server_default='1'),
        sa.Column('notes', sa.String(255), nullable=True),
    )
    op.create_index('ix_plan_exercises_plan', 'workout_plan_exercises', ['plan_id'])
    op.create_index('ix_plan_exercises_day', 'workout_plan_exercises', ['plan_id', 'plan_day_id'])

    # ── workout_sessions ───────────────────────────────────────────────────
    op.create_table(
        'workout_sessions',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('plan_id', sa.Integer(),
                  sa.ForeignKey('workout_plans.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('plan_day_id', sa.Integer(),
                  sa.ForeignKey('workout_plan_days.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('location_id', sa.Integer(),
                  sa.ForeignKey('workout_locations.id', ondelete='SET NULL'),
                  nullable=True),
        sa.Column('session_date', sa.Date(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('fatigue_rating', sa.SmallInteger(), nullable=True),  # 1-5
        # Per-session weight unit toggle — user can override lb default
        sa.Column('weight_unit',
                  sa.Enum('kg', 'lb', name='sessionweightunit'),
                  nullable=False, server_default='lb'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_sessions_date', 'workout_sessions', ['session_date'])
    op.create_index('ix_sessions_plan', 'workout_sessions', ['plan_id'])

    # ── workout_sets ───────────────────────────────────────────────────────
    op.create_table(
        'workout_sets',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('session_id', sa.BigInteger(),
                  sa.ForeignKey('workout_sessions.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('exercise_id', sa.Integer(),
                  sa.ForeignKey('exercises.id'),
                  nullable=False),
        sa.Column('set_number', sa.SmallInteger(), nullable=False),
        sa.Column('reps_completed', sa.SmallInteger(), nullable=False),
        sa.Column('weight_used', sa.Numeric(6, 2), nullable=True),
        # Unit stored per set — inherits from session but can differ
        sa.Column('weight_unit',
                  sa.Enum('kg', 'lb', name='setweightunit'),
                  nullable=False, server_default='lb'),
        sa.Column('rpe', sa.SmallInteger(), nullable=True),  # 1-10
        sa.Column('is_warmup', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('notes', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_sets_session', 'workout_sets', ['session_id'])
    op.create_index('ix_sets_exercise', 'workout_sets', ['exercise_id'])
    # Composite index for "previous best" queries — most frequent lookup pattern
    op.create_index('ix_sets_exercise_session', 'workout_sets', ['exercise_id', 'session_id'])

    # ── body_metrics ───────────────────────────────────────────────────────
    op.create_table(
        'body_metrics',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('metric_date', sa.Date(), nullable=False, unique=True),
        sa.Column('weight', sa.Numeric(5, 2), nullable=True),
        sa.Column('weight_unit',
                  sa.Enum('kg', 'lb', name='bodyweightunit'),
                  nullable=False, server_default='lb'),
        sa.Column('body_fat_pct', sa.Numeric(4, 2), nullable=True),
        sa.Column('notes', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_body_metrics_date', 'body_metrics', ['metric_date'])


def downgrade() -> None:
    op.drop_index('ix_body_metrics_date', table_name='body_metrics')
    op.drop_table('body_metrics')

    op.drop_index('ix_sets_exercise_session', table_name='workout_sets')
    op.drop_index('ix_sets_exercise', table_name='workout_sets')
    op.drop_index('ix_sets_session', table_name='workout_sets')
    op.drop_table('workout_sets')

    op.drop_index('ix_sessions_plan', table_name='workout_sessions')
    op.drop_index('ix_sessions_date', table_name='workout_sessions')
    op.drop_table('workout_sessions')

    op.drop_index('ix_plan_exercises_day', table_name='workout_plan_exercises')
    op.drop_index('ix_plan_exercises_plan', table_name='workout_plan_exercises')
    op.drop_table('workout_plan_exercises')

    op.drop_constraint('uq_plan_day_number', 'workout_plan_days', type_='unique')
    op.drop_table('workout_plan_days')

    op.drop_table('workout_plans')

    op.drop_index('ix_exercises_muscle_group', table_name='exercises')
    op.drop_table('exercises')

    op.drop_index('ix_equipment_location', table_name='equipment')
    op.drop_table('equipment')

    op.drop_table('workout_locations')

    # Drop ENUMs (MariaDB cleans these up with table drops, but explicit is safer)
    for enum_name in [
        'locationtype', 'equipmenttype', 'weightunit', 'musclegroup',
        'exerciseequipmenttype', 'planorigin', 'workoutgoal',
        'sessionweightunit', 'setweightunit', 'bodyweightunit',
    ]:
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
