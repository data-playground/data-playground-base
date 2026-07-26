"""add job scout run log for scrape health tracking

Revision ID: j0b_sc0ut_h34lth001
Revises: j0b_sc0ut_upgr4de001
Create Date: 2026-07-25

Creates job_scout_run_log — one row per DAG run for both
life_os_job_scout.py (LinkedIn) and life_os_job_scout_ats.py
(Greenhouse/Lever). Written by agents/job_scout_health.log_run(), read by
agents/job_scout_health.get_health_summary() (digest DAG, via dag_db) and
independently by routers/job_config.py (Settings page, via SQLAlchemy —
see the docstring in job_scout_health.py for why there are two read paths).
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'j0b_sc0ut_h34lth001'
down_revision: Union[str, None] = '241327ca1bae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'job_scout_run_log',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('dag_id', sa.String(100), nullable=False),
        sa.Column('run_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('items_attempted', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('items_found', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('new_items', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('items_loaded', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(20), nullable=False, server_default='ok'),
        sa.Column('message', sa.String(500), nullable=True),
    )
    op.create_index('ix_job_scout_run_log_dag_id_run_at', 'job_scout_run_log', ['dag_id', 'run_at'])


def downgrade() -> None:
    op.drop_index('ix_job_scout_run_log_dag_id_run_at', table_name='job_scout_run_log')
    op.drop_table('job_scout_run_log')
