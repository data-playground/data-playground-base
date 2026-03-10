"""add staging_jobs table

Revision ID: a1b2c3d4e5f6
Revises: bdf450bb7c4f
Create Date: 2026-03-09

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'bdf450bb7c4f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'staging_jobs',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('job_link', sa.String(511), nullable=False),

        # Status lifecycle: PENDING → PROCESSING → DONE | FAILED
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'DONE', 'FAILED', name='stagingjobstatus'), nullable=False, server_default='PENDING'),

        # Populated after scraping
        sa.Column('job_id', sa.String(64), nullable=True),
        sa.Column('job_title', sa.String(255), nullable=True),
        sa.Column('company_name', sa.String(255), nullable=True),
        sa.Column('location', sa.String(255), nullable=True),
        sa.Column('post_date', sa.Date(), nullable=True),
        sa.Column('salary', sa.String(100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),

        # Set by the user when submitting the URL
        sa.Column('job_search', sa.String(255), nullable=True),

        # Error message if enrichment fails
        sa.Column('error_message', sa.Text(), nullable=True),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_staging_jobs_status', 'staging_jobs', ['status'])


def downgrade() -> None:
    op.drop_index('ix_staging_jobs_status', table_name='staging_jobs')
    op.drop_table('staging_jobs')
    op.execute("DROP TYPE IF EXISTS stagingjobstatus")
