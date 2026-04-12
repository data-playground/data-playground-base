"""add_application_logs_table

Revision ID: bdf450bb7c4f
Revises: 
Create Date: 2026-03-09 03:01:14.312561

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'bdf450bb7c4f'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the new application_logs table
    op.create_table(
        'application_logs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'APPLIED', 'PHONE_SCREEN', 'INTERVIEWING',
                'TECHNICAL_ASSESSMENT', 'REJECTED', 'OFFER',
                name='applicationstatus'
            ),
            nullable=False,
        ),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['linkedin_jobs.ID'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_application_logs_job_id'),
        'application_logs',
        ['job_id'],
        unique=False,
    )

    # Drop the old unused staging table
    op.drop_table('linkedin_step')


def downgrade() -> None:
    # Restore linkedin_step (in case of rollback)
    op.create_table(
        'linkedin_step',
        sa.Column('ID', sa.BigInteger(), autoincrement=False, nullable=True),
        sa.Column('job_id', sa.BigInteger(), autoincrement=False, nullable=True),
        sa.Column('fit_score', sa.Integer(), autoincrement=False, nullable=True),
        sa.Column('job_title', sa.String(length=255), nullable=True),
        sa.Column('job_link', sa.String(length=511), nullable=True),
        sa.Column('company_name', sa.String(length=255), nullable=True),
        sa.Column('salary', sa.String(length=100), nullable=True),
        sa.Column('remote', sa.Boolean(), autoincrement=False, nullable=True),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('post_date', sa.Date(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('qualification_analysis', sa.Text(), nullable=True),
        sa.Column('skill_gaps', sa.Text(), nullable=True),
        sa.Column('job_search', sa.String(length=255), nullable=True),
        sa.Column('search_date', sa.Date(), nullable=True),
    )

    # Drop application_logs and its index
    op.drop_index(op.f('ix_application_logs_job_id'), table_name='application_logs')
    op.drop_table('application_logs')
