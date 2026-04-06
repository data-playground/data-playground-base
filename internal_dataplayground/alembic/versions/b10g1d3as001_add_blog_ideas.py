"""add blog_ideas table

Revision ID: b10g1d3as001
Revises: fin4lm3rg3001
Create Date: 2026-03-26

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'b10g1d3as001'
down_revision: Union[str, None] = 'fin4lm3rg3001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'blog_ideas',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('title_concept', sa.String(255), nullable=False),
        sa.Column('project_type', sa.Enum(
            'existing_asset', 'new_build',
            name='blogprojecttype'
        ), nullable=False, server_default='new_build'),

        # Core blueprint fields (AI-generated or user-supplied)
        sa.Column('the_build',         sa.Text(),     nullable=True),
        sa.Column('the_narrative',     sa.Text(),     nullable=True),
        sa.Column('the_selling_point', sa.Text(),     nullable=True),

        # BYOI raw input
        sa.Column('raw_idea_input', sa.Text(), nullable=True),

        # Human-in-the-loop evidence
        sa.Column('code_content',  sa.Text(length=4294967295), nullable=True),  # LONGTEXT
        sa.Column('author_notes',  sa.Text(), nullable=True),

        # AI artifacts
        sa.Column('draft_v1',           sa.Text(length=4294967295), nullable=True),
        sa.Column('user_review_notes',  sa.Text(), nullable=True),
        sa.Column('final_article',      sa.Text(length=4294967295), nullable=True),

        # SEO metadata
        sa.Column('seo_title',       sa.String(255), nullable=True),
        sa.Column('seo_description', sa.String(500), nullable=True),
        sa.Column('seo_tags',        sa.String(500), nullable=True),

        # State machine
        sa.Column('status', sa.Enum(
            'idea_generated',
            'waiting_for_writing_trigger',
            'writing_in_progress',
            'waiting_for_review',
            'review_completed',
            'ready_to_publish',
            'published',
            name='blogideastatus'
        ), nullable=False, server_default='idea_generated'),

        sa.Column('airflow_run_id', sa.String(255), nullable=True),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_blog_ideas_status', 'blog_ideas', ['status'])


def downgrade() -> None:
    op.drop_index('ix_blog_ideas_status', table_name='blog_ideas')
    op.drop_table('blog_ideas')
    op.execute("DROP TYPE IF EXISTS blogideastatus")
    op.execute("DROP TYPE IF EXISTS blogprojecttype")
