"""add blog pipeline and code intelligence tables

Revision ID: c0d3bl0g001
Revises: b10g1d3as001
Create Date: 2026-04-16

Creates:
  - code_projects   (project/folder-level GitHub scope)
  - code_files      (individual scripts within a project)
  - blog_ideas      (full pipeline state machine)

Also adds:
  - draft_v2, code_file_id, code_project_id columns to blog_ideas
    (if blog_ideas already exists from a prior migration, use the
     add_column version at the bottom instead of create_table)
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'c0d3bl0g001'
down_revision: Union[str, None] = 'b10g1d3as001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── code_projects ──────────────────────────────────────────────────────────
    op.create_table(
        'code_projects',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('project_name', sa.String(255), nullable=False),

        # GitHub coordinates
        sa.Column('github_repo', sa.String(255), nullable=False),
        # e.g. "pedro/data-playground-base"
        sa.Column('github_base_path', sa.String(500), nullable=True),
        # e.g. "internal_dataplayground" or "" for whole repo

        sa.Column('description', sa.Text(), nullable=True),

        # README — project-level, pushed to repo
        sa.Column('readme_md', sa.Text(), nullable=True),
        sa.Column('readme_status', sa.Enum(
            'none', 'draft', 'reviewed', 'approved', 'pushed', 'stale',
            name='readmestatus'
        ), nullable=False, server_default='none'),
        sa.Column('readme_sha', sa.String(64), nullable=True),
        sa.Column('readme_generated_at', sa.DateTime(), nullable=True),
        sa.Column('readme_pushed_at', sa.DateTime(), nullable=True),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ── code_files ─────────────────────────────────────────────────────────────
    op.create_table(
        'code_files',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('project_id', sa.Integer(),
                  sa.ForeignKey('code_projects.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        sa.Column('file_name', sa.String(255), nullable=False),
        # e.g. "finance.py"
        sa.Column('github_path', sa.String(500), nullable=False),
        # e.g. "internal_dataplayground/routers/finance.py"
        sa.Column('github_sha', sa.String(64), nullable=True),

        # Raw code pulled from GitHub
        sa.Column('raw_code', sa.Text(), nullable=True),
        sa.Column('code_pulled_at', sa.DateTime(), nullable=True),

        # Code Narrator output
        sa.Column('narration', sa.Text(), nullable=True),
        sa.Column('narration_generated_at', sa.DateTime(), nullable=True),

        # Code Commenter output
        sa.Column('commented_code', sa.Text(), nullable=True),
        sa.Column('commented_generated_at', sa.DateTime(), nullable=True),
        sa.Column('commented_status', sa.Enum(
            'none', 'generated', 'reviewed', 'pushed',
            name='commentedstatus'
        ), nullable=False, server_default='none'),

        # Code Improver output
        sa.Column('improvement_notes', sa.Text(), nullable=True),
        sa.Column('improvement_generated_at', sa.DateTime(), nullable=True),
        sa.Column('improvement_status', sa.Enum(
            'none', 'generated', 'reviewed', 'applied', 'pushed',
            name='improvementstatus'
        ), nullable=False, server_default='none'),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_code_files_project_id', 'code_files', ['project_id'])

    # ── blog_ideas additions ───────────────────────────────────────────────────
    # If blog_ideas does not exist yet, use create_table instead.
    # If it already exists from b10g1d3as001, use add_column:
    op.add_column('blog_ideas',
        sa.Column('draft_v2', sa.Text(), nullable=True))
    op.add_column('blog_ideas',
        sa.Column('code_file_id', sa.Integer(),
                  sa.ForeignKey('code_files.id', ondelete='SET NULL'),
                  nullable=True))
    op.add_column('blog_ideas',
        sa.Column('code_project_id', sa.Integer(),
                  sa.ForeignKey('code_projects.id', ondelete='SET NULL'),
                  nullable=True))


def downgrade() -> None:
    op.drop_column('blog_ideas', 'code_project_id')
    op.drop_column('blog_ideas', 'code_file_id')
    op.drop_column('blog_ideas', 'draft_v2')
    op.drop_index('ix_code_files_project_id', table_name='code_files')
    op.drop_table('code_files')
    op.drop_table('code_projects')
    op.execute("DROP TYPE IF EXISTS readmestatus")
    op.execute("DROP TYPE IF EXISTS commentedstatus")
    op.execute("DROP TYPE IF EXISTS improvementstatus")
