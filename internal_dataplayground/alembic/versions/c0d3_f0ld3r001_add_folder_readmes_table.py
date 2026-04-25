"""add folder_readmes table for per-folder README tracking

Revision ID: c0d3_f0ld3r001
Revises: c0d3bl0g001
Create Date: 2026-04-24

Replaces the original single-column approach (folder_readme_md, folder_readme_path,
folder_readme_generated_at on code_projects) with a dedicated table.

This enables:
  - Independent README state tracking per folder per project
  - Documentation coverage dashboards ("which folders need READMEs?")
  - GitHub push tracking with SHA and conflict detection per folder
  - Automatic staleness detection per folder (any file in folder pulled
    after readme_generated_at marks that folder's README as stale)

Table: folder_readmes
  - Natural key: (project_id, folder_path) — unique per project
  - Surrogate PK: id — for clean ORM relationships
  - Status lifecycle: none → draft → reviewed → pushed → stale
  - github_path: the full path where this README would live on GitHub
    e.g. "internal_dataplayground/routers/README.md"
  - folder_display_name: short human label e.g. "routers" (derived from
    folder_path but stored to avoid repeated string splits in queries)
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'c0d3_f0ld3r001'
down_revision: Union[str, None] = 'c0d3bl0g001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'folder_readmes',

        # ── Identity ──────────────────────────────────────────────────────────
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),

        # ── Ownership ─────────────────────────────────────────────────────────
        sa.Column(
            'project_id',
            sa.Integer(),
            sa.ForeignKey('code_projects.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
        ),

        # ── Folder identity ───────────────────────────────────────────────────
        # Full path within the repo, e.g. "internal_dataplayground/routers"
        sa.Column('folder_path', sa.String(500), nullable=False),

        # Short label for display: "routers", "dags", "partials", etc.
        # Stored explicitly so the UI doesn't need to parse folder_path every time.
        sa.Column('folder_display_name', sa.String(255), nullable=False),

        # Where the README.md would live on GitHub if pushed:
        # e.g. "internal_dataplayground/routers/README.md"
        sa.Column('github_path', sa.String(500), nullable=True),

        # ── Content ───────────────────────────────────────────────────────────
        sa.Column('readme_md', sa.Text(), nullable=True),

        # ── Pipeline state ────────────────────────────────────────────────────
        # Lifecycle: none → draft → reviewed → pushed → stale
        # 'stale' is set when any code_file in the folder has
        # code_pulled_at > readme_generated_at
        sa.Column(
            'status',
            sa.Enum(
                'none', 'draft', 'reviewed', 'pushed', 'stale',
                name='folderreadmestatus',
            ),
            nullable=False,
            server_default='none',
        ),

        # ── GitHub push tracking ──────────────────────────────────────────────
        # SHA of the file on GitHub after last push.
        # Required for updates — GitHub rejects a PUT without the current SHA.
        sa.Column('github_sha', sa.String(64), nullable=True),

        # ── Timestamps ────────────────────────────────────────────────────────
        sa.Column('readme_generated_at', sa.DateTime(), nullable=True),
        sa.Column('readme_pushed_at', sa.DateTime(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # ── Indexes ───────────────────────────────────────────────────────────────

    # Unique constraint: one README record per folder per project.
    # This means upserts can target (project_id, folder_path) cleanly.
    op.create_unique_constraint(
        'uq_folder_readmes_project_folder',
        'folder_readmes',
        ['project_id', 'folder_path'],
    )

    # Fast lookup by project (most common query: "give me all folders for project X")
    op.create_index(
        'ix_folder_readmes_project_id',
        'folder_readmes',
        ['project_id'],
    )

    # Fast lookup by status (for the coverage dashboard: "show me all stale READMEs")
    op.create_index(
        'ix_folder_readmes_status',
        'folder_readmes',
        ['status'],
    )


def downgrade() -> None:
    op.drop_index('ix_folder_readmes_status', table_name='folder_readmes')
    op.drop_index('ix_folder_readmes_project_id', table_name='folder_readmes')
    op.drop_constraint(
        'uq_folder_readmes_project_folder',
        'folder_readmes',
        type_='unique',
    )
    op.drop_table('folder_readmes')
    op.execute("DROP TYPE IF EXISTS folderreadmestatus")
