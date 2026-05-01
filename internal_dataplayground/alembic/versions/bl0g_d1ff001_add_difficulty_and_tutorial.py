"""add difficulty field and tutorial project type to blog_ideas

Revision ID: bl0g_d1ff001
Revises: c0d3_f0ld3r001
Create Date: 2026-04-27

Changes:
  1. blog_ideas.difficulty — new VARCHAR(20) column.
     Values: 'starter' | 'weekend' | 'ambitious'
     Nullable (existing rows will be NULL until re-scored or re-generated).

  2. blog_ideas.project_type — extends the existing ENUM to add 'tutorial'.
     MariaDB ENUM modification is done with a full ALTER TABLE ... MODIFY COLUMN,
     which preserves existing values.

     Existing values 'existing_asset' and 'new_build' are unchanged.
     New value 'tutorial' covers focused how-to posts requiring no original project.

Why VARCHAR for difficulty instead of ENUM:
  We want to be able to add difficulty levels in the future without another
  migration. VARCHAR(20) + application-level validation (the DAG and router
  both enforce the allowed values) is cleaner here.

Rollback safety:
  The downgrade removes the difficulty column and reverts project_type ENUM.
  Any 'tutorial' rows will have their project_type set to 'new_build' by the
  downgrade before the ENUM is altered, to avoid a constraint violation.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'bl0g_d1ff001'
down_revision: Union[str, None] = 'c0d3_f0ld3r001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add difficulty column
    op.add_column(
        'blog_ideas',
        sa.Column(
            'difficulty',
            sa.String(20),
            nullable=True,
            comment="starter | weekend | ambitious",
        )
    )

    # 2. Add index on difficulty for the dashboard filter query
    op.create_index(
        'ix_blog_ideas_difficulty',
        'blog_ideas',
        ['difficulty'],
    )

    # 3. Extend project_type ENUM to include 'tutorial'
    #    MariaDB requires the full ENUM list in the MODIFY COLUMN statement.
    op.execute(
        "ALTER TABLE blog_ideas MODIFY COLUMN project_type "
        "ENUM('existing_asset', 'new_build', 'tutorial') NOT NULL "
        "DEFAULT 'new_build'"
    )


def downgrade() -> None:
    # Before reverting the ENUM, update any 'tutorial' rows to 'new_build'
    # so the constraint alteration doesn't fail.
    op.execute(
        "UPDATE blog_ideas SET project_type = 'new_build' "
        "WHERE project_type = 'tutorial'"
    )

    # Revert project_type ENUM to original two values
    op.execute(
        "ALTER TABLE blog_ideas MODIFY COLUMN project_type "
        "ENUM('existing_asset', 'new_build') NOT NULL "
        "DEFAULT 'new_build'"
    )

    # Drop the difficulty index and column
    op.drop_index('ix_blog_ideas_difficulty', table_name='blog_ideas')
    op.drop_column('blog_ideas', 'difficulty')
