"""add in_development status to blog_ideas

Revision ID: bl0g_st4tus001
Revises: bl0g_d1ff001
Create Date: 2026-05-10

MariaDB ENUM columns require a full MODIFY COLUMN listing all values
to add a new one. The new value is inserted between
waiting_for_writing_trigger and writing_in_progress.
"""
from typing import Sequence, Union
from alembic import op

revision: str = 'bl0g_st4tus001'
down_revision: Union[str, None] = 'bl0g_d1ff001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE blog_ideas MODIFY COLUMN status "
        "ENUM("
        "'idea_generated',"
        "'waiting_for_writing_trigger',"
        "'in_development',"
        "'writing_in_progress',"
        "'waiting_for_review',"
        "'review_completed',"
        "'ready_to_publish',"
        "'published'"
        ") NOT NULL DEFAULT 'idea_generated'"
    )


def downgrade() -> None:
    # Move any in_development ideas back before removing the value
    op.execute(
        "UPDATE blog_ideas SET status = 'waiting_for_writing_trigger' "
        "WHERE status = 'in_development'"
    )
    op.execute(
        "ALTER TABLE blog_ideas MODIFY COLUMN status "
        "ENUM("
        "'idea_generated',"
        "'waiting_for_writing_trigger',"
        "'writing_in_progress',"
        "'waiting_for_review',"
        "'review_completed',"
        "'ready_to_publish',"
        "'published'"
        ") NOT NULL DEFAULT 'idea_generated'"
    )
