"""manual add archived status

Revision ID: 133023c99c4f
Revises: 0ecac75145ac
Create Date: 2026-05-13 02:11:29.063823

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '133023c99c4f'
down_revision: Union[str, None] = '0ecac75145ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This specifically tells MariaDB/MySQL to allow the new 'archived' value
    # without touching any of the other job or log tables.
    op.execute("ALTER TABLE blog_ideas MODIFY COLUMN status ENUM('idea_generated', 'waiting_for_writing_trigger', 'in_development', 'writing_in_progress', 'waiting_for_review', 'review_completed', 'ready_to_publish', 'published', 'archived') NOT NULL")

def downgrade() -> None:
    # Optional: logic to remove it if you ever roll back
    op.execute("ALTER TABLE blog_ideas MODIFY COLUMN status ENUM('idea_generated', 'waiting_for_writing_trigger', 'in_development', 'writing_in_progress', 'waiting_for_review', 'review_completed', 'ready_to_publish', 'published') NOT NULL")
