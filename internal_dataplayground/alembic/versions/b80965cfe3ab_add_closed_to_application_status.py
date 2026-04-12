"""add_closed_to_application_status

Revision ID: b80965cfe3ab
Revises: a1b2c3d4e5f6
Create Date: 2026-03-15 02:38:24.148936

"""
from typing import Sequence, Union
from alembic import op

revision: str = 'b80965cfe3ab'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE application_logs MODIFY COLUMN status "
        "ENUM('Applied','Phone Screen','Interviewing','Technical Assessment','Rejected','Offer','Closed') NOT NULL"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE application_logs MODIFY COLUMN status "
        "ENUM('Applied','Phone Screen','Interviewing','Technical Assessment','Rejected','Offer') NOT NULL"
    )
