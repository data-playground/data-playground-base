"""merge recipe into main

Revision ID: 7b162bedc51c
Revises: 201527ae25bc, w0rk0ut_tr4ck3r001
Create Date: 2026-05-20 02:03:00.358840

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7b162bedc51c'
down_revision: Union[str, None] = ('201527ae25bc', 'w0rk0ut_tr4ck3r001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
