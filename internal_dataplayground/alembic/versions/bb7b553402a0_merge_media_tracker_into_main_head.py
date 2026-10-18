"""merge media tracker into main head

Revision ID: bb7b553402a0
Revises: 7b162bedc51c, m3d14_tr4ck3r001
Create Date: 2026-05-24 14:32:58.840245

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bb7b553402a0'
down_revision: Union[str, None] = ('7b162bedc51c', 'm3d14_tr4ck3r001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
