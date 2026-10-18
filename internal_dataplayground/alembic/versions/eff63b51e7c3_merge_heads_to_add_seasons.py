"""merge heads to add seasons

Revision ID: eff63b51e7c3
Revises: bb7b553402a0, m3d14_s34s0ns001
Create Date: 2026-05-26 18:44:13.864021

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'eff63b51e7c3'
down_revision: Union[str, None] = ('bb7b553402a0', 'm3d14_s34s0ns001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
