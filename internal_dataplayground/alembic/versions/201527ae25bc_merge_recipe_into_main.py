"""merge recipe into main

Revision ID: 201527ae25bc
Revises: 133023c99c4f, r3c1p3_m4n4g3r001
Create Date: 2026-05-19 03:47:01.044604

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '201527ae25bc'
down_revision: Union[str, None] = ('133023c99c4f', 'r3c1p3_m4n4g3r001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
