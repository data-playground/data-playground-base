"""merge journal into main

Revision ID: 0ecac75145ac
Revises: m3rg3j0urn4l001
Create Date: 2026-05-13 01:22:16.408245

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0ecac75145ac'
down_revision: Union[str, None] = 'm3rg3j0urn4l001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
