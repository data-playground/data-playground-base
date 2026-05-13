"""merge journal into main head

Revision ID: m3rg3j0urn4l001
Revises: 12b79bb86f93, d41ly_j0urn4l001
Create Date: 2026-05-12

"""
from typing import Sequence, Union

revision: str = 'm3rg3j0urn4l001'
down_revision: Union[str, None] = ('12b79bb86f93', 'd41ly_j0urn4l001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
