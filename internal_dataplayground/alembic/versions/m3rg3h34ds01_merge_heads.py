"""merge staging and finance heads

Revision ID: m3rg3h34ds01
Revises: a1b2c3d4e5f6, f1n4nc3m0dul3
Create Date: 2026-03-24

"""
from typing import Sequence, Union

revision: str = 'm3rg3h34ds01'
down_revision: Union[str, None] = ('a1b2c3d4e5f6', 'f1n4nc3m0dul3')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
