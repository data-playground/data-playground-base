"""merge staging and finance heads

Revision ID: m3rg3h34ds02
Revises: m3rg3h34ds01, b80965cfe3ab
Create Date: 2026-03-24

"""
from typing import Sequence, Union

revision: str = 'm3rg3h34ds02'
down_revision: Union[str, None] = ('m3rg3h34ds01', 'b80965cfe3ab')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
