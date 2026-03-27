"""merge all heads into single tip

Revision ID: fin4lm3rg3001
Revises: b80965cfe3ac, m3rg3h34ds02
Create Date: 2026-03-25

"""
from typing import Sequence, Union

revision: str = 'fin4lm3rg3001'
down_revision: Union[str, None] = ('b80965cfe3ac', 'm3rg3h34ds02')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
