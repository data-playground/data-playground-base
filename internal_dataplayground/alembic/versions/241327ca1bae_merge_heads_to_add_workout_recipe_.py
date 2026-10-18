"""merge heads to add workout recipe integration

Revision ID: 241327ca1bae
Revises: eff63b51e7c3, w33kly_pl4nn3r001
Create Date: 2026-05-27 02:04:52.175011

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '241327ca1bae'
down_revision: Union[str, None] = ('eff63b51e7c3', 'w33kly_pl4nn3r001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
