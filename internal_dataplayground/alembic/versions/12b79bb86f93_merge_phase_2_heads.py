"""merge phase 2 heads

Revision ID: 12b79bb86f93
Revises: bl0g_st4tus001, h4b1t_tr4ck3r001
Create Date: 2026-05-12 03:15:01.421371

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '12b79bb86f93'
down_revision: Union[str, None] = ('bl0g_st4tus001', 'h4b1t_tr4ck3r001')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
