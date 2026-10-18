"""add payload_gz (zlib JSON) + match lookup index to soccer_raw_payloads

Revision ID: s0cc3r_r4wgz001
Revises: s0cc3r_cr3sts001
Create Date: 2026-10-17

⚠ down_revision was the single head when written (s0cc3r_cr3sts001, per the
attached versions chain). Before applying: run `alembic heads`.
  - one hash == s0cc3r_cr3sts001  -> apply as-is
  - a different single hash       -> change down_revision to it
  - more than one                 -> `alembic merge heads -m "merge before raw gz"`,
                                     then point down_revision at the merge revision

S1a of the soccer raw-first migration. Purely additive: `payload` (JSON,
NOT NULL) and the old ix_soccer_raw_payloads_lookup are untouched. No backfill.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMBLOB

revision: str = 's0cc3r_r4wgz001'
down_revision: Union[str, None] = 's0cc3r_cr3sts001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('soccer_raw_payloads', sa.Column('payload_gz', MEDIUMBLOB(), nullable=True))
    op.create_index(
        'ix_soccer_raw_payloads_match_lookup',
        'soccer_raw_payloads',
        ['fifa_match_id', 'endpoint', 'fetched_at'],
    )


def downgrade() -> None:
    op.drop_index('ix_soccer_raw_payloads_match_lookup', table_name='soccer_raw_payloads')
    op.drop_column('soccer_raw_payloads', 'payload_gz')
