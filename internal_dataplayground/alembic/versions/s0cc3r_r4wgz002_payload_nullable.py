"""make soccer_raw_payloads.payload nullable (S1b: compressed copy is authoritative)

Revision ID: s0cc3r_r4wgz002
Revises: s0cc3r_r4wgz001
Create Date: 2026-10-18

⚠ down_revision assumes s0cc3r_r4wgz001 is your current single head. Before
applying this migration (run alembic inside the life_os_web container):
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's s0cc3r_r4wgz001, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before soccer raw gz nullable"
     first, then point down_revision at that new merge revision instead.

Stage S1b of the soccer raw-first migration. Loosens `payload` from NOT NULL
to NULL so the ingest DAG can write only `payload_gz` for new rows. Nothing
is deleted or nulled by this migration; existing `payload` data is untouched.

Disk: MariaDB rebuilds the table for this ALTER, so free space on the volume
holding ./mariadb_data must exceed the table's current size (about 400 MB).
Pause DAG life_os_soccer_ingest while it runs.

Downgrade restores NOT NULL, but refuses to run while any row has a NULL
payload — run `python domains/soccer/backfill_raw_gz.py --restore-payload`
first (refills payload from payload_gz for NULL-payload rows).
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_r4wgz002'
down_revision: Union[str, None] = 's0cc3r_r4wgz001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'soccer_raw_payloads', 'payload',
        existing_type=sa.JSON(),
        existing_nullable=False,
        nullable=True,
    )


def downgrade() -> None:
    conn = op.get_bind()
    null_count = conn.execute(
        sa.text("SELECT COUNT(*) FROM soccer_raw_payloads WHERE payload IS NULL")
    ).scalar()
    if null_count and null_count > 0:
        raise RuntimeError(
            f"{null_count} soccer_raw_payloads row(s) have payload IS NULL — "
            "run backfill_raw_gz.py --restore-payload first"
        )
    op.alter_column(
        'soccer_raw_payloads', 'payload',
        existing_type=sa.JSON(),
        existing_nullable=True,
        nullable=False,
    )
