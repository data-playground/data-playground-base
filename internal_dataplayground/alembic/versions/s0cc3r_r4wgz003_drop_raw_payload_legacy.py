"""drop legacy soccer_raw_payloads.payload column (S1d: payload_gz is the only copy)

Revision ID: s0cc3r_r4wgz003
Revises: s0cc3r_r4wgz002
Create Date: 2026-10-26

⚠ down_revision assumes s0cc3r_r4wgz002 is your current single head
(confirmed by the owner on 2026-10-26). Before applying this migration (run
alembic inside the life_os_web container):
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's s0cc3r_r4wgz002, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before soccer drop legacy payload"
     first, then point down_revision at that new merge revision instead.

Stage S1d of the soccer raw-first migration. Drops the legacy, uncompressed
`payload` column from soccer_raw_payloads (about 300 MB). From here on
`payload_gz` (zlib-compressed compact JSON, see airflow/agents/payload_codec.py)
is the only stored copy of each FIFA response. S1c already stopped every
reader of `payload`; nothing in the DAG or agents touches it.

The column was created as a MariaDB JSON type, which MariaDB stores as
LONGTEXT with an inline CHECK (json_valid(`payload`)) that references only
this column. The owner's SHOW CREATE TABLE confirmed that is the only
constraint involving `payload`, so this is a plain drop_column; the CHECK
goes with the column. The owner's empty-clone DROP test (runbook G4)
confirms this before the real table is touched. No constraint names are
invented here.

Disk: MariaDB rebuilds the table for this ALTER, so free space on the volume
holding ./mariadb_data must exceed the table's current size. Pause DAG
life_os_soccer_ingest while it runs, and take a dump of soccer_raw_payloads
first.

Downgrade re-adds `payload` as a nullable JSON column. It restores the SCHEMA
ONLY, not the data: the column comes back empty. To get data back, either
load the pre-S1d dump and `alembic stamp s0cc3r_r4wgz002`, or, after the
downgrade, run `python domains/soccer/backfill_raw_gz.py --restore-payload`
(refills `payload` from `payload_gz` for rows where `payload` IS NULL).
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 's0cc3r_r4wgz003'
down_revision: Union[str, None] = 's0cc3r_r4wgz002'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('soccer_raw_payloads', 'payload')


def downgrade() -> None:
    # Schema only — the restored column is empty. See module docstring.
    op.add_column(
        'soccer_raw_payloads',
        sa.Column('payload', sa.JSON(), nullable=True),
    )
