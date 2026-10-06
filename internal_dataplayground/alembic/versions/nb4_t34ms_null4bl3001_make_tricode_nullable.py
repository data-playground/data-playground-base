"""make nba_teams.tricode nullable (preseason/exhibition opponents)

Revision ID: nb4_t34ms_null4bl3001
Revises: expl0r3r_d0ma1ns001
Create Date: 2026-09-17

⚠ down_revision assumes expl0r3r_d0ma1ns001 is your current single head. Before
applying this migration:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's expl0r3r_d0ma1ns001, you're good —
     apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before nba_teams.tricode fix"
     first, then point down_revision at that new merge revision instead.

Why: the ingest DAG (life_os_nba_ingest.py) now writes a placeholder row
to nba_teams — full_name only, no tricode — for any team_id it encounters
that isn't one of the 30 real franchises seeded by expl0r3r_d0ma1ns001. This
turned out to be a real, not theoretical, need: an October 2025 preseason
backfill hit a game against an exhibition opponent (G League squad /
international club) whose team_id had never been seeded, and the
nba_games FK violated, aborting that entire date's upsert. tricode was
NOT NULL + UNIQUE, so a placeholder row couldn't be written at all without
this change. NULL values don't count as duplicates under a UNIQUE index in
MariaDB/InnoDB, so any number of placeholder rows can now coexist without
threatening the uniqueness real franchises' tricodes still rely on.

No data migration needed — every existing row (the 30 real franchises)
already has a real, non-null tricode; this only loosens the constraint
for rows that don't exist yet.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'nb4_t34ms_null4bl3001'
down_revision: Union[str, None] = 'expl0r3r_d0ma1ns001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'nba_teams', 'tricode',
        existing_type=sa.String(3),
        nullable=True,
    )


def downgrade() -> None:
    # Reversing this requires every row to have a non-null tricode first.
    # If any placeholder rows (NULL tricode) exist when downgrading, this
    # will fail loudly rather than silently coercing them to something —
    # deliberately: there's no safe automatic tricode to assign a placeholder
    # team, so a human needs to decide (backfill a real one, or delete the
    # row) before this constraint can be restored.
    op.alter_column(
        'nba_teams', 'tricode',
        existing_type=sa.String(3),
        nullable=False,
    )
