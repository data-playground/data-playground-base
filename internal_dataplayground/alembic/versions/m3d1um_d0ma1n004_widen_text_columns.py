"""widen medium_articles text columns from TEXT to MEDIUMTEXT

Revision ID: m3d1um_d0ma1n004
Revises: s0cc3r_t34m1ds001_add_team_ids
Create Date: 2026-09-18

⚠ down_revision assumes s0cc3r_t34m1ds001_add_team_ids is your current single head.
Same verification as always, and worth taking seriously here — the
soccer lineups migration's own history shows this project's head has
drifted more than once from unrelated parallel work landing between
sessions:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's s0cc3r_t34m1ds001_add_team_ids, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before medium text widening"
     first, then point down_revision at that new merge revision instead.

Fixes a real production failure, not a preventive guess: on 2026-09-18
the daily ingest DAG failed with
  pymysql.err.DataError: (1406, "Data too long for column 'raw_item' at row 1")
sa.Text() maps to MySQL's TEXT type, capped at 65,535 bytes — a real
article's full verbatim <item> XML (raw_item) can and does exceed that.
Because every article fetched in a single ingest_sources() run lands in
one execute_many() transaction (see life_os_medium_ingest.py), this one
oversized row rolled back the *entire* batch — all 40 articles across
all 4 active sources that day, not just the one that was actually too
large. Whether that all-or-nothing batching itself is worth changing is
a separate question, not addressed by this migration.

content_html and summary are widened too, not just raw_item — same
TEXT type, and content_html in particular is a strict subset of what's
in raw_item (raw_item wraps content_html plus more), so it's exactly as
exposed to the same failure on a different, sufficiently large article.
Widening only the column that happened to fail first would just move
the next occurrence to content_html instead of preventing it.

MEDIUMTEXT (16MB) rather than a marginal bump — no realistic Medium
article approaches that, so this isn't a size someone will need to
revisit like TEXT's 64KB turned out to be.

Downgrade caveat, stated rather than glossed over: reverting to TEXT
will itself raise the same DataError this migration fixes if any row
already stores more than 64KB in these columns by the time downgrade()
runs — which, after this migration has been live for any length of
time, it likely will. Downgrading is only safe immediately after
upgrading, before new data has had a chance to need the extra room.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import MEDIUMTEXT

revision: str = 'm3d1um_d0ma1n004'
down_revision: Union[str, None] = 's0cc3r_t34m1ds001_add_team_ids'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = ['raw_item', 'content_html', 'summary']


def upgrade() -> None:
    for column in _COLUMNS:
        op.alter_column(
            'medium_articles', column,
            type_=MEDIUMTEXT(), existing_type=sa.Text(), existing_nullable=False,
        )


def downgrade() -> None:
    # See the "Downgrade caveat" note above — not a safe no-op revert
    # once real data has grown past TEXT's 64KB cap again.
    for column in _COLUMNS:
        op.alter_column(
            'medium_articles', column,
            type_=sa.Text(), existing_type=MEDIUMTEXT(), existing_nullable=False,
        )
