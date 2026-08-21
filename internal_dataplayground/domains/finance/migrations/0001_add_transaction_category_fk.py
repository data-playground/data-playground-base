# domains/finance/migrations/0001_add_transaction_category_fk.py
"""
One-time data migration for the Transaction.category -> category_id fix.

WHAT THIS DOES
---------------
1. Adds a nullable `category_id INT` column to `transactions`, with an
   index and a `FOREIGN KEY ... REFERENCES categories(id) ON DELETE
   SET NULL` constraint — only if that column doesn't already exist
   (idempotent, safe to re-run).
2. Backfills `category_id` for every existing row by matching the
   (soon-to-be-legacy) `category` string column against `categories.name`.
3. Reports how many rows were backfilled and how many rows have no
   matching Category row (these will display as "Other" going forward,
   exactly like unmatched categories did before this fix — see
   `Transaction.category`'s fallback property in domains/finance/models.py).

WHAT THIS DELIBERATELY DOES NOT DO
------------------------------------
It does NOT drop the old `transactions.category` VARCHAR column. The
ORM model no longer maps that column (it's now a read-only Python
property backed by category_id), so the app will simply stop reading
and writing it — but the column stays in the physical table, harmless
and inert, until you're confident the backfill is correct. Dropping it
is a separate, deliberately-not-automated step — see the bottom of this
file for the exact statement to run once you're ready.

USAGE
-----
Dry run (default) — reports what WOULD happen, changes nothing:
    python domains/finance/migrations/0001_add_transaction_category_fk.py

Apply for real:
    python domains/finance/migrations/0001_add_transaction_category_fk.py --apply

Run this from the same environment/working directory as the app itself
(it reuses database.py's own connection setup, so it needs the same
MARIA_DB / APP_ENV environment as `uvicorn main:app`).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

# Running this file directly (`python3 path/to/this_file.py`) puts only
# this file's OWN directory on sys.path[0] — never the caller's cwd, and
# never the app root three levels up where `database.py` actually lives.
# That's true no matter what directory you invoke this script from,
# which is why `import database` below would otherwise fail with
# `ModuleNotFoundError: No module named 'database'` even when run from
# inside internal_dataplayground/. Fix it by locating the app root
# relative to this file's own path, not relative to the cwd.
_APP_ROOT = Path(__file__).resolve().parents[3]  # .../internal_dataplayground
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import database


COLUMN_EXISTS_SQL = text("""
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'transactions'
      AND COLUMN_NAME = 'category_id'
""")

ADD_COLUMN_SQL = text("""
    ALTER TABLE transactions
        ADD COLUMN category_id INT NULL,
        ADD INDEX idx_transactions_category_id (category_id),
        ADD CONSTRAINT fk_transactions_category
            FOREIGN KEY (category_id) REFERENCES categories(id)
            ON DELETE SET NULL
""")

# Dry-run counts only — no writes.
COUNT_MATCHABLE_SQL = text("""
    SELECT COUNT(*) FROM transactions t
    JOIN categories c ON t.category = c.name
    WHERE t.category_id IS NULL
""")

COUNT_UNMATCHED_SQL = text("""
    SELECT t.category, COUNT(*) AS n
    FROM transactions t
    LEFT JOIN categories c ON t.category = c.name
    WHERE t.category_id IS NULL AND c.id IS NULL
    GROUP BY t.category
    ORDER BY n DESC
""")

BACKFILL_SQL = text("""
    UPDATE transactions t
    JOIN categories c ON t.category = c.name
    SET t.category_id = c.id
    WHERE t.category_id IS NULL
""")

# Run manually, later, once you've confirmed the backfill looks right —
# deliberately not part of this script's --apply path.
DROP_OLD_COLUMN_SQL_FOR_REFERENCE = """
    ALTER TABLE transactions DROP COLUMN category;
"""


async def main(apply: bool) -> int:
    await database.init_db()
    async with database.async_session() as session:
        already_exists = (await session.execute(COLUMN_EXISTS_SQL)).scalar_one() > 0

        if not already_exists:
            if not apply:
                print("[DRY RUN] Would add transactions.category_id (FK -> categories.id, ON DELETE SET NULL).")
            else:
                print("Adding transactions.category_id ...")
                await session.execute(ADD_COLUMN_SQL)
                await session.commit()
                print("  done.")
        else:
            print("transactions.category_id already exists — skipping column creation.")

        # Backfill counts can be computed either way (column may or may not
        # exist yet in dry-run mode) — but only actually query them if the
        # column exists, otherwise the queries below would fail.
        if already_exists or apply:
            matchable = (await session.execute(COUNT_MATCHABLE_SQL)).scalar_one()
            unmatched_rows = (await session.execute(COUNT_UNMATCHED_SQL)).all()
            unmatched_total = sum(r.n for r in unmatched_rows)

            print(f"\nRows with category_id still NULL, matchable to a real Category: {matchable}")
            if unmatched_rows:
                print(f"Rows with category_id still NULL, NO matching Category row (will show as 'Other'): {unmatched_total}")
                print("  Breakdown by original category string:")
                for r in unmatched_rows:
                    print(f"    {r.category!r}: {r.n} row(s)")
            else:
                print("Rows with no matching Category row: 0")

            if not apply:
                print("\n[DRY RUN] Would backfill category_id for the matchable rows above. Re-run with --apply to do it.")
            elif matchable:
                print(f"\nBackfilling {matchable} row(s) ...")
                await session.execute(BACKFILL_SQL)
                await session.commit()
                print("  done.")
            else:
                print("\nNothing to backfill.")

    print(
        "\nNote: the old `category` string column was NOT dropped. "
        "Once you've verified the backfill above, you can drop it manually:\n"
        f"  {DROP_OLD_COLUMN_SQL_FOR_REFERENCE.strip()}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually run the ALTER TABLE / UPDATE statements. Without this flag, only reports what would happen.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(apply=args.apply)))
