"""
Shared helpers for the explorer's table -> domain mapping.

Used by routers/explorer.py (schema browser) and routers/explorer_settings.py.
HIDDEN_TABLES lives here (rather than in explorer.py) so the settings router
doesn't have to import another router.

How new tables are handled
--------------------------
list_tables() reads information_schema live on every call — nothing here is a
hard-coded list of tables. A table created tomorrow shows up immediately.
get_domain_map() only knows about tables someone has assigned, so any table
missing from it is reported as UNASSIGNED by the callers. Assigning it on
/explorer/settings is what creates its mapping row.
"""
import logging

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from domains.explorer.models import ExplorerTableDomain

log = logging.getLogger(__name__)

UNASSIGNED = "unassigned"

# Tables hidden from the schema browser (internal Airflow/Alembic metadata)
HIDDEN_TABLES = {
    "alembic_version",
    "dag", "dag_run", "task_instance", "job", "log",
    "xcom", "serialized_dag", "import_error",
}


async def list_tables(db: AsyncSession) -> list[str]:
    """All visible base tables in the current database, alphabetical."""
    result = await db.execute(text(
        "SELECT TABLE_NAME FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' "
        "ORDER BY TABLE_NAME"
    ))
    return [r[0] for r in result.fetchall() if r[0] not in HIDDEN_TABLES]


async def get_domain_map(db: AsyncSession) -> dict[str, str]:
    """
    {table_name: domain}. Returns {} (everything 'unassigned') instead of
    raising if the mapping table doesn't exist yet, so the explorer still
    works before the migration is applied.
    """
    try:
        result = await db.execute(
            select(ExplorerTableDomain.table_name, ExplorerTableDomain.domain)
        )
        return {t: d for t, d in result.all()}
    except Exception as exc:
        log.warning("explorer_table_domains unavailable, treating all as unassigned: %s", exc)
        # A failed statement can leave the session needing a rollback before reuse.
        await db.rollback()
        return {}
