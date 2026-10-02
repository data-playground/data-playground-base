"""
Explorer domain — ORM models.

ExplorerTableDomain maps a physical table name to a free-text domain label
used only to group the SQL Explorer's table browser.

Deliberately NOT a foreign key to anything: tables can be created or dropped
outside this app (Alembic, manual DDL, Airflow), and the explorer must keep
working either way.
  - A table with no row here is shown under the "unassigned" group.
  - A row here whose table no longer exists is simply ignored.

Registered with Base.metadata the same way every other domain is: by being
imported from this domain's routers (see routers/explorer.py), which main.py
imports at startup.
"""
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from core.base_model import Base


class ExplorerTableDomain(Base):
    __tablename__ = "explorer_table_domains"

    table_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    domain: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
