from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared SQLAlchemy declarative base for every ORM model in the app.

    Lives here (rather than in the top-level models.py, where it used to be
    defined) as shared infrastructure so domain-scoped model modules —
    starting with domains/habits/models.py — can import it without
    depending on the top-level models.py module, which still has its own
    `Base = ...` re-export for backward compatibility with files that
    haven't migrated to importing this directly yet.

    All ORM classes across every domain must inherit from this exact
    object (not a separate DeclarativeBase subclass) — Base.metadata is
    the single registry Alembic/`create_all()` uses to see every table in
    the app. Importing a different Base for a new domain would silently
    split that registry and tables would go missing from migrations.
    """
