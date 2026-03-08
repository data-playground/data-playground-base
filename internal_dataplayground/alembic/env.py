from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

# 1. Import your FastAPI models and secret fetcher
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import get_key

# 2. Point Alembic to your Base metadata
from models import Base
target_metadata = Base.metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# 3. Dynamically build the DB URL using your GCP Secret logic!
# Assuming your secret is named "db_password" - adjust if necessary
mdb_json = json.loads(get_key("MariaDB"))
db_url = f"mysql+asyncmy://data_playground:{mdb_json['password']}@db:3306/jobs"
config.set_main_option("sqlalchemy.url", db_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def do_run_migrations(connection):
    # This is the line that was missing the metadata link for --autogenerate
    context.configure(
        connection=connection, 
        target_metadata=target_metadata, # Ensure this matches your variable name above
        compare_type=True # Bonus: this helps Alembic detect column type changes
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online():
    """Run migrations in 'online' mode using an async engine."""

    # Use the URL we built dynamically in the previous step
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        # We use 'run_sync' to bridge the async connection to the sync migration runner
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

if context.is_offline_mode():
    run_migrations_offline()
else:
    # This is the critical change for Asyncio
    asyncio.run(run_migrations_online())
