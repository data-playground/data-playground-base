import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from google.cloud import secretmanager
from fastapi import HTTPException
import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from secrets import get_key  # ← now imported, not defined here


# Re-export so existing callers of `from database import get_key` still work
__all__ = ["get_key", "init_db", "get_db", "Settings", "settings"]

class Settings(BaseSettings):
    app_env: str = "local"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore DB_ROOT_PASSWORD etc. — we don't need them here
    )

settings = Settings()

# Initialization (Variables will be set during FastAPI lifespan)
engine = None
async_session = None

async def init_db():
    global engine, async_session
    # Assuming your secret name is "db_password"
    mdb_json = json.loads(get_key("MariaDB"))
    
    db_host = "db" if settings.app_env == "production" else "localhost"

    db_url = f"mysql+asyncmy://data_playground:{mdb_json['password']}@{db_host}:3306/jobs"

    engine = create_async_engine(db_url, echo=(settings.app_env == "local"))
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    print(f"[Life OS] Environment: {settings.app_env.upper()} | DB host: {db_host} | Connected ✅")

# Dependency to get database session
async def get_db():
    if async_session is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    async with async_session() as session:
        yield session
