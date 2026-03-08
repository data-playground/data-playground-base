import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from google.cloud import secretmanager
from fastapi import HTTPException
import json

# GCP Secret Manager Logic
def get_key(SECRET_NAME):
    client = secretmanager.SecretManagerServiceClient()
    project_id = "impactful-post-292301"
    request = {"name": f"projects/{project_id}/secrets/{SECRET_NAME}/versions/latest"}
    response = client.access_secret_version(request)
    return response.payload.data.decode("UTF-8")

# Initialization (Variables will be set during FastAPI lifespan)
engine = None
async_session = None

async def init_db():
    global engine, async_session
    # Assuming your secret name is "db_password"
    mdb_json = json.loads(get_key("MariaDB"))

    db_url = f"mysql+asyncmy://data_playground:{mdb_json['password']}@db:3306/jobs"

    engine = create_async_engine(db_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Dependency to get database session
async def get_db():
    if async_session is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    async with async_session() as session:
        yield session
