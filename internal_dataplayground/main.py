import datetime
from fastapi import FastAPI, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy import select, BigInteger, String, Boolean, Date, Text, Integer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing import List, Optional
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager
from google.cloud import secretmanager
import json

def get_key(SECRET_NAME):
    """
        Get API Key from Google Secret Manager
    """
    # Initialize the Secret Manager client
    SMclient = secretmanager.SecretManagerServiceClient()

    # Set the project ID 
    project_id = "impactful-post-292301"

    # Build the request to access the secret version
    request = {"name": f"projects/{project_id}/secrets/{SECRET_NAME}/versions/latest"}

    # Access the secret version
    response = SMclient.access_secret_version(request)

    # Get the secret value
    SECRET_VALUE = response.payload.data.decode("UTF-8")

    return SECRET_VALUE


# Global variable to hold our engine
engine = None
async_session = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: Runs when the server starts ---
    global engine, async_session
    
    # Fetch the DB URL from GCP
    try:
        mdb_json = json.loads(get_key("MariaDB"))
        db_url = f"mariadb+asyncmy://{mdb_json['user']}:{mdb_json['password']}@{mdb_json['host']}:3306/{mdb_json['database']}"
        engine = create_async_engine(db_url, echo=True)
        async_session = async_sessionmaker(engine, expire_on_commit=False)
        print("Successfully connected to MariaDB using GCP Secret.")
    except Exception as e:
        print(f"Failed to fetch secret or connect: {e}")
        raise e
        
    yield 
    # --- SHUTDOWN: Runs when the server stops ---
    if engine:
        await engine.dispose()

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="templates")


# 2. Database Model
class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "linkedin_jobs"

	# Primary Key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Job Details
    job_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    job_title: Mapped[str] = mapped_column(String(255))
    company_name: Mapped[str] = mapped_column(String(255), nullable=True)
    job_link: Mapped[str] = mapped_column(String(511), nullable=True)
    salary: Mapped[str] = mapped_column(String(100), nullable=True)
    remote: Mapped[bool] = mapped_column(Boolean, default=False)
    location: Mapped[str] = mapped_column(String(255), nullable=True)
    post_date: Mapped[datetime.date] = mapped_column(Date, nullable=True)
    
    # AI Analysis Fields (The LONGTEXT columns)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    qualification_analysis: Mapped[str] = mapped_column(Text, nullable=True)
    skill_gaps: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Metadata
    fit_score: Mapped[int] = mapped_column(Integer, default=0)
    job_search: Mapped[str] = mapped_column(String(255), nullable=True)
    search_date: Mapped[datetime.date] = mapped_column(Date, default=datetime.date.today)

class JobResponse(BaseModel):
    id: int
    job_title: str
    company_name: Optional[str]
    fit_score: int
    remote: bool
    
    class Config:
        from_attributes = True # This tells Pydantic to read data from SQLAlchemy models

# Dependency to get database session
async def get_db():
    if async_session is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    async with async_session() as session:
        yield session

@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job))
    return result.scalars().all()
    
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job))
    jobs_list = result.scalars().all()
    
    # This sends the "jobs" list from MariaDB to the "jobs.html" file
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs_list})