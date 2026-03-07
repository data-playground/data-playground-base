import datetime
from fastapi import FastAPI, Depends
from sqlalchemy import select, BigInteger, String, Boolean, Date, Text, Integer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing import List, Optional
from pydantic import BaseModel

# 1. Database Configuration (Replace with your local credentials)
# Format: mariadb+asyncmy://user:password@localhost:3306/dbname
DB_URL = "mariadb+asyncmy://python_user:pedroPythonpass@localhost:3306/jobs"

engine = create_async_engine(DB_URL)
async_session = async_sessionmaker(engine, expire_on_commit=False)

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
		
# 3. FastAPI App Setup
app = FastAPI()

# Dependency to get database session
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job))
    return result.scalars().all()