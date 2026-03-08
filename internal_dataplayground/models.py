import datetime
from typing import Optional
from sqlalchemy import BigInteger, String, Boolean, Date, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pydantic import BaseModel

# The shared Base class for all tables
class Base(DeclarativeBase):
    pass

# --- JOBS MODULE ---
class Job(Base):
    __tablename__ = "linkedin_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(BigInteger, nullable=True)
    job_title: Mapped[str] = mapped_column(String(255))
    company_name: Mapped[str] = mapped_column(String(255), nullable=True)
    job_link: Mapped[str] = mapped_column(String(511), nullable=True)
    salary: Mapped[str] = mapped_column(String(100), nullable=True)
    remote: Mapped[bool] = mapped_column(Boolean, default=False)
    location: Mapped[str] = mapped_column(String(255), nullable=True)
    post_date: Mapped[datetime.date] = mapped_column(Date, nullable=True)

    description: Mapped[str] = mapped_column(Text, nullable=True)
    explanation: Mapped[str] = mapped_column(Text, nullable=True)
    qualification_analysis: Mapped[str] = mapped_column(Text, nullable=True)
    skill_gaps: Mapped[str] = mapped_column(Text, nullable=True)

    fit_score: Mapped[int] = mapped_column(Integer, default=0)
    job_search: Mapped[str] = mapped_column(String(255), nullable=True)
    search_date: Mapped[datetime.date] = mapped_column(Date, default=datetime.date.today)

# Pydantic model for API responses
class JobResponse(BaseModel):
    id: int
    job_title: str
    company_name: Optional[str]
    fit_score: int
    remote: bool

    class Config:
        from_attributes = True

# --- FUTURE: FINANCE MODULE ---
# You will simply add 'class Finance(Base):' here later!
