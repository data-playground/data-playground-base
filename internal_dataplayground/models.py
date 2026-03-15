import datetime
import enum
from typing import Optional
from sqlalchemy import BigInteger, String, Boolean, Date, Text, Integer, Enum, ForeignKey, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pydantic import BaseModel

# The shared Base class for all tables
class Base(DeclarativeBase):
    pass

# --- JOBS MODULE ---
class ApplicationStatus(enum.Enum):
    """
    Tracks the lifecycle of a job application.
    Order reflects the typical progression through a hiring pipeline.
    """
    APPLIED              = "Applied"
    PHONE_SCREEN         = "Phone Screen"
    INTERVIEWING         = "Interviewing"
    TECHNICAL_ASSESSMENT = "Technical Assessment"
    REJECTED             = "Rejected"
    CLOSED               = "Closed"
    OFFER                = "Offer"
    
class Job(Base):
    __tablename__ = "linkedin_jobs"

    ID: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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


    # ✅ Relationship — lets you do job.application_logs anywhere in Python
    # without writing a JOIN manually
    application_logs: Mapped[list["ApplicationLog"]] = relationship(
        "ApplicationLog",
        back_populates="job",
        order_by="desc(ApplicationLog.created_at)",
        lazy="selectin",  # Async-safe loading strategy
    )

    @property
    def latest_status(self) -> Optional[str]:
        """
        Convenience property — returns the most recent application status
        or None if the job hasn't been applied to yet.
        Used directly in Jinja2 templates: {{ job.latest_status }}
        """
        if self.application_logs:
            return self.application_logs[0].status.value
        return None


class ApplicationLog(Base):
    """
    Tracks every status change for a job application.
    One job can have many log entries, building a full history.
    e.g. Applied → Phone Screen → Interviewing → Offer
    """
    __tablename__ = "application_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Foreign key linking back to linkedin_jobs
    job_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("linkedin_jobs.ID", ondelete="CASCADE"),
        nullable=False,
        index=True,  # Speeds up lookups like "show me all logs for job X"
    )

    status: Mapped[ApplicationStatus] = mapped_column(
        Enum(ApplicationStatus),
        nullable=False,
    )

    # Free-text field for notes at each stage
    # e.g. "Spoke with recruiter Sarah, follow up Friday"
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
    )

    # Back-reference to the parent Job
    job: Mapped["Job"] = relationship("Job", back_populates="application_logs")


# Pydantic model for API responses
class JobResponse(BaseModel):
    id: int
    job_title: str
    company_name: Optional[str]
    fit_score: int
    remote: bool

    class Config:
        from_attributes = True


class ApplicationLogCreate(BaseModel):
    """Used when creating a new log entry via the API."""
    job_id: int
    status: ApplicationStatus
    notes: Optional[str] = None


class ApplicationLogResponse(BaseModel):
    """Used when returning log entries to the UI."""
    id: int
    job_id: int
    status: ApplicationStatus
    notes: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class StagingJobStatus(enum.Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    DONE       = "DONE"
    FAILED     = "FAILED"


class StagingJob(Base):
    """
    Holds job URLs submitted manually via the UI.
    Airflow polls for PENDING rows, enriches them, then promotes
    the result into linkedin_jobs and marks this row as DONE.
    """
    __tablename__ = "staging_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_link: Mapped[str] = mapped_column(String(511), nullable=False)

    status: Mapped[StagingJobStatus] = mapped_column(
        Enum(StagingJobStatus),
        nullable=False,
        default=StagingJobStatus.PENDING,
    )

    # Populated after scraping
    job_id:       Mapped[Optional[str]] = mapped_column(String(64),  nullable=True)
    job_title:    Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    location:     Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    post_date:    Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    salary:       Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description:  Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Label the user gives the search (e.g. "Senior Data Engineer")
    job_search:    Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)


# Pydantic schema for the staging endpoint
class StagingJobCreate(BaseModel):
    job_link: str
    job_search: Optional[str] = None


class StagingJobResponse(BaseModel):
    id: int
    job_link: str
    status: StagingJobStatus
    job_title: Optional[str]
    company_name: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# --- FUTURE: FINANCE MODULE ---
# You will simply add 'class Finance(Base):' here later!
