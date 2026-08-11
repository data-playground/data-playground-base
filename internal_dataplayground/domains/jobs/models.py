import datetime
import enum
from typing import Optional

from core.base_model import Base
from pydantic import BaseModel
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    __table_args__ = (
        # Guards against inserting the exact same posting twice from the
        # same source (LinkedIn, Greenhouse, or Lever). All three insertion
        # paths (life_os_job_scout.py, life_os_job_scout_ats.py, and the
        # new life_os_staging_promoter.py) already check for this in Python
        # before inserting — this is the DB-level safety net for whatever
        # a Python-level check misses (e.g. two DAG runs racing each
        # other), which matters more now that there are three separate
        # writers into this table instead of two.
        #
        # Does NOT catch cross-source duplicates (the same real job posted
        # to LinkedIn AND the company's own Greenhouse board, which get
        # different external_ref values) — that's a different problem,
        # already handled separately by the fuzzy title/company matching
        # in airflow/agents/job_dedup.py, and can't be enforced as a DB
        # constraint since it isn't an exact-match relationship.
        #
        # NULLs don't collide under this constraint (MariaDB, like most
        # SQL databases, treats each NULL as distinct) — safe for any
        # legacy rows that predate the external_ref column being populated.
        #
        # ⚠ BEFORE APPLYING TO THE REAL DATABASE: run
        #   SELECT source, external_ref, COUNT(*) FROM linkedin_jobs
        #   WHERE external_ref IS NOT NULL
        #   GROUP BY source, external_ref HAVING COUNT(*) > 1;
        # first. If that returns any rows, clean those up (or decide which
        # to keep) before adding this constraint via Alembic/ALTER TABLE,
        # or the migration will fail outright. Also worth a
        # `SHOW INDEX FROM linkedin_jobs` first — per the habits-domain
        # migration postmortem (§4.4), it's possible this already exists
        # in production without ever having been declared here.
        UniqueConstraint("source", "external_ref", name="uq_linkedin_jobs_source_external_ref"),

        # Added alongside the server-side filtering/pagination rework of
        # GET /jobs and GET /jobs/rows (see jobs.py): every job list query
        # now filters and orders by fit_score, and the keyset-pagination
        # cursor specifically orders by (fit_score DESC, ID DESC) — a
        # composite index in that same order lets the DB satisfy filtering,
        # ordering, and "give me everything after this cursor" in one index
        # scan instead of a full table scan, which matters a lot more now
        # that this runs on every filter change and every "Load More" click
        # instead of once per full page load.
        Index("ix_linkedin_jobs_fit_score_id", "fit_score", "ID"),
        # Used by the date-range filter (search_date >= / <= ...).
        Index("ix_linkedin_jobs_search_date", "search_date"),

        # ⚠ BEFORE APPLYING TO THE REAL DATABASE: these are net-new indexes,
        # not previously declared anywhere (per the habits-migration
        # postmortem's §4.4 playbook, confirm with `SHOW INDEX FROM
        # linkedin_jobs` that they genuinely don't already exist under a
        # different name first) — apply via Alembic/ALTER TABLE, same as
        # the UniqueConstraint above. Building an index on a live,
        # multi-thousand-row table takes a moment and can briefly lock
        # writes depending on your MariaDB version/settings — fine to run
        # any time, just don't expect it to be instant.
    )

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

    source: Mapped[str] = mapped_column(String(20), nullable=False, default="linkedin")
    external_ref: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

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
        Convenience property — returns the most recent application status's
        display VALUE (e.g. "Phone Screen") or None if the job hasn't been
        applied to yet. Used for human-readable display.
        """
        if self.application_logs:
            return self.application_logs[0].status.value
        return None

    @property
    def latest_status_key(self) -> Optional[str]:
        """
        Same as latest_status but returns the ApplicationStatus member's
        NAME (e.g. "PHONE_SCREEN") instead of its display value. Templates,
        JS data-attributes, and filter logic all compare against this key
        form — use this property directly instead of re-deriving it via
        `latest_status | upper | replace(' ', '_')` at each call site.
        That re-derivation is exactly what caused a case-mismatch bug in
        POST /ats/log (see ats.py) — collapsing every site onto one
        property removes the whole class of case/spacing bugs, not just
        that one instance.
        """
        if self.application_logs:
            return self.application_logs[0].status.name
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
        Enum(ApplicationStatus, values_callable=lambda x: [e.value for e in x]),
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


class JobSearchKeyword(Base):
    """
    Replaces the hardcoded DEFAULT_SEARCHES list in life_os_job_scout.py.
    The DAG reads active keywords straight from this table via dag_db raw
    SQL — editing the list here (through the Jobs > Config page) takes
    effect on the next scheduled run, no deploy required.
    """
    __tablename__ = "job_search_keywords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    keyword: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )


class WatchedCompany(Base):
    """
    The curated Greenhouse/Lever company watchlist that job_ats_agents.py
    reads from. Populated manually or via the "candidate companies" panel
    on the Jobs > Config page, which surfaces companies whose LinkedIn
    postings keep scoring high but aren't followed yet.

    greenhouse_slug / lever_slug are the board tokens from the company's
    public job board URL (e.g. boards.greenhouse.io/{slug} or
    jobs.lever.co/{slug}) — there's no directory to look these up
    automatically, they have to be found and entered once per company.
    """
    __tablename__ = "watched_companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    greenhouse_slug: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    lever_slug: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    source_note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    added_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    @property
    def has_ats_source(self) -> bool:
        return bool(self.greenhouse_slug or self.lever_slug)

    @property
    def source_badges(self) -> list[str]:
        badges = []
        if self.greenhouse_slug:
            badges.append("Greenhouse")
        if self.lever_slug:
            badges.append("Lever")
        return badges

class JobScoutRunLog(Base):
    """
    One row per DAG run for either Job Scout DAG (LinkedIn or ATS). Written
    by airflow/agents/job_scout_health.log_run(); read here (via SQLAlchemy)
    for the Settings page panel, and separately by the digest DAG in
    Airflow (via dag_db, since that container doesn't share this ORM).
    """
    __tablename__ = "job_scout_run_log"
 
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dag_id: Mapped[str] = mapped_column(String(100), nullable=False)
    run_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    items_attempted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    new_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_loaded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ok")
    message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

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
