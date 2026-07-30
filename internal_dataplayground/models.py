import datetime
import enum
from typing import Optional
from sqlalchemy import BigInteger, String, Boolean, Date, Text, Integer, Enum, ForeignKey, DateTime, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel
from decimal import Decimal

# The shared Base class for all tables now lives in core/base_model.py.
# Re-exported here (temporary — see domains/habits pilot roadmap) so every
# other file still doing `from models import Base` keeps working unchanged.
from core.base_model import Base

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
        
"""
FINANCE MODULE — updated models for dynamic categories.
Replace the existing finance section in models.py with this.
"""

class AccountType(enum.Enum):
    CHECKING     = "Checking"
    CREDIT_CARD  = "Credit Card"
    SAVINGS      = "Savings"


# ── Dynamic category — no longer an Enum, now a DB-backed table ──

class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    account_type: Mapped[AccountType] = mapped_column(
        Enum(AccountType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    last_four: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    transactions: Mapped[list["Transaction"]] = relationship(
        "Transaction", back_populates="account", lazy="selectin"
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    # category is now a plain string referencing categories.name
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="Other")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    account: Mapped["Account"] = relationship("Account", back_populates="transactions")


# ── Pydantic schemas ──

class AccountCreate(BaseModel):
    name: str
    account_type: AccountType
    last_four: Optional[str] = None


class AccountResponse(BaseModel):
    id: int
    name: str
    account_type: AccountType
    last_four: Optional[str]

    class Config:
        from_attributes = True


class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


class TransactionResponse(BaseModel):
    id: int
    account_id: int
    date: datetime.date
    description: str
    amount: Decimal
    category: str
    notes: Optional[str]

    class Config:
        from_attributes = True


"""
BLOG MODULE — append these classes to the bottom of models.py
"""

class BlogProjectType(enum.Enum):
    EXISTING_ASSET = "existing_asset"
    NEW_BUILD      = "new_build"
    TUTORIAL       = "tutorial"



class BlogIdeaStatus(enum.Enum):
    IDEA_GENERATED              = "idea_generated"
    WAITING_FOR_WRITING_TRIGGER = "waiting_for_writing_trigger"
    IN_DEVELOPMENT              = "in_development"              # new — Phase 1E
    WRITING_IN_PROGRESS         = "writing_in_progress"
    WAITING_FOR_REVIEW          = "waiting_for_review"
    REVIEW_COMPLETED            = "review_completed"
    READY_TO_PUBLISH            = "ready_to_publish"
    PUBLISHED                   = "published"
    ARCHIVED                    = "archived"
 
    @property
    def label(self) -> str:
        return {
            "idea_generated":              "Idea Generated",
            "waiting_for_writing_trigger": "Ready to Write",
            "in_development":              "In Development",   # new
            "writing_in_progress":         "Writing…",
            "waiting_for_review":          "Awaiting Review",
            "review_completed":            "Review Done",
            "ready_to_publish":            "Ready to Publish",
            "published":                   "Published",
            "archived":                    "Archived"
        }[self.value]
 
    @property
    def kanban_column(self) -> str:
        if self.value in ("idea_generated", "waiting_for_writing_trigger"):
            return "backlog"
        if self.value == "in_development":
            return "in_development"
        if self.value in ("writing_in_progress", "waiting_for_review", "review_completed"):
            return "in_progress"
        # Specifically define 'done' statuses
        if self.value in ("ready_to_publish", "published"):
            return "done"
        # Return 'archived' so it doesn't match any visible column ID
        return "archived"

# Allowed difficulty values — enforced at the application layer.
# Stored as VARCHAR(20) in the DB (not ENUM) for forward flexibility.
DIFFICULTY_LEVELS = ("starter", "weekend", "ambitious")


class BlogIdea(Base):
    __tablename__ = "blog_ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title_concept: Mapped[str] = mapped_column(String(255), nullable=False)
    project_type: Mapped[BlogProjectType] = mapped_column(
        Enum(BlogProjectType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=BlogProjectType.NEW_BUILD,
    )

    # NEW: difficulty level — 'starter' | 'weekend' | 'ambitious'
    # Nullable so legacy rows don't break. Application always sets it on insert.
    difficulty: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Blueprint fields
    the_build:         Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    the_narrative:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    the_selling_point: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # BYOI
    raw_idea_input: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Evidence (HITL checkpoint 1)
    code_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # AI artifacts
    draft_v1:          Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    draft_v2:          Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    final_article:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # SEO
    seo_title:       Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seo_description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    seo_tags:        Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # State
    status: Mapped[BlogIdeaStatus] = mapped_column(
        Enum(BlogIdeaStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=BlogIdeaStatus.IDEA_GENERATED,
    )
    airflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Code Intelligence links
    code_file_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("code_files.id", ondelete="SET NULL"), nullable=True
    )
    code_project_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("code_projects.id", ondelete="SET NULL"), nullable=True
    )
    code_file: Mapped[Optional["CodeFile"]] = relationship(
        "CodeFile", back_populates="blog_ideas",
        foreign_keys=[code_file_id],
    )
    code_project: Mapped[Optional["CodeProject"]] = relationship(
        "CodeProject", back_populates="blog_ideas",
        foreign_keys=[code_project_id],
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    @property
    def difficulty_label(self) -> str:
        """Human-readable difficulty label for templates."""
        return {
            "starter":   "⬡ Starter",
            "weekend":   "◈ Weekend",
            "ambitious": "◉ Ambitious",
        }.get(self.difficulty or "", "—")

    @property
    def difficulty_color_class(self) -> str:
        """CSS variable name for the difficulty colour in templates."""
        return {
            "starter":   "var(--green)",
            "weekend":   "var(--yellow)",
            "ambitious": "var(--red)",
        }.get(self.difficulty or "", "var(--text-muted)")


# ── Pydantic schemas ──

class BlogIdeaCreate(BaseModel):
    raw_idea_input: str
    title_concept: Optional[str] = None
    difficulty: Optional[str] = None


class BlogIdeaResponse(BaseModel):
    id: int
    title_concept: str
    project_type: BlogProjectType
    difficulty: Optional[str]
    status: BlogIdeaStatus
    the_narrative: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# ── ENUMS ─────────────────────────────────────────────────────────────────────

class ReadmeStatus(enum.Enum):
    NONE      = "none"
    DRAFT     = "draft"
    REVIEWED  = "reviewed"
    APPROVED  = "approved"
    PUSHED    = "pushed"
    STALE     = "stale"      # code changed after README was generated

class FolderReadmeStatus(enum.Enum):
    NONE     = "none"
    DRAFT    = "draft"
    REVIEWED = "reviewed"
    PUSHED   = "pushed"
    STALE    = "stale"  # set when any file in the folder was pulled after generation

class CommentedStatus(enum.Enum):
    NONE      = "none"
    GENERATED = "generated"
    REVIEWED  = "reviewed"
    PUSHED    = "pushed"


class ImprovementStatus(enum.Enum):
    NONE      = "none"
    GENERATED = "generated"
    REVIEWED  = "reviewed"
    APPLIED   = "applied"
    PUSHED    = "pushed"


# ── CODE INTELLIGENCE ─────────────────────────────────────────────────────────

class CodeProject(Base):
    """
    One row per logical project scope.
    A project can be a whole repo, a folder, or a nested subfolder.
    Defined by github_repo + github_base_path together.

    Examples:
      github_repo="pedro/data-playground-base", github_base_path=""
        → whole repo

      github_repo="pedro/data-playground-base", github_base_path="internal_dataplayground"
        → FastAPI app subfolder only

      github_repo="pedro/data-playground-base", github_base_path="internal_dataplayground/routers"
        → just the routers subfolder
    """
    __tablename__ = "code_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # GitHub coordinates
    github_repo: Mapped[str] = mapped_column(String(255), nullable=False)
    github_base_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Empty string or None = whole repo root

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # README — generated from all file narrations, pushed to GitHub
    readme_md: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    readme_status: Mapped[ReadmeStatus] = mapped_column(
        Enum(ReadmeStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ReadmeStatus.NONE,
    )
    readme_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    readme_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    readme_pushed_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    files: Mapped[list["CodeFile"]] = relationship(
        "CodeFile", back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    folder_readmes: Mapped[list["FolderReadme"]] = relationship(
        "FolderReadme",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="FolderReadme.folder_path",
    )
    blog_ideas: Mapped[list["BlogIdea"]] = relationship(
        "BlogIdea", back_populates="code_project",
        foreign_keys="BlogIdea.code_project_id",
    )

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def has_narrations(self) -> bool:
        return all(f.narration for f in self.files)

    @property
    def readme_is_stale(self) -> bool:
        """True if any file was pulled after the README was generated."""
        if not self.readme_generated_at:
            return False
        return any(
            f.code_pulled_at and f.code_pulled_at > self.readme_generated_at
            for f in self.files
        )

    @property
    def folder_readme_coverage(self) -> dict:
        """
        Returns a summary of README coverage across all tracked folders.
        Used by the coverage dashboard to show which folders need attention.

        Returns:
          {
              "total": int,       # folders with at least one tracked file
              "none": int,        # no README generated
              "draft": int,       # generated, not yet reviewed
              "reviewed": int,    # reviewed, not yet pushed
              "pushed": int,      # live on GitHub
              "stale": int,       # pushed but code has changed since
          }
        """
        counts = {"total": 0, "none": 0, "draft": 0, "reviewed": 0, "pushed": 0, "stale": 0}
        for fr in self.folder_readmes:
            counts["total"] += 1
            counts[fr.status.value] += 1
        return counts


class CodeFile(Base):
    """
    One row per individual script tracked within a CodeProject.
    Narration is the key output — consumed by Ghostwriter, Researcher,
    Idea Expander agents.
    Commented code and improvement notes are reviewed before pushing.
    """
    __tablename__ = "code_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("code_projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # e.g. "finance.py"
    github_path: Mapped[str] = mapped_column(String(500), nullable=False)
    # e.g. "internal_dataplayground/routers/finance.py"
    github_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # SHA of last pull — used for staleness detection and push

    # Raw code from GitHub
    raw_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    code_pulled_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    # Code Narrator output
    narration: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    narration_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    # Code Commenter output
    commented_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    commented_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    commented_status: Mapped[CommentedStatus] = mapped_column(
        Enum(CommentedStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=CommentedStatus.NONE,
    )

    # Code Improver output
    improvement_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    improvement_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    improvement_status: Mapped[ImprovementStatus] = mapped_column(
        Enum(ImprovementStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ImprovementStatus.NONE,
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    project: Mapped["CodeProject"] = relationship("CodeProject", back_populates="files")
    blog_ideas: Mapped[list["BlogIdea"]] = relationship(
        "BlogIdea", back_populates="code_file",
        foreign_keys="BlogIdea.code_file_id",
    )

    @property
    def narration_is_stale(self) -> bool:
        """True if code was pulled after narration was generated."""
        if not self.narration_generated_at or not self.code_pulled_at:
            return False
        return self.code_pulled_at > self.narration_generated_at


class FolderReadme(Base):
    """
    Tracks the README for a single folder within a CodeProject.

    One row per (project_id, folder_path) pair — enforced by the unique
    constraint in the migration.

    Lifecycle:
        none → draft (AI generated) → reviewed (human approved)
             → pushed (on GitHub) → stale (code changed since generation)

    The `stale` status should be set by the router or a background check
    whenever any CodeFile in folder_path has code_pulled_at > readme_generated_at.

    github_path is the full repo path where the README.md would live:
        e.g. "internal_dataplayground/routers/README.md"
    It is stored rather than computed so pushes don't need to reconstruct it.
    """
    __tablename__ = "folder_readmes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Ownership ─────────────────────────────────────────────────────────────
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("code_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Folder identity ───────────────────────────────────────────────────────
    # Full path within the repo — the unique key alongside project_id.
    # e.g. "internal_dataplayground/routers"
    folder_path: Mapped[str] = mapped_column(String(500), nullable=False)

    # Short label shown in the UI — avoids splitting strings everywhere.
    # e.g. "routers", "dags", "partials"
    folder_display_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Where the README.md would live on GitHub if pushed.
    # e.g. "internal_dataplayground/routers/README.md"
    # Nullable until the user decides the push target.
    github_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # ── Content ───────────────────────────────────────────────────────────────
    readme_md: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Pipeline state ────────────────────────────────────────────────────────
    status: Mapped[FolderReadmeStatus] = mapped_column(
        Enum(FolderReadmeStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=FolderReadmeStatus.NONE,
    )

    # ── GitHub push tracking ──────────────────────────────────────────────────
    # SHA returned by GitHub after the last push.
    # Required to update an existing file (GitHub rejects PUT without it).
    github_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    readme_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )
    readme_pushed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    project: Mapped["CodeProject"] = relationship(
        "CodeProject", back_populates="folder_readmes"
    )

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def is_stale(self) -> bool:
        """True when the README exists but the folder's code has since changed."""
        return self.status == FolderReadmeStatus.STALE

    @property
    def needs_readme(self) -> bool:
        """True when no README has been generated yet for this folder."""
        return self.status == FolderReadmeStatus.NONE

    @property
    def is_published(self) -> bool:
        """True when the README has been pushed to GitHub and is current."""
        return self.status == FolderReadmeStatus.PUSHED


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

from pydantic import BaseModel

class CodeProjectCreate(BaseModel):
    project_name: str
    github_repo: str
    github_base_path: Optional[str] = None
    description: Optional[str] = None


class CodeProjectResponse(BaseModel):
    id: int
    project_name: str
    github_repo: str
    github_base_path: Optional[str]
    description: Optional[str]
    readme_status: ReadmeStatus
    file_count: int

    class Config:
        from_attributes = True


class CodeFileResponse(BaseModel):
    id: int
    project_id: int
    file_name: str
    github_path: str
    narration: Optional[str]
    commented_status: CommentedStatus
    improvement_status: ImprovementStatus
    narration_is_stale: bool

    class Config:
        from_attributes = True

class FolderReadmeResponse(BaseModel):
    id: int
    project_id: int
    folder_path: str
    folder_display_name: str
    github_path: Optional[str]
    status: FolderReadmeStatus
    readme_generated_at: Optional[datetime.datetime]
    readme_pushed_at: Optional[datetime.datetime]

    class Config:
        from_attributes = True


class FolderReadmeCreate(BaseModel):
    """Used when the router creates or upserts a folder README record."""
    project_id: int
    folder_path: str
    folder_display_name: str
    github_path: Optional[str] = None

# ── HABIT TRACKER MODULE ─────────────────────────────────────────────────────
# Moved to domains/habits/models.py as part of the domains-folder pilot
# migration. Re-exported here so any other file still doing
# `from models import Habit` (etc.) keeps working unchanged.
# TODO: remove after all cross-references are updated
from domains.habits.models import Habit, HabitLog, HabitSettings
from domains.habits.models import HabitCreate, HabitUpdate, HabitResponse, HabitLogResponse

# ── JOURNAL MODULE ────────────────────────────────────────────────────────────
# Append these classes to the bottom of models.py
#
# PRIVACY ARCHITECTURE — HARD CONSTRAINT:
#   content, gratitude, and challenges fields are NEVER sent to external AI.
#   Weekly synthesis is generated from mood_score and energy_score ONLY.
#   Violating this constraint is a critical privacy bug.

import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, String, Boolean, Date, Text, Integer, Enum,
    ForeignKey, DateTime, Numeric, SmallInteger, JSON  # ← add SmallInteger, JSON
)
from sqlalchemy.orm import Mapped, mapped_column


class JournalEntry(Base):
    """
    One row per calendar day. Entry text is private — never sent externally.
    Numeric scores (mood, energy) feed the weekly synthesis DAG.

    Locking: entries become read-only 24 hours after created_at.
    Enforcement: router checks is_locked before accepting edits;
    the nightly DAG sets is_locked=True as a cleanup pass.
    """
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # DB-level UNIQUE constraint ensures one entry per day
    entry_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)

    # 1–5 scales. NULL = not rated for the day (perfectly valid).
    mood_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    energy_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # ── PRIVATE TEXT — NEVER SENT TO EXTERNAL AI ──────────────────────────────
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gratitude: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    challenges: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # ──────────────────────────────────────────────────────────────────────────

    # Once True, the router rejects edits. Set by router check and nightly DAG.
    is_locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    @property
    def hours_until_lock(self) -> float:
        """Hours remaining before this entry locks. Negative means already past."""
        lock_at = self.created_at + datetime.timedelta(hours=24)
        delta = lock_at - datetime.datetime.utcnow()
        return delta.total_seconds() / 3600

    @property
    def should_be_locked(self) -> bool:
        """True if 24 hours have elapsed since creation."""
        return self.hours_until_lock <= 0

    @property
    def mood_label(self) -> str:
        return {1: "😞", 2: "😕", 3: "😐", 4: "🙂", 5: "😄"}.get(self.mood_score or 0, "—")

    @property
    def energy_label(self) -> str:
        return {1: "▁", 2: "▂", 3: "▄", 4: "▆", 5: "█"}.get(self.energy_score or 0, "—")

    @property
    def mood_color_class(self) -> str:
        """CSS class suffix for mood-dot coloring on the calendar."""
        if not self.mood_score:
            return "none"
        if self.mood_score <= 2:
            return "low"
        if self.mood_score == 3:
            return "mid"
        return "high"


class WeeklySynthesis(Base):
    """
    AI-generated weekly pattern summary. Built exclusively from numeric scores
    and habit/workout counts — no personal text from journal_entries.

    data_sources JSON records which Life OS modules contributed data,
    allowing honest attribution in the UI and future auditing.
    """
    __tablename__ = "weekly_syntheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Week boundaries — Monday to Sunday, enforced by the DAG
    week_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)
    week_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    # Aggregated from journal_entries for the week
    avg_mood: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 2), nullable=True)
    avg_energy: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 2), nullable=True)

    # From habit_logs (Phase 2) — None if habit module not yet built
    habits_completion_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2), nullable=True)

    # From workout_sessions (Phase 4) — None until that phase is built
    workout_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # The AI output — generated from numbers only, never from personal text
    synthesis_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Audit: which modules fed this synthesis, and which model produced it
    data_sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    model_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    @property
    def week_label(self) -> str:
        """Human-readable week range, e.g. 'May 4 – 10, 2026'."""
        if self.week_start_date.month == self.week_end_date.month:
            return (
                f"{self.week_start_date.strftime('%b %-d')} – "
                f"{self.week_end_date.strftime('%-d, %Y')}"
            )
        return (
            f"{self.week_start_date.strftime('%b %-d')} – "
            f"{self.week_end_date.strftime('%b %-d, %Y')}"
        )

# ── RECIPE MANAGER MODULE ────────────────────────────────────────────────────
# Append everything below to the bottom of models.py.
#
# Design decisions:
#   - recipe_cook_log excluded — cook history lives on recipes.times_cooked
#     and recipes.last_cooked_at only. Simple and sufficient.
#   - PantryItem is intentionally minimal (ingredient_id only).
#     No quantity, unit, or expiry — those can be added later without
#     touching any existing code.
#   - RecipeIngredient.quantity is Decimal NULL — NULL means "to taste".
#   - Images stored as source URL strings only — no local file handling.
#   - Normalization pipeline (agent functions) lives in recipe_agents.py,
#     not in models — models stay pure data definitions.


import enum
from decimal import Decimal
from typing import Optional
import datetime

from sqlalchemy import (
    Boolean, Date, DateTime, Enum, ForeignKey, Integer,
    Numeric, SmallInteger, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class IngredientCategory(enum.Enum):
    PRODUCE    = "produce"
    PROTEIN    = "protein"
    DAIRY      = "dairy"
    GRAIN      = "grain"
    PANTRY     = "pantry"
    SPICE      = "spice"
    CONDIMENT  = "condiment"
    BEVERAGE   = "beverage"
    FROZEN     = "frozen"
    OTHER      = "other"


class RecipeSourceType(enum.Enum):
    MANUAL       = "manual"
    URL          = "url"
    PDF          = "pdf"
    IMAGE        = "image"
    AI_GENERATED = "ai_generated"


class RecipeMealType(enum.Enum):
    BREAKFAST = "breakfast"
    LUNCH     = "lunch"
    DINNER    = "dinner"
    SNACK     = "snack"
    DESSERT   = "dessert"
    SIDE      = "side"
    DRINK     = "drink"
    OTHER     = "other"


class RecipeDifficulty(enum.Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class IngredientUnit(enum.Enum):
    CUP       = "cup"
    TBSP      = "tbsp"
    TSP       = "tsp"
    ML        = "ml"
    L         = "l"
    G         = "g"
    KG        = "kg"
    OZ        = "oz"
    LB        = "lb"
    PIECE     = "piece"
    CLOVE     = "clove"
    BUNCH     = "bunch"
    SLICE     = "slice"
    CAN       = "can"
    PACKAGE   = "package"
    TO_TASTE  = "to_taste"
    AS_NEEDED = "as_needed"
    PINCH     = "pinch"
    HANDFUL   = "handful"


# ── ORM MODELS ────────────────────────────────────────────────────────────────

class Ingredient(Base):
    """
    Normalized ingredient reference table.
    Every unique ingredient appears here exactly once, by canonical name.
    "garlic", "minced garlic", and "2 garlic cloves" all resolve to
    one row with name="garlic" and category=PRODUCE.

    New rows are created automatically during recipe import by the
    normalization pipeline in recipe_agents.py — never inserted manually.
    """
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    category: Mapped[IngredientCategory] = mapped_column(
        Enum(IngredientCategory, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=IngredientCategory.OTHER,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    recipe_ingredients: Mapped[list["RecipeIngredient"]] = relationship(
        "RecipeIngredient", back_populates="ingredient"
    )
    pantry_item: Mapped[Optional["PantryItem"]] = relationship(
        "PantryItem", back_populates="ingredient", uselist=False
    )

    @property
    def in_pantry(self) -> bool:
        """True if this ingredient exists in the user's pantry."""
        return self.pantry_item is not None


class RecipeTag(Base):
    """
    Tag reference table. Tags are user-defined strings like
    "quick", "vegetarian", "meal-prep", "spicy", etc.
    """
    __tablename__ = "recipe_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)

    # Relationships
    recipes: Mapped[list["Recipe"]] = relationship(
        "Recipe", secondary="recipe_tags_junction", back_populates="tags"
    )


# Association table for the Recipe ↔ RecipeTag many-to-many.
# Defined as a plain Table (not a mapped class) because it carries
# no extra columns — just the two foreign keys.
from sqlalchemy import Table, Column
recipe_tags_junction = Table(
    "recipe_tags_junction",
    Base.metadata,
    Column(
        "recipe_id",
        Integer,
        ForeignKey("recipes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "tag_id",
        Integer,
        ForeignKey("recipe_tags.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Recipe(Base):
    """
    Core recipe table.

    Cook tracking is intentionally simple: times_cooked (INT) and
    last_cooked_at (DATE) are updated by the POST /recipes/{id}/cook
    endpoint. No per-cook log table — the simpler model is sufficient.

    Instructions are stored as Markdown — the template renders them
    with a JS markdown parser (marked.js, already loaded in base.html).

    image_url stores the source URL of an external image only.
    No local file storage — if the source URL breaks, the image disappears,
    which is acceptable for a personal tool.
    """
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    source_type: Mapped[RecipeSourceType] = mapped_column(
        Enum(RecipeSourceType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=RecipeSourceType.MANUAL,
    )
    cuisine: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    meal_type: Mapped[Optional[RecipeMealType]] = mapped_column(
        Enum(RecipeMealType, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    prep_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cook_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_time_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    servings: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    difficulty: Mapped[Optional[RecipeDifficulty]] = mapped_column(
        Enum(RecipeDifficulty, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    # Markdown formatted — numbered steps preferred for the step-mode reader.
    instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # External image URL only — no local storage.
    image_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    # 1–5 personal rating. NULL = not yet rated.
    user_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    # Cook tracking (simple — no log table).
    times_cooked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_cooked_at: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    is_favorite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Soft delete — archived recipes are excluded from library queries by default.
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    ingredients: Mapped[list["RecipeIngredient"]] = relationship(
        "RecipeIngredient",
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.sort_order",
        lazy="selectin",
    )
    tags: Mapped[list["RecipeTag"]] = relationship(
        "RecipeTag",
        secondary=recipe_tags_junction,
        back_populates="recipes",
        lazy="selectin",
    )

    @property
    def total_time(self) -> Optional[int]:
        """
        Returns total_time_minutes if set, otherwise computes from
        prep + cook if both are available.
        """
        if self.total_time_minutes is not None:
            return self.total_time_minutes
        if self.prep_time_minutes is not None and self.cook_time_minutes is not None:
            return self.prep_time_minutes + self.cook_time_minutes
        return None

    @property
    def time_display(self) -> str:
        """Human-readable total time for template display, e.g. '1h 20min'."""
        t = self.total_time
        if t is None:
            return "—"
        if t < 60:
            return f"{t}min"
        hours = t // 60
        mins = t % 60
        if mins == 0:
            return f"{hours}h"
        return f"{hours}h {mins}min"

    @property
    def rating_display(self) -> str:
        """Returns filled/empty star string for template rendering."""
        if not self.user_rating:
            return "☆☆☆☆☆"
        return "★" * self.user_rating + "☆" * (5 - self.user_rating)

    @property
    def tag_names(self) -> list[str]:
        return [t.name for t in self.tags]


class RecipeIngredient(Base):
    """
    One row per ingredient per recipe.
    Quantity is Decimal NULL — NULL means "to taste" or "as needed".
    preparation_note holds the prep method ("finely diced", "room temperature")
    separate from the canonical ingredient name — this is what the
    normalization agent strips out and stores here instead of polluting
    the ingredients table with "minced garlic" vs "garlic".
    sort_order controls display sequence in the recipe view.
    """
    __tablename__ = "recipe_ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recipe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recipes.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    ingredient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("ingredients.id"),
        nullable=False, index=True,
    )
    # NULL = "to taste" / "as needed" — rendered differently in templates.
    quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 3), nullable=True)
    unit: Mapped[Optional[IngredientUnit]] = mapped_column(
        Enum(IngredientUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=True,
    )
    # The preparation method, separated from the ingredient name.
    # e.g. "finely diced", "at room temperature", "roughly chopped"
    preparation_note: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    is_optional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="ingredients")
    ingredient: Mapped["Ingredient"] = relationship(
        "Ingredient", back_populates="recipe_ingredients", lazy="selectin"
    )


    def quantity_scaled(self, scale_factor: float) -> Optional[str]:
        """
        Returns a human-readable scaled quantity string.
        Used by the JavaScript servings scaler — this Python version is
        available for server-side rendering if needed.
        e.g. scale_factor=2.0 → "2 cups" becomes "4 cups"
        """
        if self.quantity is None:
            return None
        scaled = float(self.quantity) * scale_factor
        # Clean up trailing zeros: 2.0 → "2", 2.5 → "2.5"
        if scaled == int(scaled):
            return str(int(scaled))
        return f"{scaled:.2f}".rstrip("0")

    @property
    def display_quantity(self) -> str:
        """
        Human-readable quantity for the ingredient line.
        Returns "to taste" for NULL quantities, otherwise formats the decimal.
        """
        if self.quantity is None:
            return "to taste"
        q = float(self.quantity)
        if q == int(q):
            return str(int(q))
        return f"{q:.2f}".rstrip("0")

    @property
    def full_ingredient_line(self) -> str:
        """
        Renders the complete ingredient line as a string for display.
        e.g. "2 cups flour, sifted" or "to taste salt"
        """
        parts = []
        if self.quantity is not None:
            parts.append(self.display_quantity)
        if self.unit:
            parts.append(self.unit.value)
        parts.append(self.ingredient.name)
        if self.preparation_note:
            parts.append(f", {self.preparation_note}")
        if self.is_optional:
            parts.append(" (optional)")
        return " ".join(parts)


class PantryItem(Base):
    """
    Minimal pantry — just which ingredients the user currently has.
    One row per ingredient (enforced by UNIQUE on ingredient_id).
    No quantity, unit, or expiry by design — keeps the UX lightweight.
    A future migration can add those columns without touching this model.
    """
    __tablename__ = "pantry_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ingredient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("ingredients.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    ingredient: Mapped["Ingredient"] = relationship(
        "Ingredient", back_populates="pantry_item"
    )


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

class IngredientResponse(BaseModel):
    id: int
    name: str
    category: IngredientCategory
    in_pantry: bool

    class Config:
        from_attributes = True


class RecipeIngredientResponse(BaseModel):
    id: int
    ingredient_id: int
    ingredient_name: str
    quantity: Optional[Decimal]
    unit: Optional[IngredientUnit]
    preparation_note: Optional[str]
    is_optional: bool
    sort_order: int
    display_quantity: str
    full_ingredient_line: str

    class Config:
        from_attributes = True


class RecipeTagResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class RecipeResponse(BaseModel):
    id: int
    title: str
    source_url: Optional[str]
    source_type: RecipeSourceType
    cuisine: Optional[str]
    meal_type: Optional[RecipeMealType]
    prep_time_minutes: Optional[int]
    cook_time_minutes: Optional[int]
    total_time_minutes: Optional[int]
    servings: Optional[int]
    difficulty: Optional[RecipeDifficulty]
    user_rating: Optional[int]
    times_cooked: int
    last_cooked_at: Optional[datetime.date]
    is_favorite: bool
    image_url: Optional[str]
    time_display: str
    rating_display: str
    tag_names: list[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class RecipeCreate(BaseModel):
    """Used when creating a recipe manually from form data."""
    title: str
    source_url: Optional[str] = None
    source_type: RecipeSourceType = RecipeSourceType.MANUAL
    cuisine: Optional[str] = None
    meal_type: Optional[RecipeMealType] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    servings: Optional[int] = None
    difficulty: Optional[RecipeDifficulty] = None
    instructions: Optional[str] = None
    notes: Optional[str] = None
    image_url: Optional[str] = None


class PantryItemResponse(BaseModel):
    id: int
    ingredient_id: int
    ingredient_name: str
    ingredient_category: IngredientCategory

    class Config:
        from_attributes = True

"""
WORKOUT TRACKER MODULE — append these classes to the bottom of models.py

Imports to add at the top of models.py if not already present:
  from sqlalchemy import SmallInteger, Numeric, JSON  (Numeric + JSON likely already imported)
"""
import datetime
import enum
from decimal import Decimal
from typing import Optional

# ── ENUMS ─────────────────────────────────────────────────────────────────────

class LocationType(enum.Enum):
    HOME    = "home"
    GYM     = "gym"
    OUTDOOR = "outdoor"
    OTHER   = "other"


class EquipmentType(enum.Enum):
    BARBELL         = "barbell"
    DUMBBELL        = "dumbbell"
    MACHINE         = "machine"
    CABLE           = "cable"
    BODYWEIGHT      = "bodyweight"
    CARDIO          = "cardio"
    RESISTANCE_BAND = "resistance_band"
    KETTLEBELL      = "kettlebell"
    OTHER           = "other"


class MuscleGroup(enum.Enum):
    CHEST      = "chest"
    BACK       = "back"
    SHOULDERS  = "shoulders"
    BICEPS     = "biceps"
    TRICEPS    = "triceps"
    FOREARMS   = "forearms"
    QUADS      = "quads"
    HAMSTRINGS = "hamstrings"
    GLUTES     = "glutes"
    CALVES     = "calves"
    CORE       = "core"
    FULL_BODY  = "full_body"
    CARDIO     = "cardio"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class ExerciseEquipmentType(enum.Enum):
    BARBELL         = "barbell"
    DUMBBELL        = "dumbbell"
    MACHINE         = "machine"
    CABLE           = "cable"
    BODYWEIGHT      = "bodyweight"
    RESISTANCE_BAND = "resistance_band"
    KETTLEBELL      = "kettlebell"
    CARDIO          = "cardio"
    OTHER           = "other"
    ANY             = "any"


class PlanOrigin(enum.Enum):
    USER = "user"
    AI   = "ai"


class WorkoutGoal(enum.Enum):
    STRENGTH        = "strength"
    HYPERTROPHY     = "hypertrophy"
    ENDURANCE       = "endurance"
    GENERAL_FITNESS = "general_fitness"
    WEIGHT_LOSS     = "weight_loss"

    @property
    def label(self) -> str:
        return {
            "strength":        "Strength",
            "hypertrophy":     "Hypertrophy (Muscle Size)",
            "endurance":       "Endurance",
            "general_fitness": "General Fitness",
            "weight_loss":     "Weight Loss",
        }[self.value]


class WeightUnit(enum.Enum):
    KG = "kg"
    LB = "lb"


# ── MODELS ────────────────────────────────────────────────────────────────────

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Enum, ForeignKey,
    Integer, JSON, Numeric, SmallInteger, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
# Base is imported from wherever it's defined in models.py


class WorkoutLocation(Base):
    """
    Physical location where the user trains.
    Equipment is scoped to a location — the AI plan generator uses this
    to know what's available at each location.
    is_default=True means this location is pre-selected when starting a session.
    Only one location should have is_default=True (enforced at application layer).
    """
    __tablename__ = "workout_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    location_type: Mapped[LocationType] = mapped_column(
        Enum(LocationType, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=LocationType.GYM,
    )
    address: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    equipment: Mapped[list["Equipment"]] = relationship(
        "Equipment", back_populates="location",
        cascade="all, delete-orphan", lazy="selectin",
    )
    sessions: Mapped[list["WorkoutSession"]] = relationship(
        "WorkoutSession", back_populates="location",
    )
    plans: Mapped[list["WorkoutPlan"]] = relationship(
        "WorkoutPlan", back_populates="location",
    )

    @property
    def active_equipment(self) -> list["Equipment"]:
        return [e for e in self.equipment if e.is_active]

    @property
    def equipment_summary(self) -> str:
        """Short description of available equipment for AI prompt construction."""
        types = list({e.equipment_type.value for e in self.active_equipment})
        return ", ".join(sorted(types)) if types else "No equipment logged"


class Equipment(Base):
    """
    Individual piece of equipment available at a workout location.
    Used by the AI plan generator to scope exercise recommendations.
    """
    __tablename__ = "equipment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    location_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_locations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    equipment_type: Mapped[EquipmentType] = mapped_column(
        Enum(EquipmentType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    max_weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), nullable=True)
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    location: Mapped["WorkoutLocation"] = relationship(
        "WorkoutLocation", back_populates="equipment"
    )


class Exercise(Base):
    """
    Reference exercise library. Seeded with 75 common exercises.
    is_custom=True marks user-added exercises that are not in the seed data.
    The AI plan generator references exercise names — fuzzy matching happens
    in the router when AI suggestions don't exactly match DB entries.
    """
    __tablename__ = "exercises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    primary_muscle_group: Mapped[MuscleGroup] = mapped_column(
        Enum(MuscleGroup, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    # JSON array of secondary muscle group strings
    secondary_muscle_groups: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    equipment_type: Mapped[ExerciseEquipmentType] = mapped_column(
        Enum(ExerciseEquipmentType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    is_compound: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_custom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    plan_exercises: Mapped[list["WorkoutPlanExercise"]] = relationship(
        "WorkoutPlanExercise", back_populates="exercise",
    )
    sets: Mapped[list["WorkoutSet"]] = relationship(
        "WorkoutSet", back_populates="exercise",
    )


class WorkoutPlan(Base):
    """
    A structured training program — either user-created or AI-generated.
    Only one plan can be active at a time (enforced in the router via
    SET is_active=FALSE on all plans before activating the new one).
    target_days_per_week drives how many WorkoutPlanDay rows to generate.
    """
    __tablename__ = "workout_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_by: Mapped[PlanOrigin] = mapped_column(
        Enum(PlanOrigin, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=PlanOrigin.USER,
    )
    location_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_locations.id", ondelete="SET NULL"), nullable=True
    )
    target_days_per_week: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)
    goal: Mapped[WorkoutGoal] = mapped_column(
        Enum(WorkoutGoal, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WorkoutGoal.GENERAL_FITNESS,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    location: Mapped[Optional["WorkoutLocation"]] = relationship(
        "WorkoutLocation", back_populates="plans"
    )
    days: Mapped[list["WorkoutPlanDay"]] = relationship(
        "WorkoutPlanDay", back_populates="plan",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutPlanDay.day_number",
    )
    sessions: Mapped[list["WorkoutSession"]] = relationship(
        "WorkoutSession", back_populates="plan",
    )

    @property
    def total_exercises(self) -> int:
        return sum(len(d.exercises) for d in self.days)


class WorkoutPlanDay(Base):
    """
    A named day within a workout plan (e.g. "Day 1 — Chest & Triceps").
    day_number is 1-indexed and determines rotation order during logging.
    The user can follow any day they choose when logging — plans are guides,
    not strict schedules.
    """
    __tablename__ = "workout_plan_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    day_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # Human-readable label, e.g. "Chest & Triceps", "Pull Day", "Leg Day"
    day_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan: Mapped["WorkoutPlan"] = relationship("WorkoutPlan", back_populates="days")
    exercises: Mapped[list["WorkoutPlanExercise"]] = relationship(
        "WorkoutPlanExercise", back_populates="plan_day",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutPlanExercise.order_in_day",
    )


class WorkoutPlanExercise(Base):
    """
    An exercise prescribed within a specific day of a workout plan.
    target_weight is in lb (user's default unit) — a starting suggestion only.
    """
    __tablename__ = "workout_plan_exercises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    plan_day_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("workout_plan_days.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    exercise_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("exercises.id"), nullable=False
    )
    target_sets: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=3)
    target_reps_min: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=8)
    target_reps_max: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=12)
    target_weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), nullable=True)
    order_in_day: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan_day: Mapped["WorkoutPlanDay"] = relationship(
        "WorkoutPlanDay", back_populates="exercises"
    )
    exercise: Mapped["Exercise"] = relationship(
        "Exercise", back_populates="plan_exercises", lazy="selectin"
    )

    @property
    def rep_range(self) -> str:
        if self.target_reps_min == self.target_reps_max:
            return str(self.target_reps_min)
        return f"{self.target_reps_min}–{self.target_reps_max}"


class WorkoutSession(Base):
    """
    A single training session. Can be linked to a plan day (structured)
    or standalone (free-form logging).
    weight_unit is the per-session toggle — defaults to lb but user can
    switch to kg for a session without affecting the global default.
    duration_minutes is calculated from started_at/ended_at when ending
    the session, or can be entered manually.
    """
    __tablename__ = "workout_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    plan_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_plans.id", ondelete="SET NULL"), nullable=True
    )
    plan_day_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_plan_days.id", ondelete="SET NULL"), nullable=True
    )
    location_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workout_locations.id", ondelete="SET NULL"), nullable=True
    )
    session_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fatigue_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)  # 1-5
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    plan: Mapped[Optional["WorkoutPlan"]] = relationship(
        "WorkoutPlan", back_populates="sessions"
    )
    plan_day: Mapped[Optional["WorkoutPlanDay"]] = relationship(
        "WorkoutPlanDay", lazy="selectin"
    )
    location: Mapped[Optional["WorkoutLocation"]] = relationship(
        "WorkoutLocation", back_populates="sessions"
    )
    sets: Mapped[list["WorkoutSet"]] = relationship(
        "WorkoutSet", back_populates="session",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WorkoutSet.created_at",
    )

    @property
    def is_active(self) -> bool:
        """True if the session has been started but not ended."""
        return self.started_at is not None and self.ended_at is None

    @property
    def working_sets(self) -> list["WorkoutSet"]:
        return [s for s in self.sets if not s.is_warmup]

    @property
    def exercise_count(self) -> int:
        return len({s.exercise_id for s in self.working_sets})

    @property
    def total_volume_lb(self) -> Decimal:
        """Sum of weight × reps across all working sets."""
        total = Decimal("0")
        for s in self.working_sets:
            if s.weight_used:
                w = s.weight_used
                if s.weight_unit == WeightUnit.KG:
                    w = w * Decimal("2.20462")
                total += w * s.reps_completed
        return total

    @property
    def duration_display(self) -> str:
        if self.duration_minutes:
            h, m = divmod(self.duration_minutes, 60)
            return f"{h}h {m}m" if h else f"{m}m"
        if self.started_at and self.ended_at:
            mins = int((self.ended_at - self.started_at).total_seconds() / 60)
            h, m = divmod(mins, 60)
            return f"{h}h {m}m" if h else f"{m}m"
        return "—"

    @property
    def fatigue_label(self) -> str:
        return {1: "Easy", 2: "Light", 3: "Moderate", 4: "Hard", 5: "Brutal"}.get(
            self.fatigue_rating or 0, "—"
        )


class WorkoutSet(Base):
    """
    An individual set logged during a session.
    is_warmup=True sets are tracked separately — they don't count toward
    working volume calculations but are shown in the session log.
    rpe (Rate of Perceived Exertion) is 1-10 — optional quality signal.
    weight_unit inherits from the session but is stored per set for
    historical accuracy if the user toggles units mid-session.
    """
    __tablename__ = "workout_sets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("workout_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    exercise_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("exercises.id"), nullable=False
    )
    set_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    reps_completed: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    weight_used: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2), nullable=True)
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    rpe: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    is_warmup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    session: Mapped["WorkoutSession"] = relationship("WorkoutSession", back_populates="sets")
    exercise: Mapped["Exercise"] = relationship(
        "Exercise", back_populates="sets", lazy="selectin"
    )

    @property
    def weight_display(self) -> str:
        if self.weight_used is None:
            return "BW"
        return f"{self.weight_used:g} {self.weight_unit.value}"

    @property
    def volume(self) -> Decimal:
        """Weight × reps for this set."""
        if self.weight_used is None:
            return Decimal("0")
        return self.weight_used * self.reps_completed


class BodyMetric(Base):
    """
    Daily body weight and body fat percentage tracking.
    Unique constraint on metric_date — one entry per day, upserted by the router.
    Default unit is lb (user preference).
    """
    __tablename__ = "body_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, unique=True)
    weight: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2), nullable=True)
    weight_unit: Mapped[WeightUnit] = mapped_column(
        Enum(WeightUnit, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeightUnit.LB,
    )
    body_fat_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    @property
    def weight_display(self) -> str:
        if self.weight is None:
            return "—"
        return f"{self.weight:g} {self.weight_unit.value}"


# models_media.py
"""
Media Tracker ORM models — append to models.py.

IMPORTS TO ADD at top of models.py if not already present:
  from sqlalchemy import SmallInteger, JSON  (likely already imported)

PRIVACY NOTE: No privacy constraints here — media tracking is not sensitive.
All fields can be used freely in AI recommendation prompts.

RATING SYSTEM:
  user_rating stores 1-10. UI displays as half-stars:
  1=½★  2=★  3=1½★  4=★★  5=2½★  6=★★★  7=3½★  8=★★★★  9=4½★  10=★★★★★
  Even numbers = whole stars. Odd numbers = half-star.

PREDEFINED MOOD TAGS:
  light, cerebral, emotional, funny, tense, dark, inspiring,
  relaxing, thrilling, romantic, nostalgic, thought-provoking

  Custom tags can be added alongside predefined ones — all stored as
  a JSON string array in mood_tags.
"""

import datetime
import enum
import math
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Date, DateTime, Enum, ForeignKey,
    Integer, JSON, Numeric, SmallInteger, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel

# Base imported from wherever it's defined in models.py


# ── PREDEFINED MOOD TAGS ───────────────────────────────────────────────────────
# Used by both the UI tag selector and the ML recommendation query builder.
# Custom tags are allowed alongside these — the full list is the union of
# PREDEFINED_MOOD_TAGS and any user-defined strings in mood_tags JSON.

PREDEFINED_MOOD_TAGS = [
    "light", "cerebral", "emotional", "funny", "tense",
    "dark", "inspiring", "relaxing", "thrilling", "romantic",
    "nostalgic", "thought-provoking",
]


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class MediaExternalSource(enum.Enum):
    TMDB_MOVIE  = "tmdb_movie"
    TMDB_TV     = "tmdb_tv"
    OPENLIBRARY = "openlibrary"
    MANUAL      = "manual"


class MediaType(enum.Enum):
    MOVIE   = "movie"
    TV_SHOW = "tv_show"
    BOOK    = "book"

    @property
    def label(self) -> str:
        return {"movie": "Movie", "tv_show": "TV Show", "book": "Book"}[self.value]

    @property
    def icon(self) -> str:
        return {"movie": "🎬", "tv_show": "📺", "book": "📚"}[self.value]


class UserMediaStatus(enum.Enum):
    WANT_TO    = "want_to"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    ABANDONED  = "abandoned"

    @property
    def label(self) -> str:
        return {
            "want_to":     "Want To",
            "in_progress": "In Progress",
            "completed":   "Completed",
            "abandoned":   "Abandoned",
        }[self.value]

    @property
    def color(self) -> str:
        return {
            "want_to":     "var(--accent)",
            "in_progress": "var(--yellow)",
            "completed":   "var(--green)",
            "abandoned":   "var(--text-muted)",
        }[self.value]


class RecommendationMediaType(enum.Enum):
    MOVIE   = "movie"
    TV_SHOW = "tv_show"
    BOOK    = "book"
    ANY     = "any"


# ── STREAMING SERVICES ────────────────────────────────────────────────────────

class StreamingService(Base):
    """
    Reference table of streaming services. Seeded on migration.
    is_subscribed=True means the user actively subscribes to this service
    and it will be preferred in recommendations.
    tmdb_provider_id is used to match TMDB watch provider API responses.
    """
    __tablename__ = "streaming_services"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    tmdb_provider_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tmdb_provider_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_subscribed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )


# ── MEDIA ITEMS ───────────────────────────────────────────────────────────────

class MediaItem(Base):
    """
    The catalog table — one row per unique movie/TV show/book.
    The UNIQUE constraint on (external_id, external_source) prevents duplicates
    when the same item is searched multiple times.

    embedding is a 384-dimensional float vector from all-MiniLM-L6-v2,
    stored as a JSON array. The ML service generates these; the recommendation
    router computes cosine similarity in Python.

    streaming_provider_ids is a JSON array of TMDB provider IDs (integers)
    that have this item available for streaming in the US.
    """
    __tablename__ = "media_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── External source ────────────────────────────────────────────────────────
    external_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    external_source: Mapped[MediaExternalSource] = mapped_column(
        Enum(MediaExternalSource, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=MediaExternalSource.MANUAL,
    )

    # ── Core metadata ──────────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[MediaType] = mapped_column(
        Enum(MediaType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    genres: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    release_year: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    poster_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    external_rating: Mapped[Optional[Decimal]] = mapped_column(Numeric(3, 1), nullable=True)

    # ── Movie-specific ─────────────────────────────────────────────────────────
    runtime_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ── TV-specific ────────────────────────────────────────────────────────────
    total_seasons: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    total_episodes: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # ── Book-specific ──────────────────────────────────────────────────────────
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # ── Streaming availability ─────────────────────────────────────────────────
    # JSON array of TMDB provider IDs, e.g. [8, 119] = Netflix + Prime
    streaming_provider_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    streaming_fetched_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )
    seasons_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ── ML embedding ──────────────────────────────────────────────────────────
    # 384-dim float vector. None means the embedding job hasn't run yet.
    embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    embedding_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    user_media: Mapped[Optional["UserMedia"]] = relationship(
        "UserMedia", back_populates="media_item", uselist=False
    )

    # ── Computed properties ────────────────────────────────────────────────────

    @property
    def is_tracked(self) -> bool:
        """True if the user has added this item to their list."""
        return self.user_media is not None

    @property
    def genre_list(self) -> list[str]:
        return self.genres or []

    @property
    def runtime_display(self) -> str:
        """Human-readable runtime, e.g. '2h 15m' or '45 min'."""
        if not self.runtime_minutes:
            return "—"
        if self.runtime_minutes < 60:
            return f"{self.runtime_minutes} min"
        h, m = divmod(self.runtime_minutes, 60)
        return f"{h}h {m}m" if m else f"{h}h"

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0

    @property
    def streaming_available_on(self) -> list[int]:
        """Returns list of TMDB provider IDs where this item streams."""
        return self.streaming_provider_ids or []


class UserMedia(Base):
    """
    The user's personal tracking record for a media item.
    One row per media_item — UNIQUE constraint prevents double-tracking.

    Rating system: 1-10 stored, displayed as half-stars.
    Odd = half-star, even = whole star:
      1=½★  2=★  3=1½★  4=★★  5=2½★  6=★★★  7=3½★  8=★★★★  9=4½★  10=★★★★★

    mood_tags is a JSON string array mixing predefined and custom tags:
      ["light", "funny", "my-custom-tag"]
    """
    __tablename__ = "user_media"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    media_item_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("media_items.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    status: Mapped[UserMediaStatus] = mapped_column(
        Enum(UserMediaStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=UserMediaStatus.WANT_TO,
    )

    # 1-10; None = not yet rated
    user_rating: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    started_at: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    completed_at: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rewatch_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    # JSON string array: predefined + custom mood tags
    mood_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    media_item: Mapped["MediaItem"] = relationship(
        "MediaItem", back_populates="user_media", lazy="selectin"
    )
    season_progress: Mapped[list["TVSeasonProgress"]] = relationship(
        "TVSeasonProgress", back_populates="user_media",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="TVSeasonProgress.season_number",
    )

    # ── Rating display helpers ─────────────────────────────────────────────────

    @property
    def rating_stars(self) -> str:
        """
        Returns a star string for template display.
        Odd rating = half star. Even rating = whole star.
        1→½★  2→★  3→1½★  4→★★  5→2½★  6→★★★  7→3½★  8→★★★★  9→4½★  10→★★★★★
        """
        if not self.user_rating:
            return "☆☆☆☆☆"
        r = self.user_rating
        full = r // 2
        half = r % 2
        empty = 5 - full - half
        return "★" * full + ("½" if half else "") + "☆" * empty

    @property
    def rating_numeric(self) -> str:
        """e.g. '8/10' or '—'"""
        if not self.user_rating:
            return "—"
        return f"{self.user_rating}/10"

    @property
    def tag_list(self) -> list[str]:
        return self.mood_tags or []

    @property
    def is_rated(self) -> bool:
        return self.user_rating is not None

    @property
    def total_episodes_watched(self) -> int:
        """Total episodes watched across all seasons (TV shows only)."""
        return sum(sp.episodes_watched for sp in self.season_progress)


class TVSeasonProgress(Base):
    """
    Per-season episode progress for TV shows.
    Sparse — only seasons the user has started appear here.
    """
    __tablename__ = "tv_season_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_media_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user_media.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    season_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    episodes_watched: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    total_episodes: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    user_media: Mapped["UserMedia"] = relationship(
        "UserMedia", back_populates="season_progress"
    )

    @property
    def completion_pct(self) -> int:
        """Percentage of episodes watched in this season (0-100)."""
        if not self.total_episodes or self.total_episodes == 0:
            return 0
        return min(100, round((self.episodes_watched / self.total_episodes) * 100))

    @property
    def is_complete(self) -> bool:
        return (
            self.total_episodes is not None
            and self.episodes_watched >= self.total_episodes
        )


class MediaRecommendation(Base):
    """
    Cached recommendation sessions.
    recommendations JSON schema:
    [
      {
        "media_item_id": int,
        "title": str,
        "score": float,           # cosine similarity from ML layer
        "reasoning": str | null   # Gemini explanation (null if Gemini skipped)
      }
    ]
    """
    __tablename__ = "media_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    input_mood: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input_context: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    media_type_filter: Mapped[RecommendationMediaType] = mapped_column(
        Enum(RecommendationMediaType, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=RecommendationMediaType.ANY,
    )
    include_unsubscribed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    used_gemini: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    recommendations: Mapped[list] = mapped_column(JSON, nullable=False)
    ml_model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    @property
    def result_count(self) -> int:
        return len(self.recommendations or [])

    @property
    def age_minutes(self) -> float:
        delta = datetime.datetime.utcnow() - self.generated_at
        return delta.total_seconds() / 60


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

class MediaItemResponse(BaseModel):
    id: int
    title: str
    media_type: MediaType
    external_source: MediaExternalSource
    genres: Optional[list]
    release_year: Optional[int]
    description: Optional[str]
    poster_url: Optional[str]
    external_rating: Optional[Decimal]
    runtime_minutes: Optional[int]
    total_seasons: Optional[int]
    author: Optional[str]
    has_embedding: bool
    streaming_available_on: list

    class Config:
        from_attributes = True


class UserMediaCreate(BaseModel):
    media_item_id: int
    status: UserMediaStatus = UserMediaStatus.WANT_TO


class UserMediaUpdate(BaseModel):
    status: Optional[UserMediaStatus] = None
    user_rating: Optional[int] = None
    mood_tags: Optional[list] = None
    notes: Optional[str] = None
    started_at: Optional[datetime.date] = None
    completed_at: Optional[datetime.date] = None


class UserMediaResponse(BaseModel):
    id: int
    media_item_id: int
    status: UserMediaStatus
    user_rating: Optional[int]
    rating_stars: str
    rating_numeric: str
    tag_list: list
    notes: Optional[str]
    started_at: Optional[datetime.date]
    completed_at: Optional[datetime.date]
    rewatch_count: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class StreamingServiceResponse(BaseModel):
    id: int
    name: str
    tmdb_provider_id: Optional[int]
    logo_url: Optional[str]
    is_subscribed: bool
    sort_order: int

    class Config:
        from_attributes = True

# ── WEEKLY PLANNING MODULE ────────────────────────────────────────────────────

import enum

class FitnessGoal(enum.Enum):
    WEIGHT_LOSS    = "weight_loss"
    MUSCLE_GAIN    = "muscle_gain"
    MAINTENANCE    = "maintenance"
    ENDURANCE      = "endurance"
    GENERAL_HEALTH = "general_health"

    @property
    def label(self) -> str:
        return {
            "weight_loss":    "Weight Loss",
            "muscle_gain":    "Muscle Gain",
            "maintenance":    "Maintenance",
            "endurance":      "Endurance",
            "general_health": "General Health",
        }[self.value]


class WeeklyPlanStatus(enum.Enum):
    DRAFT     = "draft"
    CONFIRMED = "confirmed"
    ACTIVE    = "active"
    COMPLETED = "completed"


class PlanDayStatus(enum.Enum):
    PLANNED   = "planned"
    ACTIVE    = "active"
    COMPLETED = "completed"
    SKIPPED   = "skipped"


class PlanMealType(enum.Enum):
    BREAKFAST = "breakfast"
    LUNCH     = "lunch"
    DINNER    = "dinner"
    SNACK     = "snack"


class PlanMealStatus(enum.Enum):
    PLANNED  = "planned"
    EATEN    = "eaten"
    SWAPPED  = "swapped"
    OFF_PLAN = "off_plan"
    SKIPPED  = "skipped"

    @property
    def label(self) -> str:
        return {
            "planned":  "Planned",
            "eaten":    "✓ Eaten",
            "swapped":  "↔ Swapped",
            "off_plan": "↗ Off-plan",
            "skipped":  "— Skipped",
        }[self.value]


class UserIntent(Base):
    """
    Single-row table. Always query with LIMIT 1.
    Stores the user's current fitness goal and AI-generator preferences.
    All weekly plan generators read from this before producing suggestions.
    """
    __tablename__ = "user_intent"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    fitness_goal: Mapped[FitnessGoal] = mapped_column(
        Enum(FitnessGoal, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=FitnessGoal.WEIGHT_LOSS,
    )
    weekly_workout_days: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=4
    )
    target_calories: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    macro_preference: Mapped[str] = mapped_column(
        String(30), nullable=False, default="high_protein"
    )
    cooking_time_preference: Mapped[str] = mapped_column(
        String(20), nullable=False, default="moderate"
    )
    dietary_restrictions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    food_preferences: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    food_dislikes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    health_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    @property
    def cooking_time_label(self) -> str:
        return {
            "minimal":  "Minimal (<20 min)",
            "moderate": "Moderate (20–45 min)",
            "generous": "Generous (45+ min)",
        }.get(self.cooking_time_preference, self.cooking_time_preference)

    def to_ai_context(self) -> str:
        """
        Renders the intent as a plain-English block for injection into
        AI generator prompts. Called by both the workout planner and
        the meal planner agents.
        """
        lines = [
            f"Primary goal: {self.fitness_goal.label}",
            f"Target workout days per week: {self.weekly_workout_days}",
            f"Macro preference: {self.macro_preference}",
            f"Cooking time available: {self.cooking_time_label}",
        ]
        if self.target_calories:
            lines.append(f"Daily calorie target: ~{self.target_calories} kcal")
        if self.dietary_restrictions:
            lines.append(f"Dietary restrictions: {self.dietary_restrictions}")
        if self.food_preferences:
            lines.append(f"Food preferences: {self.food_preferences}")
        if self.food_dislikes:
            lines.append(f"Food dislikes: {self.food_dislikes}")
        if self.health_notes:
            lines.append(f"Health/injury notes: {self.health_notes}")
        return "\n".join(lines)


class WeeklyPlan(Base):
    """
    One row per planned week. week_start_date is always a Monday.
    Contains 7 WeeklyPlanDay children covering Mon–Sun.
    """
    __tablename__ = "weekly_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week_start_date: Mapped[datetime.date] = mapped_column(
        Date, nullable=False, unique=True
    )
    week_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    intent_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[WeeklyPlanStatus] = mapped_column(
        Enum(WeeklyPlanStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=WeeklyPlanStatus.DRAFT,
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    meals_planned: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    meals_followed: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    workouts_planned: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    workouts_completed: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow, nullable=False,
    )

    days: Mapped[list["WeeklyPlanDay"]] = relationship(
        "WeeklyPlanDay", back_populates="weekly_plan",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WeeklyPlanDay.day_number",
    )
    shopping_list: Mapped[Optional["ShoppingList"]] = relationship(
        "ShoppingList", back_populates="weekly_plan", uselist=False
    )

    @property
    def week_label(self) -> str:
        return (
            f"{self.week_start_date.strftime('%b %-d')} – "
            f"{self.week_end_date.strftime('%-d, %Y')}"
        )

    @property
    def meal_adherence_pct(self) -> int:
        if not self.meals_planned:
            return 0
        return min(100, round((self.meals_followed / self.meals_planned) * 100))

    @property
    def workout_adherence_pct(self) -> int:
        if not self.workouts_planned:
            return 0
        return min(100, round((self.workouts_completed / self.workouts_planned) * 100))


class WeeklyPlanDay(Base):
    """One calendar day within a weekly plan."""
    __tablename__ = "weekly_plan_days"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    weekly_plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weekly_plans.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    plan_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    day_number: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 1=Mon

    workout_session_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("workout_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_rest_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    day_status: Mapped[PlanDayStatus] = mapped_column(
        Enum(PlanDayStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=PlanDayStatus.PLANNED,
    )
    override_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    journal_entry_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("journal_entries.id", ondelete="SET NULL"), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    weekly_plan: Mapped["WeeklyPlan"] = relationship(
        "WeeklyPlan", back_populates="days"
    )
    meals: Mapped[list["WeeklyPlanMeal"]] = relationship(
        "WeeklyPlanMeal", back_populates="plan_day",
        cascade="all, delete-orphan", lazy="selectin",
        order_by="WeeklyPlanMeal.sort_order",
    )
    workout_session: Mapped[Optional["WorkoutSession"]] = relationship(
        "WorkoutSession", foreign_keys=[workout_session_id], lazy="selectin"
    )

    @property
    def day_name(self) -> str:
        return self.plan_date.strftime("%A")  # "Monday", "Tuesday", etc.

    @property
    def is_today(self) -> bool:
        return self.plan_date == datetime.date.today()

    @property
    def dinner(self) -> Optional["WeeklyPlanMeal"]:
        for m in self.meals:
            if m.meal_type == PlanMealType.DINNER:
                return m
        return None

    @property
    def workout_is_done(self) -> bool:
        if self.is_rest_day:
            return True
        if not self.workout_session:
            return False
        return self.workout_session.ended_at is not None


class WeeklyPlanMeal(Base):
    """One planned meal within a single day of a weekly plan."""
    __tablename__ = "weekly_plan_meals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_day_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weekly_plan_days.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    recipe_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True
    )
    meal_type: Mapped[PlanMealType] = mapped_column(
        Enum(PlanMealType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    status: Mapped[PlanMealStatus] = mapped_column(
        Enum(PlanMealStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False, default=PlanMealStatus.PLANNED,
    )
    swap_recipe_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("recipes.id", ondelete="SET NULL"), nullable=True
    )
    off_plan_note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    plan_day: Mapped["WeeklyPlanDay"] = relationship(
        "WeeklyPlanDay", back_populates="meals"
    )
    recipe: Mapped[Optional["Recipe"]] = relationship(
        "Recipe", foreign_keys=[recipe_id], lazy="selectin"
    )
    swap_recipe: Mapped[Optional["Recipe"]] = relationship(
        "Recipe", foreign_keys=[swap_recipe_id], lazy="selectin"
    )

    @property
    def active_recipe(self) -> Optional["Recipe"]:
        """Returns the swap recipe if swapped, otherwise the original."""
        if self.status == PlanMealStatus.SWAPPED and self.swap_recipe:
            return self.swap_recipe
        return self.recipe


class ShoppingList(Base):
    """Aggregated shopping list derived from a confirmed weekly plan."""
    __tablename__ = "shopping_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    weekly_plan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("weekly_plans.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    items: Mapped[list] = mapped_column(JSON, nullable=False)
    generated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    weekly_plan: Mapped["WeeklyPlan"] = relationship(
        "WeeklyPlan", back_populates="shopping_list"
    )
