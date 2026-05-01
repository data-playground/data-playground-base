import datetime
import enum
from typing import Optional
from sqlalchemy import BigInteger, String, Boolean, Date, Text, Integer, Enum, ForeignKey, DateTime, Numeric
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pydantic import BaseModel
from decimal import Decimal

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


class BlogIdeaStatus(enum.Enum):
    IDEA_GENERATED              = "idea_generated"
    WAITING_FOR_WRITING_TRIGGER = "waiting_for_writing_trigger"
    WRITING_IN_PROGRESS         = "writing_in_progress"
    WAITING_FOR_REVIEW          = "waiting_for_review"
    REVIEW_COMPLETED            = "review_completed"
    READY_TO_PUBLISH            = "ready_to_publish"
    PUBLISHED                   = "published"

    @property
    def label(self) -> str:
        return {
            "idea_generated":              "Idea Generated",
            "waiting_for_writing_trigger": "Ready to Write",
            "writing_in_progress":         "Writing…",
            "waiting_for_review":          "Awaiting Review",
            "review_completed":            "Review Done",
            "ready_to_publish":            "Ready to Publish",
            "published":                   "Published",
        }[self.value]

    @property
    def kanban_column(self) -> str:
        if self.value in ("idea_generated", "waiting_for_writing_trigger"):
            return "backlog"
        if self.value in ("writing_in_progress", "waiting_for_review", "review_completed"):
            return "in_progress"
        return "done"


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
