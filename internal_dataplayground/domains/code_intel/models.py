# domains/code_intel/models.py
"""
Code Intelligence domain models — extracted from the top-level models.py as
part of the domain-folder migration (see routers/ci_projects.py,
routers/ci_files.py, routers/ci_readme.py for usage).

CodeFile.blog_ideas / CodeProject.blog_ideas use relationship("BlogIdea", ...)
with a STRING class name, so it resolves against the shared SQLAlchemy mapper
registry at query time rather than at import time. No import of
domains.blog.models is needed here for the same reason described in
domains/blog/models.py.
"""

import datetime
import enum
from typing import Optional

from core.base_model import Base
from pydantic import BaseModel
from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship


# ── ENUMS ─────────────────────────────────────────────────────────────────────

class ReadmeStatus(enum.Enum):
    NONE      = "none"
    DRAFT     = "draft"
    REVIEWED  = "reviewed"
    APPROVED  = "approved"
    PUSHED    = "pushed"
    STALE     = "stale"

class FolderReadmeStatus(enum.Enum):
    NONE     = "none"
    DRAFT    = "draft"
    REVIEWED = "reviewed"
    PUSHED   = "pushed"
    STALE    = "stale"

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
    __tablename__ = "code_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_name: Mapped[str] = mapped_column(String(255), nullable=False)

    github_repo: Mapped[str] = mapped_column(String(255), nullable=False)
    github_base_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

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
        if not self.readme_generated_at:
            return False
        return any(
            f.code_pulled_at and f.code_pulled_at > self.readme_generated_at
            for f in self.files
        )

    @property
    def folder_readme_coverage(self) -> dict:
        counts = {"total": 0, "none": 0, "draft": 0, "reviewed": 0, "pushed": 0, "stale": 0}
        for fr in self.folder_readmes:
            counts["total"] += 1
            counts[fr.status.value] += 1
        return counts


class CodeFile(Base):
    __tablename__ = "code_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("code_projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    github_path: Mapped[str] = mapped_column(String(500), nullable=False)
    github_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    raw_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    code_pulled_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    narration: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    narration_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    commented_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    commented_generated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    commented_status: Mapped[CommentedStatus] = mapped_column(
        Enum(CommentedStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=CommentedStatus.NONE,
    )

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

    project: Mapped["CodeProject"] = relationship("CodeProject", back_populates="files")
    blog_ideas: Mapped[list["BlogIdea"]] = relationship(
        "BlogIdea", back_populates="code_file",
        foreign_keys="BlogIdea.code_file_id",
    )

    @property
    def narration_is_stale(self) -> bool:
        if not self.narration_generated_at or not self.code_pulled_at:
            return False
        return self.code_pulled_at > self.narration_generated_at


class FolderReadme(Base):
    __tablename__ = "folder_readmes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("code_projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    folder_path: Mapped[str] = mapped_column(String(500), nullable=False)
    folder_display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    github_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    readme_md: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[FolderReadmeStatus] = mapped_column(
        Enum(FolderReadmeStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=FolderReadmeStatus.NONE,
    )

    github_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

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

    project: Mapped["CodeProject"] = relationship(
        "CodeProject", back_populates="folder_readmes"
    )

    @property
    def is_stale(self) -> bool:
        return self.status == FolderReadmeStatus.STALE

    @property
    def needs_readme(self) -> bool:
        return self.status == FolderReadmeStatus.NONE

    @property
    def is_published(self) -> bool:
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
    project_id: int
    folder_path: str
    folder_display_name: str
    github_path: Optional[str] = None

