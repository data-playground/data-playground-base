# domains/blog/models.py
"""
Blog Ideation domain models — extracted from the top-level models.py as part
of the domain-folder migration (see routers/blog.py for usage).

BlogIdea.code_file / code_project use relationship("CodeFile", ...) /
relationship("CodeProject", ...) with STRING class names, so they resolve
against the shared SQLAlchemy mapper registry at query time rather than at
import time. No import of domains.code_intel.models is needed here — the
only requirement is that both domains/blog/models.py and
domains/code_intel/models.py get imported by something before the first
query runs, which the models.py re-export shim guarantees.
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
        if self.value in ("ready_to_publish", "published"):
            return "done"
        return "archived"

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

    difficulty: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    the_build:         Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    the_narrative:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    the_selling_point: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    raw_idea_input: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    code_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    draft_v1:          Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    draft_v2:          Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    final_article:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    seo_title:       Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    seo_description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    seo_tags:        Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    status: Mapped[BlogIdeaStatus] = mapped_column(
        Enum(BlogIdeaStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=BlogIdeaStatus.IDEA_GENERATED,
    )
    airflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

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
        return {
            "starter":   "⬡ Starter",
            "weekend":   "◈ Weekend",
            "ambitious": "◉ Ambitious",
        }.get(self.difficulty or "", "—")

    @property
    def difficulty_color_class(self) -> str:
        return {
            "starter":   "var(--green)",
            "weekend":   "var(--yellow)",
            "ambitious": "var(--red)",
        }.get(self.difficulty or "", "var(--text-muted)")


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

