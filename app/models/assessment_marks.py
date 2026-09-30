"""Assessment-level "marks released" record (owner decision 2026-09-29).

One row per authorization scope and assessment. ``marks_released_at`` set
means marks for that assessment are released: further searching can no longer
affect it, so automatic retries and "Search again" stop for its papers, and a
Judgment reserve kept "until_grades_released" is purged. An instructor or
administrator sets it now (``release_source`` "manual"); the planned LMS
grade-release signal sets the same row ("lms"). Clearing keeps the row and
records who cleared it.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


def _utcnow():
    return datetime.now(timezone.utc)


class AssessmentMarksRelease(Base):
    __tablename__ = "assessment_marks_releases"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", "assessment_id",
                                       name="uq_assessment_marks_release_scope_assessment"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    assessment_id: Mapped[str] = mapped_column(String(255), nullable=False)
    marks_released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    released_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    release_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cleared_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    clear_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
