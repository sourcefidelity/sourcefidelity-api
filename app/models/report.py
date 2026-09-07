"""Report model – stores citation checking results."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint("job_id", "report_version", name="uq_report_job_version"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True
    )
    total_references = Column(Text, nullable=True)  # JSON string
    verified_references = Column(Text, nullable=True)  # JSON string
    summary = Column(Text, nullable=True)  # JSON or Markdown summary
    report_markdown = Column(Text, nullable=True)
    report_json = Column(JSON, nullable=True)
    report_version = Column(Integer, nullable=False, default=1)
    previous_report_id = Column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="RESTRICT"), nullable=True
    )
    amendment_reason = Column(String(100), nullable=True)
    # The legacy reports column is timestamp-without-time-zone; preserve that
    # schema while avoiding deprecated utcnow().
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"<Report(id={self.id}, job_id={self.job_id}, "
            f"refs={self.total_references})>"
        )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VerificationReportRecord(Base):
    """Immutable, exact-scope evidence snapshot for one verification."""

    __tablename__ = "verification_reports"
    __table_args__ = (
        UniqueConstraint(
            "scope_type",
            "scope_id",
            "verification_id",
            "report_version",
            name="uq_verification_report_scope_version",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    verification_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    paper_version_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    report_version: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_report_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("verification_reports.id", ondelete="RESTRICT"),
        nullable=True,
    )
    verification_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("verification_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    artifact_version: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String(40), nullable=False)
    evidence_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    report_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class ReportPaperArtifactRecord(Base):
    """Sanitized marking copy shared by immutable reports for one paper job."""

    __tablename__ = "report_paper_artifacts"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_report_paper_artifact_job"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="RESTRICT"), nullable=False
    )
    report_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="RESTRICT"), nullable=True,
        index=True,
    )
    paper_version_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    presentation_status: Mapped[str] = mapped_column(String(40), nullable=False)
    presentation_storage_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    presentation_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    presentation_media_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    presentation_byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    presentation_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sanitization_evidence: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class PaperAnnotationRecord(Base):
    """Append-only authored annotation revision over one immutable paper surface."""

    __tablename__ = "paper_annotation_revisions"
    __table_args__ = (
        UniqueConstraint(
            "scope_type",
            "scope_id",
            "report_id",
            "annotation_id",
            "revision",
            name="uq_paper_annotation_scope_revision",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    annotation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="RESTRICT"), nullable=False,
        index=True,
    )
    paper_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("report_paper_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    paper_version_id: Mapped[str] = mapped_column(String(255), nullable=False)
    paper_content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("paper_annotation_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    annotation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    anchor_kind: Mapped[str] = mapped_column(String(30), nullable=False)
    anchor_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    anchor_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    author_provider: Mapped[str] = mapped_column(String(60), nullable=False)
    author_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    visibility: Mapped[str] = mapped_column(String(20), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
