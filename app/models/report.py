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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True
    )
    total_references = Column(Text, nullable=True)  # JSON string
    verified_references = Column(Text, nullable=True)  # JSON string
    summary = Column(Text, nullable=True)  # JSON or Markdown summary
    report_markdown = Column(Text, nullable=True)
    report_json = Column(JSON, nullable=True)
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
