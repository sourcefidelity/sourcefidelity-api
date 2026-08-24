"""Job tracking model – PostgreSQL."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.models import Base


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage:
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    RETRIEVING = "retrieving"
    RETRIEVED = "retrieved"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("paper_version_id", name="uq_jobs_paper_version_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    title = Column(String(500), nullable=True)
    status = Column(
        String(20), nullable=False, default=JobStatus.PENDING, index=True
    )
    stage = Column(String(30), nullable=False, default=JobStage.UPLOADED, index=True)
    paper_version_id = Column(String(255), nullable=False, index=True)
    scope_type = Column(String(30), nullable=False, default="personal_owner")
    scope_id = Column(String(255), nullable=False)
    input_storage_key = Column(String(500), nullable=True)
    input_sha256 = Column(String(64), nullable=False)
    input_media_type = Column(String(100), nullable=False)
    input_byte_size = Column(Integer, nullable=False)
    input_expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    input_deleted_at = Column(DateTime(timezone=True), nullable=True)
    store_only = Column(Boolean, nullable=False, default=False)
    task_id = Column(String(255), nullable=True)
    upload_evidence = Column(JSON, nullable=False, default=dict)
    extraction_payload = Column(JSON, nullable=True)
    source_results = Column(JSON, nullable=True)
    verification_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Job(id={self.id}, status={self.status}, filename={self.filename})>"
