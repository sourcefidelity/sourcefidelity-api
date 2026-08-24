"""Minimal operational/audit record for one transient verification run."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class VerificationRunRecord(Base):
    """Lease-backed transient execution; never a reusable source admission."""

    __tablename__ = "verification_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    audit_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="verification-run-audit-v1"
    )
    paper_version_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    canonical_work_id: Mapped[str] = mapped_column(String(255), nullable=False)
    representation_kind: Mapped[str] = mapped_column(String(30), nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    acquisition_route: Mapped[str] = mapped_column(String(100), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    transient_object_count: Mapped[int] = mapped_column(Integer, nullable=False)
    transient_byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    # Active locators exist only until cleanup succeeds. The surviving audit
    # retains counts/hashes, not reusable object locations.
    transient_objects: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    identity_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    completeness_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    cleanliness_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    text_quality: Mapped[str] = mapped_column(String(30), nullable=False)
    edition_or_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    terminal_outcome: Mapped[str | None] = mapped_column(String(30), nullable=True)
    cleanup_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_cleanup_error_code: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    cleaned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
