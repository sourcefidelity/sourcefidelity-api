"""Durable canonical-work, immutable-object, and representation records."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CanonicalWorkRecord(Base):
    """Bibliographic work identity, independent of any acquired file."""

    __tablename__ = "canonical_works"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    isbn: Mapped[str | None] = mapped_column(String(32), nullable=True, unique=True)
    normalized_title: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    display_title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[str | None] = mapped_column(String(10), nullable=True)
    work_type: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    representations: Mapped[list["SourceRepresentationRecord"]] = relationship(
        back_populates="canonical_work"
    )


class ContentObjectRecord(Base):
    """One immutable content-addressed object in the configured object store."""

    __tablename__ = "content_objects"
    __table_args__ = (
        UniqueConstraint(
            "license_class", "content_sha256", name="uq_content_object_license_hash"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    license_class: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    deletion_pending: Mapped[bool] = mapped_column(
        nullable=False, default=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    representations: Mapped[list["SourceRepresentationRecord"]] = relationship(
        back_populates="content_object"
    )


class SourceRepresentationRecord(Base):
    """A scoped, validated use of an immutable object for one canonical work."""

    __tablename__ = "source_representations"
    __table_args__ = (
        UniqueConstraint(
            "canonical_work_id",
            "content_object_id",
            "provenance",
            "scope_type",
            "scope_id",
            name="uq_representation_work_object_provenance_scope",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    canonical_work_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("canonical_works.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    content_object_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("content_objects.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    representation_kind: Mapped[str] = mapped_column(String(30), nullable=False)
    original_kind: Mapped[str | None] = mapped_column(String(30), nullable=True)
    provenance: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    admission_state: Mapped[str] = mapped_column(
        String(30), nullable=False, default="needs_review", index=True
    )
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    edition_or_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    identity_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    identity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    completeness_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    cleanliness_verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    text_quality: Mapped[str | None] = mapped_column(String(30), nullable=True)
    validation_evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    admitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    admitted_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    canonical_work: Mapped[CanonicalWorkRecord] = relationship(
        back_populates="representations"
    )
    content_object: Mapped[ContentObjectRecord] = relationship(
        back_populates="representations"
    )
