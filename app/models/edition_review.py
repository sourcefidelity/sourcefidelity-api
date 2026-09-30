"""Append-only Personal edition reviews; not source admission records."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class EditionReviewSnapshot(Base):
    __tablename__ = "edition_review_snapshots"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    representation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_representations.id", ondelete="CASCADE"), nullable=False)
    scope_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc))


class EditionReviewDecision(Base):
    __tablename__ = "edition_review_decisions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True),
        ForeignKey("edition_review_snapshots.id", ondelete="CASCADE"), nullable=False, unique=True)
    reviewer_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc))
