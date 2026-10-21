"""A scope's storage settings (owner decision 2026-10-09).

One row per authorization scope (a Personal user, or an institution's course
or assessment scope) holding only the settings that differ from the
deployment defaults; `app.services.storage_policy` merges and validates them.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


def _utcnow():
    return datetime.now(timezone.utc)


class StoragePolicyRecord(Base):
    __tablename__ = "storage_policies"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", name="uq_storage_policy_scope"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    settings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
