"""Scoped memo of a completed paid web search that found no source.

Owner decision 2026-09-29 (`search-reuse-memo-v1`). A row says that, in one
authorization scope, a reference's required web search completed on a given
date without an acceptable source. It holds a reference-key hash and an
operational outcome summary only: no URLs and no search-result content.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


def _utcnow():
    return datetime.now(timezone.utc)


class SearchMemoRecord(Base):
    __tablename__ = "search_memos"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", "reference_key_sha256",
                                       name="uq_search_memo_scope_reference"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    reference_key_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    memo_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    search_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome_summary: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
