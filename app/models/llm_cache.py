"""A remembered model answer for one exact request (owner decision 2026-10-09).

Scoped like the paper that asked: an answer is reused only within the same
authorization scope. Rows hold the parsed answer and the calling module, never
the request text.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


def _utcnow():
    return datetime.now(timezone.utc)


class LLMCallCacheRecord(Base):
    __tablename__ = "llm_call_cache"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", "cache_key", name="uq_llm_call_cache_scope_key"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    caller: Mapped[str] = mapped_column(String(120), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    response: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
