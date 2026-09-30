"""Experimental Judgment layout records (ARCHITECTURE §7, owner decisions 2026-09-24/25).

Separate from every evidence record: nothing here is written into Report,
VerificationReportRecord or the Evidence Package, and the Sources report never
reads these tables. Every row carries its authorization scope.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


def _utcnow():
    return datetime.now(timezone.utc)


class JudgmentAcknowledgement(Base):
    """Historical: a viewer accepted or saw one notice version.

    The notice was removed (owner decision 2026-09-28). The table is kept for
    its existing rows and is no longer written or read.
    """

    __tablename__ = "judgment_acknowledgements"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", "principal_subject", "notice_version",
                                       name="uq_judgment_ack_principal_notice"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    principal_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    principal_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    notice_version: Mapped[str] = mapped_column(String(80), nullable=False)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class JudgmentRun(Base):
    """One on-demand Judgment run over one report's eligible citations."""

    __tablename__ = "judgment_runs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    requested_by: Mapped[str] = mapped_column(String(255), nullable=False)
    notice_version: Mapped[str] = mapped_column(String(80), nullable=False)
    # queued | running | completed | unavailable | failed
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="queued")
    reason_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    panel_version: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    policy_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    candidate_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    candidate_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    spend_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unpriced_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class JudgmentArmResult(Base):
    """A validated arm response, reusable only for the identical arm and prompt.

    Failures are never stored here, so a failure is never reused as a result.
    Reuse never crosses an authorization scope.
    """

    __tablename__ = "judgment_arm_results"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", "cache_key",
                                       name="uq_judgment_arm_result_scope_key"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    arm_id: Mapped[str] = mapped_column(String(30), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    returned_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    returned_provider: Mapped[str | None] = mapped_column(String(80), nullable=True)
    contract_version: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    response: Mapped[dict] = mapped_column(JSON, nullable=False)
    usage: Mapped[dict] = mapped_column(JSON, nullable=False)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_basis: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class JudgmentCandidateResult(Base):
    """One clause candidate's panel result for one cited source, in arrival order."""

    __tablename__ = "judgment_candidate_results"
    __table_args__ = (UniqueConstraint("run_id", "verification_report_id", "candidate_id",
                                       name="uq_judgment_candidate_per_run"),
                      UniqueConstraint("run_id", "seq", name="uq_judgment_candidate_seq"))
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("judgment_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    citation_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verification_report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("verification_reports.id", ondelete="CASCADE"), nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    display_state: Mapped[str] = mapped_column(String(30), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(60), nullable=False)
    panel: Mapped[dict] = mapped_column(JSON, nullable=False)
    wider_search: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    arm_result_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    coaching: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    spend_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class JudgmentSourceReserve(Base):
    """New source sentences for the wider search, kept beside one verification report.

    Deleted with its verification report. Built only while Judgment is enabled
    and JUDGMENT_RESERVE_RETENTION is not "off".
    """

    __tablename__ = "judgment_source_reserves"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    verification_report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("verification_reports.id", ondelete="CASCADE"),
        nullable=False, unique=True)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    reserve_version: Mapped[str] = mapped_column(String(64), nullable=False)
    retention_policy: Mapped[str] = mapped_column(String(40), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)


class JudgmentCoachingNote(Base):
    """A checked coaching note (or the fixed fallback), reused for the identical request."""

    __tablename__ = "judgment_coaching_notes"
    __table_args__ = (UniqueConstraint("scope_type", "scope_id", "cache_key",
                                       name="uq_judgment_coaching_scope_key"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
