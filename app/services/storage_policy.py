"""What a scope keeps, and for how long: sources, papers and reports.

Owner decision 2026-10-09 (PLAN §2 *Shared lifecycle*): individuals and
institutions differ in what they keep, so storage is configurable per scope
(a Personal user, or an institution's course or assessment scope), separately
for sources, papers and reports. The deployment defaults reproduce the
behaviour before settings existed, so nothing changes until a scope chooses
otherwise:

* sources: validated publicly downloadable copies are kept with no time
  limit; copies obtained through a library licence are not kept unless the
  deployment enables it (then for `CAMPUS_ACCESS_TTL_DAYS`);
* papers: the copy shown beside the report is kept for
  `FORMATIVE_REPORT_PAPER_RETENTION_DAYS` after the report was last built
  or returned again; the submitted original is deleted when the check ends;
* reports: kept until deleted (0 days).

A day count of 0 means "until deleted". A scope's row holds only its
differences from the defaults; anything invalid fails closed to the defaults.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings

POLICY_VERSION = "storage-policy-v1"
Profile = Literal["personal", "institutional"]


class SourceStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Validated publicly downloadable copies, whatever route found them:
    # "all", only those with an open-access or licence signal, or none.
    keep_public_copies: Literal["all", "open_access_only", "none"] = "all"
    public_copy_days: int = Field(default=0, ge=0)
    # Copies obtained through a library or campus licence.
    keep_licensed_copies: bool = False
    licensed_copy_days: int = Field(default=90, ge=0)


class PaperStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The copy of the paper shown beside its report.
    paper_copy_days: int = Field(default=7, ge=1)


class ReportStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The whole paper record (report versions, checks, judgments) is deleted
    # this many days after its report was last built or returned again.
    report_days: int = Field(default=0, ge=0)


class StoragePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["storage-policy-v1"] = POLICY_VERSION
    sources: SourceStorage = Field(default_factory=SourceStorage)
    papers: PaperStorage = Field(default_factory=PaperStorage)
    reports: ReportStorage = Field(default_factory=ReportStorage)

    @model_validator(mode="after")
    def _paper_copy_within_report(self):
        if self.reports.report_days and self.papers.paper_copy_days > self.reports.report_days:
            raise ValueError("the paper copy cannot outlive its report")
        return self


def profile_for(scope_type: str) -> Profile:
    return "personal" if scope_type == "personal_owner" else "institutional"


def deployment_defaults(profile: Profile = "personal") -> StoragePolicy:
    """The deployment's settings as a policy; both profiles start from them.
    Institutional sharing of kept copies across users (open or library
    licence only) is a later step and is not expressed here."""
    return StoragePolicy(
        sources=SourceStorage(
            keep_public_copies=("all" if settings.PUBLIC_RETRIEVAL_RETENTION_POLICY == "store_scoped"
                                else "open_access_only"),
            keep_licensed_copies=bool(settings.CACHE_PAYWALLED_PDFS),
            licensed_copy_days=max(0, int(settings.CAMPUS_ACCESS_TTL_DAYS)),
        ),
        papers=PaperStorage(paper_copy_days=max(1, int(settings.FORMATIVE_REPORT_PAPER_RETENTION_DAYS))),
    )


def _merged(base: dict, changes: dict) -> dict:
    out = dict(base)
    for key, value in (changes or {}).items():
        out[key] = _merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def policy_for(session: Session | None, scope_type: str, scope_id: str) -> StoragePolicy:
    """The scope's settings over the deployment defaults; the defaults when
    the scope has none, the row is invalid or the database is unavailable."""
    defaults = deployment_defaults(profile_for(scope_type))
    if session is None:
        return defaults
    from app.models.storage_policy import StoragePolicyRecord
    try:
        record = session.scalar(select(StoragePolicyRecord).where(
            StoragePolicyRecord.scope_type == scope_type, StoragePolicyRecord.scope_id == scope_id))
    except Exception:
        return defaults
    if record is None or not record.settings:
        return defaults
    try:
        return StoragePolicy.model_validate(_merged(defaults.model_dump(), record.settings))
    except ValidationError:
        return defaults


def set_policy(session: Session, scope_type: str, scope_id: str, changes: dict, *, updated_by: str | None = None
               ) -> StoragePolicy:
    """Store a scope's differences from the defaults; refuses an invalid policy."""
    from datetime import datetime, timezone
    from app.models.storage_policy import StoragePolicyRecord
    defaults = deployment_defaults(profile_for(scope_type)).model_dump()
    record = session.scalar(select(StoragePolicyRecord).where(
        StoragePolicyRecord.scope_type == scope_type, StoragePolicyRecord.scope_id == scope_id))
    combined = _merged(record.settings if record is not None else {}, changes)
    policy = StoragePolicy.model_validate(_merged(defaults, combined))
    if record is None:
        record = StoragePolicyRecord(scope_type=scope_type, scope_id=scope_id, settings=combined)
        session.add(record)
    record.settings = combined
    record.updated_by = updated_by
    record.updated_at = datetime.now(timezone.utc)
    session.flush()
    return policy
