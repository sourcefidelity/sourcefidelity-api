"""What the user's deployment keeps, and for how long: sources, papers and reports.

Owner decisions 2026-10-09 and 2026-10-19 (PLAN §2 *Shared lifecycle*;
Personal only). Sources follow one setting for every kept copy, the user's own
uploads included: keep all articles, open-access articles only, or none, and
keep each copy 3, 6, 9 or 12 months after its last use in a check, or
permanently (`keep_months` 0). The default is all articles, permanently.
Papers: the copy shown beside the report is kept for
`FORMATIVE_REPORT_PAPER_RETENTION_DAYS` after the report was last built or
returned again. Reports: kept until deleted (0 days).
A day count of 0 means "until deleted". The row holds only the differences
from the defaults; anything invalid fails closed to the defaults. Rows written
under storage-policy-v1 are read with their source fields translated.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings

POLICY_VERSION = "storage-policy-v2"
PAPER_COPY_OUTLIVES_REPORT = "the paper copy cannot outlive its report"
SUBMITTED_FILE_OUTLIVES_REPORT = "the submitted file cannot outlive its report"
KEEP_MONTHS = (0, 3, 6, 9, 12)
_DAYS_PER_MONTH = 30.44


class SourceStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Every kept copy, uploads included: all, only those with an open-access
    # or public-domain signal, or none.
    keep_copies: Literal["all", "open_access_only", "none"] = "all"
    # Months after the copy's last use in a check; 0 keeps it permanently.
    keep_months: Literal[0, 3, 6, 9, 12] = 0


class PaperStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The copy of the paper shown beside its report.
    paper_copy_days: int = Field(default=7, ge=1)
    # The submitted file, kept so the paper can be checked again without a new
    # upload (`recheck-copy-v1`, owner decision 2026-10-13). An operational
    # copy inside temporary paper retention: never a comparison corpus.
    keep_submitted_file: bool = False
    submitted_file_days: int = Field(default=30, ge=0)


class ReportStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The whole paper record (report versions, checks, judgments) is deleted
    # this many days after its report was last built or returned again.
    report_days: int = Field(default=0, ge=0)


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Incomplete searches are finished by paid re-searches on their own only
    # when the user turns it on (owner decision 2026-10-18).
    finish_incomplete_automatically: bool = False


class StoragePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["storage-policy-v2"] = POLICY_VERSION
    sources: SourceStorage = Field(default_factory=SourceStorage)
    papers: PaperStorage = Field(default_factory=PaperStorage)
    reports: ReportStorage = Field(default_factory=ReportStorage)
    searches: SearchSettings = Field(default_factory=SearchSettings)

    @model_validator(mode="after")
    def _paper_copy_within_report(self):
        if self.reports.report_days and self.papers.paper_copy_days > self.reports.report_days:
            raise ValueError(PAPER_COPY_OUTLIVES_REPORT)
        days = self.papers.submitted_file_days
        if self.reports.report_days and self.papers.keep_submitted_file and (
                days == 0 or days > self.reports.report_days):
            raise ValueError(SUBMITTED_FILE_OUTLIVES_REPORT)
        return self


def deployment_defaults() -> StoragePolicy:
    """The deployment's settings as a policy."""
    return StoragePolicy(
        papers=PaperStorage(paper_copy_days=max(1, int(settings.FORMATIVE_REPORT_PAPER_RETENTION_DAYS)),
                            keep_submitted_file=True),
    )


def copy_expiry(policy: StoragePolicy, now=None):
    """When a kept copy used now expires; None when copies are kept permanently."""
    from datetime import datetime, timedelta, timezone
    months = policy.sources.keep_months
    if not months:
        return None
    return (now or datetime.now(timezone.utc)) + timedelta(days=round(months * _DAYS_PER_MONTH))


def upload_expiry(policy: StoragePolicy, now=None):
    """When a user's upload expires (owner decision 2026-10-19: one setting for
    every kept copy). Under "all" it follows the chosen months; otherwise an
    upload, having no open-access signal, is kept only as long as the paper
    copy beside its report, so the check that uses it can still run."""
    from datetime import datetime, timedelta, timezone
    if policy.sources.keep_copies == "all":
        return copy_expiry(policy, now)
    return (now or datetime.now(timezone.utc)) + timedelta(days=policy.papers.paper_copy_days)


def finishes_searches_automatically(session: Session | None, scope_type: str, scope_id: str) -> bool:
    """Whether the user's incomplete searches are re-searched on their own."""
    return policy_for(session, scope_type, scope_id).searches.finish_incomplete_automatically


def _upgrade_v1_row(row: dict) -> dict:
    """A storage-policy-v1 row's source fields in the v2 shape: the public-copy
    choice becomes the one choice, its days become the nearest allowed months
    at or above them (permanently above 12), library-licence fields drop."""
    sources = row.get("sources")
    if not isinstance(sources, dict) or not ({"keep_public_copies", "public_copy_days",
                                              "keep_licensed_copies", "licensed_copy_days"} & set(sources)):
        return row
    upgraded = {}
    if "keep_public_copies" in sources:
        upgraded["keep_copies"] = sources["keep_public_copies"]
    days = sources.get("public_copy_days")
    if isinstance(days, int) and days > 0:
        upgraded["keep_months"] = next((m for m in KEEP_MONTHS[1:] if m * _DAYS_PER_MONTH >= days), 0)
    out = {k: v for k, v in row.items() if k != "version"}
    out["sources"] = upgraded
    return out


def _merged(base: dict, changes: dict) -> dict:
    out = dict(base)
    for key, value in (changes or {}).items():
        out[key] = _merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def policy_for(session: Session | None, scope_type: str, scope_id: str) -> StoragePolicy:
    """The user's settings over the deployment defaults; the defaults when
    there are none, the row is invalid or the database is unavailable."""
    defaults = deployment_defaults()
    if session is None:
        return defaults
    try:
        record = _record(session, scope_type, scope_id)
    except Exception:
        return defaults
    row = getattr(record, "settings", None)
    merged = defaults.model_dump()
    if isinstance(row, dict) and row:
        merged = _merged(merged, _upgrade_v1_row(row))
    try:
        return StoragePolicy.model_validate(merged)
    except ValidationError:
        return defaults


def _record(session: Session, scope_type: str, scope_id: str):
    from app.models.storage_policy import StoragePolicyRecord
    return session.scalar(select(StoragePolicyRecord).where(
        StoragePolicyRecord.scope_type == scope_type, StoragePolicyRecord.scope_id == scope_id))


def set_policy(session: Session, scope_type: str, scope_id: str, changes: dict, *, updated_by: str | None = None
               ) -> StoragePolicy:
    """Store a scope's differences from the defaults; refuses an invalid policy."""
    from datetime import datetime, timezone
    from app.models.storage_policy import StoragePolicyRecord
    defaults = deployment_defaults().model_dump()
    record = session.scalar(select(StoragePolicyRecord).where(
        StoragePolicyRecord.scope_type == scope_type, StoragePolicyRecord.scope_id == scope_id))
    combined = _merged(_upgrade_v1_row(record.settings) if record is not None and record.settings else {}, changes)
    policy = StoragePolicy.model_validate(_merged(defaults, combined))
    if record is None:
        record = StoragePolicyRecord(scope_type=scope_type, scope_id=scope_id, settings=combined)
        session.add(record)
    record.settings = combined
    record.updated_by = updated_by
    record.updated_at = datetime.now(timezone.utc)
    session.flush()
    return policy
