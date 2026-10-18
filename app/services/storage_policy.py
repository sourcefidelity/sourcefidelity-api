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

Who sets them (owner decision 2026-10-13): the Personal owner for their own
scope; in Institutional, the administrator who deploys the application, for
the whole deployment (`INSTITUTIONAL_SCOPE`), because the choices must follow
university policy. Instructors cannot change them, and the settings page
writes no course or assessment scope row.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings

POLICY_VERSION = "storage-policy-v1"
Profile = Literal["personal", "institutional"]
PAPER_COPY_OUTLIVES_REPORT = "the paper copy cannot outlive its report"
SUBMITTED_FILE_OUTLIVES_REPORT = "the submitted file cannot outlive its report"
# The deployment-wide Institutional settings row every institutional scope inherits.
INSTITUTIONAL_SCOPE = ("deployment", "institutional")


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
    # Incomplete searches are finished by paid re-searches on their own:
    # always in Institutional; in Personal only when the owner turns it on
    # (owner decision 2026-10-18).
    finish_incomplete_automatically: bool = True


class StoragePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["storage-policy-v1"] = POLICY_VERSION
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
        # The submitted file is kept 30 days in Personal and not kept in
        # Institutional until the administrator decides (owner decision 2026-10-13).
        papers=PaperStorage(paper_copy_days=max(1, int(settings.FORMATIVE_REPORT_PAPER_RETENTION_DAYS)),
                            keep_submitted_file=profile == "personal"),
        searches=SearchSettings(finish_incomplete_automatically=profile != "personal"),
    )


def finishes_searches_automatically(session: Session | None, scope_type: str, scope_id: str) -> bool:
    """Whether a scope's incomplete searches are re-searched on their own."""
    if profile_for(scope_type) != "personal":
        return True
    return policy_for(session, scope_type, scope_id).searches.finish_incomplete_automatically


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
    keys = [(scope_type, scope_id)]
    if profile_for(scope_type) == "institutional" and (scope_type, scope_id) != INSTITUTIONAL_SCOPE:
        keys.insert(0, INSTITUTIONAL_SCOPE)
    merged = defaults.model_dump()
    for key in keys:
        try:
            record = _record(session, *key)
        except Exception:
            return defaults
        settings_row = getattr(record, "settings", None)
        if isinstance(settings_row, dict) and settings_row:
            merged = _merged(merged, settings_row)
    try:
        return StoragePolicy.model_validate(merged)
    except ValidationError:
        return defaults


def settings_scope_for(scope_type: str, scope_id: str) -> tuple[str, str]:
    """The row a settings-page save writes: the Personal owner's own scope,
    or the deployment-wide Institutional row."""
    return (scope_type, scope_id) if profile_for(scope_type) == "personal" else INSTITUTIONAL_SCOPE


def _record(session: Session, scope_type: str, scope_id: str):
    from app.models.storage_policy import StoragePolicyRecord
    return session.scalar(select(StoragePolicyRecord).where(
        StoragePolicyRecord.scope_type == scope_type, StoragePolicyRecord.scope_id == scope_id))


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
