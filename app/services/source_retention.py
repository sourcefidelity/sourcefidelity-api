"""Apply the user's source storage setting to copies already kept (owner decision 2026-10-19).

One setting covers every kept copy, uploads included (`storage_policy.SourceStorage`).
A copy the new setting no longer keeps is marked to expire now, so the scheduled
clean-up deletes it; the settings page shows how many first and asks to confirm.
A copy still kept gets the chosen months from now (never longer than it already
had), or no expiry when copies are kept permanently.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services.storage_policy import StoragePolicy, copy_expiry

OPEN_CLASSES = frozenset({"open_access", "public_domain"})


def kept_by(policy: StoragePolicy, license_class: str) -> bool:
    """Whether the setting keeps a copy of this licence class."""
    keep = policy.sources.keep_copies
    if keep == "none":
        return False
    if keep == "open_access_only":
        return license_class in OPEN_CLASSES
    return True


def _live_copies(session: Session, scope_type: str, scope_id: str, now: datetime):
    rows = session.execute(
        select(SourceRepresentationRecord, ContentObjectRecord.license_class)
        .join(ContentObjectRecord, ContentObjectRecord.id == SourceRepresentationRecord.content_object_id)
        .where(SourceRepresentationRecord.scope_type == scope_type,
               SourceRepresentationRecord.scope_id == scope_id,
               (SourceRepresentationRecord.expires_at.is_(None))
               | (SourceRepresentationRecord.expires_at > now))
    ).all()
    return rows


def copies_to_delete(session: Session, scope_type: str, scope_id: str, policy: StoragePolicy,
                     now: datetime | None = None) -> int:
    """How many kept copies the setting would delete."""
    current = now or datetime.now(timezone.utc)
    return sum(1 for _, license_class in _live_copies(session, scope_type, scope_id, current)
               if not kept_by(policy, license_class))


def apply_source_setting(session: Session, scope_type: str, scope_id: str, policy: StoragePolicy,
                         now: datetime | None = None) -> int:
    """Re-date the user's kept copies under the setting; returns how many were marked for deletion."""
    current = now or datetime.now(timezone.utc)
    target = copy_expiry(policy, current)
    removed = 0
    for record, license_class in _live_copies(session, scope_type, scope_id, current):
        if not kept_by(policy, license_class):
            record.expires_at = current
            removed += 1
        elif target is None:
            record.expires_at = None
        elif record.expires_at is None or record.expires_at > target:
            record.expires_at = target
    session.flush()
    return removed
