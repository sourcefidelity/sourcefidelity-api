"""Re-run the current identity rule over already-admitted search-found PDFs.

Read-only (opening-byline-identity-v1): nothing is re-admitted, retired or
rewritten. Each accepted representation of the given provenance is checked
again against its canonical work's cited fields, and those that would no
longer reach `high` identity are listed by id and hash only.
"""
from __future__ import annotations

from sqlalchemy import select

from app.models.source_repository import SourceRepresentationRecord
from app.services.source_validator import OPENING_BYLINE_POLICY, validate_retrieved_pdf

DEFAULT_PROVENANCE = ("web_search",)


def recheck_admitted_representations(session, backend, *,
                                     provenance: tuple[str, ...] = DEFAULT_PROVENANCE) -> dict:
    """Return counts plus the representations no longer at high identity.

    Lists identifiers, content hashes and the rule's outcome codes only; never
    a title, author, address or any text from the file.
    """
    records = session.scalars(
        select(SourceRepresentationRecord).where(
            SourceRepresentationRecord.provenance.in_(tuple(provenance)),
            SourceRepresentationRecord.admission_state == "accepted",
        ).order_by(SourceRepresentationRecord.created_at)
    ).all()
    flagged: list[dict] = []
    checked = skipped_not_pdf = unavailable = 0
    for record in records:
        if record.representation_kind != "pdf":
            skipped_not_pdf += 1
            continue
        try:
            data = backend.download(record.content_object.storage_key)
        except FileNotFoundError:
            unavailable += 1
            continue
        work = record.canonical_work
        validation = validate_retrieved_pdf(
            data, expected_doi=work.doi, expected_title=work.display_title,
            expected_author=work.author, expected_year=work.year, expected_isbn=work.isbn,
            expected_source_kind=work.work_type, skip_completeness=True,
        )
        checked += 1
        if validation.identity_confidence == "high":
            continue
        flagged.append({
            "representation_id": str(record.id),
            "canonical_work_id": str(work.id),
            "content_sha256": record.content_object.content_sha256,
            "identity_confidence": validation.identity_confidence,
            "author_conflict": bool(validation.author_conflict),
            "reason_code": validation.reason_code,
        })
    return {
        "policy": OPENING_BYLINE_POLICY,
        "provenance": list(provenance),
        "checked": checked,
        "skipped_not_pdf": skipped_not_pdf,
        "object_unavailable": unavailable,
        "flagged": flagged,
    }
