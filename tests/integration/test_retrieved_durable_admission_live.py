"""Opt-in live PostgreSQL/MinIO/ClamAV validation for retrieved admission."""

import os
from datetime import datetime, timedelta, timezone

import fitz
import pytest
from sqlalchemy import func, select

from app.database import SessionLocal
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
    SourceRepresentation,
)
from app.services.source_repository import (
    AdmissionRequest,
    WorkIdentity,
    admit_representation,
    expire_representations,
    finalize_pending_object_deletions,
    find_accepted_representation,
)
from app.services.source_resolver import SourceResolver


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_RETRIEVED_ADMISSION") != "1",
    reason="requires isolated live PostgreSQL, MinIO, and ClamAV services",
)


class _LocalRetrievedSource(RetrievalSource):
    name = "live_stub_provider"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(source_name=self.name, success=False)

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        return RetrievalResult(source_name=self.name, success=False)


def _pdf(title: str, doi: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(40, 40, 555, 780),
        f"{title}\nDOI {doi}\nRivera, Alex (2026)\n"
        + "Complete generated article content for live validation. " * 25,
        fontsize=10,
    )
    payload = document.tobytes()
    document.close()
    return payload


def _result(content: bytes, title: str, doi: str) -> RetrievalResult:
    url = "https://repository.invalid/generated-validation.pdf"
    return RetrievalResult(
        source_name="live_stub_provider",
        success=True,
        doi=doi,
        title=title,
        year="2026",
        authors=["Rivera, Alex"],
        representation=SourceRepresentation(
            kind=RepresentationKind.PDF,
            media_type="application/pdf",
            content=content,
            source_url=url,
            completeness="complete",
        ),
        locations=[
            AcquisitionLocation(
                url=url,
                provider="live_stub_provider",
                representation_kind=RepresentationKind.PDF,
                media_type="application/pdf",
                access_type="open_access",
                is_best=True,
            )
        ],
    )


@pytest.mark.integration
def test_live_retrieved_pdf_admission_cache_dedup_rejection_and_expiry() -> None:
    title = "SourceFidelity Live Retrieved Admission"
    doi = "10.5555/sourcefidelity-live-retrieved-admission"
    resolver = SourceResolver()
    source = _LocalRetrievedSource()
    clean_pdf = _pdf(title, doi)

    first = resolver._download_and_cache(
        source,
        _result(clean_pdf, title, doi),
        ref_doi=doi,
        ref_title=title,
        ref_author="Rivera, Alex",
        ref_year="2026",
    )
    assert first.metadata["durable_admission"]["state"] == "accepted"
    representation_id = first.metadata["durable_admission"]["representation_id"]

    cached = resolver._check_local_cache(doi, None, title)
    assert cached.success is True
    assert cached.full_text == clean_pdf
    assert cached.metadata["repository_representation_id"] == representation_id

    second = resolver._download_and_cache(
        source,
        _result(clean_pdf, title, doi),
        ref_doi=doi,
        ref_title=title,
        ref_author="Rivera, Alex",
        ref_year="2026",
    )
    assert second.metadata["durable_admission"]["representation_id"] == representation_id

    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 1
        accepted = find_accepted_representation(
            session,
            scope_type="personal_owner",
            scope_id="retrieved-validation",
            doi=doi,
        )
        assert accepted is not None
        storage_key = accepted.content_object.storage_key
        record_id = accepted.id

        scoped_request = {
            "work": WorkIdentity(
                title=title,
                work_type="journal_article",
                doi=doi,
                author="Rivera, Alex",
                year="2026",
            ),
            "representation": SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=clean_pdf,
                source_url="https://repository.invalid/generated-validation.pdf",
                completeness="complete",
            ),
            "provenance": source.name,
            "license_class": "open_access",
            "identity_verdict": "verified",
            "identity_confidence": 1.0,
            "completeness_verdict": "complete",
            "cleanliness_verdict": "clean",
            "text_quality": "digital",
            "admitted_by": "live_retention_test",
        }
        assessment = admit_representation(
            session,
            resolver._backend,
            AdmissionRequest(
                **scoped_request,
                scope_type="assessment",
                scope_id="assessment-live-1",
                expires_at=now + timedelta(days=30),
            ),
        )
        course = admit_representation(
            session,
            resolver._backend,
            AdmissionRequest(
                **scoped_request,
                scope_type="course_offering",
                scope_id="course-live-2026",
                expires_at=now + timedelta(days=120),
            ),
        )
        session.commit()

        assert assessment.content_object_id == accepted.content_object_id
        assert course.content_object_id == accepted.content_object_id
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 3
        assert find_accepted_representation(
            session,
            scope_type="assessment",
            scope_id="assessment-live-1",
            doi=doi,
            now=now,
        ).id == assessment.id
        assert find_accepted_representation(
            session,
            scope_type="course_offering",
            scope_id="course-live-2026",
            doi=doi,
            now=now,
        ).id == course.id
        assert find_accepted_representation(
            session,
            scope_type="course_offering",
            scope_id="course-other-2026",
            doi=doi,
            now=now,
        ) is None

        course.expires_at = now - timedelta(seconds=1)
        session.commit()
        assert find_accepted_representation(
            session,
            scope_type="course_offering",
            scope_id="course-live-2026",
            doi=doi,
            now=now,
        ) is None

        assert expire_representations(session, now=now) == 1
        session.commit()
        assert session.get(SourceRepresentationRecord, course.id) is None
        assert finalize_pending_object_deletions(session, resolver._backend) == 0
        session.commit()
        assert resolver._backend.exists(storage_key) is True
        assert session.get(SourceRepresentationRecord, assessment.id) is not None

        accepted.expires_at = now - timedelta(seconds=1)
        assessment.expires_at = now - timedelta(seconds=1)
        session.commit()

    hostile = _result(
        clean_pdf + b"\n/JavaScript /OpenAction ",
        "Rejected Retrieved Admission",
        "10.5555/sourcefidelity-live-rejected",
    )
    rejected = resolver._download_and_cache(
        source,
        hostile,
        ref_doi="10.5555/sourcefidelity-live-rejected",
        ref_title="Rejected Retrieved Admission",
    )
    assert rejected.full_text is None
    assert rejected.metadata["durable_admission"]["state"] == "rejected"
    with SessionLocal() as session:
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 2

    with SessionLocal() as session:
        assert expire_representations(session, now=now) == 2
        session.commit()
        assert session.get(SourceRepresentationRecord, record_id) is None
        assert finalize_pending_object_deletions(session, resolver._backend) == 1
        session.commit()
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 0
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 0
    assert resolver._backend.exists(storage_key) is False
