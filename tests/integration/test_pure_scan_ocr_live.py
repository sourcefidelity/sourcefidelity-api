"""Opt-in live ClamAV/OCR/PostgreSQL/MinIO lifecycle validation."""

import os
from pathlib import Path
import uuid

import pytest
from sqlalchemy import func, select

from app.database import SessionLocal, engine
from app.models import Base
from app.models.source_repository import (
    ContentObjectRecord,
    SourceRepresentationRecord,
)
from app.services.retrieval.base import (
    RepresentationKind,
    RetrievalResult,
    SourceRepresentation,
)
from app.services.source_repository import (
    delete_representation,
    finalize_pending_object_deletions,
)
from app.services.source_resolver import SourceResolver
from app.services.source_type import SourceKindAssessment
from app.services.verification_evidence import authorize_representation


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_PURE_SCAN_OCR") != "1",
    reason="requires isolated PostgreSQL, MinIO, ClamAV, Tesseract, and scan PDF",
)


@pytest.mark.integration
def test_live_pure_scan_ocr_parent_derivative_admission_and_cleanup() -> None:
    source_path = Path(os.environ["OCR_LIVE_PDF_PATH"])
    source_bytes = source_path.read_bytes()
    Base.metadata.create_all(bind=engine)
    expected_kind = SourceKindAssessment(
        "journal_article", "high", ("live OCR acceptance control",)
    )
    result = RetrievalResult(
        source_name="live_owner_scan",
        success=True,
        title=(
            "Rejecting the Center: Radical Grassroots Politics in the 1970s—"
            "Second-Wave Feminism as a Case Study"
        ),
        year="2008",
        authors=["Joshua Zeitz"],
        representation=SourceRepresentation(
            kind=RepresentationKind.PDF,
            media_type="application/pdf",
            content=source_bytes,
            source_url="file:///private/live-ocr-control.pdf",
        ),
        metadata={
            "page": "673-688",
            "license_class": "commercial_user_upload",
        },
    )
    resolver = SourceResolver()

    accepted, outcome, reason = resolver._preflight_acquired_representation(
        result,
        expected_doi=None,
        expected_title=result.title,
        expected_author="Joshua Zeitz",
        expected_year="2008",
        expected_source_kind=expected_kind,
    )

    assert accepted, (outcome, reason)
    assert result.parent_representation is not None
    assert result.representation is not None
    assert result.representation.kind is RepresentationKind.PLAIN_TEXT
    assert result.metadata["text_quality"] == "scan_ocr"
    assert result.metadata["ocr_derivative"]["page_labels"][1:] == [
        str(page) for page in range(673, 689)
    ]

    resolver._persist_retrieved_representation(
        result,
        ref_doi=None,
        ref_title=result.title,
        ref_author="Joshua Zeitz",
        ref_year="2008",
        identity_confidence="high",
        identity_reason=result.metadata["identity_reason"],
        downloaded_via_publisher=False,
        safety_report=None,
        expected_source_kind=expected_kind,
    )
    admission = result.metadata["durable_admission"]
    assert admission["state"] == "accepted"
    assert admission["parent_representation_id"] is not None

    with SessionLocal() as session:
        derivative = session.get(
            SourceRepresentationRecord, uuid.UUID(admission["representation_id"])
        )
        parent = session.get(
            SourceRepresentationRecord,
            uuid.UUID(admission["parent_representation_id"]),
        )
        assert derivative is not None and derivative.admission_state == "accepted"
        assert parent is not None and parent.admission_state == "needs_review"
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 2
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 2
        authorized = authorize_representation(
            session,
            resolver._backend,
            representation_id=derivative.id,
            scope_type="personal_owner",
            scope_id=os.environ["SOURCE_REPOSITORY_SCOPE_ID"],
        )
        assert authorized.parent_content_sha256 == parent.content_object.content_sha256
        assert authorized.derivation_manifest_sha256 == (
            result.metadata["ocr_derivative"]["derivation_manifest_sha256"]
        )

        assert delete_representation(session, derivative.id)
        assert delete_representation(session, parent.id)
        session.commit()
        assert finalize_pending_object_deletions(session, resolver._backend) == 2
        session.commit()
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 0
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 0
