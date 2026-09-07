from datetime import datetime, timedelta, timezone
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.report import Report, ReportPaperArtifactRecord
from app.services.paper_annotations import (
    PaperAnnotationError,
    create_paper_annotation,
    list_current_paper_annotations,
    revise_paper_annotation,
)


def _anchor():
    return {
        "anchor_id": "a" * 64,
        "localization_level": "exact_rectangle",
        "page_indexes": [0],
        "rectangles": [
            {
                "page_index": 0,
                "x0": 72.0,
                "y0": 100.0,
                "x1": 240.0,
                "y1": 118.0,
            }
        ],
    }


@pytest.fixture
def annotation_store():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        job = Job(
            filename="paper.pdf",
            status=JobStatus.COMPLETED,
            stage=JobStage.COMPLETED,
            paper_version_id="paper-v1",
            scope_type="personal_owner",
            scope_id="owner-1",
            input_sha256="1" * 64,
            input_media_type="application/pdf",
            input_byte_size=10,
            input_expires_at=datetime.now(timezone.utc) + timedelta(days=1),
            input_storage_key=None,
        )
        session.add(job)
        session.flush()
        report = Report(job_id=job.id, report_json={})
        session.add(report)
        session.flush()
        artifact = ReportPaperArtifactRecord(
            job_id=job.id,
            report_id=report.id,
            paper_version_id="paper-v1",
            scope_type="personal_owner",
            scope_id="owner-1",
            storage_key="paper.pdf",
            content_sha256="2" * 64,
            media_type="application/pdf",
            byte_size=10,
            artifact_kind="submitted_pdf",
            presentation_status="page_faithful_ready",
            presentation_storage_key="paper.pdf",
            presentation_sha256="3" * 64,
            presentation_media_type="application/pdf",
            presentation_byte_size=10,
            presentation_evidence={},
            sanitization_evidence={},
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        session.add(artifact)
        session.commit()
        yield session, report, artifact


def test_annotation_revisions_are_append_only_and_current_view_is_scope_bound(
    annotation_store,
):
    session, report, artifact = annotation_store
    created = create_paper_annotation(
        session,
        artifact=artifact,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="personal-owner",
        annotation_type="comment",
        anchor=_anchor(),
        content="Check this distinction.",
    )

    assert created["revision"] == 1
    assert created["visibility"] == "private"
    assert created["anchor"]["rectangles"][0]["x0"] == 72.0
    assert len(
        list_current_paper_annotations(
            session,
            report_id=report.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )
    ) == 1
    assert list_current_paper_annotations(
        session,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="another-owner",
    ) == []

    released = revise_paper_annotation(
        session,
        report_id=report.id,
        annotation_id=created["annotation_id"],
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="personal-owner",
        expected_revision=1,
        content="Check the actor distinction.",
        visibility="released",
    )
    assert released["revision"] == 2
    assert released["visibility"] == "released"
    assert released["content"] == "Check the actor distinction."

    with pytest.raises(PaperAnnotationError, match="changed"):
        revise_paper_annotation(
            session,
            report_id=report.id,
            annotation_id=created["annotation_id"],
            scope_type="personal_owner",
            scope_id="owner-1",
            author_provider="personal_local",
            author_subject="personal-owner",
            expected_revision=1,
        )

    deleted = revise_paper_annotation(
        session,
        report_id=report.id,
        annotation_id=created["annotation_id"],
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="personal-owner",
        expected_revision=2,
        state="deleted",
    )
    assert deleted["revision"] == 3
    assert list_current_paper_annotations(
        session,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    ) == []


def test_annotation_creation_rejects_fragmentary_or_unbound_inputs(annotation_store):
    session, report, artifact = annotation_store
    with pytest.raises(PaperAnnotationError, match="requires text"):
        create_paper_annotation(
            session,
            artifact=artifact,
            report_id=report.id,
            scope_type="personal_owner",
            scope_id="owner-1",
            author_provider="personal_local",
            author_subject="personal-owner",
            annotation_type="comment",
            anchor=_anchor(),
        )
    bad_anchor = _anchor()
    bad_anchor["rectangles"] = []
    with pytest.raises(PaperAnnotationError, match="exact paper geometry"):
        create_paper_annotation(
            session,
            artifact=artifact,
            report_id=report.id,
            scope_type="personal_owner",
            scope_id="owner-1",
            author_provider="personal_local",
            author_subject="personal-owner",
            annotation_type="highlight",
            anchor=bad_anchor,
        )


def test_page_region_annotation_is_hash_bound_and_rejects_out_of_page_geometry(
    annotation_store,
):
    session, report, artifact = annotation_store
    region = {
        "anchor_version": "page-region-anchor-v1",
        "anchor_kind": "page_region",
        "localization_level": "exact_rectangle",
        "rectangles": [
            {
                "page_index": 0,
                "x0": 80.0,
                "y0": 120.0,
                "x1": 260.0,
                "y1": 148.0,
            }
        ],
    }
    created = create_paper_annotation(
        session,
        artifact=artifact,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="personal-owner",
        annotation_type="highlight",
        anchor=region,
        page_dimensions={0: (612.0, 792.0)},
    )

    assert created["anchor_kind"] == "page_region"
    assert created["anchor"]["anchor_version"] == "page-region-anchor-v1"
    assert created["anchor"]["page_indexes"] == [0]
    assert len(created["anchor"]["anchor_id"]) == 64

    repeated = create_paper_annotation(
        session,
        artifact=artifact,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="personal-owner",
        annotation_type="comment",
        anchor=region,
        content="Same paper region.",
        page_dimensions={0: (612.0, 792.0)},
    )
    assert repeated["anchor"]["anchor_id"] == created["anchor"]["anchor_id"]

    outside = dict(region)
    outside["rectangles"] = [dict(region["rectangles"][0], x1=700.0)]
    with pytest.raises(PaperAnnotationError, match="outside the paper"):
        create_paper_annotation(
            session,
            artifact=artifact,
            report_id=report.id,
            scope_type="personal_owner",
            scope_id="owner-1",
            author_provider="personal_local",
            author_subject="personal-owner",
            annotation_type="highlight",
            anchor=outside,
            page_dimensions={0: (612.0, 792.0)},
        )
