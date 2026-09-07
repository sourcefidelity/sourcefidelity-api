"""Released annotated PDF export acceptance tests."""

from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace
import uuid

from fastapi.testclient import TestClient
import fitz
from pydantic import SecretStr
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.main import app
from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.report import Report, ReportPaperArtifactRecord
from app.services.paper_annotations import (
    create_paper_annotation,
    revise_paper_annotation,
)
from app.services.report_export import (
    ReleasedReportExport,
    ReportExportError,
    build_released_report_export,
)
from app.services.storage.backend import StorageBackend, get_storage_backend


class MemoryStorage(StorageBackend):
    def __init__(self):
        self.objects = {}

    def upload(self, file_bytes, key):
        self.objects[key] = file_bytes
        return key

    def download(self, key):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete(self, key):
        self.objects.pop(key, None)
        return True

    def exists(self, key):
        return key in self.objects

    def list_keys(self, prefix):
        return [key for key in self.objects if key.startswith(prefix)]


def _pdf_bytes():
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 72), "Synthetic paper content for export acceptance.")
    content = document.tobytes(no_new_id=True)
    document.close()
    return content


def _anchor(anchor_id, *, y0=60.0, y1=78.0, kind="citation_anchor"):
    return {
        "anchor_version": (
            "page-region-anchor-v1" if kind == "page_region" else "citation-anchor-v1"
        ),
        "anchor_kind": kind,
        "anchor_id": anchor_id,
        "localization_level": "exact_rectangle",
        "page_indexes": [0],
        "rectangles": [
            {
                "page_index": 0,
                "x0": 70.0,
                "y0": y0,
                "x1": 280.0,
                "y1": y1,
            }
        ],
    }


def _view(artifact, *, source_secret="Inspectable source excerpt."):
    anchor_id = "a" * 64
    return {
        "report_id": str(artifact.report_id),
        "paper_version_id": artifact.paper_version_id,
        "title": "Synthetic acceptance paper",
        "citation_format": "APA",
        "word_counts": {"total": 7, "body": 7, "references": 0},
        "paper_surface": {
            "artifact_id": str(artifact.id),
            "anchor_reason_code": "presentation_hash_bound",
            "page_dimensions": [
                {"page_index": 0, "width": 612.0, "height": 792.0}
            ],
        },
        "citations": [
            {
                "claim_id": "claim-1",
                "tone": "evidence_available",
                "paper_location": _anchor(anchor_id),
                "quotation_difference_rectangles": [
                    {
                        "page_index": 0,
                        "x0": 150.0,
                        "y0": 60.0,
                        "x1": 180.0,
                        "y1": 78.0,
                    }
                ],
                "members": [{"coverage_level":"full_text", "source":{"author":"Researcher", "year":"2020", "title":"Synthetic source", "raw_reference":"Researcher (2020). Synthetic source."}, "best_evidence": {"text": source_secret}}],
            }
        ],
        "reference_practice": [
            {
                "finding_id": "format-1",
                "rectangles": [
                    {
                        "page_index": 0,
                        "x0": 70.0,
                        "y0": 700.0,
                        "x1": 320.0,
                        "y1": 720.0,
                    }
                ],
            }
        ],
        "role_summaries": {},
        "gauges": [],
        "limits": [],
    }


@pytest.fixture
def export_store(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    storage = MemoryStorage()
    paper = _pdf_bytes()
    with Session(engine) as session:
        job = Job(
            filename="synthetic.pdf",
            status=JobStatus.COMPLETED,
            stage=JobStage.COMPLETED,
            paper_version_id="paper-v1",
            scope_type="personal_owner",
            scope_id="owner-1",
            input_sha256=hashlib.sha256(paper).hexdigest(),
            input_media_type="application/pdf",
            input_byte_size=len(paper),
            input_expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        session.add(job)
        session.flush()
        first = Report(job_id=job.id, report_json={}, report_version=1)
        session.add(first)
        session.flush()
        report = Report(
            job_id=job.id,
            report_json={},
            report_version=2,
            previous_report_id=first.id,
            amendment_reason="user_source_upload",
        )
        session.add(report)
        session.flush()
        artifact = ReportPaperArtifactRecord(
            job_id=job.id,
            report_id=report.id,
            paper_version_id="paper-v1",
            scope_type="personal_owner",
            scope_id="owner-1",
            storage_key="paper.pdf",
            content_sha256=hashlib.sha256(paper).hexdigest(),
            media_type="application/pdf",
            byte_size=len(paper),
            artifact_kind="submitted_pdf",
            presentation_status="page_faithful_ready",
            presentation_storage_key="paper.pdf",
            presentation_sha256=hashlib.sha256(paper).hexdigest(),
            presentation_media_type="application/pdf",
            presentation_byte_size=len(paper),
            presentation_evidence={},
            sanitization_evidence={},
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        session.add(artifact)
        session.commit()
        storage.upload(paper, "paper.pdf")
        view = _view(artifact)
        monkeypatch.setattr(
            "app.services.report_export.load_authorized_evidence_report_bundle",
            lambda *args, **kwargs: (view, artifact, paper),
        )
        yield session, storage, report, artifact, view, paper


def _create_annotation(
    session,
    report,
    artifact,
    *,
    annotation_type,
    anchor,
    content=None,
    visibility="private",
):
    return create_paper_annotation(
        session,
        artifact=artifact,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="owner",
        annotation_type=annotation_type,
        anchor=anchor,
        content=content,
        visibility=visibility,
        page_dimensions={0: (612.0, 792.0)},
    )


def test_released_export_is_deterministic_hash_bound_and_source_free(export_store):
    session, storage, report, artifact, _view_data, paper = export_store
    citation_anchor = _anchor("a" * 64)
    _create_annotation(
        session,
        report,
        artifact,
        annotation_type="highlight",
        anchor=citation_anchor,
        visibility="released",
    )
    _create_annotation(
        session,
        report,
        artifact,
        annotation_type="comment",
        anchor=_anchor("", y0=120, y1=145, kind="page_region"),
        content="Released comment text.",
        visibility="released",
    )
    _create_annotation(
        session,
        report,
        artifact,
        annotation_type="comment",
        anchor=citation_anchor,
        content="PRIVATE-COMMENT-MUST-NOT-EXPORT",
        visibility="private",
    )

    first = build_released_report_export(
        session,
        storage,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    second = build_released_report_export(
        session,
        storage,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )

    assert first.content == second.content
    assert first.manifest == second.manifest
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest["report_id"] == str(report.id)
    assert first.manifest["report_version"] == 2
    assert first.manifest["previous_report_id"] == str(report.previous_report_id)
    assert first.manifest["paper_presentation_sha256"] == hashlib.sha256(paper).hexdigest()
    assert first.manifest["source_policy"] == {
        "source_bytes_embedded": False,
        "source_excerpts_embedded": True,
        "authorization_bound_source_actions_embedded": False,
    }
    assert {key:first.manifest["overlay_counts"][key] for key in ("citation_underlines", "quotation_differences", "reference_practice_findings", "released_highlights", "released_comments")} == {
        "citation_underlines": 1,
        "quotation_differences": 1,
        "reference_practice_findings": 1,
        "released_highlights": 1,
        "released_comments": 1,
    }
    serialized_manifest = str(first.manifest)
    assert "Released comment text." not in serialized_manifest
    assert "PRIVATE-COMMENT-MUST-NOT-EXPORT" not in serialized_manifest
    assert "SOURCE-CONTENT-MUST-NOT-EXPORT" not in first.content.decode(
        "latin-1", errors="ignore"
    )

    exported = fitz.open(stream=first.content, filetype="pdf")
    try:
        assert exported.page_count > 1
        appendix = "\n".join(page.get_text() for page in list(exported)[1:])
        assert "Inspectable source excerpt." in appendix
        assert "Released comment text." in appendix
        assert "PRIVATE-COMMENT-MUST-NOT-EXPORT" not in appendix
        assert any(link.get("page",0) > 0 for link in exported[0].get_links())
        assert any(link.get("page") == 0 for page in list(exported)[1:] for link in page.get_links())
        notes = list(exported[0].annots() or [])
        assert len(notes) == 1
        assert notes[0].info["content"] == "Released comment text."
        assert "PRIVATE-COMMENT-MUST-NOT-EXPORT" not in str(notes[0].info)
        assert len(exported[0].get_drawings()) >= 4
        assert "SOURCE-CONTENT-MUST-NOT-EXPORT" not in exported[0].get_text()
        assert f"report={report.id}" in exported.metadata["subject"]
        original = fitz.open(stream=paper, filetype="pdf")
        try:
            assert exported[0].get_text() == original[0].get_text()
        finally:
            original.close()
    finally:
        exported.close()


def test_export_uses_only_latest_released_revision_and_handles_empty_set(export_store):
    session, storage, report, artifact, _view_data, _paper = export_store
    private = _create_annotation(
        session,
        report,
        artifact,
        annotation_type="comment",
        anchor=_anchor("a" * 64),
        content="Initially private.",
    )
    released = revise_paper_annotation(
        session,
        report_id=report.id,
        annotation_id=private["annotation_id"],
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="owner",
        expected_revision=1,
        content="Released successor.",
        visibility="released",
    )
    revise_paper_annotation(
        session,
        report_id=report.id,
        annotation_id=released["annotation_id"],
        scope_type="personal_owner",
        scope_id="owner-1",
        author_provider="personal_local",
        author_subject="owner",
        expected_revision=2,
        content="Returned to private.",
        visibility="private",
    )

    exported = build_released_report_export(
        session,
        storage,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    assert exported.manifest["released_annotation_revisions"] == []
    assert exported.manifest["overlay_counts"]["released_comments"] == 0
    document = fitz.open(stream=exported.content, filetype="pdf")
    try:
        assert list(document[0].annots() or []) == []
    finally:
        document.close()


def test_export_fails_closed_on_annotation_paper_hash_mismatch(export_store, monkeypatch):
    session, storage, report, artifact, _view_data, _paper = export_store
    monkeypatch.setattr(
        "app.services.report_export.list_current_paper_annotations",
        lambda *args, **kwargs: [
            {
                "annotation_id": str(uuid.uuid4()),
                "revision": 1,
                "paper_artifact_id": str(artifact.id),
                "paper_version_id": artifact.paper_version_id,
                "paper_content_sha256": "0" * 64,
                "annotation_type": "highlight",
                "anchor_kind": "citation_anchor",
                "anchor_sha256": "1" * 64,
                "anchor": _anchor("a" * 64),
                "content": None,
                "user_label": None,
                "visibility": "released",
            }
        ],
    )
    with pytest.raises(ReportExportError, match="does not match"):
        build_released_report_export(
            session,
            storage,
            report_id=report.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )


def test_failed_export_leaves_no_partial_object_and_retry_succeeds(
    export_store, monkeypatch
):
    session, storage, report, artifact, view, paper = export_store
    original_objects = dict(storage.objects)
    malformed = b"not a pdf"
    artifact.presentation_sha256 = hashlib.sha256(malformed).hexdigest()
    monkeypatch.setattr(
        "app.services.report_export.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, artifact, malformed),
    )
    with pytest.raises(ReportExportError, match="readable PDF"):
        build_released_report_export(
            session,
            storage,
            report_id=report.id,
            scope_type="personal_owner",
            scope_id="owner-1",
        )
    assert storage.objects == original_objects

    artifact.presentation_sha256 = hashlib.sha256(paper).hexdigest()
    monkeypatch.setattr(
        "app.services.report_export.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, artifact, paper),
    )
    retried = build_released_report_export(
        session,
        storage,
        report_id=report.id,
        scope_type="personal_owner",
        scope_id="owner-1",
    )
    assert retried.content.startswith(b"%PDF")
    assert storage.objects == original_objects


def test_paginated_appendix_retains_every_heading_and_evidence_span():
    from app.services.report_export import _render_pdf

    citations = []
    for index in range(65):
        citations.append({
            "student_text": f"Selected paper passage {index}.",
            "members": [{
                "coverage_level": "full_text",
                "source": {"raw_reference": f"Reference record {index}."},
                "best_evidence": {"text": (
                    f"Unique evidence start {index}. "
                    + "A complete inspectable source proposition. " * (index % 13 + 1)
                    + f"Unique evidence end {index}."
                )},
            }],
        })
    content, counts = _render_pdf(
        _pdf_bytes(), citations=citations, reference_practice=[],
        annotations=[], export_binding="synthetic-pagination",
    )
    with fitz.open(stream=content, filetype="pdf") as document:
        text = " ".join(" ".join(page.get_text().split()) for page in list(document)[1:])
        assert counts["evidence_appendix_pages"] > 1
        assert len(document.get_toc()) == 67
        for index in range(65):
            assert f"Selected paper passage {index}." in text
            assert f"Unique evidence start {index}." in text
            assert f"Unique evidence end {index}." in text


def test_export_routes_require_authorization_and_return_bound_artifacts(monkeypatch):
    token = "e" * 48
    manifest = {
        "export_sha256": "1" * 64,
        "released_annotation_set_sha256": "2" * 64,
    }
    exported = ReleasedReportExport(
        content=b"%PDF-1.7\nreleased",
        manifest=manifest,
        manifest_sha256="3" * 64,
    )
    view = {
        "title": "Synthetic report",
        "citation_format": "APA",
        "word_counts": {},
        "paper_surface": {
            "page_dimensions": [],
            "citation_anchors": [],
        },
        "overview": {},
        "citations": [],
        "reference_practice": [],
        "role_summaries": {},
        "gauges": [],
        "limits": [],
    }
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    monkeypatch.setattr(
        "app.routers.report.build_released_report_export",
        lambda *args, **kwargs: exported,
    )
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, SimpleNamespace(id="artifact-1"), _pdf_bytes()),
    )
    monkeypatch.setattr(
        "app.routers.report.list_current_paper_annotations",
        lambda *args, **kwargs: [],
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        assert client.get("/report/report-1/export.pdf").status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        pdf = client.get("/report/report-1/export.pdf", headers=headers)
        assert pdf.status_code == 200
        assert pdf.content == exported.content
        assert pdf.headers["x-sourcefidelity-export-sha256"] == "1" * 64
        assert pdf.headers["x-sourcefidelity-manifest-sha256"] == "3" * 64
        assert "attachment" in pdf.headers["content-disposition"]

        details = client.get("/report/report-1/export/manifest", headers=headers)
        assert details.status_code == 200
        assert details.json()["manifest_sha256"] == "3" * 64
        assert details.headers["cache-control"] == "no-store, private"

        printable = client.get("/report/report-1/export/print", headers=headers)
        assert printable.status_code == 200
        assert printable.content == pdf.content
        assert 'inline' in printable.headers['content-disposition']
    finally:
        app.dependency_overrides.clear()
