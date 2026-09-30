"""PDF export acceptance tests: one report, no annotations (owner decision 2026-09-25)."""

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
from app.services.paper_annotations import create_paper_annotation
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


def test_pdf_carries_no_processing_details(monkeypatch):
    import app.services.report_export as module
    original = module._write_evidence_blocks
    observed = []
    def inspect_blocks(blocks, css, position):
        assert not any('class="metrics"' in block for block in blocks)
        result = original(blocks, css, position)
        observed.append(len(result))
        return result
    monkeypatch.setattr(module, '_write_evidence_blocks', inspect_blocks)
    with fitz.open() as document:
        document.new_page()
        counts = module._append_evidence(document, [], [], {})
        assert counts['evidence_appendix_pages'] == observed[0]
        # Processing time, requests and cost are instructor-only HTML details.
        text = ''.join(page.get_text() for page in document)
        assert 'Processing time' not in text and 'Estimated cost' not in text and 'CPU time' not in text


def test_pdf_is_the_one_report_without_audience(export_store, monkeypatch):
    session, storage, report, artifact, view, paper = export_store
    view['summary'] = {'evidence': ['Finding first.', 'Finding second.', 'Finding third.']}
    monkeypatch.setattr('app.services.report_export.project_reference_flags', lambda value, *args: value)
    exported = build_released_report_export(session, storage, report_id=report.id,
        scope_type='personal_owner', scope_id='owner-1')
    assert 'audience' not in exported.manifest
    assert not any(key.startswith('released_annotation') for key in exported.manifest)
    with fitz.open(stream=exported.content, filetype='pdf') as document:
        import unicodedata
        text = ' '.join(unicodedata.normalize('NFKC', ' '.join(page.get_text() for page in document)).split())
    assert 'Finding third.' in text and 'Patterns and issues' in text
    assert 'Student report' not in text and 'Instructor report' not in text


def test_html_export_links_carry_no_audience_and_no_print():
    from app.services.evidence_report import render_evidence_report_html
    view = {'title': 'Export checks', 'citation_format': 'APA', 'citations': [],
        'reference_practice': [], 'summary': {}, 'word_counts': {},
        'overview': {}, 'gauges': [], 'limits': [], 'paper_surface': {},
        'export_action': {'report_href': '/report/test',
            'pdf_href': '/report/test/export.pdf', 'manifest_href': '/report/test/export/manifest'}}
    html = render_evidence_report_html(view, csp_nonce='test-nonce-123456789')
    assert '/report/test/export.pdf"' in html and '/report/test/export.html"' in html
    assert '/report/test/export/manifest"' in html and 'audience=' not in html
    assert 'Preview / print PDF' not in html
    assert '/export/print' not in html


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
        "summary": {},
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


def test_released_export_is_deterministic_hash_bound_and_source_free(export_store, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, 'REPORT_AUTH_MODE', 'institutional_adapter')
    session, storage, report, artifact, _view_data, paper = export_store
    citation_anchor = _anchor("a" * 64)
    # Rows stored before the annotation tools were removed stay untouched and
    # are never exported.
    stored = _create_annotation(session, report, artifact, annotation_type="comment", anchor=citation_anchor,
                                content="STORED-COMMENT-MUST-NOT-EXPORT", visibility="released")

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
    assert {key:first.manifest["overlay_counts"][key] for key in ("citation_underlines", "quotation_differences", "reference_practice_findings")} == {
        "citation_underlines": 1,
        "quotation_differences": 1,
        "reference_practice_findings": 1,
    }
    assert "STORED-COMMENT-MUST-NOT-EXPORT" not in str(first.manifest)
    from app.models.report import PaperAnnotationRecord
    from sqlalchemy import select
    assert session.execute(select(PaperAnnotationRecord).where(
        PaperAnnotationRecord.annotation_id == uuid.UUID(stored["annotation_id"]))).scalars().first() is not None
    assert "SOURCE-CONTENT-MUST-NOT-EXPORT" not in first.content.decode(
        "latin-1", errors="ignore"
    )

    exported = fitz.open(stream=first.content, filetype="pdf")
    try:
        assert exported.page_count > 1
        appendix = "\n".join(page.get_text() for page in list(exported)[1:])
        assert "Inspectable source excerpt." in appendix
        assert "STORED-COMMENT-MUST-NOT-EXPORT" not in appendix
        assert any(link.get("page",0) > 0 for link in exported[0].get_links())
        assert any(link.get("page") == 0 for page in list(exported)[1:] for link in page.get_links())
        assert list(exported[0].annots() or []) == []
        assert len(exported[0].get_drawings()) >= 3  # underline, quotation difference, reference mark
        assert "SOURCE-CONTENT-MUST-NOT-EXPORT" not in exported[0].get_text()
        assert f"report={report.id}" in exported.metadata["subject"]
        original = fitz.open(stream=paper, filetype="pdf")
        try:
            assert exported[0].get_text() == original[0].get_text()
        finally:
            original.close()
    finally:
        exported.close()


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
        export_binding="synthetic-pagination",
    )
    with fitz.open(stream=content, filetype="pdf") as document:
        text = " ".join(" ".join(page.get_text().split()) for page in list(document)[1:])
        assert counts["evidence_appendix_pages"] > 1
        assert len(document.get_toc()) == 132  # Citation and source-specific destinations.
        for index in range(65):
            assert f"Selected paper passage {index}." in text
            assert f"Unique evidence start {index}." in text
            assert f"Unique evidence end {index}." in text


def test_export_routes_require_authorization_and_return_bound_artifacts(monkeypatch):
    token = "e" * 48
    manifest = {
        "export_sha256": "1" * 64,
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
        "summary": {},
        "gauges": [],
        "limits": [],
    }
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        return exported
    monkeypatch.setattr("app.routers.report.build_released_report_export", build)
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, SimpleNamespace(id="artifact-1"), _pdf_bytes()),
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
        # An old audience link still works and gets the one report.
        for old in ("instructor", "student", "invalid"):
            legacy = client.get(f"/report/report-1/export.pdf?audience={old}", headers=headers)
            assert legacy.status_code == 200 and 'sourcefidelity-report.pdf' in legacy.headers['content-disposition']
        assert all('audience' not in call for call in calls)

        details = client.get("/report/report-1/export/manifest", headers=headers)
        assert details.status_code == 200
        assert details.json()["manifest_sha256"] == "3" * 64
        assert details.headers["cache-control"] == "no-store, private"

        printable = client.get("/report/report-1/export/print", headers=headers)
        assert printable.status_code == 404
        assert client.get("/report/report-1/export.pdf?inline=true", headers=headers).headers['content-disposition'].startswith('attachment')
        assert client.get("/report/report-1/export/manifest?audience=instructor", headers=headers).status_code == 200
    finally:
        app.dependency_overrides.clear()
