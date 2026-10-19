"""Report export tests. The PDF report and the verification record were dropped
(owner decision 2026-10-19); the interactive HTML export remains."""

from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace

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


def test_html_export_links_carry_no_audience_and_no_print():
    from app.services.evidence_report import render_evidence_report_html
    view = {'title': 'Export checks', 'citation_format': 'APA', 'citations': [],
        'reference_practice': [], 'summary': {}, 'word_counts': {},
        'overview': {}, 'gauges': [], 'limits': [], 'paper_surface': {},
        'export_action': {'report_href': '/report/test'}}
    html = render_evidence_report_html(view, csp_nonce='test-nonce-123456789')
    # The PDF report was dropped (owner decision 2026-10-19).
    assert 'export.pdf' not in html and 'Download PDF' not in html and '/report/test/export.html"' in html
    # The verification record (JSON) was dropped too (owner decision 2026-10-19).
    assert 'export/manifest' not in html and 'Verification Record' not in html
    assert 'Technical Export Record' not in html and 'audience=' not in html
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
        yield session, storage, report, artifact, view, paper


def test_dropped_pdf_and_record_routes_are_gone(monkeypatch):
    token = "e" * 48
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
    monkeypatch.setattr(
        "app.routers.report.load_authorized_evidence_report_bundle",
        lambda *args, **kwargs: (view, SimpleNamespace(id="artifact-1"), _pdf_bytes()),
    )
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    try:
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {token}"}
        # The PDF report and the verification record were dropped (owner decision 2026-10-19).
        assert client.get("/report/report-1/export.pdf", headers=headers).status_code == 404
        assert client.get("/report/report-1/export/manifest", headers=headers).status_code == 404
        assert client.get("/report/report-1/export/print", headers=headers).status_code == 404
    finally:
        app.dependency_overrides.clear()
