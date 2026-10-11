"""An unchanged paper returns its existing report (owner decision 2026-10-09)."""
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.report import Report, ReportPaperArtifactRecord
from app.services import paper_reuse
from app.services.paper_reuse import FINGERPRINT_KEY, REUSE_KEY, find_reusable_report, record_reuse

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
CONTENT = b"%PDF-1.7 harbour cranes in river ports"
CONFIG = {"version": "assessment-configuration-v1", "require_reference_links": False}


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(paper_reuse, "pipeline_fingerprint", lambda: "f" * 64)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as database_session:
        yield database_session


def _job(session, *, content=CONTENT, fingerprint="f" * 64, status=JobStatus.COMPLETED, config=CONFIG,
         updated=NOW - timedelta(days=1), copy_expires=NOW + timedelta(days=3), filename="ports.pdf"):
    job = Job(filename=filename, status=status, stage=JobStage.COMPLETED, paper_version_id="paper:x",
              scope_type="personal_owner", scope_id="owner-1", input_sha256=hashlib.sha256(content).hexdigest(),
              input_media_type="application/pdf", input_byte_size=len(content), input_expires_at=NOW,
              upload_evidence={"assessment_configuration": config}, updated_at=updated)
    session.add(job)
    session.flush()
    report = Report(job_id=job.id, report_json={FINGERPRINT_KEY: fingerprint}, report_version=1)
    session.add(report)
    session.flush()
    session.add(ReportPaperArtifactRecord(
        job_id=job.id, report_id=report.id, paper_version_id="paper:x", scope_type="personal_owner",
        scope_id="owner-1", storage_key="paper.pdf", content_sha256="b" * 64, media_type="application/pdf",
        byte_size=100, artifact_kind="submitted_pdf", presentation_status="page_faithful_ready",
        sanitization_evidence={}, expires_at=copy_expires))
    session.commit()
    return job, report


def _find(session, **changes):
    args = dict(content=CONTENT, filename="ports.pdf", scope_type="personal_owner", scope_id="owner-1",
                assessment_configuration=CONFIG, assessment_id=None, now=NOW)
    return find_reusable_report(session, **{**args, **changes})


def test_the_same_file_under_the_same_rules_gets_its_report_back(session):
    job, report = _job(session)
    assert _find(session) == (job, report)
    record_reuse(session, job, report, now=NOW)
    assert job.upload_evidence[REUSE_KEY][-1]["report_id"] == str(report.id)
    copy = session.query(ReportPaperArtifactRecord).one()
    assert copy.expires_at.replace(tzinfo=timezone.utc) > NOW + timedelta(days=6)


@pytest.mark.parametrize("change", [
    {"content": b"%PDF-1.7 a revised draft"},
    {"filename": "other.pdf"},
    {"scope_id": "owner-2"},
    {"assessment_configuration": {**CONFIG, "require_reference_links": True}},
    {"assessment_id": "course-1"},
])
def test_any_difference_in_the_submission_checks_it_again(session, change):
    _job(session)
    assert _find(session, **change) is None


@pytest.mark.parametrize("change", [
    {"fingerprint": "e" * 64},
    {"status": JobStatus.FAILED},
    {"updated": NOW - timedelta(days=31)},
    {"copy_expires": NOW - timedelta(minutes=1)},
])
def test_changed_rules_a_failed_or_old_run_or_an_expired_copy_checks_it_again(session, change):
    _job(session, **change)
    assert _find(session) is None


def test_the_upload_route_returns_the_existing_report_without_a_new_run(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import check
    from app.security import PAPER_CHECK_CAPABILITY, AuthenticatedPrincipal
    app = FastAPI()
    app.include_router(check.router, prefix="/check")
    app.dependency_overrides[check.get_authenticated_principal] = lambda: AuthenticatedPrincipal(
        provider="test", subject="owner", scope_type="personal_owner", scope_id="owner-1",
        capabilities=frozenset({PAPER_CHECK_CAPABILITY}))
    app.dependency_overrides[check.get_db] = lambda: Mock()
    app.dependency_overrides[check.get_storage_backend] = lambda: Mock()
    job = SimpleNamespace(id="job-1", paper_version_id="paper", status="completed", stage="completed",
                          store_only=False, upload_evidence={"assessment_configuration": CONFIG})
    found = Mock(return_value=(job, SimpleNamespace(id="report-1")))
    monkeypatch.setattr(paper_reuse, "find_reusable_report", found)
    monkeypatch.setattr(paper_reuse, "record_reuse", Mock())
    create = Mock()
    monkeypatch.setattr(check, "create_paper_job", create)
    with TestClient(app) as client:
        response = client.post("/check/", headers={"origin": "http://testserver"},
                               files={"file": ("ports.pdf", CONTENT, "application/pdf")})
    assert response.status_code == 202
    assert response.json()["job_id"] == "job-1" and response.json()["report_id"] == "report-1"
    assert response.json()["reused_existing_report"] is True
    assert not create.called
    assert found.call_args.kwargs["content"] == CONTENT and found.call_args.kwargs["scope_id"] == "owner-1"


def test_a_run_excluded_from_reuse_checks_it_again(session, monkeypatch):
    job, _ = _job(session)
    monkeypatch.setattr(paper_reuse.settings, "SEARCH_REUSE_EXCLUDED_JOBS", f" other , {str(job.id).upper()}")
    assert _find(session) is None
