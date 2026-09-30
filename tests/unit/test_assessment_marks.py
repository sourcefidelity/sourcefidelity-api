"""Assessment-level "marks released" setting (owner decision 2026-09-29).

Synthetic throughout: no student or source text.
"""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select

from app.config import settings
from app.models import Base
from app.models.assessment_marks import AssessmentMarksRelease
from app.models.job import Job, JobStatus
from app.models.judgment import JudgmentSourceReserve
from app.models.report import VerificationReportRecord
from app.security import (
    ASSESSMENT_MARKS_RELEASE_CAPABILITY,
    AuthenticatedPrincipal,
    REPORT_VIEW_CAPABILITY,
    _personal_principal,
    _report_session_principal,
)
from app.services.assessment_marks import (
    AssessmentMarksError,
    clear_marks_released,
    job_marks_released,
    normalize_assessment_id,
    set_marks_released,
)
from app.services.paper_workflow import TARGETED_SOURCE_REFRESH_KEY, PaperWorkflowError
from app.services.search_retry import (
    AUTO_RETRY_KEY,
    job_accepts_search_refresh,
    prepare_search_again_refresh,
)
from test_incomplete_search_retry import (  # noqa: F401  (fixture)
    NOW,
    _add_job,
    _citations,
    _full_text_incomplete,
    _identity_incomplete,
    _run_task,
    factory,
)

TOKEN = "s" * 48
OWNER = dict(scope_type="personal_owner", scope_id="owner-1")


@pytest.fixture(autouse=True)
def _institutional(monkeypatch):
    """Marks release exists only in an Institutional deployment (owner decision
    2026-09-29); these tests exercise that behaviour."""
    import app.services.assessment_marks as marks
    monkeypatch.setattr(marks, "institutional_deployment", lambda: True)


def _job(factory, results, *, assessment_id=None, scope_id="owner-1", name="paper"):
    job_id, report_id = _add_job(factory, results, name=name, scope_id=scope_id)
    with factory() as session:
        session.get(Job, job_id).assessment_id = assessment_id
        session.commit()
    return job_id, report_id


def _release(factory, assessment_id="A1", scope_id="owner-1", **kwargs):
    with factory() as session:
        record, purged = set_marks_released(
            session, scope_type="personal_owner", scope_id=scope_id, assessment_id=assessment_id,
            released_by="personal-owner", now=NOW, **kwargs)
        return record, purged


# --- the record -------------------------------------------------------------


def test_set_is_idempotent_scoped_and_clearable(factory):
    _release(factory)
    with factory() as session:
        again, _ = set_marks_released(session, **OWNER, assessment_id=" A1 ", released_by="lms-bot",
                                      source="lms", now=NOW + timedelta(days=1))
        assert again.marks_released_at.replace(tzinfo=None) == NOW.replace(tzinfo=None)
        assert again.released_by == "personal-owner" and again.release_source == "manual"
    with factory() as session:
        assert len(session.scalars(select(AssessmentMarksRelease)).all()) == 1
    released, _ = _job(factory, [], assessment_id="A1")
    other_scope, _ = _job(factory, [], assessment_id="A1", scope_id="owner-2", name="other-scope")
    other_assessment, _ = _job(factory, [], assessment_id="A2", name="other-assessment")
    unassigned, _ = _job(factory, [], name="unassigned")
    with factory() as session:
        states = {job_id: job_marks_released(session, session.get(Job, job_id))
                  for job_id in (released, other_scope, other_assessment, unassigned)}
    assert states == {released: True, other_scope: False, other_assessment: False, unassigned: False}
    with factory() as session:
        cleared = clear_marks_released(session, **OWNER, assessment_id="A1", cleared_by="personal-owner",
                                       now=NOW + timedelta(hours=1))
        assert cleared.marks_released_at is None and cleared.cleared_by == "personal-owner"
        assert not job_marks_released(session, session.get(Job, released))
    with factory() as session:  # an LMS release sets the same record
        record, _ = set_marks_released(session, **OWNER, assessment_id="A1", released_by="lms",
                                       source="lms", now=NOW + timedelta(hours=2))
        assert record.release_source == "lms" and job_marks_released(session, session.get(Job, released))


@pytest.mark.parametrize("value", ["", "   ", "a\nb", "x" * 256, 5])
def test_invalid_assessment_identifiers_are_refused(factory, value):
    with factory() as session, pytest.raises(AssessmentMarksError):
        set_marks_released(session, **OWNER, assessment_id=value, released_by="owner")
    with factory() as session, pytest.raises(AssessmentMarksError):
        set_marks_released(session, **OWNER, assessment_id="A1", released_by="owner", source="email")


def test_upload_records_the_optional_assessment(monkeypatch):
    from sqlalchemy.orm import Session
    from app.services import paper_upload
    from app.services.file_safety import SafetyVerdict
    from app.services.paper_upload import DOCX_MEDIA_TYPE, PaperUploadError, create_paper_job
    from test_paper_upload import MemoryStorage, _docx_bytes
    monkeypatch.setattr(paper_upload, "scan_with_clamd", lambda _c: (SafetyVerdict.CLEAN, "stream: OK"))
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        call = dict(content=_docx_bytes(), filename="paper.docx", media_type=DOCX_MEDIA_TYPE,
                    scope_id="owner-1")
        assert create_paper_job(session, MemoryStorage(), assessment_id=" A1 ", **call).assessment_id == "A1"
        assert create_paper_job(session, MemoryStorage(), **call).assessment_id is None
        with pytest.raises(PaperUploadError) as refused:
            create_paper_job(session, MemoryStorage(), assessment_id="a\x00b", **call)
        assert refused.value.code == "assessment_id_invalid"
    assert normalize_assessment_id(None) is None and normalize_assessment_id("  ") is None


# --- automatic retries ------------------------------------------------------


def test_retries_skip_released_assessments_only(monkeypatch, factory):
    monkeypatch.setattr(settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", True)
    due = [_identity_incomplete("ref-1", NOW - timedelta(hours=2))]
    released, _ = _job(factory, deepcopy(due), assessment_id="A1", name="released")
    unassigned, _ = _job(factory, deepcopy(due), name="unassigned")
    other_scope, _ = _job(factory, deepcopy(due), assessment_id="A1", scope_id="owner-2", name="other-scope")
    _release(factory)

    summary, dispatched = _run_task(monkeypatch, factory)

    assert summary["jobs_requeued"] == 2 and dispatched.call_count == 2
    with factory() as session:
        job = session.get(Job, released)
        assert job.status == JobStatus.COMPLETED and AUTO_RETRY_KEY not in job.upload_evidence
        assert not job_accepts_search_refresh(session, job, now=NOW)
        for job_id in (unassigned, other_scope):
            assert TARGETED_SOURCE_REFRESH_KEY in session.get(Job, job_id).upload_evidence


# --- Search again -----------------------------------------------------------


def test_search_again_is_refused_after_release(factory):
    _, report_id = _job(factory, [_full_text_incomplete("ref-1", NOW)], assessment_id="A1")
    _, unassigned_report = _job(factory, [_full_text_incomplete("ref-1", NOW)], name="unassigned")
    _release(factory)
    with factory() as session, pytest.raises(PaperWorkflowError) as refused:
        prepare_search_again_refresh(session, report_id=report_id, reference_id="ref-1", **OWNER, now=NOW)
    assert refused.value.code == "assessment_marks_released"
    with factory() as session:
        prepared = prepare_search_again_refresh(session, report_id=unassigned_report, reference_id="ref-1",
                                                **OWNER, now=NOW)
    assert prepared["reference_id"] == "ref-1"


def test_button_is_not_attached_after_release(factory):
    from app.routers.report import _report_marks_released
    from app.services.evidence_report import enable_authenticated_paper_actions
    view = {"paper_surface": {}, "citations": _citations()}
    members = enable_authenticated_paper_actions(view, report_id="r", search_again_enabled=False)[
        "citations"][0]["members"]
    assert all("search_again_action" not in member for member in members)
    _, report_id = _job(factory, [], assessment_id="A1")
    _, unassigned_report = _job(factory, [], name="unassigned")
    principal = SimpleNamespace(**OWNER)
    with factory() as session:
        assert not _report_marks_released(session, str(report_id), principal)
    _release(factory)
    with factory() as session:
        assert _report_marks_released(session, str(report_id), principal)
        assert not _report_marks_released(session, str(unassigned_report), principal)
        assert not _report_marks_released(session, str(report_id), SimpleNamespace(
            scope_type="personal_owner", scope_id="owner-2"))


@pytest.fixture
def api(monkeypatch, factory):
    from app.database import get_db
    from app.main import app
    from app.services.storage.backend import get_storage_backend
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(TOKEN))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")

    def db():
        with factory() as session:
            yield session

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_storage_backend] = lambda: object()
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": "http://testserver",
               "Sec-Fetch-Site": "same-origin"}
    # Stand-in for an Institutional administrator (the manual fallback until Moodle).
    from app.security import get_authenticated_principal
    base = _personal_principal()
    admin = AuthenticatedPrincipal(provider=base.provider, subject=base.subject, scope_type=base.scope_type,
                                   scope_id=base.scope_id,
                                   capabilities=base.capabilities | {ASSESSMENT_MARKS_RELEASE_CAPABILITY})
    app.dependency_overrides[get_authenticated_principal] = lambda: admin
    try:
        yield TestClient(app), headers, app
    finally:
        app.dependency_overrides.clear()


def test_report_page_and_route_refuse_search_again_after_release(monkeypatch, factory, api):
    client, headers, _app = api
    _, report_id = _job(factory, [_full_text_incomplete("ref-text", NOW)], assessment_id="A1")
    view = {"paper_surface": {}, "citations": _citations()}
    monkeypatch.setattr("app.routers.report.load_authorized_evidence_report_bundle",
                        lambda *a, **k: (view, SimpleNamespace(id="artifact-1"), b""))
    seen = {}

    def capture(_view, **kwargs):
        seen.update(kwargs)
        raise HTTPException(status_code=418)

    monkeypatch.setattr("app.routers.report.enable_authenticated_paper_actions", capture)
    assert client.get(f"/report/{report_id}", headers=headers).status_code == 418
    assert seen["search_again_enabled"] is True
    assert client.post("/assessments/A1/marks-released", headers=headers).status_code == 200
    client.get(f"/report/{report_id}", headers=headers)
    assert seen["search_again_enabled"] is False
    response = client.post(f"/report/{report_id}/reference/ref-text/search-again", headers=headers)
    # 410: the page removes the button, no message (owner decision 2026-09-29).
    assert response.status_code == 410 and response.json()["detail"] == "assessment_marks_released"


# --- endpoints --------------------------------------------------------------


def test_endpoints_set_read_and_clear_in_the_caller_scope(factory, api):
    client, headers, _app = api
    with factory() as session:  # the same identifier in another scope stays released
        set_marks_released(session, scope_type="personal_owner", scope_id="owner-2", assessment_id="A1",
                           released_by="someone", now=NOW)
    url = "/assessments/A1/marks-released"
    assert client.get(url, headers=headers).json()["marks_released"] is False
    body = client.post(url, headers=headers).json()
    assert body["marks_released"] is True and body["release_source"] == "manual"
    assert body["released_by"] == "personal-owner" and body["judgment_reserves_purged"] == 0
    assert client.get(url, headers=headers).json()["marks_released"] is True
    assert client.delete(url, headers=headers).json()["marks_released"] is False
    assert client.get(url, headers=headers).json()["marks_released"] is False
    with factory() as session:
        rows = {(r.scope_id, r.marks_released_at is not None)
                for r in session.scalars(select(AssessmentMarksRelease))}
    assert rows == {("owner-1", False), ("owner-2", True)}


def test_endpoints_require_the_capability_and_same_origin(api):
    from app.security import get_authenticated_principal
    client, headers, app = api
    url = "/assessments/A1/marks-released"
    assert client.post(url, headers={**headers, "Origin": "https://elsewhere.invalid",
                                     "Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.delete(url, headers={**headers, "Sec-Fetch-Site": "cross-site"}).status_code == 403
    app.dependency_overrides.pop(get_authenticated_principal)
    assert client.post(url).status_code == 401
    # No marks release in a Personal deployment (owner decision 2026-09-29).
    assert ASSESSMENT_MARKS_RELEASE_CAPABILITY not in _personal_principal().capabilities
    # The report browser session never carries it.
    assert ASSESSMENT_MARKS_RELEASE_CAPABILITY not in _report_session_principal().capabilities
    app.dependency_overrides[get_authenticated_principal] = lambda: AuthenticatedPrincipal(
        provider="test", subject="viewer", **OWNER, capabilities=frozenset({REPORT_VIEW_CAPABILITY}))
    for method in (client.get, client.post, client.delete):
        assert method(url, headers=headers).status_code == 403


# --- Judgment reserve kept until grades are released -------------------------


def _reserve(session, paper_version_id, retention, scope_id="owner-1"):
    record = VerificationReportRecord(
        verification_id=f"v-{uuid.uuid4()}", paper_version_id=paper_version_id, scope_type="personal_owner",
        scope_id=scope_id, report_version=1, artifact_version="x", verdict="x", evidence_sha256="c" * 64,
        report_payload={})
    session.add(record)
    session.flush()
    session.add(JudgmentSourceReserve(verification_report_id=record.id, scope_type="personal_owner",
                                      scope_id=scope_id, reserve_version="judgment-reserve-v1",
                                      retention_policy=retention, payload={}))
    session.flush()
    return record.id


def test_release_purges_only_reserves_kept_until_grades_released(factory):
    _job(factory, [], assessment_id="A1", name="released")
    _job(factory, [], name="unassigned")
    _job(factory, [], assessment_id="A1", scope_id="owner-2", name="other-scope")
    with factory() as session:
        purged_id = _reserve(session, "released", "until_grades_released")
        kept = {_reserve(session, "released", "paper_retention"),
                _reserve(session, "unassigned", "until_grades_released"),
                _reserve(session, "other-scope", "until_grades_released", scope_id="owner-2")}
        session.commit()
    _, purged = _release(factory)
    assert purged == 1
    with factory() as session:
        remaining = set(session.scalars(select(JudgmentSourceReserve.verification_report_id)))
    assert purged_id not in remaining and kept <= remaining


def test_no_reserve_kept_until_release_is_stored_after_release(factory):
    from app.services.judgment_reserve import store_reserve
    _job(factory, [], assessment_id="A1", name="released")
    _job(factory, [], name="unassigned")
    _release(factory)
    reserve = {"reserve_version": "judgment-reserve-v1", "retention": "until_grades_released"}
    with factory() as session:
        for paper in ("released", "unassigned"):
            record = VerificationReportRecord(
                verification_id=f"v-{paper}", paper_version_id=paper, **OWNER, report_version=1,
                artifact_version="x", verdict="x", evidence_sha256="c" * 64, report_payload={})
            session.add(record)
            session.flush()
            store_reserve(session, record, reserve)
        session.commit()
        stored = session.scalars(select(VerificationReportRecord.paper_version_id).join(
            JudgmentSourceReserve,
            JudgmentSourceReserve.verification_report_id == VerificationReportRecord.id)).all()
    assert stored == ["unassigned"]


def test_migration_roundtrip_isolated_database():
    from importlib import import_module
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect
    migration = import_module("app.alembic.versions.f3c9a1d7b852_assessment_marks_released")
    assert migration.down_revision == "e2b7c4d9a613"
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Job.__table__])
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP INDEX ix_jobs_assessment_id")
        connection.exec_driver_sql("ALTER TABLE jobs DROP COLUMN assessment_id")
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            inspector = inspect(connection)
            assert {c["name"] for c in inspector.get_columns("assessment_marks_releases")} == set(
                AssessmentMarksRelease.__table__.columns.keys())
            assert "assessment_id" in {c["name"] for c in inspector.get_columns("jobs")}
            migration.downgrade()
            assert "assessment_marks_releases" not in inspect(connection).get_table_names()
            assert "assessment_id" not in {c["name"] for c in inspect(connection).get_columns("jobs")}
    engine.dispose()


def test_check_route_passes_the_optional_assessment(monkeypatch):
    from unittest.mock import Mock
    from fastapi import FastAPI
    from app.routers import check
    from app.security import PAPER_CHECK_CAPABILITY
    app = FastAPI()
    app.include_router(check.router, prefix="/check")
    app.dependency_overrides[check.get_authenticated_principal] = lambda: AuthenticatedPrincipal(
        provider="test", subject="owner", **OWNER, capabilities=frozenset({PAPER_CHECK_CAPABILITY}))
    app.dependency_overrides[check.get_db] = lambda: Mock()
    app.dependency_overrides[check.get_storage_backend] = lambda: Mock()
    create = Mock(return_value=SimpleNamespace(id="job", paper_version_id="paper", status="pending",
                                               stage="uploaded", store_only=False, upload_evidence={}))
    monkeypatch.setattr(check, "create_paper_job", create)
    monkeypatch.setattr(check.check_paper_task, "delay", Mock(return_value=SimpleNamespace(id="task")))
    with TestClient(app) as client:
        for data, expected in (({"assessment_id": "A1"}, "A1"), ({}, None)):
            response = client.post("/check/", data=data, headers={"origin": "http://testserver"},
                                   files={"file": ("paper.pdf", b"fixture", "application/pdf")})
            assert response.status_code == 202
            assert create.call_args.kwargs["assessment_id"] == expected



def test_personal_deployment_has_no_marks_release_and_keeps_search_again(monkeypatch, factory):
    import app.services.assessment_marks as marks
    monkeypatch.undo()
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    assert marks.institutional_deployment() is False
    job_id, _ = _job(factory, [_full_text_incomplete("ref-text", NOW)], assessment_id="A1")
    with factory() as session:
        set_marks_released(session, scope_type="personal_owner", scope_id="owner-1", assessment_id="A1",
                           released_by="someone", now=NOW)
        assert job_marks_released(session, session.get(Job, job_id)) is False



def test_page_removes_the_button_when_marks_were_released_meanwhile():
    from pathlib import Path
    script = Path("app/services/report_interactions.js").read_text()
    assert "if(response.status===410){form.remove();return;}" in script
