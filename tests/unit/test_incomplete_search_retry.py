"""Automatic retries of incomplete searches and the "Search Again" button.

Synthetic throughout: no student or source text.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import fitz
import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.models import Base
from app.models.job import Job, JobStage, JobStatus
from app.models.report import Report, ReportPaperArtifactRecord
from app.services import search_retry
from app.services.paper_workflow import TARGETED_SOURCE_REFRESH_KEY, PaperWorkflowError
from app.services.search_retry import (
    AUTO_RETRY_KEY,
    SEARCH_AGAIN_KEY,
    SearchAgainRateLimited,
    due_retry_references,
    prepare_search_again_refresh,
)
from app.tasks import incomplete_search_retry as task_module
from test_report_export import export_store  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _institutional_deployment(monkeypatch):
    """Automatic re-searching happens only in an Institutional deployment
    (owner decision 2026-09-29); these tests exercise that behaviour."""
    import app.services.assessment_marks as marks
    monkeypatch.setattr(marks, "institutional_deployment", lambda: True)


NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
DELAYS = [timedelta(hours=1), timedelta(hours=24), timedelta(hours=72)]
WINDOW = timedelta(hours=24)


def _identity_incomplete(reference_id, searched_at):
    return {"reference_id": reference_id, "status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": {"outcome": "search_incomplete", "created_at": searched_at.isoformat()}}


def _full_text_incomplete(reference_id, searched_at):
    return {"reference_id": reference_id, "status": "abstract_only",
            "reason_code": "full_text_search_incomplete", "full_text_search_incomplete_providers": ["exa"],
            "reference_discovery": {"outcome": "confirmed", "created_at": searched_at.isoformat()}}


def _completed(reference_id, searched_at):
    return {"reference_id": reference_id, "status": "unavailable", "reason_code": "full_text_unavailable",
            "reference_discovery": {"outcome": "unlocated_after_search", "created_at": searched_at.isoformat()}}


@pytest.fixture
def factory():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _add_job(factory, results, *, name="paper", status=JobStatus.COMPLETED, scope_id="owner-1",
             artifact_expires=None, artifact_deleted=False, upload_evidence=None):
    with factory() as session:
        job = Job(filename=f"{name}.pdf", status=status, stage=JobStage.COMPLETED, paper_version_id=name,
                  scope_type="personal_owner", scope_id=scope_id, input_sha256="a" * 64,
                  input_media_type="application/pdf", input_byte_size=100,
                  input_expires_at=NOW + timedelta(days=1), upload_evidence=upload_evidence or {},
                  extraction_payload={"checkpoint": "present"}, source_results=results,
                  verification_summary={"reports_persisted": 0, "report_ids": []})
        session.add(job)
        session.flush()
        report = Report(job_id=job.id, report_json={}, report_version=1)
        session.add(report)
        session.flush()
        session.add(ReportPaperArtifactRecord(
            job_id=job.id, report_id=report.id, paper_version_id=name, scope_type="personal_owner",
            scope_id=scope_id, storage_key="paper.pdf", content_sha256="b" * 64,
            media_type="application/pdf", byte_size=100, artifact_kind="submitted_pdf",
            presentation_status="page_faithful_ready", sanitization_evidence={},
            expires_at=artifact_expires or datetime.now(timezone.utc) + timedelta(days=7),
            deleted_at=NOW if artifact_deleted else None))
        session.commit()
        return job.id, report.id


def _run_task(monkeypatch, factory, now=NOW):
    dispatched = Mock()
    monkeypatch.setattr(task_module, "SessionLocal", factory)
    monkeypatch.setattr(task_module, "dispatch_paper_workflow", dispatched)
    monkeypatch.setattr(task_module, "datetime", SimpleNamespace(now=lambda tz=None: now))
    return task_module.retry_incomplete_searches.run(), dispatched


# --- scheduling -----------------------------------------------------------


@pytest.mark.parametrize("age,attempts,due", [
    (timedelta(minutes=30), 0, False),   # first retry waits an hour
    (timedelta(hours=2), 0, True),
    (timedelta(hours=2), 1, False),      # second retry waits a day after the search it retries
    (timedelta(hours=25), 1, True),
    (timedelta(hours=25), 2, False),     # third waits three days
    (timedelta(hours=73), 2, True),
    (timedelta(hours=100), 3, False),    # at most three automatic retries
    (timedelta(hours=26), 0, False),     # due time passed more than the window ago
])
def test_retry_is_due_by_delay_and_attempt_count(age, attempts, due):
    job = SimpleNamespace(
        source_results=[_identity_incomplete("ref-1", NOW - age)],
        upload_evidence={AUTO_RETRY_KEY: {"references": {"ref-1": {"attempts": [{}] * attempts}}}})
    assert due_retry_references(job, now=NOW, delays=DELAYS, window=WINDOW) == (["ref-1"] if due else [])


def test_completed_searches_and_held_sources_are_never_retried():
    old = NOW - timedelta(hours=2)
    held = {**_identity_incomplete("ref-held", old), "status": "durable_authorized"}
    author = {**_identity_incomplete("ref-author", old), "author_metadata_lookup": {}}
    job = SimpleNamespace(source_results=[_completed("ref-done", old), held, author,
                                          _full_text_incomplete("ref-text", old)], upload_evidence={})
    assert due_retry_references(job, now=NOW, delays=DELAYS, window=WINDOW) == ["ref-text"]


def test_task_groups_due_references_of_one_job_into_one_refresh(monkeypatch, factory):
    monkeypatch.setattr(settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", True)
    results = [_identity_incomplete("ref-1", NOW - timedelta(hours=2)),
               _full_text_incomplete("ref-2", NOW - timedelta(hours=3)),
               _identity_incomplete("ref-later", NOW - timedelta(minutes=10)),
               _completed("ref-done", NOW - timedelta(hours=5))]
    job_id, _ = _add_job(factory, results)

    summary, dispatched = _run_task(monkeypatch, factory)

    assert summary["jobs_requeued"] == 1 and summary["reference_members"] == 2
    dispatched.assert_called_once()
    with factory() as session:
        job = session.get(Job, job_id)
        refresh = job.upload_evidence[TARGETED_SOURCE_REFRESH_KEY]
        assert job.status == JobStatus.RUNNING and job.stage == JobStage.EXTRACTED
        assert refresh["reference_ids"] == ["ref-1", "ref-2"]
        assert refresh["reason"] == "incomplete_search_retry" and refresh["force_search"] is False
        assert dispatched.call_args.args == (str(job_id), refresh["attempt_id"])
        assert sorted(item["reference_id"] for item in job.source_results) == ["ref-done", "ref-later"]
        state = job.upload_evidence[AUTO_RETRY_KEY]
        assert state["runs"] == [{"attempt_id": refresh["attempt_id"], "reference_ids": ["ref-1", "ref-2"],
                                  "started_at": NOW.isoformat()}]
        assert [len(state["references"][r]["attempts"]) for r in ("ref-1", "ref-2")] == [1, 1]
        assert "ref-later" not in state["references"]


def test_task_skips_busy_pending_and_retention_expired_jobs(monkeypatch, factory):
    monkeypatch.setattr(settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", True)
    due = [_identity_incomplete("ref-1", NOW - timedelta(hours=2))]
    running, _ = _add_job(factory, deepcopy(due), name="running", status=JobStatus.RUNNING)
    pending, _ = _add_job(factory, deepcopy(due), name="pending",
                          upload_evidence={TARGETED_SOURCE_REFRESH_KEY: {"reference_ids": ["x"]}})
    expired, _ = _add_job(factory, deepcopy(due), name="expired",
                          artifact_expires=NOW - timedelta(hours=1))
    deleted, _ = _add_job(factory, deepcopy(due), name="deleted", artifact_deleted=True)

    summary, dispatched = _run_task(monkeypatch, factory)

    assert summary["jobs_requeued"] == 0
    dispatched.assert_not_called()
    with factory() as session:
        for job_id in (running, pending, expired, deleted):
            job = session.get(Job, job_id)
            assert [item["reference_id"] for item in job.source_results] == ["ref-1"]
            assert AUTO_RETRY_KEY not in (job.upload_evidence or {})


def test_third_retry_is_the_last(monkeypatch, factory):
    monkeypatch.setattr(settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", True)
    monkeypatch.setattr(settings, "INCOMPLETE_SEARCH_RETRY_WINDOW_HOURS", 0)
    evidence = {AUTO_RETRY_KEY: {"references": {"ref-1": {"attempts": [{}, {}, {}]}}}}
    job_id, _ = _add_job(factory, [_identity_incomplete("ref-1", NOW - timedelta(days=30))],
                         upload_evidence=evidence)
    summary, dispatched = _run_task(monkeypatch, factory)
    assert summary["jobs_requeued"] == 0
    dispatched.assert_not_called()
    with factory() as session:
        assert session.get(Job, job_id).status == JobStatus.COMPLETED


def test_disabled_flag_starts_nothing(monkeypatch, factory):
    monkeypatch.setattr(settings, "PROVIDER_RECOVERY_REFRESH_ENABLED", False)
    _add_job(factory, [_identity_incomplete("ref-1", NOW - timedelta(hours=2))])
    summary, dispatched = _run_task(monkeypatch, factory)
    assert summary["disabled"] == "PROVIDER_RECOVERY_REFRESH_ENABLED"
    dispatched.assert_not_called()


def test_beat_schedule_runs_the_scan():
    from app.tasks.celery_app import celery_app
    entry = celery_app.conf.beat_schedule["retry-incomplete-searches"]
    assert entry["task"] == "retry_incomplete_searches"
    assert entry["schedule"] == max(60, settings.INCOMPLETE_SEARCH_RETRY_SCAN_SECONDS)
    assert settings.INCOMPLETE_SEARCH_RETRY_DELAYS_HOURS == "1,24,72"


# --- Search again: service -------------------------------------------------


def test_search_again_starts_a_forced_refresh_once_per_day(factory):
    job_id, report_id = _add_job(factory, [_full_text_incomplete("ref-1", NOW - timedelta(minutes=5)),
                                           _completed("ref-2", NOW)])
    with factory() as session:
        prepared = prepare_search_again_refresh(session, report_id=report_id, reference_id="ref-1",
                                                scope_type="personal_owner", scope_id="owner-1", now=NOW)
    with factory() as session:
        job = session.get(Job, job_id)
        refresh = job.upload_evidence[TARGETED_SOURCE_REFRESH_KEY]
        assert refresh["reference_ids"] == ["ref-1"] and refresh["force_search"] is True
        assert refresh["reason"] == "user_search_again" and refresh["attempt_id"] == prepared["attempt_id"]
        assert [item["reference_id"] for item in job.source_results] == ["ref-2"]
        assert job.upload_evidence[SEARCH_AGAIN_KEY]["ref-1"][0]["requested_at"] == NOW.isoformat()
        # The refresh finishes (or is rolled back) and the search is still incomplete.
        evidence = dict(job.upload_evidence)
        evidence.pop(TARGETED_SOURCE_REFRESH_KEY)
        job.upload_evidence = evidence
        job.status = JobStatus.COMPLETED
        job.source_results = [_full_text_incomplete("ref-1", NOW + timedelta(minutes=5)), _completed("ref-2", NOW)]
        session.commit()
    with factory() as session, pytest.raises(SearchAgainRateLimited) as refused:
        prepare_search_again_refresh(session, report_id=report_id, reference_id="ref-1",
                                     scope_type="personal_owner", scope_id="owner-1",
                                     now=NOW + timedelta(hours=23))
    assert refused.value.retry_after_seconds == 3600
    with factory() as session:
        again = prepare_search_again_refresh(session, report_id=report_id, reference_id="ref-1",
                                             scope_type="personal_owner", scope_id="owner-1",
                                             now=NOW + timedelta(days=1, minutes=1))
    assert again["attempt_id"] != prepared["attempt_id"]


def test_search_again_refuses_other_scopes_completed_searches_and_busy_jobs(factory):
    _, report_id = _add_job(factory, [_full_text_incomplete("ref-1", NOW), _completed("ref-2", NOW)])
    call = dict(report_id=report_id, scope_type="personal_owner", now=NOW)
    with factory() as session, pytest.raises(PaperWorkflowError) as other_scope:
        prepare_search_again_refresh(session, reference_id="ref-1", scope_id="owner-2", **call)
    assert other_scope.value.code == "report_missing"
    with factory() as session, pytest.raises(PaperWorkflowError) as completed:
        prepare_search_again_refresh(session, reference_id="ref-2", scope_id="owner-1", **call)
    assert completed.value.code == "search_not_incomplete"
    _, busy_report = _add_job(factory, [_full_text_incomplete("ref-1", NOW)], name="busy",
                              status=JobStatus.RUNNING)
    with factory() as session, pytest.raises(PaperWorkflowError) as busy:
        prepare_search_again_refresh(session, reference_id="ref-1", scope_id="owner-1",
                                     **{**call, "report_id": busy_report})
    assert busy.value.code == "report_reanalysis_busy"


# --- Search again: button and route ---------------------------------------


def _member(reference_id, **extra):
    return {"reference_id": reference_id, "status": "source_unavailable", "coverage_level": "unavailable",
            "source": {"author": "Author", "year": "2020", "title": "Synthetic source", "raw_reference": "Author (2020). Synthetic source."},
            "reference_identity": {"status": "confirmed"}, **extra}


def _citations():
    return [{"claim_id": "claim-1", "paper_location": {"localization_level": "semantic_only"}, "members": [
        _member("ref-text", reason_code="full_text_search_incomplete", coverage_level="abstract_only"),
        _member("ref-identity", reason_code="source_not_found",
                reference_identity={"status": "search_incomplete"}),
        _member("ref-done", reason_code="full_text_unavailable"),
        {**_member("ref-full", reference_identity={"status": "search_incomplete"}),
         "status": "evidence_package_persisted", "coverage_level": "full_text"},
    ]}]


def test_button_is_offered_only_for_incomplete_members_in_the_connected_report():
    from app.services.evidence_report import (
        _render_member, enable_authenticated_paper_actions, _render_search_again)
    view = {"paper_surface": {}, "citations": _citations()}
    members = enable_authenticated_paper_actions(view, report_id="report-1")["citations"][0]["members"]
    actions = {m["reference_id"]: m.get("search_again_action") for m in members}
    assert actions["ref-text"]["href"] == "/report/report-1/reference/ref-text/search-again"
    assert actions["ref-identity"]["enabled"] is True
    assert actions["ref-done"] is None and actions["ref-full"] is None
    soup = BeautifulSoup(_render_member(members[0]), "html.parser")
    form = soup.select_one("form.search-again")
    assert form["method"] == "post" and form["action"].endswith("/ref-text/search-again")
    assert form.button.get_text() == "Search Again" and form.button["type"] == "submit"
    assert form.select_one(".upload-status").get_text() == ""
    # The Reference N window showing the same member offers the same button.
    from app.services.evidence_report import _render_reference_window_template
    entry = {"number": 1, "template_id": "reference-entry-panel-1", "source": members[0]["source"],
             "member": members[0], "first_citation": (1, 0), "citation_numbers": [1]}
    template = BeautifulSoup(_render_reference_window_template(entry, [], [{"members": members}]),
                             "html.parser").template
    window = BeautifulSoup(template.decode_contents(), "html.parser")
    assert [b.get_text() for b in window.select("form.search-again button")] == ["Search Again"]
    # The persisted (export) projection carries no action, so no button.
    assert "search-again" not in _render_member(view["citations"][0]["members"][0])
    assert _render_search_again({}) == ""


def test_exports_never_contain_the_button(export_store, monkeypatch):  # noqa: F811
    import hashlib
    from app.services.interactive_report_export import build_interactive_report_html
    from app.services.report_export import build_released_report_export
    session, storage, report, _artifact, view, paper = export_store
    member = view["citations"][0]["members"][0]
    member.update(status="source_unavailable", coverage_level="unavailable",
                  reason_code="full_text_search_incomplete",
                  search_again_action={"enabled": True, "href": "/report/x/reference/r/search-again"})
    portable = deepcopy(view)
    portable["reference_practice"] = []
    portable["paper_surface"]["presentation_sha256"] = hashlib.sha256(paper).hexdigest()
    html = build_interactive_report_html(portable, paper).decode()
    assert "Search Again" not in html and "search-again\"" not in html
    monkeypatch.setattr("app.services.report_export.project_reference_flags", lambda value, *a: value)
    exported = build_released_report_export(session, storage, report_id=report.id,
                                            scope_type="personal_owner", scope_id="owner-1")
    with fitz.open(stream=exported.content, filetype="pdf") as document:
        assert "Search Again" not in " ".join(page.get_text() for page in document)


@pytest.fixture
def client(monkeypatch):
    from app.database import get_db
    from app.main import app
    from app.services.storage.backend import get_storage_backend
    token = "s" * 48
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_bearer")
    monkeypatch.setattr(settings, "REPORT_PERSONAL_ACCESS_TOKEN", SecretStr(token))
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "owner-1")
    view = {"paper_surface": {}, "citations": _citations()}
    monkeypatch.setattr("app.routers.report.load_authorized_evidence_report_bundle",
                        lambda *a, **k: (view, SimpleNamespace(id="artifact-1"), b""))
    app.dependency_overrides[get_db] = lambda: object()
    app.dependency_overrides[get_storage_backend] = lambda: object()
    headers = {"Authorization": f"Bearer {token}", "Origin": "http://testserver",
               "Sec-Fetch-Site": "same-origin"}
    try:
        yield TestClient(app), headers
    finally:
        app.dependency_overrides.clear()


def test_route_starts_a_forced_refresh_and_follows_the_successor(monkeypatch, client):
    test_client, headers = client
    captured = {}

    def prepare(_session, **kwargs):
        captured.update(kwargs)
        return {"job_id": "job-1", "attempt_id": "attempt-1", "report_id": "report-1", "reference_id": "ref-text"}

    dispatched = Mock(return_value="task-1")
    monkeypatch.setattr("app.routers.report.prepare_search_again_refresh", prepare)
    monkeypatch.setattr("app.routers.report.dispatch_paper_workflow", dispatched)
    response = test_client.post("/report/report-1/reference/ref-text/search-again", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"reanalysis_status": "scheduled", "base_report_id": "report-1",
                               "attempt_id": "attempt-1", "task_id": "task-1", "publication_pending": False,
                               "status_url": "/report/report-1/successor"}
    assert captured == {"report_id": "report-1", "reference_id": "ref-text",
                        "scope_type": "personal_owner", "scope_id": "owner-1"}
    dispatched.assert_called_once_with("job-1", "attempt-1")


def test_route_refuses_repeat_cross_site_and_complete_searches(monkeypatch, client):
    test_client, headers = client

    def limited(_session, **_kwargs):
        raise SearchAgainRateLimited(1800)

    monkeypatch.setattr("app.routers.report.prepare_search_again_refresh", limited)
    monkeypatch.setattr("app.routers.report.dispatch_paper_workflow", Mock())
    repeat = test_client.post("/report/report-1/reference/ref-identity/search-again", headers=headers)
    assert repeat.status_code == 429 and repeat.headers["retry-after"] == "1800"
    cross = test_client.post("/report/report-1/reference/ref-text/search-again",
                             headers={**headers, "Origin": "https://attacker.invalid",
                                      "Sec-Fetch-Site": "cross-site"})
    assert cross.status_code == 403
    assert test_client.post("/report/report-1/reference/ref-done/search-again",
                            headers=headers).status_code == 409
    assert test_client.post("/report/report-1/reference/ref-missing/search-again",
                            headers=headers).status_code == 404


def test_route_refuses_a_report_outside_the_viewer_scope(monkeypatch, client):
    from app.services.evidence_report import EvidenceReportAuthorizationError
    test_client, headers = client

    def outside(*_a, **_k):
        raise EvidenceReportAuthorizationError("Report does not exist in the authorized scope")

    prepare = Mock()
    monkeypatch.setattr("app.routers.report.load_authorized_evidence_report_bundle", outside)
    monkeypatch.setattr("app.routers.report.prepare_search_again_refresh", prepare)
    response = test_client.post(f"/report/{uuid.uuid4()}/reference/ref-text/search-again", headers=headers)
    assert response.status_code == 404
    prepare.assert_not_called()


def test_page_script_follows_only_its_own_search_attempt():
    from pathlib import Path
    script = Path("app/services/report_interactions.js").read_text()
    handler = script[script.index("classList.contains('search-again')"):script.index("// view-state:start")]
    # Only the owner-approved messages (2026-09-29) are written.
    assert "Searching again…" in handler
    assert "awaitSuccessor(data.status_url,message,String(data.attempt_id))" in handler
    assert "failure.attempt_id===attemptId" in script


def test_search_again_messages_use_the_owner_wording():
    from pathlib import Path
    script = Path("app/services/report_interactions.js").read_text()
    for text in ("Searching again…", "This reference was already searched again today. Try again tomorrow.",
                 "The new search could not update the report.", "The report is already being updated."):
        assert text in script


def test_personal_deployment_never_searches_again_automatically(monkeypatch):
    # Owner decision 2026-09-29: in Personal, searching again is the user's choice.
    import app.services.assessment_marks as marks
    from app.tasks.incomplete_search_retry import retry_incomplete_searches
    from app.tasks.provider_recovery import requeue_recovered_provider_work
    monkeypatch.setattr(marks, "institutional_deployment", lambda: False)
    assert retry_incomplete_searches()["disabled"] == "personal_deployment"
    assert requeue_recovered_provider_work("brave")["disabled"] == "personal_deployment"
