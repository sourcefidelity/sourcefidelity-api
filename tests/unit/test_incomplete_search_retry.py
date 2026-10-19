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
    due_retry_references,
)
from app.tasks import incomplete_search_retry as task_module
from test_report_export import export_store  # noqa: F401  (fixture)




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


def test_no_search_again_button_is_offered():
    # Owner decision 2026-10-18: searches complete on their own; no button.
    from app.services.evidence_report import (
        _render_member, enable_authenticated_paper_actions, _render_reference_window_template)
    view = {"paper_surface": {}, "citations": _citations()}
    members = enable_authenticated_paper_actions(view, report_id="report-1")["citations"][0]["members"]
    assert all(m.get("search_again_action") is None for m in members)
    assert "search-again" not in _render_member(members[0])
    entry = {"number": 1, "template_id": "reference-entry-panel-1", "source": members[0]["source"],
             "member": members[0], "first_citation": (1, 0), "citation_numbers": [1]}
    assert "search-again" not in _render_reference_window_template(entry, [], [{"members": members}])


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


def test_personal_deployment_also_completes_searches_on_its_own(monkeypatch):
    # Owner decision 2026-10-18 (was the user's choice in Personal, 2026-09-29).
    from app.tasks.incomplete_search_retry import retry_incomplete_searches
    from app.tasks.provider_recovery import requeue_recovered_provider_work
    monkeypatch.setattr("app.tasks.incomplete_search_retry.settings.PROVIDER_RECOVERY_REFRESH_ENABLED", False)
    monkeypatch.setattr("app.tasks.provider_recovery.settings.PROVIDER_RECOVERY_REFRESH_ENABLED", False)
    # Only the one deployment switch stops them now, never the profile.
    assert retry_incomplete_searches()["disabled"] == "PROVIDER_RECOVERY_REFRESH_ENABLED"
    assert requeue_recovered_provider_work("brave")["disabled"] == "PROVIDER_RECOVERY_REFRESH_ENABLED"


def test_personal_automatic_searching_is_the_owners_setting(factory):
    from app.services.storage_policy import finishes_searches_automatically, set_policy
    with factory() as session:
        assert finishes_searches_automatically(session, "personal_owner", "owner-1") is False
        set_policy(session, "personal_owner", "owner-1", {"searches": {"finish_incomplete_automatically": True}})
        session.commit()
        assert finishes_searches_automatically(session, "personal_owner", "owner-1") is True


def test_the_settings_page_offers_the_search_line():
    from app.routers.storage_settings import _changes, _page
    from app.services.storage_policy import deployment_defaults
    page = _page(deployment_defaults()).body.decode()
    assert "<legend>Searches</legend>" in page and "Finish incomplete searches automatically" in page
    assert 'value="off" checked' in page.split("Finish incomplete searches automatically")[1]
    form = {"keep_copies": "all", "keep_months": "0", "keep_submitted_file": "off",
            "paper_copy_days": "7", "report_days_until_deleted": "true", "finish_incomplete_automatically": "on"}
    assert _changes(form)["searches"] == {"finish_incomplete_automatically": True}
    assert "searches" not in _changes({k: v for k, v in form.items() if k != "finish_incomplete_automatically"})


@pytest.fixture(autouse=True)
def _automatic_searching_on(monkeypatch, request):
    # These papers' scopes finish searches on their own; the Personal default
    # (off, owner decision 2026-10-18) is tested on its own.
    if "owners_setting" not in request.node.name:
        monkeypatch.setattr("app.services.storage_policy.finishes_searches_automatically",
                            lambda session, scope_type, scope_id: True)
