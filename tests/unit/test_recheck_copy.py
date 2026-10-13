"""Keeping the submitted file so a paper can be checked again (recheck-copy-v1, owner decision 2026-10-13)."""
from datetime import datetime, timedelta, timezone

import pytest

from app.models.job import Job, JobStage, JobStatus
from app.services import paper_upload, recheck_copy
from app.services.file_safety import SafetyVerdict
from app.services.paper_upload import DOCX_MEDIA_TYPE, cleanup_stale_paper_job_inputs, create_paper_job
from tests.unit.test_paper_upload import MemoryStorage, _docx_bytes, session  # noqa: F401 - fixture

NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_scan(monkeypatch):
    monkeypatch.setattr(paper_upload, "scan_with_clamd", lambda _content: (SafetyVerdict.CLEAN, "OK"))


def _completed(session, storage, *, keep: dict | None, scope="personal_owner"):
    job = create_paper_job(session, storage, content=_docx_bytes(), filename="paper.docx",
                           media_type=DOCX_MEDIA_TYPE, scope_type=scope, scope_id="owner-1", recheck_copy=keep)
    if keep:
        recheck_copy.mark_kept(job, NOW)
    job.status, job.stage = JobStatus.COMPLETED, JobStage.COMPLETED
    session.commit()
    return job


def test_the_upload_records_the_scopes_choice():
    assert recheck_copy.retention_at_upload(None, "personal_owner", "owner-1") == {
        "policy": recheck_copy.POLICY, "days": 30}
    # Institutional keeps nothing until the administrator turns it on.
    assert recheck_copy.retention_at_upload(None, "assessment", "assessment-1") is None


def test_a_kept_file_waits_for_its_period_and_an_unkept_one_goes_at_once(session):
    storage = MemoryStorage()
    kept = _completed(session, storage, keep={"policy": recheck_copy.POLICY, "days": 30})
    plain = _completed(session, storage, keep=None)
    assert recheck_copy.available(kept, NOW) and not recheck_copy.available(plain, NOW)
    cleanup_stale_paper_job_inputs(session, storage, now=NOW + timedelta(days=1))
    assert kept.input_storage_key in storage.objects and plain.input_storage_key is None
    cleanup_stale_paper_job_inputs(session, storage, now=NOW + timedelta(days=30, seconds=1))
    assert kept.input_storage_key is None and storage.objects == {}


def test_until_deleted_keeps_the_file_with_the_record(session):
    storage = MemoryStorage()
    job = _completed(session, storage, keep={"policy": recheck_copy.POLICY, "days": 0})
    assert job.input_expires_at.replace(tzinfo=timezone.utc) == recheck_copy.UNTIL_DELETED
    assert job.upload_evidence[recheck_copy.EVIDENCE_KEY]["kept_until"] is None
    cleanup_stale_paper_job_inputs(session, storage, now=NOW + timedelta(days=3650))
    assert job.input_storage_key in storage.objects
    from app.services.record_expiry import delete_paper_record
    assert delete_paper_record(session, storage, job, now=NOW) is True
    assert storage.objects == {}


def test_check_again_starts_a_linked_check_of_the_same_file(session, monkeypatch):
    storage = MemoryStorage()
    job = _completed(session, storage, keep={"policy": recheck_copy.POLICY, "days": 30})
    monkeypatch.setattr("app.services.paper_reuse.find_reusable_report", lambda *a, **k: None)
    monkeypatch.setattr(recheck_copy, "available", lambda item, now=None: item.status == JobStatus.COMPLETED
                        and bool(item.input_storage_key))
    outcome, again = recheck_copy.start_recheck(session, storage, job, scope_type="personal_owner",
                                                scope_id="owner-1")
    assert outcome == "started" and again.input_sha256 == job.input_sha256 and again.filename == job.filename
    assert again.upload_evidence["recheck_of_job_id"] == str(job.id)
    assert again.upload_evidence["submission_root_job_id"] == str(job.id)
    assert job.upload_evidence["recheck_job_id"] == str(again.id)
    assert again.upload_evidence[recheck_copy.EVIDENCE_KEY]["days"] == 30
    assert [item.id for item in recheck_copy.lineage(session, job)] == [job.id, again.id]
    with pytest.raises(recheck_copy.RecheckInProgress):
        recheck_copy.start_recheck(session, storage, job, scope_type="personal_owner", scope_id="owner-1")


def test_an_unchanged_paper_returns_its_report_and_a_lapsed_file_cannot_be_checked(session, monkeypatch):
    storage = MemoryStorage()
    job = _completed(session, storage, keep={"policy": recheck_copy.POLICY, "days": 30})
    report = object()
    monkeypatch.setattr("app.services.paper_reuse.find_reusable_report", lambda *a, **k: (job, report))
    monkeypatch.setattr("app.services.paper_reuse.record_reuse", lambda *a, **k: None)
    assert recheck_copy.start_recheck(session, storage, job, scope_type="personal_owner",
                                      scope_id="owner-1") == ("reused", report)
    job.input_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session.commit()
    with pytest.raises(recheck_copy.RecheckUnavailable):
        recheck_copy.start_recheck(session, storage, job, scope_type="personal_owner", scope_id="owner-1")
