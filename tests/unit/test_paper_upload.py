"""Temporary student-paper intake, immutable loading, and cleanup tests."""

from datetime import datetime, timedelta, timezone
import io
from zipfile import ZipFile

from docx import Document
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models import Base
from app.models.job import Job, JobStatus
from sqlalchemy import select
from app.services import paper_upload
from app.services.file_safety import SafetyVerdict
from app.services.paper_upload import (
    DOCX_MEDIA_TYPE,
    PaperUploadError,
    cleanup_stale_paper_job_inputs,
    create_paper_job,
    load_paper_job_input,
)
from app.services.storage.backend import StorageBackend


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


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as active:
        yield active


def _docx_bytes(text="A paper"):
    document = Document()
    document.add_paragraph(text)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def test_docx_job_commits_locator_verifies_hash_and_cleans(monkeypatch, session):
    monkeypatch.setattr(
        paper_upload,
        "scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    storage = MemoryStorage()
    content = _docx_bytes()
    job = create_paper_job(
        session,
        storage,
        content=content,
        filename="paper.docx",
        media_type=DOCX_MEDIA_TYPE,
        scope_id="personal-default",
    )

    assert job.input_storage_key in storage.objects
    loaded_job, loaded = load_paper_job_input(session, storage, job.id)
    assert loaded_job.id == job.id
    assert loaded == content
    assert job.upload_evidence["structural_verdict"] == "clean"
    assert job.upload_evidence["paper_retention_mode"] == "temporary"
    assert job.upload_evidence["paper_input_upload_state"] == "ready"

    job.input_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session.commit()
    result = cleanup_stale_paper_job_inputs(session, storage)
    assert result == {"jobs_cleaned": 1, "jobs_pending": 0}
    assert storage.objects == {}
    assert job.input_storage_key is None
    assert job.input_deleted_at is not None


def test_interrupted_input_upload_retains_locator_and_fails_on_expiry(monkeypatch, session):
    monkeypatch.setattr(paper_upload, "scan_with_clamd", lambda _content: (SafetyVerdict.CLEAN, "OK"))
    monkeypatch.setattr(paper_upload, "prepare_dispatch", lambda _job: (_ for _ in ()).throw(RuntimeError("exit boundary")))
    storage = MemoryStorage()
    with pytest.raises(RuntimeError, match="exit boundary"):
        create_paper_job(session, storage, content=_docx_bytes(), filename="paper.docx",
                         media_type=DOCX_MEDIA_TYPE, scope_id="test")
    session.rollback()
    job = session.scalar(select(Job))
    assert job.input_storage_key in storage.objects
    assert job.upload_evidence["paper_input_upload_state"] == "pending"
    assert "workflow_dispatch_v1" not in job.upload_evidence
    result = cleanup_stale_paper_job_inputs(session, storage, now=job.input_expires_at + timedelta(seconds=1))
    assert result["jobs_cleaned"] == 1
    assert job.status == JobStatus.FAILED
    assert job.error_message == "paper_upload_interrupted"
    assert storage.objects == {}
    assert job.input_storage_key is None
    assert job.input_deleted_at is not None


def test_docx_rejects_embedded_active_content(monkeypatch):
    monkeypatch.setattr(
        paper_upload,
        "scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    source = _docx_bytes()
    rebuilt = io.BytesIO()
    with ZipFile(io.BytesIO(source)) as original, ZipFile(rebuilt, "w") as target:
        for item in original.infolist():
            target.writestr(item, original.read(item.filename))
        target.writestr("word/embeddings/payload.exe", b"not executable")

    with pytest.raises(PaperUploadError) as captured:
        paper_upload.inspect_paper_upload(
            rebuilt.getvalue(),
            filename="paper.docx",
            media_type=DOCX_MEDIA_TYPE,
        )
    assert captured.value.code == "active_docx_content"


def test_immutable_load_rejects_changed_object(monkeypatch, session):
    monkeypatch.setattr(
        paper_upload,
        "scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    storage = MemoryStorage()
    job = create_paper_job(
        session,
        storage,
        content=_docx_bytes(),
        filename="paper.docx",
        media_type=DOCX_MEDIA_TYPE,
        scope_id="personal-default",
    )
    storage.objects[job.input_storage_key] = b"changed"
    with pytest.raises(PaperUploadError) as captured:
        load_paper_job_input(session, storage, job.id)
    assert captured.value.code == "paper_input_tampered"


@pytest.mark.parametrize("mode", ["assessment", "course", "institutional"])
def test_unimplemented_paper_retention_modes_reject_before_storage(
    monkeypatch, session, mode
):
    monkeypatch.setattr(
        paper_upload,
        "scan_with_clamd",
        lambda _content: (SafetyVerdict.CLEAN, "stream: OK"),
    )
    storage = MemoryStorage()
    with pytest.raises(PaperUploadError) as captured:
        create_paper_job(
            session,
            storage,
            content=_docx_bytes(),
            filename="paper.docx",
            media_type=DOCX_MEDIA_TYPE,
            scope_id="personal-default",
            paper_retention_mode=mode,
        )
    assert captured.value.code == "paper_retention_mode_not_implemented"
    assert storage.objects == {}
