"""Sanitization and lifecycle tests for report-scoped marking copies."""

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json

import fitz
from docx import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.models.job import Job
from app.services.docx_presentation import DocxPresentationRender
from app.services.paper_upload import DOCX_MEDIA_TYPE, PDF_MEDIA_TYPE
from app.services.report_paper_artifact import (
    cleanup_expired_report_paper_artifacts,
    ensure_report_paper_artifact,
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


def _pdf_bytes():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Visible student paper content")
    document.set_metadata({"author": "Private Author", "title": "Private Title"})
    value = document.tobytes()
    document.close()
    return value


def _docx_bytes():
    document = Document()
    document.core_properties.author = "Private Author"
    document.add_paragraph("Native semantic paragraph")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _rendered_pdf_bytes():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Native semantic paragraph")
    value = document.tobytes()
    document.close()
    return value


def test_pdf_marking_copy_removes_metadata_and_preserves_page_render(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()
    content = _pdf_bytes()
    with factory() as session:
        job = Job(
            filename="paper.pdf",
            paper_version_id="paper-v1",
            scope_type="personal_owner",
            scope_id="owner-1",
            input_sha256=hashlib.sha256(content).hexdigest(),
            input_media_type=PDF_MEDIA_TYPE,
            input_byte_size=len(content),
            input_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        session.add(job)
        session.commit()
        record = ensure_report_paper_artifact(
            session, storage, job=job, content=content
        )
        sanitized = storage.download(record.storage_key)
        assert record.presentation_status == "page_faithful_ready"
        assert record.content_sha256 == hashlib.sha256(sanitized).hexdigest()
        assert record.content_sha256 != job.input_sha256
        checked = fitz.open(stream=sanitized, filetype="pdf")
        assert checked.metadata["author"] == ""
        assert "Visible student paper content" in checked[0].get_text()
        checked.close()
        storage.objects.pop(record.storage_key)
        recovered = ensure_report_paper_artifact(
            session, storage, job=job, content=content
        )
        assert recovered.id == record.id
        assert hashlib.sha256(storage.download(record.storage_key)).hexdigest() == (
            record.content_sha256
        )


def test_expired_pending_marking_copy_is_physically_removed():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()
    content = _pdf_bytes()
    with factory() as session:
        job = Job(
            filename="paper.pdf",
            paper_version_id="paper-v2",
            scope_type="personal_owner",
            scope_id="owner-1",
            input_sha256=hashlib.sha256(content).hexdigest(),
            input_media_type=PDF_MEDIA_TYPE,
            input_byte_size=len(content),
            input_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        session.add(job)
        session.commit()
        record = ensure_report_paper_artifact(
            session, storage, job=job, content=content
        )
        record.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
        result = cleanup_expired_report_paper_artifacts(session, storage)
        assert result == {"artifacts_cleaned": 1, "artifacts_pending": 0}
        assert storage.objects == {}
        assert record.storage_key is None
        assert record.deleted_at is not None


def test_docx_retains_native_source_and_separately_hashed_pdf(monkeypatch):
    from app.services.schemas import ParsedReference
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()
    content = _docx_bytes()
    rendered = _rendered_pdf_bytes()
    render_calls = []
    monkeypatch.setattr(
        "app.services.report_paper_artifact.settings.DOCX_PRESENTATION_RENDERING_ENABLED",
        True,
    )
    def render_once(value):
        render_calls.append(hashlib.sha256(value).hexdigest())
        return DocxPresentationRender(
            pdf_bytes=rendered,
            provenance={
                "renderer": "test_renderer",
                "renderer_version": "test-1",
                "font_substitution_risk": False,
            },
        )

    monkeypatch.setattr(
        "app.services.report_paper_artifact.render_docx_to_pdf",
        render_once,
    )
    with factory() as session:
        job = Job(
            filename="paper.docx",
            paper_version_id="paper-docx-v1",
            scope_type="personal_owner",
            scope_id="owner-1",
            input_sha256=hashlib.sha256(content).hexdigest(),
            input_media_type=DOCX_MEDIA_TYPE,
            input_byte_size=len(content),
            input_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        session.add(job)
        session.commit()
        record = ensure_report_paper_artifact(session, storage, job=job, content=content,
            references=[ParsedReference(reference_id='r', raw_ref='Smith, J. (2020). A book.')])
        navigation = record.presentation_evidence['submitted_reference_navigation']
        assert navigation['input_sha256'] == job.input_sha256
        assert navigation['layout']['content_sha256'] == record.presentation_sha256
        saved_navigation = json.dumps(navigation, sort_keys=True)
        ensure_report_paper_artifact(session, storage, job=job, content=content,
            references=[ParsedReference(reference_id='changed', raw_ref='Changed reference')])
        assert json.dumps(record.presentation_evidence['submitted_reference_navigation'],sort_keys=True) == saved_navigation
        assert record.presentation_status == "page_faithful_ready"
        assert record.storage_key != record.presentation_storage_key
        source = storage.download(record.storage_key)
        presentation = storage.download(record.presentation_storage_key)
        assert source[:2] == b"PK"
        assert presentation.startswith(b"%PDF")
        assert record.content_sha256 == hashlib.sha256(source).hexdigest()
        assert record.presentation_sha256 == hashlib.sha256(presentation).hexdigest()
        assert record.presentation_evidence["source_sha256"] == record.content_sha256
        assert record.presentation_evidence["page_count"] == 1
        assert record.presentation_evidence["per_document_render_count"] == 1
        assert len(render_calls) == 1

        record.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
        result = cleanup_expired_report_paper_artifacts(session, storage)
        assert result == {"artifacts_cleaned": 1, "artifacts_pending": 0}
        assert storage.objects == {}
