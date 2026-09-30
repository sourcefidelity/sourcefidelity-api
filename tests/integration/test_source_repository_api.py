"""Source API integration with durable representation storage."""

import hashlib
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import get_db
from app.main import app
from app.models import Base
from app.models.source_repository import SourceRepresentationRecord
from app.services.file_safety import (
    FileSafetyReport,
    FileSafetyUnavailable,
    SafetyVerdict,
)
from app.services.storage.backend import StorageBackend


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.objects[key] = file_bytes
        return key

    def download(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


def test_reconciled_upload_retains_source_inspection_and_original_title(monkeypatch):
    from app.services.source_validator import ValidationResult
    monkeypatch.setattr(settings,'REPORT_AUTH_MODE','personal_local')
    monkeypatch.setattr(settings,'SOURCE_REPOSITORY_ENABLED',True)
    monkeypatch.setattr(settings,'SOURCE_REPOSITORY_SCOPE_ID','inspection-test-owner')
    monkeypatch.setattr(settings,'COMPLETENESS_CHECK_ENABLED',False)
    engine=create_engine('sqlite+pysqlite:///:memory:',connect_args={'check_same_thread':False},poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions=sessionmaker(bind=engine,expire_on_commit=False)
    storage=MemoryStorage()
    def db_override():
        with sessions() as session:yield session
    previous=dict(app.dependency_overrides)
    app.dependency_overrides[get_db]=db_override
    monkeypatch.setattr('app.routers.sources.get_storage_backend',lambda:storage)
    clean=FileSafetyReport(verdict=SafetyVerdict.CLEAN,structural_verdict=SafetyVerdict.CLEAN,malware_verdict=SafetyVerdict.CLEAN)
    monkeypatch.setattr('app.routers.sources.inspect_uploaded_pdf',lambda _:clean)
    monkeypatch.setattr('app.routers.sources.verify_instructor_upload',lambda *a,**k:(False,['insufficient_identity_evidence']))
    monkeypatch.setattr('app.routers.sources.classify_text_quality',lambda _:type('Quality',(),{'verdict':'digital'})())
    finding={'decision_applied':True,'reconciliation_version':'single-title-typo-v1',
        'original_expected_title':'A sufficiently long original tittle',
        'observed_title':'A sufficiently long original title'}
    fallback=ValidationResult(True,'high','complete','digital','Accepted',source_inspection=finding)
    kwargs=dict(files={'file':('source.pdf',b'%PDF-1.7 reconciled fixture','application/pdf')},
        data={'title':finding['original_expected_title'],'author':'Writer','year':'2020','source_kind':'monograph'})
    try:
        client=TestClient(app)
        monkeypatch.setattr('app.routers.sources._inspect_uncorroborated_personal_upload',lambda *a,**k:None)
        assert client.post('/sources/upload',**kwargs).status_code == 422
        assert not storage.objects
        monkeypatch.setattr('app.routers.sources._inspect_uncorroborated_personal_upload',lambda *a,**k:fallback)
        response=client.post('/sources/upload',**kwargs)
        assert response.status_code == 200, response.text
        with sessions() as session:
            record=session.scalars(select(SourceRepresentationRecord)).one()
            assert record.validation_evidence['source_inspection'] == finding
            assert record.canonical_work.display_title == finding['original_expected_title']
            assert record.identity_confidence == .9
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def test_upload_search_and_delete_use_durable_records(monkeypatch) -> None:
    monkeypatch.setattr(settings, "REPORT_AUTH_MODE", "personal_local")
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()

    def override_db():
        with sessions() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_ENABLED", True)
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "test-owner")
    monkeypatch.setattr(settings, "COMPLETENESS_CHECK_ENABLED", False)
    monkeypatch.setattr(
        "app.routers.sources.get_storage_backend", lambda: storage
    )
    clean_report = FileSafetyReport(
        verdict=SafetyVerdict.CLEAN,
        structural_verdict=SafetyVerdict.CLEAN,
        malware_verdict=SafetyVerdict.CLEAN,
    )
    monkeypatch.setattr(
        "app.routers.sources.inspect_uploaded_pdf", lambda _data: clean_report
    )
    monkeypatch.setattr(
        "app.routers.sources.verify_instructor_upload", lambda *_args, **_kwargs: (True, ["doi_match"])
    )
    quality = type("Quality", (), {"verdict": "digital"})()
    monkeypatch.setattr("app.routers.sources.classify_text_quality", lambda _data: quality)
    monkeypatch.setattr("app.routers.sources.is_edited_collection", lambda _data: False)

    try:
        client = TestClient(app)
        upload_args = {
            "files": {
                "file": ("source.pdf", b"%PDF-1.7 API source", "application/pdf")
            },
            "data": {
                "doi": "10.5555/durable-api",
                "title": "Durable API Source",
                "author": "A. Author",
                "year": "2026",
                "source_kind": "journal_article",
                "edition_or_version": "author_accepted_manuscript",
            },
        }

        conflicting_kind = client.post(
            "/sources/upload",
            files=upload_args["files"],
            data={
                **upload_args["data"],
                "source_kind": "monograph",
                "document_kind": "article",
            },
        )
        assert conflicting_kind.status_code == 400
        assert storage.objects == {}

        monkeypatch.setattr(
            "app.routers.sources.inspect_uploaded_pdf",
            lambda _data: FileSafetyReport(
                verdict=SafetyVerdict.REJECTED,
                structural_verdict=SafetyVerdict.REJECTED,
                malware_verdict=SafetyVerdict.NOT_ASSESSED,
                findings=("PDF JavaScript",),
            ),
        )
        rejected = client.post("/sources/upload", **upload_args)
        assert rejected.status_code == 422
        assert storage.objects == {}

        def unavailable(_data):
            raise FileSafetyUnavailable("scanner offline")

        monkeypatch.setattr("app.routers.sources.inspect_uploaded_pdf", unavailable)
        unavailable_response = client.post("/sources/upload", **upload_args)
        assert unavailable_response.status_code == 503
        assert storage.objects == {}

        monkeypatch.setattr(
            "app.routers.sources.inspect_uploaded_pdf", lambda _data: clean_report
        )
        upload = client.post(
            "/sources/upload",
            **upload_args,
        )
        assert upload.status_code == 200
        document = upload.json()["documents"][0]
        assert document["admission_state"] == "needs_review"
        assert document["cleanliness_verdict"] == "clean"
        assert document["retention_mode"] == "durable"
        assert document["edition_or_version"] == "author_accepted_manuscript"
        assert "s3_key" not in document
        assert "scope_type" not in document
        assert "scope_id" not in document
        assert len(storage.objects) == 1
        assert next(iter(storage.objects)).startswith("commercial_user_upload/")
        assert upload.json()["source_kind"] == "journal_article"

        with sessions() as session:
            persisted = session.scalar(select(SourceRepresentationRecord))
            assert persisted is not None
            assert persisted.scope_id == "test-owner"
            assert persisted.edition_or_version == "author_accepted_manuscript"

        search = client.get("/sources/search", params={"doi": "10.5555/durable-api"})
        assert search.status_code == 200
        assert search.json()["count"] == 1

        with sessions() as session:
            persisted = session.scalar(select(SourceRepresentationRecord))
            persisted.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            session.commit()

        expired_search = client.get(
            "/sources/search", params={"doi": "10.5555/durable-api"}
        )
        assert expired_search.status_code == 200
        assert expired_search.json()["count"] == 0
        expired_review = client.post(
            f"/sources/{document['id']}/review",
            data={"decision": "accept"},
        )
        assert expired_review.status_code == 410

        deleted = client.delete(f"/sources/{document['id']}")
        assert deleted.status_code == 200
        assert storage.objects == {}
        assert client.get("/sources/search").json()["count"] == 0
    finally:
        app.dependency_overrides.clear()


def _supplied_page() -> bytes:
    body = (
        '<p>The reviewed study compares documented archival practice across '
        'three national collections and records each difference in an '
        'appendix.</p>'
    ) * 14
    return (
        '<html><head>'
        '<meta property="og:title" content="A Reviewed Work. Ada Perez. '
        'Oxford: Oxford University Press, 2016. Pp. xii+204. | Journal of '
        'Records: Vol 94, No 2">'
        '<meta property="og:site_name" content="Journal of Records">'
        '<meta property="og:type" content="article">'
        '<meta name="dc.creator" content="Morgan Chen">'
        '<meta name="dc.identifier" content="10.1086/695968">'
        '<meta name="dc.date" content="2018-05-01">'
        '<script src="https://cdn.example/app.js"></script>'
        '</head><body><article><div class="article-body">'
        f'{body}'
        '</div></article></body></html>'
    ).encode('utf-8')


def test_supplied_html_intake_is_disabled_by_default(monkeypatch) -> None:
    """A saved page is a separate provenance boundary and must be opted into."""
    monkeypatch.setattr(settings, 'REPORT_AUTH_MODE', 'personal_local')
    monkeypatch.setattr(settings, 'SOURCE_REPOSITORY_ENABLED', True)
    assert settings.SUPPLIED_HTML_SOURCE_INTAKE_ENABLED is False
    client = TestClient(app)
    response = client.post(
        '/sources/upload',
        files={'file': ('saved.html', _supplied_page(), 'text/html')},
        data={'title': 'A Reviewed Work', 'author': 'Chen, M', 'year': '2018'},
    )
    assert response.status_code == 415
    assert 'disabled' in response.text


def test_supplied_html_admits_extracted_text_only(monkeypatch) -> None:
    monkeypatch.setattr(settings, 'REPORT_AUTH_MODE', 'personal_local')
    monkeypatch.setattr(settings, 'SOURCE_REPOSITORY_ENABLED', True)
    monkeypatch.setattr(settings, 'SUPPLIED_HTML_SOURCE_INTAKE_ENABLED', True)
    monkeypatch.setattr(settings, 'SOURCE_REPOSITORY_SCOPE_ID', 'supplied-html-owner')
    engine = create_engine('sqlite+pysqlite:///:memory:',
                           connect_args={'check_same_thread': False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()

    def override_db():
        with sessions() as session:
            yield session

    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr('app.routers.sources.get_storage_backend', lambda: storage)
    clean = FileSafetyReport(verdict=SafetyVerdict.CLEAN,
                             structural_verdict=SafetyVerdict.NOT_ASSESSED,
                             malware_verdict=SafetyVerdict.CLEAN)
    monkeypatch.setattr('app.routers.sources.inspect_uploaded_html', lambda _:
                        clean)
    content = _supplied_page()
    data = {
        'title': 'A Reviewed Work. Ada Perez. Oxford: Oxford University Press, '
                 '2016. Pp. xii+204',
        'author': 'Chen, M', 'year': '2018', 'doi': '10.1086/695968',
        'source_kind': 'journal_article',
    }
    try:
        client = TestClient(app)
        # A wrong work is still refused through the shared identity standard.
        rejected = client.post(
            '/sources/upload',
            files={'file': ('saved.html', content, 'text/html')},
            data={**data, 'doi': '10.1086/000000'},
        )
        assert rejected.status_code == 422
        assert rejected.json()['detail']['reason_code'] == 'bibliographic_fields_conflict'
        assert not storage.objects

        response = client.post(
            '/sources/upload',
            files={'file': ('saved.html', content, 'text/html')},
            data=data,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['supplied_html']['supplied_document'] is True
        assert payload['supplied_html']['network_access'] == 'none'
        assert 'script element' in payload['supplied_html']['active_content_markers']

        stored = list(storage.objects.values())
        assert len(stored) == 1
        # Raw HTML is never stored; only the extracted article text is.
        assert b'<script' not in stored[0]
        assert b'cdn.example' not in stored[0]
        assert b'<html' not in stored[0]
        assert b'archival practice' in stored[0]

        with sessions() as session:
            record = session.scalars(select(SourceRepresentationRecord)).one()
            evidence = record.validation_evidence['supplied_html']
            assert evidence['html_sha256'] == hashlib.sha256(content).hexdigest()
            assert evidence['retained_representation'] == 'extracted_plain_text'
            assert record.representation_kind == 'plain_text'
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
