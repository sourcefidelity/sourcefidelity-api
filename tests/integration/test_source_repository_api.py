"""Source API integration with durable representation storage."""

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
        self.objects.setdefault(key, file_bytes)
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


def test_upload_search_and_delete_use_durable_records(monkeypatch) -> None:
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
        assert document["s3_key"].startswith("commercial_user_upload/")
        assert document["s3_key"] in storage.objects
        assert upload.json()["source_kind"] == "journal_article"

        with sessions() as session:
            persisted = session.scalar(select(SourceRepresentationRecord))
            assert persisted is not None
            assert persisted.scope_id == "test-owner"

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
