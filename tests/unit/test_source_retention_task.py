"""Periodic source-retention cleanup wiring tests."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.source_repository import AdmissionRequest, WorkIdentity, admit_representation
from app.services.storage.backend import StorageBackend
from app.tasks import source_retention
from app.tasks.celery_app import celery_app


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


def test_retention_cleanup_has_periodic_schedule() -> None:
    schedule = celery_app.conf.beat_schedule["cleanup-expired-source-representations"]

    assert schedule["task"] == "cleanup_expired_source_representations"
    assert schedule["schedule"] >= 60


def test_retention_task_removes_expired_record_and_object(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    storage = MemoryStorage()
    with sessions() as session:
        record = admit_representation(
            session,
            storage,
            AdmissionRequest(
                work=WorkIdentity(
                    title="Expired Task Source",
                    work_type="journal_article",
                    doi="10.1234/expired-task",
                ),
                representation=SourceRepresentation(
                    kind=RepresentationKind.PDF,
                    media_type="application/pdf",
                    content=b"%PDF-1.7 expired task source",
                ),
                provenance="test_provider",
                license_class="paywalled_db_retrieved",
                scope_type="personal_owner",
                scope_id="owner-1",
                identity_verdict="verified",
                completeness_verdict="complete",
                cleanliness_verdict="clean",
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
        )
        storage_key = record.content_object.storage_key
        session.commit()

    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_ENABLED", True)
    monkeypatch.setattr(source_retention, "SessionLocal", sessions)
    monkeypatch.setattr(source_retention, "get_storage_backend", lambda: storage)

    result = source_retention.cleanup_expired_source_representations.run()

    assert result == {
        "status": "ok",
        "representations_expired": 1,
        "objects_deleted": 1,
    }
    with sessions() as session:
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 0
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 0
    assert not storage.exists(storage_key)
