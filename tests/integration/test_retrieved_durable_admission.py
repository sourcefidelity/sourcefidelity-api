"""Retrieved representations enter the durable repository, not legacy keys."""

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base
from app.models.source_repository import ContentObjectRecord, SourceRepresentationRecord
from app.services.file_safety import FileSafetyReport, SafetyVerdict
from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    SourceRepresentation,
)
from app.services.source_resolver import SourceResolver, _CANONICAL_GRAPH_SOURCE
from app.services.storage.backend import StorageBackend


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.objects.setdefault(key, file_bytes)
        return key

    def download(self, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise FileNotFoundError(key) from exc

    def delete(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


@pytest.fixture
def repository_resolver(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    storage = MemoryStorage()
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = storage
    resolver._repository_session_factory = factory
    resolver._retrieval_sources = []
    resolver._acquisition_capabilities = None
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_ENABLED", True)
    monkeypatch.setattr(settings, "SOURCE_REPOSITORY_SCOPE_ID", "retrieval-test")
    yield resolver, factory, storage
    engine.dispose()


def _text_result(*, completeness: str = "complete") -> RetrievalResult:
    text = (
        "A Durable Retrieved Source by Rivera, 2024. "
        "This is complete normalized academic source evidence. " * 20
    )
    return RetrievalResult(
        source_name="gutenberg",
        success=True,
        title="A Durable Retrieved Source",
        authors=["Rivera, Alex"],
        year="2024",
        representation=SourceRepresentation(
            kind=RepresentationKind.PLAIN_TEXT,
            media_type="text/plain",
            content=text.encode(),
            original_kind=RepresentationKind.EPUB,
            completeness=completeness,
        ),
        metadata={"license_class": "public_domain"},
    )


@pytest.mark.integration
def test_retrieved_text_is_admitted_and_served_from_durable_cache(
    repository_resolver,
) -> None:
    resolver, factory, storage = repository_resolver
    result = resolver._download_and_cache(
        _CANONICAL_GRAPH_SOURCE,
        _text_result(),
        ref_title="A Durable Retrieved Source",
        ref_author="Rivera, Alex",
        ref_year="2024",
    )

    assert result.metadata["durable_admission"]["state"] == "accepted"
    assert all(not key.startswith("by-") for key in storage.objects)
    assert next(iter(storage.objects)).startswith("public_domain/")

    cached = resolver._check_local_cache(None, None, "A Durable Retrieved Source")
    assert cached.success is True
    assert cached.full_text == result.full_text
    assert cached.metadata["admission_state"] == "accepted"

    with factory() as session:
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 1
        assert session.scalar(select(func.count(SourceRepresentationRecord.id))) == 1


@pytest.mark.integration
def test_uncertain_retrieved_completeness_is_durable_but_not_cache_eligible(
    repository_resolver,
) -> None:
    resolver, factory, _ = repository_resolver
    result = resolver._download_and_cache(
        _CANONICAL_GRAPH_SOURCE,
        _text_result(completeness="not_assessed"),
        ref_title="A Durable Retrieved Source",
        ref_author="Rivera, Alex",
        ref_year="2024",
    )

    assert result.metadata["durable_admission"]["state"] == "needs_review"
    assert resolver._check_local_cache(
        None, None, "A Durable Retrieved Source"
    ).success is False
    with factory() as session:
        record = session.scalar(select(SourceRepresentationRecord))
        assert record is not None
        assert record.completeness_verdict == "not_assessed"


@pytest.mark.integration
def test_retrieved_pdf_rejected_by_safety_is_not_stored(
    repository_resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, factory, storage = repository_resolver
    monkeypatch.setattr(
        "app.services.source_resolver.inspect_uploaded_pdf",
        lambda _: FileSafetyReport(
            verdict=SafetyVerdict.REJECTED,
            structural_verdict=SafetyVerdict.REJECTED,
            malware_verdict=SafetyVerdict.NOT_ASSESSED,
            findings=("PDF JavaScript",),
        ),
    )
    result = RetrievalResult(
        source_name="core",
        success=True,
        doi="10.1234/retrieved",
        title="Retrieved PDF",
        representation=SourceRepresentation(
            kind=RepresentationKind.PDF,
            media_type="application/pdf",
            content=b"%PDF-1.7 /JavaScript /OpenAction",
            completeness="not_assessed",
        ),
        locations=[
            AcquisitionLocation(
                url="https://repository.example/retrieved.pdf",
                provider="core",
                representation_kind=RepresentationKind.PDF,
                access_type="open_access",
            )
        ],
    )

    resolved = resolver._download_and_cache(
        _CANONICAL_GRAPH_SOURCE,
        result,
        ref_doi="10.1234/retrieved",
        ref_title="Retrieved PDF",
    )

    assert resolved.full_text is None
    assert resolved.metadata["durable_admission"]["state"] == "rejected"
    assert storage.objects == {}
    with factory() as session:
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 0


@pytest.mark.integration
def test_clean_retrieved_pdf_with_unknown_completeness_needs_review(
    repository_resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, factory, storage = repository_resolver
    monkeypatch.setattr(
        "app.services.source_resolver.inspect_uploaded_pdf",
        lambda _: FileSafetyReport(
            verdict=SafetyVerdict.CLEAN,
            structural_verdict=SafetyVerdict.CLEAN,
            malware_verdict=SafetyVerdict.CLEAN,
        ),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        lambda *args, **kwargs: SimpleNamespace(
            identity_confidence="high", reason="exact DOI"
        ),
    )
    result = RetrievalResult(
        source_name="core",
        success=True,
        doi="10.1234/retrieved",
        title="Retrieved PDF",
        representation=SourceRepresentation(
            kind=RepresentationKind.PDF,
            media_type="application/pdf",
            content=b"%PDF-1.7 clean retrieved bytes",
            completeness="not_assessed",
        ),
        locations=[
            AcquisitionLocation(
                url="https://repository.example/retrieved.pdf",
                provider="core",
                representation_kind=RepresentationKind.PDF,
                access_type="open_access",
            )
        ],
    )

    resolved = resolver._download_and_cache(
        _CANONICAL_GRAPH_SOURCE,
        result,
        ref_doi="10.1234/retrieved",
        ref_title="Retrieved PDF",
    )

    assert resolved.metadata["durable_admission"]["state"] == "needs_review"
    assert next(iter(storage.objects)).startswith("open_access/")
    with factory() as session:
        record = session.scalar(select(SourceRepresentationRecord))
        assert record is not None
        assert record.cleanliness_verdict == "clean"
        assert record.completeness_verdict == "not_assessed"


def _public_download_result() -> RetrievalResult:
    return RetrievalResult(
        source_name="openalex",
        success=True,
        title="Publicly Downloaded Article",
        representation=SourceRepresentation(
            kind=RepresentationKind.PDF,
            media_type="application/pdf",
            content=b"%PDF-1.7 clean public download",
            source_url="https://repository.example/article-file/123",
            completeness="not_assessed",
        ),
        locations=[
            AcquisitionLocation(
                url="https://repository.example/article/123",
                provider="openalex",
                representation_kind=RepresentationKind.HTML,
            )
        ],
    )


@pytest.mark.integration
def test_public_download_is_retained_without_making_oa_claim(
    repository_resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, factory, storage = repository_resolver
    monkeypatch.setattr(settings, "PUBLIC_RETRIEVAL_RETENTION_POLICY", "store_scoped")
    monkeypatch.setattr(
        "app.services.source_resolver.inspect_uploaded_pdf",
        lambda _: FileSafetyReport(
            verdict=SafetyVerdict.CLEAN,
            structural_verdict=SafetyVerdict.CLEAN,
            malware_verdict=SafetyVerdict.CLEAN,
        ),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        lambda *args, **kwargs: SimpleNamespace(
            identity_confidence="high", reason="matching title and author"
        ),
    )

    resolved = resolver._download_and_cache(
        _CANONICAL_GRAPH_SOURCE,
        _public_download_result(),
        ref_title="Publicly Downloaded Article",
    )

    admission = resolved.metadata["durable_admission"]
    assert admission["state"] == "needs_review"
    assert admission["license_class"] == "rights_unclassified"
    assert admission["retention_basis"] == "deployment_public_download_policy"
    assert next(iter(storage.objects)).startswith("rights_unclassified/")
    with factory() as session:
        record = session.scalar(select(SourceRepresentationRecord))
        assert record is not None
        assert record.validation_evidence["retention_basis"] == (
            "deployment_public_download_policy"
        )


@pytest.mark.integration
def test_public_download_retention_can_require_explicit_licence(
    repository_resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver, factory, storage = repository_resolver
    monkeypatch.setattr(
        settings,
        "PUBLIC_RETRIEVAL_RETENTION_POLICY",
        "explicit_license_only",
    )
    monkeypatch.setattr(
        "app.services.source_resolver.inspect_uploaded_pdf",
        lambda _: FileSafetyReport(
            verdict=SafetyVerdict.CLEAN,
            structural_verdict=SafetyVerdict.CLEAN,
            malware_verdict=SafetyVerdict.CLEAN,
        ),
    )
    monkeypatch.setattr(
        "app.services.source_resolver.validate_retrieved_pdf",
        lambda *args, **kwargs: SimpleNamespace(
            identity_confidence="high", reason="matching title and author"
        ),
    )

    resolved = resolver._download_and_cache(
        _CANONICAL_GRAPH_SOURCE,
        _public_download_result(),
        ref_title="Publicly Downloaded Article",
    )

    admission = resolved.metadata["durable_admission"]
    assert admission["state"] == "not_stored"
    assert admission["retention_basis"] == (
        "deployment_policy_did_not_authorize_retention"
    )
    assert storage.objects == {}
    with factory() as session:
        assert session.scalar(select(func.count(ContentObjectRecord.id))) == 0
