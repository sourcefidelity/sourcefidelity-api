import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from app.services.evidence_package import build_evidence_package
from app.services.source_navigation import (
    SourceNavigationAnchor,
    SourceNavigationDescriptor,
    authorize_source_navigation_document,
    build_source_navigation_descriptor,
    validate_source_navigation_descriptor,
)
from app.services.source_repository import AdmissionRequest, WorkIdentity, admit_representation
from app.services.retrieval.base import RepresentationKind, SourceRepresentation
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import EvidenceAuthorizationError
from tests.unit.test_evidence_package import _artifact


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, file_bytes: bytes, key: str) -> str:
        self.objects.setdefault(key, file_bytes)
        return key

    def download(self, key: str) -> bytes:
        return self.objects[key]

    def delete(self, key: str) -> bool:
        self.objects.pop(key, None)
        return True

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


def test_navigation_descriptor_binds_displayed_passages_without_public_url():
    package = build_evidence_package(_artifact())

    descriptor = build_source_navigation_descriptor(package)

    assert descriptor.status == "ready"
    assert descriptor.required_capability == "authorized_source_content"
    assert descriptor.source_absence_claim_permitted is False
    assert [item.passage_id for item in descriptor.anchors] == package.retrieval.displayed_passage_ids
    assert all(item.target_kind == "text_document" for item in descriptor.anchors)
    assert not any("url" in key.casefold() for key in descriptor.model_dump())
    validate_source_navigation_descriptor(descriptor)


def test_navigation_descriptor_retains_more_than_sixteen_authorized_anchors():
    anchors = [
        SourceNavigationAnchor(
            passage_id=f"passage-{index}",
            target_kind="pdf_page",
            page_index=0,
            character_start=index * 10,
            character_end=index * 10 + 5,
            passage_text_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
        )
        for index in range(17)
    ]

    descriptor = SourceNavigationDescriptor(
        descriptor_sha256="0" * 64,
        status="ready",
        representation_id="representation-1",
        content_sha256="1" * 64,
        authorization_scope_type="personal_owner",
        authorization_scope_id="personal-default",
        representation_kind="pdf",
        media_type="application/pdf",
        pages_total=1,
        anchors=anchors,
    )

    assert len(descriptor.anchors) == 17


def test_navigation_descriptor_hash_rejects_mutation():
    descriptor = build_source_navigation_descriptor(build_evidence_package(_artifact()))
    tampered = descriptor.model_copy(update={"pages_total": 999})

    with pytest.raises(EvidenceAuthorizationError, match="content-hash"):
        validate_source_navigation_descriptor(tampered)


def test_navigation_document_rechecks_scope_and_immutable_source():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    backend = MemoryStorage()
    content = b"source-navigation-test"
    with session_factory() as session:
        record = admit_representation(
            session,
            backend,
            AdmissionRequest(
                work=WorkIdentity(
                    title="Navigation",
                    work_type="journal_article",
                    author="Smith",
                    year="2020",
                ),
                representation=SourceRepresentation(
                    kind=RepresentationKind.PLAIN_TEXT,
                    media_type="text/plain",
                    content=content,
                    source_url="https://repository.example/navigation.txt",
                ),
                provenance="test",
                license_class="commercial_user_upload",
                scope_type="personal_owner",
                scope_id="personal-default",
                identity_verdict="verified",
                identity_confidence=1.0,
                completeness_verdict="complete",
                cleanliness_verdict="clean",
                text_quality="digital",
            ),
        )
        package = build_evidence_package(_artifact())
        descriptor = build_source_navigation_descriptor(package).model_copy(
            update={
                "representation_id": str(record.id),
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "authorization_scope_type": "personal_owner",
                "authorization_scope_id": "personal-default",
            }
        )
        payload = descriptor.model_dump(mode="json", exclude={"descriptor_sha256"})
        descriptor = descriptor.model_copy(
            update={
                "descriptor_sha256": hashlib.sha256(
                    __import__("json").dumps(
                        payload, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
            }
        )

        authorized = authorize_source_navigation_document(
            session,
            backend,
            descriptor,
            scope_type="personal_owner",
            scope_id="personal-default",
        )
        assert authorized.representation.content == content

        with pytest.raises(EvidenceAuthorizationError, match="requesting scope"):
            authorize_source_navigation_document(
                session,
                backend,
                descriptor,
                scope_type="personal_owner",
                scope_id="another-owner",
            )
