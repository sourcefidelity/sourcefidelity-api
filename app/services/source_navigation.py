"""Authorized whole-source navigation derived from an Evidence Package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.services.evidence_package import EvidencePackageV1, validate_evidence_package
from app.services.storage.backend import StorageBackend
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    EvidenceAuthorizationError,
    authorize_representation,
)


SOURCE_NAVIGATION_VERSION = "source-navigation-v1"
# Evidence Package candidate retrieval can retain up to ten passages for each
# of sixteen candidates, plus the protected whole-citation set. Navigation
# must bind that complete display union rather than silently truncating it.
MAX_SOURCE_NAVIGATION_ANCHORS = 192


class SourceNavigationAnchor(BaseModel):
    """One exact report passage target inside the authorized source."""

    passage_id: str = Field(min_length=1, max_length=128)
    target_kind: Literal["pdf_page", "text_document"]
    page_index: int | None = Field(default=None, ge=0)
    page_label: str | None = Field(default=None, max_length=100)
    character_start: int = Field(ge=0)
    character_end: int = Field(gt=0)
    passage_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_target(self):
        if self.character_end <= self.character_start:
            raise ValueError("Source navigation anchor is empty")
        if self.target_kind == "pdf_page" and self.page_index is None:
            raise ValueError("PDF navigation anchors require a page index")
        return self


class SourceNavigationDescriptor(BaseModel):
    """Persistable navigation contract; contains no source bytes or URL."""

    navigation_version: str = SOURCE_NAVIGATION_VERSION
    descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["ready", "not_assessable"]
    required_capability: Literal["authorized_source_content"] = (
        "authorized_source_content"
    )
    representation_id: str = Field(min_length=1, max_length=255)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_scope_type: str = Field(min_length=1, max_length=100)
    authorization_scope_id: str = Field(min_length=1, max_length=255)
    representation_kind: str = Field(min_length=1, max_length=100)
    media_type: str = Field(min_length=1, max_length=200)
    pages_total: int | None = Field(default=None, ge=1)
    anchors: list[SourceNavigationAnchor] = Field(
        default_factory=list, max_length=MAX_SOURCE_NAVIGATION_ANCHORS
    )
    source_absence_claim_permitted: Literal[False] = False
    limitations: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_navigation_shape(self):
        if self.status == "ready" and not self.anchors:
            raise ValueError("Ready source navigation requires an anchor")
        expected_kind = (
            "pdf_page" if self.media_type == "application/pdf" else "text_document"
        )
        if any(anchor.target_kind != expected_kind for anchor in self.anchors):
            raise ValueError("Source navigation anchor kind conflicts with media type")
        if self.pages_total is not None and any(
            anchor.page_index is not None and anchor.page_index >= self.pages_total
            for anchor in self.anchors
        ):
            raise ValueError("Source navigation anchor exceeds the source page count")
        return self


@dataclass(frozen=True)
class AuthorizedSourceNavigationDocument:
    """Runtime-only authorized source bytes plus their validated descriptor."""

    descriptor: SourceNavigationDescriptor
    representation: AuthorizedRepresentation


def build_source_navigation_descriptor(
    package: EvidencePackageV1,
) -> SourceNavigationDescriptor:
    """Derive exact evidence anchors without creating a public content URL."""
    validate_evidence_package(package)
    displayed = set(package.retrieval.displayed_passage_ids)
    passages = [item for item in package.passages if item.passage_id in displayed]
    media_type = package.coverage.media_type
    pdf_mode = media_type == "application/pdf"
    anchors = []
    for passage in passages:
        if pdf_mode and passage.page_index is None:
            continue
        anchors.append(
            SourceNavigationAnchor(
                passage_id=passage.passage_id,
                target_kind="pdf_page" if pdf_mode else "text_document",
                page_index=passage.page_index,
                page_label=passage.page_label,
                character_start=passage.character_start,
                character_end=passage.character_end,
                passage_text_sha256=passage.passage_text_sha256,
            )
        )
    status = "ready" if anchors else "not_assessable"
    payload = {
        "navigation_version": SOURCE_NAVIGATION_VERSION,
        "status": status,
        "required_capability": "authorized_source_content",
        "representation_id": package.source_identity.representation_id,
        "content_sha256": package.source_identity.content_sha256,
        "authorization_scope_type": package.source_identity.authorization_scope_type,
        "authorization_scope_id": package.source_identity.authorization_scope_id,
        "representation_kind": package.coverage.representation_kind,
        "media_type": media_type,
        "pages_total": package.coverage.pages_total,
        "anchors": [item.model_dump(mode="json") for item in anchors],
        "source_absence_claim_permitted": False,
        "limitations": [
            "Navigation exposes the authorized source for human inspection and never establishes that an unretrieved proposition is absent.",
            "No public source URL is persisted; runtime content requires exact-scope authorization.",
            *(
                ["No displayed passage has a navigable source location."]
                if not anchors
                else []
            ),
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SourceNavigationDescriptor(descriptor_sha256=digest, **payload)


def validate_source_navigation_descriptor(
    descriptor: SourceNavigationDescriptor,
) -> None:
    payload = descriptor.model_dump(mode="json", exclude={"descriptor_sha256"})
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != descriptor.descriptor_sha256:
        raise EvidenceAuthorizationError(
            "Source navigation descriptor failed content-hash validation"
        )


def authorize_source_navigation_document(
    session: Session,
    backend: StorageBackend,
    descriptor: SourceNavigationDescriptor,
    *,
    scope_type: str,
    scope_id: str,
) -> AuthorizedSourceNavigationDocument:
    """Load the complete source only after descriptor and repository checks."""
    validate_source_navigation_descriptor(descriptor)
    if descriptor.status != "ready":
        raise EvidenceAuthorizationError("Source navigation is not assessable")
    if (
        descriptor.authorization_scope_type != scope_type.strip().casefold()
        or descriptor.authorization_scope_id != scope_id.strip()
    ):
        raise EvidenceAuthorizationError(
            "Source navigation is not authorized in the requesting scope"
        )
    representation = authorize_representation(
        session,
        backend,
        representation_id=descriptor.representation_id,
        scope_type=scope_type,
        scope_id=scope_id,
    )
    if (
        representation.content_sha256 != descriptor.content_sha256
        or representation.representation_kind != descriptor.representation_kind
        or representation.media_type != descriptor.media_type
    ):
        raise EvidenceAuthorizationError(
            "Authorized source does not match the navigation descriptor"
        )
    return AuthorizedSourceNavigationDocument(
        descriptor=descriptor,
        representation=representation,
    )
