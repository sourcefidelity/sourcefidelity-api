"""Authorized pages: one fixed order of checks before any source text is read (authorized-pages-v1, 2026-10-21).

One case for each situation where the six former readers differed.
"""
import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from app.services import verification_evidence as ve
from app.services.authorized_pages import AuthorizedPages, PagesUnavailable, open_authorized_pages
from app.services.cross_language_whole_source import attach_whole_source
from app.services.paper_workflow import _unavailable_source_reason
from app.services.verification_evidence import (
    AuthorizedRepresentation, ClaimEvidence, EvidenceAuthorizationError, VerificationVerdict,
    build_passage_evidence,
)

TEXT = "Harbour cranes changed port labour. Dockworkers organised in response to the new machines."


def _source(text: str = TEXT, **changes) -> AuthorizedRepresentation:
    content = text.encode("utf-8")
    source = AuthorizedRepresentation(
        representation_id="pages-source", canonical_work_id="pages-work", content_object_id="object-1",
        content_sha256=hashlib.sha256(content).hexdigest(), content=content,
        representation_kind="plain_text", media_type="text/plain", provenance="test",
        scope_type="personal_owner", scope_id="personal-default", identity_verdict="verified",
        identity_confidence=0.9, completeness_verdict="complete", text_quality="digital",
        edition_or_version="v1", created_at=datetime(2026, 10, 21, tzinfo=timezone.utc), admitted_at=None,
    )
    return replace(source, **changes)


def _claim() -> ClaimEvidence:
    text = "Harbour cranes changed port labour (Writer, 2020)."
    return ClaimEvidence(claim_id="claim-1", paper_version_id="paper-v1", text=text, claim_type="paraphrase",
                         reference_ids=["ref-1"], page_locator="", passage_start=0, passage_end=len(text))


def test_a_valid_source_is_extracted_once_for_every_reader(monkeypatch):
    calls = []
    extract = ve._extract_pages
    monkeypatch.setattr(ve, "_extract_pages", lambda source: calls.append(1) or extract(source))
    source = _source()
    first = open_authorized_pages(source)
    assert isinstance(first, AuthorizedPages) and first.pages
    artifact = build_passage_evidence(source, claim=_claim())
    assert open_authorized_pages(source, artifact) is first
    assert calls == [1]


def test_invalid_provenance_is_refused():
    source = _source(parent_content_sha256="a" * 64, derivation_method="local-pdf-ocr-derivative-v1")
    opened = open_authorized_pages(source)
    assert isinstance(opened, PagesUnavailable) and opened.reason_code == "derivative_provenance_invalid"
    with pytest.raises(EvidenceAuthorizationError, match="provenance"):
        build_passage_evidence(source, claim=_claim())


def test_changed_bytes_are_refused():
    source = replace(_source(), content=b"Other text entirely.")
    assert open_authorized_pages(source).reason_code == "authorized_source_mismatch"


def test_another_source_than_the_artifact_was_built_from_is_refused():
    artifact = build_passage_evidence(_source(), claim=_claim())
    other = _source(representation_id="other-source")
    assert open_authorized_pages(other, artifact).reason_code == "authorized_source_mismatch"


def test_a_changed_page_label_is_refused():
    artifact = build_passage_evidence(_source(page_labels=("12",)), claim=_claim())
    relabelled = _source(page_labels=("13",))
    assert ve._source_matches_artifact(relabelled, artifact)  # identity alone does not see labels
    assert open_authorized_pages(relabelled, artifact).reason_code == "source_extraction_mismatch"


def test_an_artifact_from_another_extraction_version_is_refused():
    source = _source()
    artifact = build_passage_evidence(source, claim=_claim())
    stale = artifact.model_copy(update={"coverage": artifact.coverage.model_copy(
        update={"extraction_version": "source-text-extraction-v0"})})
    assert open_authorized_pages(source, stale).reason_code == "stale_source_extraction"


def test_empty_text_opens_with_no_usable_page():
    source = _source("   ")
    opened = open_authorized_pages(source)
    assert isinstance(opened, AuthorizedPages)
    assert not any(page.text.strip() for page in opened.pages)
    artifact = build_passage_evidence(source, claim=_claim())
    assert artifact.verdict == VerificationVerdict.NOT_ASSESSED
    assert "source_text_unavailable" in artifact.reason_codes


def test_a_reader_records_the_refusal_instead_of_reading():
    source = _source()
    artifact = build_passage_evidence(source, claim=_claim(), cross_language={"differs": True})
    record = attach_whole_source(_source(page_labels=("2",)), artifact).cross_language["whole_source"]
    assert record["status"] == "not_available" and record["reason"] == "source_extraction_mismatch"


def test_the_paper_workflow_drops_only_a_refused_source():
    assert _unavailable_source_reason(_source()) is None
    bad = _source(parent_content_sha256="a" * 64, derivation_method="local-pdf-ocr-derivative-v1")
    assert _unavailable_source_reason(bad) == "derivative_provenance_invalid"
