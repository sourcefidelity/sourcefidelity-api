"""Authoritative Evidence Package v1 contract tests."""

from datetime import datetime, timezone
import hashlib

import pytest

from app.services.evidence_package import (
    EVIDENCE_PACKAGE_VERSION,
    EvidencePackageError,
    _ordered_retrieval_ids,
    build_evidence_package,
    validate_evidence_package,
)
from app.services.reference_discovery import (
    ExpectedBibliographicFields,
    ReferenceRouteAttempt,
    ReferenceSearchQuery,
    derive_reference_discovery_record,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    build_passage_evidence,
)
from app.services.verification_report import build_inspectable_report_payload


def _artifact(source_text: str = "Careful checking improves accuracy."):
    content = source_text.encode("utf-8")
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="representation-package-1",
        canonical_work_id="work-package-1",
        content_object_id="object-package-1",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        representation_kind="plain_text",
        media_type="text/plain",
        provenance="authorized_upload",
        scope_type="personal_owner",
        scope_id="owner-1",
        identity_verdict="verified",
        identity_confidence=0.99,
        completeness_verdict="complete",
        text_quality="digital",
        edition_or_version="edition-1",
        created_at=now,
        admitted_at=now,
    )
    claim_text = "Careful checking improves accuracy (Smith, 2020)."
    claim = ClaimEvidence(
        claim_id="claim-package-1",
        paper_version_id="paper-package-v1",
        text=claim_text,
        reference_ids=["reference-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
        passage_start=100,
        passage_end=100 + len(claim_text),
    )
    return build_passage_evidence(source, claim=claim)


def test_evidence_package_is_source_bound_hashed_and_judgment_free():
    artifact = _artifact()
    package = build_evidence_package(artifact)

    assert package.package_version == EVIDENCE_PACKAGE_VERSION
    assert len(package.package_id) == 64
    assert len(package.package_sha256) == 64
    assert package.student_text == artifact.claim.text
    assert package.student_text_sha256 == hashlib.sha256(
        artifact.claim.text.encode("utf-8")
    ).hexdigest()
    assert package.source_binding.reference_id == "reference-1"
    assert package.source_identity.content_sha256 == artifact.source_identity.content_sha256
    assert package.extracted_text_sha256 == artifact.coverage.extracted_text_sha256
    assert package.retrieval.source_absence_claim_permitted is False
    assert package.retrieval.candidate_availability == "candidates_retrieved"
    assert package.reference_discovery.status == "not_run"
    assert package.quotation_check.outcome == "not_applicable"
    assert package.locator_check.outcome == "not_applicable"
    assert "relationship" not in package.model_dump(mode="json")
    assert "judgment" not in package.model_dump(mode="json")
    validate_evidence_package(package)


def test_evidence_package_preserves_semantic_retrieval_provenance():
    artifact = _artifact()
    retrieval = artifact.candidate_passage_retrieval.model_copy(
        update={
            "semantic_rescue_status": "complete",
            "semantic_rescue_version": "bm25-prefilter-local-nli-v1",
            "semantic_model_id": "local-model",
            "semantic_model_revision": "revision-1",
            "semantic_prefilter_count": 32,
            "semantic_addition_count": 2,
        }
    )
    package = build_evidence_package(
        artifact.model_copy(update={"candidate_passage_retrieval": retrieval})
    )

    assert package.retrieval.semantic_rescue_status == "complete"
    assert package.retrieval.semantic_rescue_version == "bm25-prefilter-local-nli-v1"
    assert package.retrieval.semantic_model_id == "local-model"
    assert package.retrieval.semantic_model_revision == "revision-1"
    assert package.retrieval.semantic_prefilter_count == 32
    assert package.retrieval.semantic_addition_count == 2
    validate_evidence_package(package)


def test_evidence_package_hash_rejects_mutation():
    package = build_evidence_package(_artifact())
    tampered = package.model_copy(
        update={"student_text": "Different student wording (Smith, 2020)."}
    )

    with pytest.raises(EvidencePackageError, match="content hash"):
        validate_evidence_package(tampered)


def test_evidence_package_can_bind_completed_reference_discovery():
    artifact = _artifact()
    normalized_query = "title:careful checking improves accuracy"
    query = ReferenceSearchQuery(
        query_id="query-1",
        route_category="academic_adapter",
        provider="crossref",
        normalized_query=normalized_query,
        query_sha256=hashlib.sha256(normalized_query.encode()).hexdigest(),
    )
    discovery = derive_reference_discovery_record(
        reference_id="reference-1",
        expected=ExpectedBibliographicFields(
            title="Careful checking improves accuracy"
        ),
        required_route_categories=["academic_adapter"],
        queries=[query],
        attempts=[
            ReferenceRouteAttempt(
                attempt_id="attempt-1",
                route_category="academic_adapter",
                provider="crossref",
                required=True,
                permitted=True,
                query_ids=["query-1"],
                outcome="no_match",
                started_at=artifact.created_at,
                completed_at=artifact.created_at,
            )
        ],
        candidates=[],
        created_at=artifact.created_at,
    )

    package = build_evidence_package(
        artifact,
        reference_discovery=discovery,
    )

    assert package.reference_discovery.outcome == "unlocated_after_search"
    assert package.reference_discovery.contributes_to_neutral_pattern is True
    validate_evidence_package(package)


def test_evidence_package_rejects_discovery_for_another_reference():
    artifact = _artifact()
    discovery = derive_reference_discovery_record(
        reference_id="different-reference",
        expected=ExpectedBibliographicFields(title="Too little"),
        required_route_categories=[],
        queries=[],
        attempts=[],
        candidates=[],
        created_at=artifact.created_at,
    )

    with pytest.raises(EvidencePackageError, match="does not match"):
        build_evidence_package(artifact, reference_discovery=discovery)


def test_evidence_package_records_completed_full_span_quotation_check():
    artifact = _artifact("Careful checking improves accuracy.")
    marker = "(Smith, 2020)"
    claim_text = f'"Careful checking improves accuracy." {marker}'
    claim = artifact.claim.model_copy(
        update={
            "text": claim_text,
            "claim_type": "quotation",
            "citation_marker": marker,
            "passage_end": 100 + len(claim_text),
        }
    )
    rebuilt = build_passage_evidence(
        AuthorizedRepresentation(
            representation_id=artifact.source_identity.representation_id,
            canonical_work_id=artifact.source_identity.canonical_work_id,
            content_object_id="object-package-quote",
            content_sha256=artifact.source_identity.content_sha256,
            content=b"Careful checking improves accuracy.",
            representation_kind="plain_text",
            media_type="text/plain",
            provenance="authorized_upload",
            scope_type="personal_owner",
            scope_id="owner-1",
            identity_verdict="verified",
            identity_confidence=0.99,
            completeness_verdict="complete",
            text_quality="digital",
            edition_or_version="edition-1",
            created_at=artifact.created_at,
            admitted_at=artifact.created_at,
        ),
        claim=claim,
    )
    package = build_evidence_package(rebuilt)

    assert package.quotation_check.status == "complete"
    assert package.quotation_check.outcome == "all_spans_literal_match"
    assert package.quotation_check.evidence_passage_ids == [
        rebuilt.passages[0].passage_id
    ]


def test_evidence_package_rejects_displayed_reference_list_material():
    artifact = _artifact()
    passage = artifact.passages[0].model_copy(update={"passage_role": "reference_list"})

    with pytest.raises(EvidencePackageError, match="excluded source role"):
        build_evidence_package(artifact.model_copy(update={"passages": [passage]}))


def test_evidence_package_rejects_unknown_workflow_evidence_id():
    artifact = _artifact()
    quotation_check = artifact.quotation_check.model_copy(
        update={"evidence_passage_ids": ["unknown-passage"]}
    )

    with pytest.raises(EvidencePackageError, match="workflow check"):
        build_evidence_package(
            artifact.model_copy(update={"quotation_check": quotation_check})
        )


def test_evidence_package_consolidates_overlapping_visible_slots_without_losing_ids():
    artifact = _artifact("Careful checking improves accuracy across repeated reviews.")
    first = artifact.passages[0]
    second = first.model_copy(
        update={
            "passage_id": "overlapping-passage-2",
            "character_start": first.character_start + 5,
            "character_end": first.character_end - 1,
            "text": first.text[5:-1],
        }
    )

    package = build_evidence_package(
        artifact.model_copy(update={"passages": [first, second]})
    )

    assert package.retrieval.displayed_passage_ids == [first.passage_id]
    assert package.retrieval.display_consolidations[first.passage_id] == [
        first.passage_id,
        second.passage_id,
    ]


def test_candidate_evidence_leads_without_dropping_protected_baseline():
    order = _ordered_retrieval_ids(
        ["baseline-1", "baseline-2", "baseline-3", "baseline-4"],
        ["candidate-1", "baseline-2", "candidate-2"],
        ["baseline-1", "baseline-2", "baseline-3", "baseline-4", "candidate-1", "candidate-2"],
    )

    assert order == [
        "candidate-1",
        "baseline-2",
        "baseline-1",
        "candidate-2",
        "baseline-3",
        "baseline-4",
    ]


def test_report_persists_authoritative_package_separately_from_experimental_views():
    artifact = _artifact()
    payload = build_inspectable_report_payload(artifact)
    package = payload["authoritative_evidence_package"]

    assert package["package_version"] == EVIDENCE_PACKAGE_VERSION
    assert package["package_sha256"]
    assert package["retrieval"]["source_absence_claim_permitted"] is False
    assert payload["structured_judgment"]["decision_applied"] is False
