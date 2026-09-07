"""Evidence Package retrieval acceptance contract tests."""

from datetime import datetime, timezone
import hashlib

import pytest

from app.services.evidence_package import build_evidence_package
from app.services.retrieval_acceptance import (
    GoldEvidenceSpan,
    MaterialEvidenceFacet,
    RetrievalAcceptanceCase,
    RetrievalAcceptanceCaseResult,
    RetrievalAcceptanceCorpus,
    RetrievalAcceptanceError,
    RetrievedAcceptancePassage,
    evaluate_retrieval_acceptance,
    retrieval_result_from_package,
)
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    build_passage_evidence,
)


_CONTENT_SHA = "a" * 64
_EXTRACTION_SHA = "b" * 64
_QUERY_SHA = "c" * 64


def _case(**updates):
    claim_text = "Careful checking improves accuracy."
    facet_text = "Careful checking improves accuracy."
    span_text = "checking improves"
    values = {
        "case_id": "case-1",
        "source_group_id": "source-1",
        "content_sha256": _CONTENT_SHA,
        "extracted_text_sha256": _EXTRACTION_SHA,
        "query_sha256": _QUERY_SHA,
        "paper_version_id": "paper-1",
        "citation_id": "citation-1",
        "claim_text": claim_text,
        "claim_sha256": hashlib.sha256(claim_text.encode("utf-8")).hexdigest(),
        "source_extraction_version": "test-extractor-v1",
        "review_scope": "complete_source",
        "reviewer_provenance": "complete-source review",
        "reviewed_at": datetime.now(timezone.utc),
        "acceptance_basis": "complete_source_human_review",
        "task_types": ["quotation", "supplied_locator"],
        "material_facet_ids": ["facet-1"],
        "material_facets": [
            MaterialEvidenceFacet(
                facet_id="facet-1",
                text=facet_text,
                text_sha256=hashlib.sha256(facet_text.encode("utf-8")).hexdigest(),
                kind="proposition",
                derivation="human_review",
            )
        ],
        "gold_spans": [
            GoldEvidenceSpan(
                gold_span_id="gold-1",
                page_index=0,
                character_start=10,
                character_end=20,
                text=span_text,
                text_sha256=hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                material_facet_ids=["facet-1"],
                passage_role="body_prose",
            )
        ],
        "protected_baseline_passage_ids": ["passage-1"],
        "supplied_locator_page_indices": [0],
    }
    values.update(updates)
    return RetrievalAcceptanceCase(**values)


def _result(**updates):
    values = {
        "case_id": "case-1",
        "source_group_id": "source-1",
        "content_sha256": _CONTENT_SHA,
        "extracted_text_sha256": _EXTRACTION_SHA,
        "query_sha256": _QUERY_SHA,
        "passages": [
            RetrievedAcceptancePassage(
                passage_id="passage-1",
                rank=1,
                page_index=0,
                character_start=0,
                character_end=40,
                passage_role="body_prose",
                channels=["whole_citation_protected"],
                protected_baseline=True,
            )
        ],
    }
    values.update(updates)
    return RetrievalAcceptanceCaseResult(**values)


def test_acceptance_metrics_apply_all_evidence_first_gates():
    corpus = RetrievalAcceptanceCorpus(
        corpus_id="acceptance-v1", partition="acceptance", cases=[_case()]
    )

    metrics = evaluate_retrieval_acceptance(corpus, [_result()])

    assert metrics.source_binding_rate == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.material_facet_coverage_at_10 == 1.0
    assert metrics.exact_quote_retrieval_rate == 1.0
    assert metrics.supplied_locator_retrieval_rate == 1.0
    assert metrics.protected_baseline_retention == 1.0
    assert all(metrics.gates.values())


def test_acceptance_rejects_hash_or_query_drift():
    corpus = RetrievalAcceptanceCorpus(
        corpus_id="acceptance-v1", partition="acceptance", cases=[_case()]
    )

    with pytest.raises(RetrievalAcceptanceError, match="binding mismatch"):
        evaluate_retrieval_acceptance(
            corpus, [_result(extracted_text_sha256="e" * 64)]
        )


def test_v2_rejects_uninspectable_or_hash_drifting_gold() -> None:
    with pytest.raises(ValueError, match="text hash"):
        GoldEvidenceSpan(
            gold_span_id="gold-1",
            page_index=0,
            character_start=10,
            character_end=20,
            text="checking improves",
            text_sha256="d" * 64,
            material_facet_ids=["facet-1"],
            passage_role="body_prose",
        )

    legacy_case = _case(
        paper_version_id=None,
        citation_id=None,
        claim_text=None,
        claim_sha256=None,
        source_extraction_version=None,
        reviewed_at=None,
        acceptance_basis=None,
        material_facets=[],
        gold_spans=[
            GoldEvidenceSpan(
                gold_span_id="gold-1",
                page_index=0,
                character_start=10,
                character_end=20,
                text_sha256="d" * 64,
                material_facet_ids=["facet-1"],
                passage_role="body_prose",
            )
        ],
    )
    with pytest.raises(ValueError, match="acceptance v2 requires"):
        RetrievalAcceptanceCorpus(
            corpus_id="acceptance-v2", partition="acceptance", cases=[legacy_case]
        )


def test_acceptance_counts_excluded_and_redundant_display_slots():
    passages = [
        RetrievedAcceptancePassage(
            passage_id="passage-1",
            rank=1,
            page_index=0,
            character_start=0,
            character_end=40,
            passage_role="body_prose",
            channels=["lexical"],
            protected_baseline=True,
        ),
        RetrievedAcceptancePassage(
            passage_id="passage-2",
            rank=2,
            page_index=0,
            character_start=5,
            character_end=35,
            passage_role="reference_list",
            channels=["semantic"],
        ),
    ]
    corpus = RetrievalAcceptanceCorpus(
        corpus_id="acceptance-v1", partition="acceptance", cases=[_case()]
    )

    metrics = evaluate_retrieval_acceptance(corpus, [_result(passages=passages)])

    assert metrics.excluded_role_slot_count == 1
    assert metrics.redundant_slot_pair_count == 1
    assert metrics.gates["zero_excluded_role_slots"] is False
    assert metrics.gates["zero_redundant_slots"] is False


def test_package_projection_preserves_protected_passages_and_hashes():
    source_text = "Careful checking improves accuracy."
    content = source_text.encode("utf-8")
    now = datetime.now(timezone.utc)
    source = AuthorizedRepresentation(
        representation_id="representation-1",
        canonical_work_id="source-1",
        content_object_id="object-1",
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
    claim = ClaimEvidence(
        claim_id="claim-1",
        paper_version_id="paper-1",
        text="Careful checking improves accuracy (Smith, 2020).",
        reference_ids=["reference-1"],
        citation_marker="(Smith, 2020)",
        citation_marker_type="parenthetical",
    )
    package = build_evidence_package(build_passage_evidence(source, claim=claim))

    result = retrieval_result_from_package(
        package,
        case_id="case-1",
        source_group_id="source-1",
        query_sha256=_QUERY_SHA,
    )

    assert result.content_sha256 == package.source_identity.content_sha256
    assert result.extracted_text_sha256 == package.extracted_text_sha256
    assert result.passages
    assert all(passage.protected_baseline for passage in result.passages)
