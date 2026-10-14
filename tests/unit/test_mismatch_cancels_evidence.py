"""A source ruled out on subject should not also be quoted as evidence for it.

Selecting passages from a work and presenting them as support for a claim the
same report marks as topically unsupported tells the reader two opposite
things about one source. It also spends two model calls producing material the
report then withholds.
"""
import hashlib

import pytest

from app.services import paper_workflow
from app.services.evidence_report import _withhold_mismatched_evidence
from app.services.report_layers import scope_mark_qualifies

CLAIM = "Canadian telecommunications policy favours incumbents."
EXCERPT = "This article examines United States platform markets and their regulation."


def _scope(**overrides):
    scope = {
        "status": "complete",
        "relevance": "apparent_mismatch", "attention": True, "confidence": "high",
        "discrepancy": "incompatible_stated_scope", "stated_scope_conflict": "present",
        "scope_dimension": "jurisdiction", "source_scope": "United States",
        "claim_scope": "Canada", "broad_subject_relation": "compatible",
        "subject_comparison": "US platforms against Canadian telecom",
        "rationale": "The source states a United States scope.",
        "abstract_span": EXCERPT[:24], "claim_span": CLAIM[:14],
        "scope_policy_version": "fulltext-topic-v1", "scope_coverage": "partial_text",
        "abstract_sha256": hashlib.sha256(EXCERPT.encode()).hexdigest(),
        "claim_sha256": hashlib.sha256(CLAIM.encode()).hexdigest(),
    }
    scope.update(overrides)
    return scope


def _member(**overrides):
    scope = overrides.pop("scope", _scope())
    member = {
        "coverage_level": "partial_text",
        "reference_identity": {"status": "confirmed"},
        "scope_source": {"text": EXCERPT},
        "best_evidence": {"text": "A passage selected from the work."},
        "additional_evidence": [{"text": "Another passage."}],
        "relevance_status": "connected",
        "abstract_relevance": {
            "status": "complete", "scope_assessment": scope,
            "abstract_sha256": scope["abstract_sha256"],
            "claim_sha256": scope["claim_sha256"]},
    }
    member.update(overrides)
    return member


class TestWithholdingTheDisplay:
    def test_passages_are_withheld_from_a_mismatched_source(self):
        item = _withhold_mismatched_evidence(_member(), {"student_text": CLAIM})
        assert item["best_evidence"] is None
        assert item["additional_evidence"] == []
        assert item["topical_mismatch_withheld_evidence"] == 2
        assert "topical mismatch" in item["availability"]

    def test_the_comparison_that_raised_it_is_kept(self):
        """Withholding the explanation would leave an unexplained mark."""
        item = _withhold_mismatched_evidence(_member(), {"student_text": CLAIM})
        assert item["scope_source"]["text"] == EXCERPT
        assert item["abstract_relevance"]["scope_assessment"]["rationale"]

    def test_an_unmarked_source_keeps_its_evidence(self):
        item = _withhold_mismatched_evidence(
            _member(scope=_scope(relevance="generally_relevant", attention=False)),
            {"student_text": CLAIM})
        assert item["best_evidence"] is not None
        assert item["additional_evidence"]

    def test_an_abstract_member_is_untouched(self):
        """There the displayed abstract IS the text the judgment was made on."""
        member = _member(coverage_level="abstract_only")
        assert _withhold_mismatched_evidence(member, {"student_text": CLAIM}) is member


class TestSharedPredicate:
    """The workflow must never skip work the report would not have marked."""

    def test_a_qualifying_judgment_is_recognised(self):
        assert scope_mark_qualifies(_scope(), "partial_text")

    @pytest.mark.parametrize("override", [
        {"relevance": "generally_relevant"}, {"attention": False},
        {"confidence": "medium"}, {"stated_scope_conflict": "absent"},
    ])
    def test_a_non_qualifying_judgment_is_not(self, override):
        assert not scope_mark_qualifies(_scope(**override), "partial_text")

    def test_unsupported_coverage_never_qualifies(self):
        assert not scope_mark_qualifies(_scope(), "metadata_only")

    def test_a_different_subject_needs_whole_document_support(self):
        ground = _scope(discrepancy="different_subject", topic_relation="disjoint",
                        broad_subject_relation="incompatible",
                        plausible_connection="absent", stated_scope_conflict="absent")
        assert not scope_mark_qualifies({**ground, "claim_terms_present": 10,
                                         "claim_terms_total": 12}, "partial_text")
        assert scope_mark_qualifies({**ground, "claim_terms_present": 0,
                                     "claim_terms_total": 12}, "partial_text")


class TestWorkflowShortCircuit:
    def _artifact(self, status, scope, coverage="partial_text"):
        from app.services.verification_evidence import SourceScopeAssessmentEvidence
        record = SourceScopeAssessmentEvidence(
            status=status, coverage=coverage, excerpt=EXCERPT,
            excerpt_sha256="x" * 64,
            assessment={"status": "complete", "scope_assessment": scope},
            claim_terms_present=0, claim_terms_total=12)
        return type("A", (), {"source_scope_assessment": record})()

    def test_an_established_mismatch_stops_the_evidence_work(self):
        assert paper_workflow._scope_mismatch_established(
            self._artifact("complete", _scope()))

    def test_an_ordinary_judgment_does_not(self):
        assert not paper_workflow._scope_mismatch_established(
            self._artifact("complete", _scope(relevance="generally_relevant",
                                              attention=False)))

    def test_an_unrun_comparison_does_not(self):
        assert not paper_workflow._scope_mismatch_established(
            self._artifact("not_run", _scope()))

    def test_a_missing_record_does_not(self):
        assert not paper_workflow._scope_mismatch_established(
            type("A", (), {"source_scope_assessment": None})())


class TestOwnerDecision20261014:
    """Topical mismatch is for abstracts and limited text, never for a fully
    retrieved text, and never for a citation of three or more sources
    (owner decisions 2026-10-14 and 2026-09-30; Harlow citation 7)."""

    def test_a_fully_retrieved_text_is_never_a_mismatch(self):
        assert not scope_mark_qualifies(_scope(scope_coverage="full_text"), "full_text")
        member = _member(coverage_level="full_text", scope=_scope(scope_coverage="full_text"))
        assert _withhold_mismatched_evidence(member, {"student_text": CLAIM})["best_evidence"] is not None

    def test_limited_text_can_still_be_a_mismatch(self):
        assert scope_mark_qualifies(_scope(), "partial_text")
        item = _withhold_mismatched_evidence(_member(coverage_level="partial_text"), {"student_text": CLAIM})
        assert item["best_evidence"] is None

    def test_three_sources_skip_no_evidence_work(self):
        assert scope_mark_qualifies(_scope(), "partial_text", sources=2)
        assert not scope_mark_qualifies(_scope(), "partial_text", sources=3)
        member = _member(coverage_level="partial_text")
        three = {"student_text": CLAIM, "members": [member, {}, {}]}
        assert _withhold_mismatched_evidence(member, three)["best_evidence"] is not None
