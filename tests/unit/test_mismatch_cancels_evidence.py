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
from app.services.report_layers import topical_mismatch

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


class TestTheMarkDecidesTheWithholding:
    """The report's mark alone decides; the run no longer has a rule of its own
    (owner decision 2026-10-15)."""

    @pytest.mark.parametrize("override", [
        {"relevance": "generally_relevant"}, {"attention": False},
        {"confidence": "medium"}, {"stated_scope_conflict": "absent"},
    ])
    def test_a_non_qualifying_judgment_keeps_the_evidence(self, override):
        member = _member(scope=_scope(**override))
        assert not topical_mismatch(member, {"student_text": CLAIM})
        assert _withhold_mismatched_evidence(member, {"student_text": CLAIM})["best_evidence"] is not None

    def test_a_different_subject_needs_whole_document_support(self):
        ground = _scope(discrepancy="different_subject", topic_relation="disjoint",
                        broad_subject_relation="incompatible",
                        plausible_connection="absent", stated_scope_conflict="absent")
        common = {"student_text": CLAIM}
        assert not topical_mismatch(_member(scope={**ground, "claim_terms_present": 10, "claim_terms_total": 12}), common)
        assert topical_mismatch(_member(scope={**ground, "claim_terms_present": 0, "claim_terms_total": 12}), common)


class TestOwnerDecision20261014:
    """Topical mismatch is for abstracts and limited text, never for a fully
    retrieved text, and never for a citation of three or more sources
    (owner decisions 2026-10-14 and 2026-10-18; Harlow citation 7)."""

    def test_a_fully_retrieved_text_is_never_a_mismatch(self):
        member = _member(coverage_level="full_text", scope=_scope(scope_coverage="full_text"))
        assert not topical_mismatch(member, {"student_text": CLAIM})
        assert _withhold_mismatched_evidence(member, {"student_text": CLAIM})["best_evidence"] is not None

    def test_limited_text_can_still_be_a_mismatch(self):
        item = _withhold_mismatched_evidence(_member(coverage_level="partial_text"), {"student_text": CLAIM})
        assert item["best_evidence"] is None

    def test_three_sources_keep_their_evidence(self):
        member = _member(coverage_level="partial_text")
        assert topical_mismatch(member, {"student_text": CLAIM, "members": [member, {}]})
        three = {"student_text": CLAIM, "members": [member, {}, {}]}
        assert _withhold_mismatched_evidence(member, three)["best_evidence"] is not None


def test_the_run_gathers_evidence_even_for_a_topical_mismatch(monkeypatch):
    # Owner decision 2026-10-15: the run never skips evidence work on a topical
    # comparison; only the report decides the mark and withholds the evidence.
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from app.services import cross_language_evidence, evidence_sentence_selection, judgment_reserve
    from app.services.verification_evidence import SourceScopeAssessmentEvidence
    evidence = MagicMock(cross_language={})
    evidence.model_copy.return_value = evidence
    mismatch = SourceScopeAssessmentEvidence(
        status="complete", coverage="partial_text", excerpt=EXCERPT, excerpt_sha256="x" * 64,
        assessment={"status": "complete", "scope_assessment": _scope()},
        claim_terms_present=0, claim_terms_total=12)
    def scoped(found, *a, **k):
        found.source_scope_assessment = mismatch
        return found
    called = []
    same = lambda found, *a, **k: found
    for name in ("build_passage_evidence", "attach_verification_candidates", "attach_citation_use_routes",
                 "attach_source_blind_interpretations", "attach_evidence_obligations",
                 "apply_passage_relevance_gate", "attach_facet_evidence_foundation"):
        monkeypatch.setattr(paper_workflow, name, (lambda *a, **k: evidence) if name == "build_passage_evidence" else same)
    monkeypatch.setattr(paper_workflow, "_attach_source_scope", scoped)
    monkeypatch.setattr(paper_workflow, "attach_candidate_passage_retrieval",
                        lambda source, found, **k: called.append("retrieval") or found)
    monkeypatch.setattr(evidence_sentence_selection, "attach_sentence_evidence",
                        lambda found: called.append("selection") or found)
    monkeypatch.setattr(cross_language_evidence, "claim_inputs", lambda *a, **k: (None, []))
    monkeypatch.setattr(judgment_reserve, "reserve_enabled", lambda: False)
    source = SimpleNamespace(identity_verdict="verified")
    claim = SimpleNamespace(text=CLAIM, reference_ids=["r1"])
    paper_workflow._shadow_artifact(source, claim, True, active_reference_id="r1", cited_author_label="Ames")
    assert called == ["retrieval", "selection"]


def test_a_limited_text_excerpt_with_line_breaks_keeps_its_mark():
    # 2026-10-15: the assessment binds to the excerpt with its whitespace
    # collapsed, so the report must compare the same form (944 of 1,030 stored
    # retrieved-text assessments failed the binding; reference 13, two papers).
    stored = "This article examines United States\n\nplatform markets  and their regulation."
    scope = _scope()
    member = _member(coverage_level="partial_text", scope_source={"text": stored}, scope=scope)
    assert topical_mismatch(member, {"student_text": CLAIM})
