"""Subject comparison after the Belton and Khan cases.

Two questions are deliberately separated here. Whether the app knows which
work a reference names, which decides if its subject may be compared at all,
and whether the scope judgment is bound to the text it was made against.
"""
import hashlib

from app.services.evidence_report import _title_and_author_agree
from app.services.report_layers import _reference_identified, topical_mismatch
from app.services.passage_relevance import assess_retrieved_text_scope


def _candidate(agreed, **flags):
    return {
        "plausible_identity_match": flags.get("plausible", True),
        "comparisons": [{"field_name": name, "outcome": "agreement"} for name in agreed]
        + [{"field_name": "year", "outcome": "material_conflict"}],
    }


class TestWhichWorkIsThis:
    def test_title_and_author_agreement_is_recognised(self):
        assert _title_and_author_agree({"candidates": [_candidate(["title", "author"])]})

    def test_title_alone_is_not_enough(self):
        assert not _title_and_author_agree({"candidates": [_candidate(["title"])]})

    def test_an_implausible_candidate_does_not_count(self):
        # The Belton failure: records were returned, none was the cited work.
        assert not _title_and_author_agree(
            {"candidates": [_candidate(["title", "author"], plausible=False)]})

    def test_no_candidates_at_all(self):
        assert not _title_and_author_agree({"candidates": []})


class TestSubjectComparisonEligibility:
    def test_confirmed_reference_qualifies(self):
        assert _reference_identified({"reference_identity": {"status": "confirmed"}})

    def test_wrong_year_on_a_recognised_work_qualifies(self):
        """Khan: title and author agree, the student's year is wrong."""
        assert _reference_identified({"reference_identity": {
            "status": "bibliographic_conflict", "title_and_author_agree": True}})

    def test_conflict_without_a_recognised_work_does_not(self):
        assert not _reference_identified({"reference_identity": {
            "status": "bibliographic_conflict", "title_and_author_agree": False}})

    def test_unlocated_reference_never_qualifies(self):
        """Belton: never identified, so its displayed subject proves nothing."""
        for status in ("search_incomplete", "unlocated_after_search",
                       "possible_match", "insufficient_metadata"):
            assert not _reference_identified({"reference_identity": {
                "status": status, "title_and_author_agree": True}}), status


def _scope_member(coverage, text, claim, *, version, scope_coverage, identity="confirmed"):
    scope = {
        "status": "complete", "relevance": "apparent_mismatch", "attention": True,
        "confidence": "high", "discrepancy": "incompatible_stated_scope",
        "stated_scope_conflict": "present", "scope_dimension": "jurisdiction",
        "source_scope": "United States", "claim_scope": "Canada",
        "broad_subject_relation": "compatible", "subject_comparison": "platforms vs telecom",
        "rationale": "The source states a United States scope.",
        "abstract_span": text[:24], "claim_span": claim[:14],
        "scope_policy_version": version, "scope_coverage": scope_coverage,
        "abstract_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
    }
    member = {
        "coverage_level": coverage,
        "reference_identity": {"status": identity, "title_and_author_agree": True},
        "abstract_relevance": {
            "status": "complete", "scope_assessment": scope,
            "abstract_sha256": scope["abstract_sha256"],
            "claim_sha256": scope["claim_sha256"]},
    }
    if coverage == "abstract_only":
        member["best_evidence"] = {"text": text}
    else:
        member["scope_source"] = {"text": text}
    return member


CLAIM = "Canadian telecommunications policy favours incumbents."
SOURCE = "This article examines United States platform markets and their regulation."


class TestRetrievedDocumentScope:
    def test_full_text_mismatch_is_now_marked(self):
        """Khan: the check used to be skipped entirely once text was retrieved."""
        member = _scope_member("full_text", SOURCE, CLAIM,
                               version="fulltext-topic-v1", scope_coverage="full_text")
        assert topical_mismatch(member, {"student_text": CLAIM})

    def test_partial_text_is_also_comparable(self):
        member = _scope_member("partial_text", SOURCE, CLAIM,
                               version="fulltext-topic-v1", scope_coverage="partial_text")
        assert topical_mismatch(member, {"student_text": CLAIM})

    def test_abstract_contract_still_holds(self):
        member = _scope_member("abstract_only", SOURCE, CLAIM,
                               version="abstract-topic-v5", scope_coverage="abstract_only")
        assert topical_mismatch(member, {"student_text": CLAIM})

    def test_legacy_abstract_record_without_coverage_still_reads(self):
        member = _scope_member("abstract_only", SOURCE, CLAIM,
                               version="abstract-topic-v4", scope_coverage="abstract_only")
        member["abstract_relevance"]["scope_assessment"].pop("scope_coverage")
        member["abstract_relevance"]["scope_assessment"].update(
            topic_relation="disjoint", broad_subject_relation="incompatible",
            plausible_connection="absent", discrepancy="different_subject")
        assert topical_mismatch(member, {"student_text": CLAIM})

    def test_a_judgment_made_on_an_abstract_cannot_mark_a_full_text_member(self):
        member = _scope_member("full_text", SOURCE, CLAIM,
                               version="abstract-topic-v5", scope_coverage="abstract_only")
        assert not topical_mismatch(member, {"student_text": CLAIM})

    def test_unbound_scope_text_never_marks(self):
        member = _scope_member("full_text", SOURCE, CLAIM,
                               version="fulltext-topic-v1", scope_coverage="full_text")
        member["scope_source"] = {}
        assert not topical_mismatch(member, {"student_text": CLAIM})

    def test_text_altered_after_assessment_never_marks(self):
        member = _scope_member("full_text", SOURCE, CLAIM,
                               version="fulltext-topic-v1", scope_coverage="full_text")
        member["scope_source"] = {"text": SOURCE + " Revised."}
        assert not topical_mismatch(member, {"student_text": CLAIM})

    def test_unidentified_full_text_member_never_marks(self):
        member = _scope_member("full_text", SOURCE, CLAIM,
                               version="fulltext-topic-v1", scope_coverage="full_text",
                               identity="search_incomplete")
        assert not topical_mismatch(member, {"student_text": CLAIM})


class TestRetrievedScopeEntryPoint:
    def test_unsupported_coverage_is_refused(self):
        result = assess_retrieved_text_scope(object(), "text", coverage="abstract_only")
        assert result == {"status": "not_assessed", "outcome": "unsupported_coverage"}

    def test_empty_document_is_not_assessed(self):
        result = assess_retrieved_text_scope(object(), "   ", coverage="full_text")
        assert result["status"] == "not_assessed"
