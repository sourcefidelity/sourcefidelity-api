"""One flag, with the disagreement shown as evidence rather than a weaker mark.

A mark means the signals agreed. Anything less goes to the reader as the
comparison itself plus the reason it is not a mark -- in the evidence window,
not beside the citation. An abstract is a partial view of a work by
construction, so a disagreement over one is information, not a finding.
"""
import hashlib

import pytest

from app.services.report_layers import scope_disagreement, topical_mismatch

CLAIM = "Distribution in postclassical Hollywood uses multi-platform releases."
TEXT = "This article examines next-generation filmmaking in Australia and digital distribution."


def _scope(**over):
    scope = {
        "status": "complete", "relevance": "apparent_mismatch", "attention": True,
        "confidence": "high", "discrepancy": "incompatible_stated_scope",
        "stated_scope_conflict": "present", "scope_dimension": "geography",
        "source_scope": "filmmaking in Australia", "claim_scope": "postclassical Hollywood",
        "broad_subject_relation": "compatible",
        "subject_comparison": "The abstract concerns Australian filmmaking; the statement concerns Hollywood.",
        "rationale": "Stated scopes differ.",
        "abstract_span": TEXT[:24], "claim_span": CLAIM[:14],
        "scope_policy_version": "abstract-topic-v6", "scope_coverage": "abstract_only",
        "abstract_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
        "claim_sha256": hashlib.sha256(CLAIM.encode()).hexdigest(),
    }
    scope.update(over)
    return scope


def _member(scope=None, identity="confirmed", coverage="abstract_only"):
    scope = scope or _scope()
    return {
        "coverage_level": coverage,
        "reference_identity": {"status": identity},
        "best_evidence": {"text": TEXT},
        "abstract_relevance": {"status": "complete", "scope_assessment": scope,
                               "abstract_sha256": scope["abstract_sha256"],
                               "claim_sha256": scope["claim_sha256"]},
    }


CITATION = {"student_text": CLAIM}


class TestAMarkSpeaksForItself:
    def test_no_note_when_the_signals_agree(self):
        member = _member()
        assert topical_mismatch(member, CITATION)
        assert scope_disagreement(member, CITATION) is None


class TestDisagreementBecomesEvidence:
    @pytest.mark.parametrize("over, reason", [
        ({"confidence": "medium"}, "confidence not high"),
        ({"stated_scope_conflict": "absent"}, "ground unmet"),
        # On the different-subject ground a plausible connection dissents. On
        # the stated-scope ground it does not: that ground overrides topic
        # overlap by design, because the reader is being sent to a source
        # about somewhere else whatever the subjects share.
        ({"discrepancy": "different_subject", "stated_scope_conflict": "absent",
          "topic_relation": "disjoint", "broad_subject_relation": "incompatible",
          "plausible_connection": "present"}, "connection present"),
    ])
    def test_each_dissent_is_named(self, over, reason):
        note = scope_disagreement(_member(_scope(**over)), CITATION)
        assert note is not None
        assert reason in note["reasons"]
        assert note["comparison"].rstrip(". ") in note["note"]
        # The reader is given the comparison, not the rules behind it.
        assert "not marked as a mismatch" not in note["note"]
        assert "grounds" not in note["note"]

    def test_the_scopes_travel_with_the_note(self):
        note = scope_disagreement(_member(_scope(confidence="medium")), CITATION)
        assert note["source_scope"] == "filmmaking in Australia"
        assert note["claim_scope"] == "postclassical Hollywood"
        assert note["scope_dimension"] == "geography"

    def test_distinct_scopes_without_a_conflict_are_shown(self):
        """Ryan: the model named Australia and Hollywood and called it no conflict."""
        note = scope_disagreement(
            _member(_scope(relevance="generally_relevant", attention=False,
                           stated_scope_conflict="absent", discrepancy=None)), CITATION)
        assert note is not None
        assert "scopes differ without conflict" in note["reasons"]

    def test_the_wording_carries_no_accusation(self):
        note = scope_disagreement(_member(_scope(confidence="medium")), CITATION)
        lowered = note["note"].lower()
        for word in ("fabricat", "false", "incorrect", "wrong", "misciting", "dishonest"):
            assert word not in lowered


class TestAbstentions:
    def test_an_unidentified_reference_is_never_described(self):
        """Belton's abstract belonged to a review of another book."""
        for status in ("search_incomplete", "unlocated_after_search", "possible_match"):
            assert scope_disagreement(
                _member(_scope(confidence="medium"), identity=status), CITATION) is None

    def test_an_ordinary_relevant_judgment_says_nothing(self):
        note = scope_disagreement(
            _member(_scope(relevance="generally_relevant", attention=False,
                           stated_scope_conflict="absent", discrepancy=None,
                           source_scope="", claim_scope="", scope_dimension="")), CITATION)
        assert note is None

    def test_an_incomplete_assessment_says_nothing(self):
        assert scope_disagreement(_member(_scope(status="not_assessed")), CITATION) is None

    def test_a_missing_comparison_says_nothing(self):
        assert scope_disagreement(
            _member(_scope(confidence="medium", subject_comparison="", rationale="")),
            CITATION) is None

    def test_unsupported_coverage_says_nothing(self):
        assert scope_disagreement(
            _member(_scope(confidence="medium"), coverage="metadata_only"), CITATION) is None


class TestTheNoteReachesTheReader:
    """Computing a note and never rendering it is the waste it was meant to end.

    The note was attached to the member and read by nothing, so the design
    reached no reader at all.
    """

    def _member_with_note(self):
        return {
            "reference_id": "r1",
            "coverage_level": "abstract_only",
            "availability": "",
            "source": {"author": "Ryan, M. D.", "year": "2010",
                       "title": "Next-generation filmmaking",
                       "raw_reference": "Ryan, M. D. (2010). Next-generation filmmaking."},
            "best_evidence": None,
            "additional_evidence": [],
            "quotation_check": {}, "locator_check": {},
            "scope_disagreement": {
                "note": "The abstract concerns Australian filmmaking; the statement "
                        "concerns Hollywood. Read the source to judge whether it "
                        "supports the statement.",
                "reasons": ["scopes differ without conflict"],
                "comparison": "The abstract concerns Australian filmmaking.",
                "source_scope": "Australia", "claim_scope": "Hollywood",
                "scope_dimension": "geography"},
        }

    def test_the_interactive_report_shows_it(self):
        from app.services.evidence_report import _render_member
        html = _render_member(self._member_with_note())
        assert "Australian filmmaking" in html
        # The closing advice is implied, and dropped even from stored notes
        # (owner decision 2026-10-18).
        assert "Read the source to judge" not in html

    def test_it_is_styled_below_a_mark(self):
        """A mark carries `attention`; a disagreement is visibly weaker."""
        from app.services.evidence_report import _render_member
        html = _render_member(self._member_with_note())
        assert 'class="muted"' in html
        assert 'class="attention">The abstract appears unrelated' not in html

    def test_a_mark_takes_precedence_over_the_note(self):
        from app.services.evidence_report import _render_member
        member = self._member_with_note()
        member["abstract_scope_attention"] = True
        member["abstract_relevance"] = {"scope_assessment": {"rationale": "Stated scopes differ."}}
        html = _render_member(member)
        # The label names the mismatch; the window gives only the reason (2026-09-30).
        assert "appears unrelated" not in html and "Stated scopes differ." in html
        assert "Read the source to judge" not in html

    def test_the_portable_export_shows_it(self):
        from app.services.report_export import _portable_member_html
        html = _portable_member_html(self._member_with_note())
        assert "Australian filmmaking" in html

    def test_a_member_without_a_note_renders_nothing_extra(self):
        from app.services.evidence_report import _render_member
        member = self._member_with_note()
        member.pop("scope_disagreement")
        assert "Read the source to judge" not in _render_member(member)


def test_a_member_carries_no_field_without_a_consumer():
    """`source_completeness` duplicated a verdict already surfaced as prose.

    The completeness verdict reaches the reader through the availability
    statement ("Only part of the source is available..."), which is derived
    from the same value. The raw copy on the member was read by nothing --
    not by a renderer, a decision, the export, the interactive script or a
    test. A field with no consumer is weight the report carries and a reader
    never sees.
    """
    import inspect
    from app.services import evidence_report
    source = inspect.getsource(evidence_report)
    assert '"source_completeness"' not in source
    # The verdict itself still reaches the reader, through the message.
    assert "Only part of the source is available" in source


class TestEveryAbstractShowsItsComparison:
    """Owner decision 2026-10-12: the stored subject comparison is shown for
    every abstract-backed source, also where nothing disagrees."""

    RELEVANT = dict(relevance="generally_relevant", attention=False, stated_scope_conflict="absent",
                    discrepancy=None, source_scope="", claim_scope="", scope_dimension="")

    def test_a_relevant_abstract_with_empty_scopes_gets_the_note(self):
        from app.services.report_layers import abstract_scope_note
        member = _member(_scope(**self.RELEVANT))
        assert scope_disagreement(member, CITATION) is None
        note = abstract_scope_note(member, CITATION)
        assert note == {"note": "The abstract concerns Australian filmmaking; the statement concerns "
                                "Hollywood."}

    @pytest.mark.parametrize("change", ["abstract_hash", "claim_hash", "identity", "llm_off",
                                        "incomplete", "no_comparison", "full_text", "coverage_mismatch",
                                        "truncated_display"])
    def test_unbound_or_unidentified_gets_none(self, change):
        from app.services.report_layers import abstract_scope_note
        scope = _scope(**self.RELEVANT)
        member = _member(scope)
        if change == "abstract_hash":
            member["best_evidence"] = {"text": TEXT + " More."}
        elif change == "claim_hash":
            scope["claim_sha256"] = member["abstract_relevance"]["claim_sha256"] = "0" * 64
        elif change == "identity":
            member["reference_identity"] = {"status": "possible_match"}
        elif change == "llm_off":
            member["abstract_relevance"] = {"status": "not_assessed",
                                            "outcome": "abstract_relevance_unavailable"}
        elif change == "incomplete":
            scope["status"] = "not_assessed"
        elif change == "no_comparison":
            scope["subject_comparison"] = ""
        elif change == "full_text":
            member["coverage_level"] = "full_text"
        elif change == "coverage_mismatch":
            scope["scope_coverage"] = "full_text"
        elif change == "truncated_display":
            member["best_evidence"]["excerpt_truncated"] = True
        assert abstract_scope_note(member, CITATION) is None

    def test_a_topical_mismatch_keeps_its_mark(self):
        from app.services.report_layers import abstract_scope_note
        member = _member()
        assert topical_mismatch(member, CITATION)
        assert abstract_scope_note(member, CITATION) is None

    @staticmethod
    def _with_note(**over):
        member = TestTheNoteReachesTheReader()._member_with_note()
        member.pop("scope_disagreement")
        member["abstract_scope_note"] = {"note": "The abstract concerns Australian filmmaking. "
                                                 "Read the source to judge whether it supports the statement."}
        member.update(over)
        return member

    def test_the_window_shows_it_muted(self):
        from app.services.evidence_report import _render_member
        html = _render_member(self._with_note())
        assert '<p class="muted">The abstract concerns Australian filmmaking.</p>' in html
        assert "Judgment" not in html

    def test_a_limited_text_window_shows_it(self):
        # Owner decision 2026-10-13: limited text gets the note as well.
        from app.services.evidence_report import _render_member
        assert "Australian filmmaking" in _render_member(self._with_note(coverage_level="partial_text"))

    def test_a_full_text_source_does_not_show_it(self):
        from app.services.evidence_report import _render_member
        assert "Australian filmmaking" not in _render_member(self._with_note(coverage_level="full_text"))

    def test_a_mark_takes_precedence(self):
        from app.services.evidence_report import _render_member
        html = _render_member(self._with_note(
            abstract_scope_attention=True,
            abstract_relevance={"scope_assessment": {"rationale": "Stated scopes differ."}}))
        assert "Stated scopes differ." in html and "Read the source to judge" not in html

    def test_the_portable_export_shows_it(self):
        from app.services.report_export import _portable_member_html
        assert "Australian filmmaking" in _portable_member_html(self._with_note())
