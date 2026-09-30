"""The retrieved-document scope path, end to end through the projection.

Khan's case: full text was retrieved, which used to remove the scope check
entirely. These fix the wiring so the judgment reaches the report member bound
to the text it was made against.
"""
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

from app.services import paper_workflow
from app.services.report_layers import topical_mismatch
from app.services.verification_evidence import (
    CoverageLevel, SourceScopeAssessmentEvidence, leading_source_excerpt,
)

CLAIM_TEXT = "Canadian telecommunications policy favours incumbents."
OPENING = "This article examines United States platform markets and their regulation."


def _claim(text: str = "Canadian telecommunications policy favours incumbents."):
    """A claim stub with the fields the scope envelope and term count read."""
    return SimpleNamespace(text=text, antecedent_context=[], source_segments=[],
                           citation_marker="(Author, 2020)",
                           citation_marker_type="parenthetical")


class _Page:
    def __init__(self, text):
        self.text = text


class TestLeadingExcerpt:
    def test_opening_pages_are_joined_and_bounded(self, monkeypatch):
        monkeypatch.setattr(paper_workflow, "leading_source_excerpt", leading_source_excerpt)
        import app.services.verification_evidence as ve
        monkeypatch.setattr(ve, "_extract_pages",
                            lambda s: ([_Page("A" * 2000), _Page("B" * 2000)], []))
        excerpt = leading_source_excerpt(Mock(), limit=3_500)
        assert len(excerpt) == 3_500
        assert excerpt.startswith("A")

    def test_extraction_failure_abstains(self, monkeypatch):
        import app.services.verification_evidence as ve
        monkeypatch.setattr(ve, "_extract_pages",
                            Mock(side_effect=RuntimeError("unreadable")))
        assert leading_source_excerpt(Mock()) == ""

    def test_whitespace_is_normalised(self, monkeypatch):
        import app.services.verification_evidence as ve
        monkeypatch.setattr(ve, "_extract_pages",
                            lambda s: ([_Page("two\n\n  words")], []))
        assert leading_source_excerpt(Mock()) == "two words"


class TestWorkflowAttachment:
    def _evidence(self, level):
        """The real enum, not a string.

        `CoverageLevel` is an Enum whose str() is its member name; a stringly
        typed stand-in hid that and the attachment silently never ran.
        """
        evidence = Mock()
        evidence.coverage.level = CoverageLevel(level)
        evidence.model_copy = Mock(side_effect=lambda update: update)
        return evidence

    def test_full_text_coverage_is_assessed(self, monkeypatch):
        called = {}

        def fake(claim, text, *, source_title, coverage):
            called.update(text=text, coverage=coverage, source_title=source_title)
            return {"status": "complete", "scope_assessment": {"status": "complete"}}

        monkeypatch.setattr(paper_workflow, "assess_retrieved_text_scope", fake)
        result = paper_workflow._attach_source_scope(
            self._evidence("full_text"), _claim(), OPENING, source_title="Platforms")

        record = result["source_scope_assessment"]
        assert isinstance(record, SourceScopeAssessmentEvidence)
        # The stored text is what was sent: the opening plus any further
        # passages from the same document.
        assert record.excerpt.startswith(OPENING)
        assert record.excerpt_sha256 == hashlib.sha256(OPENING.encode()).hexdigest()
        assert called["coverage"] == "full_text"

    def test_coverage_enum_is_read_by_value_not_name(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(paper_workflow, "assess_retrieved_text_scope",
                            lambda c, t, *, source_title, coverage: seen.setdefault(
                                "coverage", coverage) and None or {"status": "complete"})
        paper_workflow._attach_source_scope(
            self._evidence("partial_text"), _claim(), OPENING, source_title="")
        assert seen["coverage"] == "partial_text"

    def test_abstract_coverage_is_left_to_its_own_contract(self, monkeypatch):
        monkeypatch.setattr(paper_workflow, "assess_retrieved_text_scope",
                            Mock(side_effect=AssertionError("must not run")))
        evidence = self._evidence("abstract")
        assert paper_workflow._attach_source_scope(
            evidence, Mock(), OPENING, source_title="") is evidence

    def test_missing_excerpt_leaves_no_comparison(self, monkeypatch):
        monkeypatch.setattr(paper_workflow, "assess_retrieved_text_scope",
                            Mock(side_effect=AssertionError("must not run")))
        evidence = self._evidence("full_text")
        assert paper_workflow._attach_source_scope(
            evidence, Mock(), "", source_title="") is evidence


class TestProjectionReachesTheGate:
    """The projected member must satisfy the gate it was built for."""

    def _scope(self, text, claim):
        return {
            "status": "complete", "relevance": "apparent_mismatch", "attention": True,
            "confidence": "high", "discrepancy": "incompatible_stated_scope",
            "stated_scope_conflict": "present", "scope_dimension": "jurisdiction",
            "source_scope": "United States", "claim_scope": "Canada",
            "broad_subject_relation": "compatible",
            "subject_comparison": "US platforms against Canadian telecom",
            "rationale": "The source states a United States scope.",
            "abstract_span": text[:24], "claim_span": claim[:14],
            "scope_policy_version": "fulltext-topic-v1", "scope_coverage": "full_text",
            "abstract_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
        }

    def test_projected_member_marks_the_mismatch(self):
        scope = self._scope(OPENING, CLAIM_TEXT)
        member = {
            "coverage_level": "full_text",
            "reference_identity": {"status": "bibliographic_conflict",
                                   "title_and_author_agree": True},
            "scope_source": {"text": OPENING},
            "abstract_relevance": {
                "status": "complete", "scope_assessment": scope,
                "abstract_sha256": scope["abstract_sha256"],
                "claim_sha256": scope["claim_sha256"]},
        }
        assert topical_mismatch(member, {"student_text": CLAIM_TEXT})

    def test_member_without_a_scope_source_is_inert(self):
        assert not topical_mismatch(
            {"coverage_level": "full_text", "scope_source": {},
             "reference_identity": {"status": "confirmed"}, "abstract_relevance": {}},
            {"student_text": CLAIM_TEXT})


class TestPayloadCarriesTheJudgment:
    """The projection is explicit, so a new artifact field must be added to it.

    The first wired run produced zero comparisons because the field existed on
    the artifact and was silently dropped when the report payload was built.
    """

    def test_scope_assessment_is_projected_into_the_payload(self):
        import inspect
        from app.services import verification_report
        source = inspect.getsource(verification_report.build_inspectable_report_payload)
        assert '"source_scope_assessment"' in source

    def test_excerpt_bound_matches_stored_source_text_bound(self):
        from app.services.verification_evidence import MAX_SCOPE_EXCERPT_CHARACTERS
        from app.services.verification_report import MAX_REPORT_PASSAGE_CHARACTERS
        assert MAX_SCOPE_EXCERPT_CHARACTERS <= MAX_REPORT_PASSAGE_CHARACTERS


class TestChainFromArtifactToMark:
    """Walk the whole path with a real artifact: attach, persist, project, mark.

    Two silent failures reached a paid run before this existed: the coverage
    enum was compared as a string, and the payload projection dropped the
    field. Both looked like success and produced nothing.
    """

    def _artifact_with_scope(self, monkeypatch, excerpt, claim_text):
        from tests.unit.test_verification_report import _artifact
        artifact = _artifact()
        artifact = artifact.model_copy(update={
            "coverage": artifact.coverage.model_copy(
                update={"level": CoverageLevel("full_text")})})

        scope = {
            "status": "complete", "relevance": "apparent_mismatch", "attention": True,
            "confidence": "high", "discrepancy": "incompatible_stated_scope",
            "stated_scope_conflict": "present", "scope_dimension": "jurisdiction",
            "source_scope": "United States", "claim_scope": "Canada",
            "broad_subject_relation": "compatible",
            "subject_comparison": "US platforms against Canadian telecom",
            "rationale": "The source states a United States scope.",
            "abstract_span": excerpt[:24], "claim_span": claim_text[:14],
            "scope_policy_version": "fulltext-topic-v1", "scope_coverage": "full_text",
            "abstract_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
            "claim_sha256": hashlib.sha256(claim_text.encode()).hexdigest(),
        }
        def fake(c, t, *, source_title, coverage):
            # Bind to the text actually sent, as the real call does.
            bound = {**scope, "abstract_span": t[:24],
                     "abstract_sha256": hashlib.sha256(t.encode()).hexdigest()}
            return {"status": "complete", "scope_assessment": bound,
                    "abstract_sha256": bound["abstract_sha256"],
                    "claim_sha256": bound["claim_sha256"]}
        monkeypatch.setattr(paper_workflow, "assess_retrieved_text_scope", fake)
        return paper_workflow._attach_source_scope(
            artifact, _claim(), excerpt, source_title="Platforms")

    def test_excerpt_survives_into_the_persisted_payload(self, monkeypatch):
        from app.services.verification_report import build_inspectable_report_payload
        claim_text = "Canadian telecommunications policy favours incumbents."
        artifact = self._artifact_with_scope(monkeypatch, OPENING, claim_text)

        assert artifact.source_scope_assessment.excerpt.startswith(OPENING)
        payload = build_inspectable_report_payload(artifact)

        record = payload["source_scope_assessment"]
        assert record["excerpt"].startswith(OPENING)
        assert record["coverage"] == "full_text"
        assert record["assessment"]["scope_assessment"]["relevance"] == "apparent_mismatch"

    def test_payload_projects_into_a_member_the_gate_marks(self, monkeypatch):
        from app.services.verification_report import build_inspectable_report_payload
        claim_text = "Canadian telecommunications policy favours incumbents."
        artifact = self._artifact_with_scope(monkeypatch, OPENING, claim_text)
        payload = build_inspectable_report_payload(artifact)

        # The projection evidence_report performs for a persisted member.
        record = payload["source_scope_assessment"]
        member = {
            "coverage_level": payload["coverage"]["level"],
            "reference_identity": {"status": "confirmed"},
            "scope_source": {"text": record["excerpt"]},
            "abstract_relevance": {
                **record["assessment"],
                "abstract_sha256": record["assessment"]["abstract_sha256"]},
        }
        assert topical_mismatch(member, {"student_text": claim_text})


class TestProjectionReadsTheRightLevel:
    """`package` is the evidence sub-dict; the scope record is at the root.

    Reading the wrong level produced a run where every payload carried a
    complete assessment with a 1,200-character excerpt and every report member
    showed none.
    """

    def test_scope_record_is_read_from_the_payload_root(self):
        import inspect
        from app.services import evidence_report
        source = inspect.getsource(evidence_report._available_member_view)
        line = next(l for l in source.splitlines() if "scope_record =" in l)
        assert "report_payload" in line
        assert "package.get" not in line

    def test_root_and_package_levels_are_distinct_in_a_real_payload(self, monkeypatch):
        from app.services.verification_report import build_inspectable_report_payload
        from tests.unit.test_verification_report import _artifact
        artifact = _artifact().model_copy(update={"source_scope_assessment":
            SourceScopeAssessmentEvidence(status="complete", coverage="full_text",
                                          excerpt=OPENING, excerpt_sha256="x" * 64,
                                          assessment={"status": "complete"})})
        payload = build_inspectable_report_payload(artifact)

        assert "source_scope_assessment" in payload
        assert "source_scope_assessment" not in payload["authoritative_evidence_package"]
