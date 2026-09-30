"""Judging "a different subject" from a document's opening was the mistake.

Khan's article runs 126 pages; the scope model saw 1,200 characters of it and
concluded the work does not discuss the Telecommunications Act. It discusses
it at length. Pallant's opening omitted narrative; that work is about
narrative. Both full-text different-subject marks in the corpus were this.

The ground is now allowed only when the COMPLETE document also lacks the
vocabulary the citation attributed to it.
"""
import pytest

from app.services.report_layers import (
    MAX_CLAIM_TERM_PRESENCE_FOR_DIFFERENT_SUBJECT,
    MIN_CLAIM_TERMS_FOR_ABSENCE,
    _claim_absent_from_source,
    _qualifying_scope_ground,
)
from app.services.verification_evidence import (
    claim_terms_present_in_source, claim_topic_terms,
)


class TestClaimTerms:
    def test_reporting_verbs_and_stopwords_are_dropped(self):
        terms = claim_topic_terms(
            "According to Pallant (2010), Disney's narrative follows the "
            "convention of the hero's journey.")
        assert "according" not in terms and "follows" not in terms
        assert "narrative" in terms and "convention" in terms

    def test_short_words_are_not_distinctive(self):
        assert all(len(t) >= 5 for t in claim_topic_terms("The act set a new tax on tea."))

    def test_terms_are_bounded_and_unique(self):
        terms = claim_topic_terms(" ".join(f"term{i:03d}word" for i in range(40)))
        assert len(terms) <= 12 and len(terms) == len(set(terms))


class TestUnmeasurableAbstains:
    """A total of zero means not measured, never "the source lacks it"."""

    def test_no_source_abstains(self):
        assert claim_terms_present_in_source(None, "some claim about telecoms") == (0, 0)

    def test_a_non_string_claim_abstains(self):
        assert claim_terms_present_in_source(object(), None) == (0, 0)

    def test_extraction_failure_abstains(self, monkeypatch):
        import app.services.verification_evidence as ve
        monkeypatch.setattr(ve, "_extract_pages",
                            lambda s: (_ for _ in ()).throw(RuntimeError("unreadable")))
        assert claim_terms_present_in_source(object(), "telecommunications policy markets") == (0, 0)

    def test_an_empty_document_abstains(self, monkeypatch):
        import app.services.verification_evidence as ve
        monkeypatch.setattr(ve, "_extract_pages", lambda s: ([], []))
        assert claim_terms_present_in_source(object(), "telecommunications policy markets") == (0, 0)


class TestPresenceCounting:
    def _pages(self, monkeypatch, text):
        import app.services.verification_evidence as ve
        monkeypatch.setattr(ve, "_extract_pages",
                            lambda s: ([type("P", (), {"text": text})()], []))

    def test_a_discussed_topic_is_found_beyond_the_opening(self, monkeypatch):
        self._pages(monkeypatch, "Opening about formalism, with no other topic named. " * 20 +
                    "Later the article turns to Disney narrative and to the conventions "
                    "of the mythic journey.")
        present, total = claim_terms_present_in_source(
            object(), "Disney's narrative follows the convention of the hero's journey.")
        # disney, narrative, convention and journey all appear further in, even
        # though the opening names none of them.
        assert total >= 4 and present >= 4

    def test_inflection_is_absorbed(self, monkeypatch):
        self._pages(monkeypatch, "The narratives of these films rely on archetypes.")
        present, _ = claim_terms_present_in_source(object(), "narrative archetype")
        assert present == 2

    def test_a_genuinely_different_work_shares_little(self, monkeypatch):
        self._pages(monkeypatch, "Asbestos fibre tensile strength under thermal load.")
        present, total = claim_terms_present_in_source(
            object(), "Canadian telecommunications policy favours incumbent carriers.")
        assert total >= 4 and present == 0


def _scope(present, total, ground="different_subject"):
    base = {"scope_policy_version": "fulltext-topic-v1",
            "claim_terms_present": present, "claim_terms_total": total}
    if ground == "different_subject":
        base.update(topic_relation="disjoint", broad_subject_relation="incompatible",
                    plausible_connection="absent", discrepancy="different_subject")
    else:
        base.update(stated_scope_conflict="present",
                    discrepancy="incompatible_stated_scope", scope_dimension="jurisdiction",
                    source_scope="United States", claim_scope="Canada",
                    broad_subject_relation="compatible")
    return base


class TestTheGate:
    def test_a_work_that_discusses_the_claim_cannot_be_a_different_subject(self):
        """Khan: 10 of 12 claim terms present in the document."""
        assert not _qualifying_scope_ground(_scope(10, 12))

    def test_a_work_that_shares_almost_nothing_may_be(self):
        assert _qualifying_scope_ground(_scope(0, 12))

    def test_the_threshold_sits_below_ordinary_use(self):
        """Measured: 42%-100% present for judgments of general relevance."""
        assert MAX_CLAIM_TERM_PRESENCE_FOR_DIFFERENT_SUBJECT < 0.42
        assert _qualifying_scope_ground(_scope(3, 12))      # 25%
        assert not _qualifying_scope_ground(_scope(4, 12))  # 33%

    def test_too_few_terms_to_measure_abstains(self):
        assert not _qualifying_scope_ground(_scope(0, MIN_CLAIM_TERMS_FOR_ABSENCE - 1))

    def test_missing_counts_abstain(self):
        scope = _scope(0, 12)
        del scope["claim_terms_present"]
        assert not _qualifying_scope_ground(scope)

    def test_nonsense_counts_abstain(self):
        assert not _claim_absent_from_source({"claim_terms_present": 9, "claim_terms_total": 4})
        assert not _claim_absent_from_source({"claim_terms_present": -1, "claim_terms_total": 8})

    def test_a_stated_scope_needs_no_term_support(self):
        """An affirmative claim the text makes about itself, found in the opening."""
        assert _qualifying_scope_ground(_scope(12, 12, ground="stated_scope"))


class TestScopeEvidenceBlock:
    """The opening states what a work covers; the passages show what it discusses.

    Measured over 409 retained documents: a clear geographic scope was evident
    in 63% of document text and in only 3% of openings. Judging Khan's article
    from its first page missed that it is United States law throughout.
    """

    class _P:
        def __init__(self, text, page=0, start=0):
            self.text, self.page_index, self.character_start = text, page, start

    def test_passages_are_ordered_by_position_not_score(self):
        from app.services.verification_evidence import scope_evidence_block
        block = scope_evidence_block([self._P("second", 2), self._P("first", 1)])
        assert block.index("first") < block.index("second")

    def test_the_block_is_bounded(self):
        from app.services.verification_evidence import scope_evidence_block
        block = scope_evidence_block([self._P("x" * 9000, 1)], limit=500)
        assert len(block) <= 500

    def test_whitespace_is_normalised(self):
        from app.services.verification_evidence import scope_evidence_block
        assert scope_evidence_block([self._P("two\n\n  words", 1)]) == "two words"

    def test_a_non_sequence_yields_nothing(self):
        from app.services.verification_evidence import scope_evidence_block
        from unittest.mock import Mock
        assert scope_evidence_block(Mock()) == ""
        assert scope_evidence_block(None) == ""

    def test_empty_passages_are_skipped(self):
        from app.services.verification_evidence import scope_evidence_block
        assert scope_evidence_block([self._P("   ", 1), self._P("real", 2)]) == "real"


class TestCompositionFitsTheBudget:
    """Prompt text and source text share one allowance.

    A composed block that overruns it is truncated, which sets
    `abstract_truncated` and withdraws every mark -- silently.
    """

    def test_the_worst_case_composition_leaves_margin(self):
        from app.services.paper_workflow import _SCOPE_COMPOSITION_MARGIN
        from app.services.passage_relevance import (
            FULL_TEXT_SCOPE_POLICY_VERSION, _scope_character_budget,
        )
        from app.services.verification_evidence import (
            MAX_SCOPE_EXCERPT_CHARACTERS, _SCOPE_EVIDENCE_HEADING,
        )
        budget = _scope_character_budget(FULL_TEXT_SCOPE_POLICY_VERSION)
        room = (budget - MAX_SCOPE_EXCERPT_CHARACTERS
                - len(_SCOPE_EVIDENCE_HEADING) - _SCOPE_COMPOSITION_MARGIN)
        assert room > 200
        composed = MAX_SCOPE_EXCERPT_CHARACTERS + len(_SCOPE_EVIDENCE_HEADING) + room
        assert composed + _SCOPE_COMPOSITION_MARGIN <= budget


class TestCompositionFitsTheRealPrompt:
    """The budget is shared with the claim, its context and the source title.

    Sizing the composition against a character constant overran the real
    envelope and every full-text judgment returned `prompt_budget_exceeded`,
    so the run produced no scope assessment at all.
    """

    def _claim(self, text="Canadian telecommunications policy favours incumbents."):
        from types import SimpleNamespace
        return SimpleNamespace(text=text, antecedent_context=[], source_segments=[],
                               citation_marker="(Khan, 2018)",
                               citation_marker_type="parenthetical")

    def _fits(self, claim, text, title):
        from app.config import settings
        from app.services.passage_relevance import (
            _SYSTEM_PROMPT, _scope_prompt, _scope_prompt_envelope)
        from app.services.llm_input_boundary import (
            enforce_complete_prompt_budget, LLMInputBudgetExceeded)
        try:
            enforce_complete_prompt_budget(
                _SYSTEM_PROMPT + _scope_prompt("fulltext-topic-v1"),
                _scope_prompt_envelope(claim, text, title),
                max_input_tokens=settings.VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS)
            return True
        except LLMInputBudgetExceeded:
            return False

    def test_the_fitted_text_passes_the_real_budget_check(self):
        from app.services.passage_relevance import fit_scope_text
        claim = self._claim()
        fitted = fit_scope_text(claim, "O" * 1200, "E" * 5000, "fulltext-topic-v1",
                                source_title="A source title of ordinary length")
        assert self._fits(claim, fitted, "A source title of ordinary length")

    def test_evidence_is_actually_included_when_there_is_room(self):
        from app.services.passage_relevance import fit_scope_text
        claim = self._claim()
        fitted = fit_scope_text(claim, "O" * 1200, "E" * 5000, "fulltext-topic-v1",
                                source_title="Title")
        assert len(fitted) > 1200

    def test_a_long_claim_leaves_less_room_for_evidence(self):
        """The envelope competes with the source text for one allowance."""
        from app.services.passage_relevance import fit_scope_text
        short = fit_scope_text(self._claim(), "O" * 1200, "E" * 5000,
                               "fulltext-topic-v1", source_title="T")
        long = fit_scope_text(self._claim("word " * 400), "O" * 1200, "E" * 5000,
                              "fulltext-topic-v1", source_title="T")
        assert len(long) < len(short)

    def test_no_evidence_yields_the_opening_unchanged(self):
        from app.services.passage_relevance import fit_scope_text
        assert fit_scope_text(self._claim(), "O" * 1200, "", "fulltext-topic-v1") == "O" * 1200

    def test_an_opening_that_cannot_fit_is_returned_for_the_existing_path(self):
        from app.services.passage_relevance import fit_scope_text
        claim = self._claim("word " * 900)
        assert fit_scope_text(claim, "O" * 3000, "E" * 2000, "fulltext-topic-v1") == "O" * 3000
