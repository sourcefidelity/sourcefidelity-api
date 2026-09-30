"""The scope prompt is versioned so a tuning can be reverted by configuration.

v6 clarifies one thing only: the general-point exclusion was swallowing a
statement about one national industry supported by a source that states a
different one (Ryan & Hearn, an Australian filmmaking study cited for a claim
about postclassical Hollywood distribution).
"""
import pytest

from app.config import settings
from app.services import passage_relevance as pr
from app.services.report_layers import _qualifying_scope_ground, _scope_coverage_agrees


class TestVersionSelection:
    def test_v5_returns_the_untouched_baseline(self):
        assert pr._scope_prompt("abstract-topic-v5") == pr._ABSTRACT_SCOPE_PROMPT_V5

    def test_v6_extends_the_baseline_rather_than_replacing_it(self):
        v6 = pr._scope_prompt("abstract-topic-v6")
        assert v6.startswith(pr._ABSTRACT_SCOPE_PROMPT_V5)
        assert len(v6) > len(pr._ABSTRACT_SCOPE_PROMPT_V5)

    def test_the_abstract_path_follows_the_setting(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(pr, "_assess_scope",
                            lambda *a, **k: seen.update(k) or {"status": "not_assessed"})
        monkeypatch.setattr(settings, "ABSTRACT_SCOPE_POLICY_VERSION", "abstract-topic-v5")
        pr.assess_abstract_relevance(object(), "text")
        assert seen["policy_version"] == "abstract-topic-v5"


class TestSharedInputBudget:
    """Prompt text and source text share one budget; growth must cost room."""

    def test_a_longer_prompt_reduces_the_source_budget(self):
        v5 = pr._scope_character_budget("abstract-topic-v5")
        v6 = pr._scope_character_budget("abstract-topic-v6")
        assert v5 == pr.MAX_ABSTRACT_SCOPE_CHARACTERS
        assert v6 == v5 - (len(pr._ABSTRACT_SCOPE_PROMPT_V6)
                           - len(pr._ABSTRACT_SCOPE_PROMPT_V5))
        assert v6 < v5

    def test_the_budget_never_collapses(self, monkeypatch):
        monkeypatch.setattr(pr, "_scope_prompt", lambda v: "x" * 99_999)
        assert pr._scope_character_budget("abstract-topic-v6") == 1_000


class TestStoredRecordsRemainReadable:
    """A record is read under the version it was written with."""

    @pytest.mark.parametrize("version", [
        "abstract-topic-v4", "abstract-topic-v5", "abstract-topic-v6"])
    def test_every_abstract_version_is_accepted_for_abstract_coverage(self, version):
        assert _scope_coverage_agrees(
            {"coverage_level": "abstract_only"},
            {"scope_policy_version": version, "scope_coverage": "abstract_only"})

    def test_v6_carries_both_grounds_like_v5(self):
        stated = {"scope_policy_version": "abstract-topic-v6",
                  "stated_scope_conflict": "present",
                  "discrepancy": "incompatible_stated_scope",
                  "scope_dimension": "national film industry",
                  "source_scope": "Australia", "claim_scope": "postclassical Hollywood",
                  "broad_subject_relation": "compatible"}
        assert _qualifying_scope_ground(stated)

    def test_v4_still_only_has_the_different_subject_ground(self):
        stated = {"scope_policy_version": "abstract-topic-v4",
                  "stated_scope_conflict": "present",
                  "discrepancy": "incompatible_stated_scope",
                  "scope_dimension": "x", "source_scope": "a", "claim_scope": "b",
                  "broad_subject_relation": "compatible"}
        assert not _qualifying_scope_ground(stated)

    def test_an_unknown_version_never_qualifies(self):
        assert not _qualifying_scope_ground({"scope_policy_version": "abstract-topic-v9"})


class TestRetrievedDocumentGroundIsNarrower:
    """A leading excerpt states what a work covers; it does not exhaust it.

    Pallant's Disney-formalism paper was marked a different subject from a
    claim about Disney narrative convention, because the first 1,200
    characters did not mention narrative. Absence from an opening is not
    evidence of a different subject, so the retrieved-document contract keeps
    only the ground its evidence actually supports.
    """

    def _different_subject(self, version):
        return {"scope_policy_version": version, "topic_relation": "disjoint",
                "broad_subject_relation": "incompatible",
                "plausible_connection": "absent", "discrepancy": "different_subject"}

    def _stated_scope(self, version):
        return {"scope_policy_version": version, "stated_scope_conflict": "present",
                "discrepancy": "incompatible_stated_scope",
                "scope_dimension": "jurisdiction", "source_scope": "United States",
                "claim_scope": "Canada", "broad_subject_relation": "compatible"}

    def test_full_text_cannot_mark_on_different_subject(self):
        assert not _qualifying_scope_ground(self._different_subject("fulltext-topic-v1"))

    def test_full_text_can_still_mark_on_a_stated_scope(self):
        assert _qualifying_scope_ground(self._stated_scope("fulltext-topic-v1"))

    @pytest.mark.parametrize("version", ["abstract-topic-v5", "abstract-topic-v6"])
    def test_an_abstract_keeps_both_grounds(self, version):
        assert _qualifying_scope_ground(self._different_subject(version))
        assert _qualifying_scope_ground(self._stated_scope(version))
