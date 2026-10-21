"""Judgment reasons (2026-10-15): one table for how each reason code reads."""
import pytest

from app.services import judgment_reasons as reasons
from app.services.facet_evidence_judgment import PRE_CALL_REASONS, _failed_finding
from app.services.judgment_report import display_state
from app.services.judgment_runs import _pre_call_reason


@pytest.mark.parametrize("code", ["judge_undecided", "samples_split", "judges_undecided",
                                  "wider_search_judge_undecided", "wider_search_samples_split"])
def test_every_undecided_judge_reads_llm_undecided(code):
    # The older panel's unanimous "uncertain" now reads like the single judge's
    # (owner decision 2026-10-15).
    assert display_state("not_judged", code) == "undecided"


@pytest.mark.parametrize("code", ["wider_search_unavailable", "judges_disagree", "not_complete_full_text"])
def test_other_reasons_stay_not_judged(code):
    assert display_state("not_judged", code) == "not_judged"


def test_an_undecided_wider_search_gets_the_uncertain_judge_note():
    # Owner decision 2026-10-15: the same situation one search later.
    assert reasons.uncertain_judge("judge_undecided") and reasons.uncertain_judge("wider_search_judge_undecided")
    assert not reasons.uncertain_judge("samples_split") and reasons.samples_split("wider_search_samples_split")


def test_fixed_notes_and_try_again():
    assert reasons.fixed_note("referral_citation").startswith("Not judged: a “see” citation")
    assert reasons.fixed_note("antecedent_unresolved") == (
        "Not judged: it could not be determined what this statement refers to.")
    assert reasons.fixed_note("wider_search_unavailable").startswith("Not judged: the judge found no evidence")
    assert reasons.fixed_note("not_complete_full_text") is None
    assert reasons.retryable("judge_failed") and reasons.retryable("wider_search_judge_failed")
    assert not reasons.retryable("judge_undecided")


def test_a_claim_stopped_before_the_judge_keeps_its_exact_reason():
    bundle = type("B", (), {"candidate_id": "c1"})()
    for message, code in PRE_CALL_REASONS.items():
        assert _pre_call_reason(_failed_finding(bundle, message)) == code
    # A message no table entry names is not guessed from its words.
    assert _pre_call_reason(_failed_finding(bundle, "An antecedent note in other words.")) == "not_assessed_before_judging"
