"""Judgment reasons: why a claim has the result it has, and how the window treats it (2026-10-15).

Every reason code a Judgment result carries, in one table: whether it reads
LLM Undecided, whether the run's note treats it as a judge that left part of
the statement uncertain, its fixed note, and whether Try Again is offered. A
result after the wider search carries its base code with the ``wider_search_``
prefix; ``base`` strips it wherever the two mean the same thing.
"""
from __future__ import annotations

WIDER_SEARCH_PREFIX = "wider_search_"

# Owner wording 2026-10-07.
REFERRAL_NOTE = "Not judged: a “see” citation points to further reading rather than supporting a statement."
# Owner wording 2026-10-14 (Harlow review).
NOT_JUDGED_NOTES = {
    "referral_citation": REFERRAL_NOTE,
    "antecedent_unresolved": "Not judged: it could not be determined what this statement refers to.",
    "wider_search_unavailable": ("Not judged: the judge found no evidence in the sentences it read, and no "
                                 "further sentences from the source were available to check."),
}

# The judge could not decide: the single judge's "uncertain", the older
# panel's unanimous "uncertain" (owner decision 2026-10-15) and three answers
# with no shared result (owner decision 2026-10-03).
_UNDECIDED = frozenset({"judge_undecided", "judges_undecided", "samples_split"})
# A judge left part of the statement uncertain: the run writes its note from
# the uncertain answer (owner decision 2026-09-30), on either reading
# (owner decision 2026-10-15).
_UNCERTAIN_JUDGE = frozenset({"judge_undecided", "judges_undecided"})
_RETRY = frozenset({"judge_failed", "wider_search_judge_failed", "judge_unavailable"})
QUOTATION_ONLY = "quotation_only"


def base(code: str | None) -> str:
    return str(code or "").removeprefix(WIDER_SEARCH_PREFIX)


def after_wider_search(code: str) -> str:
    return WIDER_SEARCH_PREFIX + code


def undecided(code: str | None) -> bool:
    """Shown as LLM Undecided (when the result is not judged)."""
    return base(code) in _UNDECIDED


def uncertain_judge(code: str | None) -> bool:
    return base(code) in _UNCERTAIN_JUDGE


def samples_split(code: str | None) -> bool:
    return base(code) == "samples_split"


def fixed_note(code: str | None) -> str | None:
    """The owner-worded note for a Not Judged reason, if it has one."""
    return NOT_JUDGED_NOTES.get(str(code or ""))


def retryable(code: str | None) -> bool:
    return str(code or "") in _RETRY


def quotation_only(code: str | None) -> bool:
    return str(code or "") == QUOTATION_ONLY
