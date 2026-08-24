"""Sentence-boundary regressions for academic prose."""

from app.services.sentence_splitter import split_sentences


def test_person_middle_initial_does_not_truncate_sentence():
    text = (
        "Leaming notes that her manager Edward C. Judson controlled publicity. "
        "A second sentence follows."
    )

    assert split_sentences(text) == [
        "Leaming notes that her manager Edward C. Judson controlled publicity.",
        "A second sentence follows.",
    ]


def test_corporate_bros_abbreviation_does_not_split_one_sentence():
    text = (
        "This was an attempt by Warner Bros. to capitalize on the series. "
        "The promotion was extensive."
    )

    assert split_sentences(text) == [
        "This was an attempt by Warner Bros. to capitalize on the series.",
        "The promotion was extensive.",
    ]
