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


def test_a_citation_after_a_closing_quotation_stays_with_its_sentence():
    # Owner review 2026-10-14 (Harlow citation 5): the quotation's own
    # parenthetical followed by the full stop ends the quoted sentence.
    for end in ("?", "!", "."):
        text = f"She asked, “Will it rain{end}” (Ames et al., 1930, 34:56). Next one starts."
        assert split_sentences(text) == [
            f"She asked, “Will it rain{end}” (Ames et al., 1930, 34:56).", "Next one starts."]
    # A parenthetical that is its own sentence still stands alone.
    assert split_sentences("He left. (See the appendix for the full list.) Then he returned.") == [
        "He left.", "(See the appendix for the full list.)", "Then he returned."]
