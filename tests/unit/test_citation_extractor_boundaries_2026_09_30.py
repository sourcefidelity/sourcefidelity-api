"""A parenthetical citation placed after the sentence's full stop (Paper 2, 2026-09-30)."""
import pytest

from app.services.citation_extractor import _extract_attributed_text_and_index
from app.services.sentence_splitter import split_sentences


@pytest.mark.parametrize('paragraph,marker,claim', [
    ('Genre films calm audiences with the status quo. (Hess,1974) These films produce satisfaction, not revolt.',
     '(Hess,1974)', 'Genre films calm audiences with the status quo. (Hess,1974)'),
    ('“It is a dystopian film that draws on the crime thriller.” (Cornea,2007) Drawing on the novel, it tells a story.',
     '(Cornea,2007)', '“It is a dystopian film that draws on the crime thriller.” (Cornea,2007)'),
    # Correct placement and a citation opening a paragraph are unchanged.
    ('Genre films calm audiences with the status quo (Hess, 1974). These films produce satisfaction.',
     '(Hess, 1974)', 'Genre films calm audiences with the status quo (Hess, 1974).'),
    ('(Hess,1974) These films produce feelings of satisfaction.',
     '(Hess,1974)', '(Hess,1974) These films produce feelings of satisfaction.'),
])
def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before_it(paragraph, marker, claim):
    start = paragraph.index(marker)
    text, _index, local_start, local_end = _extract_attributed_text_and_index(
        paragraph, start, start + len(marker), split_sentences(paragraph), 'parenthetical')
    assert text == claim and paragraph[local_start:local_end] == claim
