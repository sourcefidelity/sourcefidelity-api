from app.services.parsers.apa_parser import ApaParser
from app.services.parsers.mla_parser import MlaParser


def test_page_continuation_does_not_swallow_next_apa_reference():
    text = '''Smith, J. (2020). A title. A Long Journal

Journal, 20(5), 384–394.
https://doi.org/10.1234/first
Jones, J. (2021). Another title. Journal, 3, 1–9.
https://doi.org/10.1234/second'''
    refs = ApaParser.split_references(text)
    assert len(refs) == 2
    assert '10.1234/first' in refs[0] and '10.1234/first' not in refs[1]


def test_explicit_numbering_preserves_multiword_mla_author():
    text = '''Sources
1. Smith, John. A title. Journal, 2020.
https://example.org/first
2. Jones, Mary Beth. “Another title.”
An Edited Volume. Edited by Jane Brown, 2021.
3.“An episode.” A Show,
2022. A Network.'''
    refs = MlaParser.split_references(text)
    assert len(refs) == 3
    assert refs[1].startswith('Jones, Mary Beth.')
    assert 'Jones' not in refs[0]
