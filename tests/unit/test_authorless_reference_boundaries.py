from app.services.parsers.apa_parser import ApaParser
from app.services.reference_parser import extract_and_parse_references


def test_separately_linked_authorless_blocks_do_not_inherit_previous_author():
    text = ('Smith, J. (2020). A complete first work. https://example.org/first\n\n'
            '(1985). A sufficiently descriptive article title. Journal, 14(2), 3-12. https://example.org/second\n\n'
            'A sufficiently descriptive book title about cinema. Example Press.\nhttps://example.org/third')
    refs = extract_and_parse_references(text, 'apa', use_llm_fallback=False)
    assert len(refs) == 3
    assert refs[0].author == 'Smith, J'
    assert all(not r.author for r in refs[1:])
    assert refs[1].extraction_method == 'authorless_journal_regex' and not refs[1].needs_review
    assert refs[2].needs_review
    assert 'second' not in refs[0].raw_ref
    assert refs[1].year == '1985'
    assert refs[1].title == 'A sufficiently descriptive article title'
    assert refs[2].title == 'A sufficiently descriptive book title about cinema'


def test_url_only_and_retrieval_continuations_remain_attached():
    text = ('Smith, J. (2020). A complete first work. https://example.org/first\n\n'
            'https://example.org/second\n\n'
            'Retrieved from a library collection containing more material. https://example.org/third')
    assert len(ApaParser.split_references(text)) == 1


def test_lowercase_surname_particle_starts_separate_reference():
    from app.services.parsers.apa_parser import ApaParser
    previous = ['Example, A. (2004). A separate book. Example Press.']
    assert ApaParser._starts_new_reference(
        'von Example, J. (Director). (1932). Another film [Film]. Studio.', previous)
    assert not ApaParser._starts_new_reference(
        'von Example discusses the film (1932) in this continuation.', previous)
    assert not ApaParser._starts_new_reference(
        'https://example.invalid/von/Example/1932/article', previous)


def test_paragraph_gap_before_lowercase_author_survives_text_cleanup():
    from app.services.text_extractor import _clean_text
    original = ('Example, A. (2004). A separate book. Example Press. https://example.invalid/book.pdf\n\n'
                'von Example, J. (Director). (1932). Another film [Film]. Studio.')
    cleaned = _clean_text(original)
    assert len(ApaParser.split_references(cleaned)) == 2
    assert _clean_text('a wrapped\n sentence') == 'a wrapped sentence'


def test_doi_year_in_wrapped_journal_tail_does_not_start_reference():
    text = ('Smith, J. (2020). An article about historical practice. Historical Studies\n'
            'Review, 29(5), 860-874. https://doi.org/10.1080/09612025.2019.1703540\n'
            'Jones, A. (2021). Another distinct article. Journal, 1(2), 3-9. https://example.org/2021')
    refs = ApaParser.split_references(text)
    assert len(refs) == 2
    assert 'Historical Studies Review, 29(5)' in refs[0]
    assert '2019.1703540' in refs[0]
    assert refs[1].startswith('Jones, A. (2021)')
