import pytest

from app.services.ref_field_extractor import (
    extract_fields_apa,
    extract_fields_from_llm_response,
    extract_fields_mla,
)


@pytest.mark.parametrize('mark', ['?', '!'])
@pytest.mark.parametrize('volume', ['22(5), 787–804.', '22, 787–804.'])
def test_apa_terminal_title_punctuation_before_journal_volume(mark, volume):
    raw = f'Writer, W. (2020). Is policy changing{mark} Journal of Policy, {volume}'
    ref = extract_fields_apa(raw)
    assert ref.title == f'Is policy changing{mark}'
    assert ref.raw_ref == raw


@pytest.mark.parametrize('title', [
    'Why now? A study of policy',
    'Change! A historical account',
    'Policy (why now? A question)',
    'Why now? Why here?',
])
def test_apa_question_does_not_truncate_subtitle_or_parentheses(title):
    raw = f'Writer, W. (2020). {title}. Journal of Policy, 22(5), 787–804.'
    ref = extract_fields_apa(raw)
    assert ref.title == title
    assert ref.raw_ref == raw


@pytest.mark.parametrize('credit', [
    '(A. Example, Ed.)', '(A. Example (Ed.))',
    '(A. Example & B. Sample, Eds.)', '(A. B. Example, Trans.)',
    '(A. Example (Ed.); B. Sample (Trans.))',
])
def test_apa_contributor_initials_do_not_split_title(credit):
    raw = f'Writer, W. (2020). A translated work {credit}. Sample Press.'
    ref = extract_fields_apa(raw)
    assert ref is not None and ref.title == 'A translated work'
    assert ref.raw_ref == raw and credit in ref.raw_ref
    assert ref.author == 'Writer, W' and ref.year == '2020'


@pytest.mark.parametrize('title', [
    'A normal article', 'A title (with a subtitle)',
    'A title (including A. Example and B. Sample)',
    'A title (with nested (A. Example) details)',
    'A title (2nd ed.)',
])
def test_apa_balanced_title_parentheses_are_preserved(title):
    ref = extract_fields_apa(f'Writer, W. (2020). {title}. Sample Press.')
    assert ref is not None and ref.title == title


@pytest.mark.parametrize('title', ['An unfinished (A. Example', 'An unmatched) title'])
def test_apa_unbalanced_title_does_not_claim_regex_success(title):
    assert extract_fields_apa(f'Writer, W. (2020). {title}. Sample Press.') is None


def test_apa_regex_extracts_identity_fields() -> None:
    raw = (
        "Smith, J. (2024). Platform governance and academic integrity. "
        "Journal of Source Studies, 8(2). https://doi.org/10.1234/example"
    )

    reference = extract_fields_apa(raw)

    assert reference is not None
    assert reference.author == "Smith, J"
    assert reference.year == "2024"
    assert reference.title == "Platform governance and academic integrity"
    assert reference.doi == "10.1234/example"
    assert reference.extraction_method == "regex"
    assert reference.needs_review is False


@pytest.mark.parametrize('container',[
    '*Journal of Source Studies*, 8(2), 12-34.',
    '*Journal of Source Studies, 8*(2), 12-34.',
    '*Journal of Source Studies*, 8, 12-34.',
])
def test_apa_marked_journal_is_not_part_of_article_title(container):
    raw=f'Writer, A. (2020). Source verification in practice. {container}'
    reference=extract_fields_apa(raw)
    assert reference.title=='Source verification in practice'
    assert reference.raw_ref==raw


@pytest.mark.parametrize('publisher',['Springer','Sage','Wiley'])
def test_apa_joined_terminal_publisher_is_not_part_of_book_title(publisher):
    raw=f'Writer, A. (2020). Studies of culture and society.{publisher}.'
    reference=extract_fields_apa(raw)
    assert reference.title=='Studies of culture and society'
    assert reference.raw_ref==raw


def test_publisher_boundary_does_not_repair_joined_words_in_student_title():
    raw='Writer, A. (2020). Culture and mediain society.Springer.'
    reference=extract_fields_apa(raw)
    assert reference.title=='Culture and mediain society'
    assert reference.raw_ref==raw


@pytest.mark.parametrize('title',[
    'A study. *Another perspective*',
    'A study of Mr.Smith',
    'The U.S. press',
    'A study (vol. *Two*)',
])
def test_new_boundary_cues_do_not_trim_title_content(title):
    reference=extract_fields_apa(f'Writer, A. (2020). {title}. Sample Press.')
    assert reference.title==title


def test_apa_monograph_title_excludes_trailing_cited_page_range() -> None:
    raw = "Hardy, J. (2010). Cross-media promotion (pp. 3-32). New York: Peter Lang."

    reference = extract_fields_apa(raw)

    assert reference is not None
    assert reference.title == "Cross-media promotion"
    assert "pp. 3-32" in reference.raw_ref


def test_mla_regex_extracts_quoted_article_title() -> None:
    raw = (
        'Smith, Jane. "Platform Governance and Academic Integrity." '
        "Journal of Source Studies, vol. 8, no. 2, 2024, pp. 1-20."
    )

    reference = extract_fields_mla(raw)

    assert reference is not None
    assert reference.author == "Smith, Jane"
    assert reference.year == "2024"
    assert reference.title == "Platform Governance and Academic Integrity."
    assert reference.extraction_method == "regex"


def test_llm_fallback_is_always_marked_for_review() -> None:
    response = """Author: Smith, J.
Year: Published 2024
Title: Platform governance
DOI: https://doi.org/10.1234/example
URL: none
"""

    reference = extract_fields_from_llm_response(response, "original reference")

    assert reference.extraction_method == "llm"
    assert reference.needs_review is True
    assert reference.year == "2024"
    assert reference.doi == "10.1234/example"
    assert reference.raw_ref == "original reference"
def test_explicit_month_date_without_comma_retains_identity_fields():
    from app.services.ref_field_extractor import extract_fields_apa, _APA_YEAR
    raw = 'Example Herald. (1939 January 7). A news report. Example Herald, 134, pp. 53.'
    result = extract_fields_apa(raw)
    assert result is not None
    assert result.author == 'Example Herald'
    assert result.year == '1939'
    assert result.title == 'A news report'
    assert result.raw_ref == raw
    for date in ['(1939 commentary)', '(1939 January 99)', '(1939 January 7 extra)', '(1939 January 0)']:
        assert _APA_YEAR.fullmatch(date) is None


def test_chapter_without_editors_keeps_its_book_and_pages():
    # 2026-09-29: an editor-less "In <book> (pp. x-y)" was read as a whole book,
    # so the containing book was never looked up and a real chapter was flagged.
    from app.services.ref_field_extractor import extract_fields_apa
    parsed = extract_fields_apa(
        'Rivers, A. & Stone, B. (2006). The changing shape of an industry. In The Business of '
        'Things: Corporate goods and the public (pp.75-116). London: Example Press.')
    assert parsed.title == 'The changing shape of an industry'
    assert parsed.container_title == 'The Business of Things: Corporate goods and the public'
    assert parsed.pages == '75-116' and parsed.source_kind == 'book_section'
    whole = extract_fields_apa('Rivers, A. (2006). In the shadow of things. Example Press.')
    assert whole.source_kind != 'book_section' and not whole.container_title


# 2026-10-12: other-language references. Invented fixtures.
ROMANISED_CHAPTER = (
    'Ren, B., Tao, M., & Wen, K. (2018). Meiti ronghe beijing xia de xinwen chuanbo yanjiu '
    '(A study of news communication under media convergence). In C. Lo (Ed.), Report on the Example '
    'Media Industry {year} (pp. 117-123). Example Academic Press. https://books.example.org/item/123')


@pytest.mark.parametrize('year', ['(2018)', '（2018）'])
def test_a_bracketed_year_in_a_book_title_keeps_the_container_and_pages(year):
    parsed = extract_fields_apa(ROMANISED_CHAPTER.format(year=year))
    assert parsed.container_title == f'Report on the Example Media Industry {year}'
    assert parsed.pages == '117-123' and parsed.source_kind == 'book_section'


def test_a_bracket_other_than_a_year_still_ends_the_container():
    parsed = extract_fields_apa(
        'Ames, T. (2017). A part. In B. Poore (Ed.), The book (draft copy) (pp. 1-9). Example Press.')
    assert parsed.container_title != 'The book (draft copy)'


def test_a_translation_tool_note_is_ignored_for_parsing_and_kept_in_the_raw_reference():
    raw = ('Pan, Y., & Fu, S. (2016). Shuju quanli yu falv baohu yanjiu (Research on data rights and legal '
           'protection). Example Law and Economy, (11), 6-9.[数据权利与法律保护研究 (Example Translate).] '
           'https://link.example.net/urlid/abc123')
    parsed = extract_fields_apa(raw)
    assert parsed.raw_ref == raw
    assert parsed.title == 'Shuju quanli yu falv baohu yanjiu (Research on data rights and legal protection)'
    assert (parsed.container_title, parsed.volume, parsed.issue, parsed.pages) == ('Example Law and Economy', '', '11', '6-9')
    assert parsed.source_kind == 'journal_article' and parsed.publisher == ''
    assert parsed.url == 'https://link.example.net/urlid/abc123'


def test_an_issue_only_journal_repeated_after_its_url_keeps_the_clean_name():
    parsed = extract_fields_apa(
        'Quill, I. (2012). Screen roles. Example Journal of Screen Media, (4), 97-115. Available online: '
        'https://www.examplejournal.com/Example Journal of Screen Media, (4), 97-115.')
    assert (parsed.container_title, parsed.issue, parsed.pages) == ('Example Journal of Screen Media', '4', '97-115')


def test_only_a_bracket_naming_a_translation_tool_is_a_note():
    from app.services.source_type import has_translation_tool_note, without_translation_tool_notes
    assert has_translation_tool_note('Title. Journal, 2(1), 1-9. [原题 (Example Translator)]')
    assert has_translation_tool_note('Title. Journal, 2(1), 1-9.【原题（DeepL）。】')
    for kept in ('A study [Doctoral dissertation, Example University].',
                 'Why we translate [Why We Translate]. Example Press.',
                 'Title [Translated by A. Person]. Example Press.'):
        assert not has_translation_tool_note(kept) and without_translation_tool_notes(kept) == kept


def test_a_chapter_with_an_edition_note_before_its_pages_is_a_chapter():
    # 2026-10-16 (Rita Hayworth reference 3): "(2nd ed.)" between the book
    # title and the page range made the entry a monograph titled "Introduction".
    from app.services.source_type import classify_reference_source_kind
    raw = ("Stone, B. (2004). Introduction. In River towns: Water and society (2nd ed.) (pp. 1-16). "
           "Harbour Press.")
    reference = extract_fields_apa(raw)
    assert reference.container_title == "River towns: Water and society"
    assert classify_reference_source_kind(raw, title=reference.title).kind == "book_section"


def test_a_title_ending_in_an_exclamation_mark_is_not_part_of_the_magazine_name():
    from app.services.ref_field_extractor import extract_fields_apa
    parsed = extract_fields_apa("Quillby, O. (1945, August). How the star has changed! Lanternland, 49(10), 30–31, 74–75.")
    assert parsed.container_title == "Lanternland" and parsed.volume == "49" and parsed.issue == "10"


def test_a_dated_magazine_article_without_a_volume_names_its_magazine():
    from app.services.ref_field_extractor import extract_fields_apa
    from app.services.source_type import classify_reference_source_kind
    raw = "Quillby, O. (1947, March). Ask the editor. Lanternland, 50–51, 90–92."
    parsed = extract_fields_apa(raw)
    assert parsed.container_title == "Lanternland" and parsed.pages == "50-51, 90-92"
    assert parsed.publisher == ""
    assert classify_reference_source_kind(raw, title=parsed.title).kind == "news_article"
    # A book's year carries no month; its trailing element is not a magazine.
    assert extract_fields_apa("Quillby, O. (1947). Ask the editor. Harbour Press.").container_title == ""


def test_a_chapter_title_ending_in_a_question_mark_is_a_chapter():
    from app.services.ref_field_extractor import extract_fields_apa
    from app.services.source_type import classify_reference_source_kind
    raw = "Quillby, O. (2005b). Why this star? In Being a star: Labor and fame (pp. 1–28). Harbour Press."
    parsed = extract_fields_apa(raw)
    assert parsed.title == "Why this star?" and parsed.container_title == "Being a star: Labor and fame"
    assert parsed.pages == "1-28"
    assert classify_reference_source_kind(raw, title=parsed.title).kind == "book_section"


def test_a_month_dated_magazine_without_an_issue_lists_pages_not_a_volume():
    from app.services.ref_field_extractor import extract_fields_apa
    parsed = extract_fields_apa("Quillby, O. (1946b, August). Lanterns. Fanmag, 42, 86–87.")
    assert parsed.container_title == "Fanmag" and parsed.volume == "" and parsed.pages == "42, 86-87"
    # A journal year with a volume and pages is unchanged.
    journal = extract_fields_apa("Quillby, O. (2019). Lanterns. Journal of Light, 42, 86–87.")
    assert journal.volume == "42" and journal.pages == "86-87"
