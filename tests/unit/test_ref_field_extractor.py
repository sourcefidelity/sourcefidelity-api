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
