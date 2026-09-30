"""Publisher review-header titles and letter-prefixed journal locators.

A journal prints a book review's title as the reviewed work followed by its
author, imprint and whole-work extent. Parsing that span as a book publisher
element truncated the title and misclassified the work, so the reference could
never match the publisher's own record.
"""

import pytest

from app.services.ref_field_extractor import extract_fields_apa
from app.services.source_type import classify_reference_source_kind
from app.services.web_source_metadata import _strip_declared_site_suffix

REVIEW_HEADERS = [
    (
        'Orlemanski, J. (2018). Character and Person. John Frow. Oxford: Oxford '
        'University Press, 2014. Pp. xvi+331. Modern Philology, 115(4), '
        'E276–E280. https://doi.org/10.1086/695968',
        'Character and Person. John Frow. Oxford: Oxford University Press, 2014. '
        'Pp. xvi+331',
    ),
    (
        'Vance, M. (2019). The Long Silence. Ada Perez. Cambridge: Cambridge '
        'University Press, 2016. Pp. xii+204. Speculum, 94(2), 501–503.',
        'The Long Silence. Ada Perez. Cambridge: Cambridge University Press, '
        '2016. Pp. xii+204',
    ),
    (
        'Doe, A. (2021). Water and Stone. Lin Chen. London: Routledge, 2018. '
        'Pp. 288. Journal of Rural History, 33(1), 77–79.',
        'Water and Stone. Lin Chen. London: Routledge, 2018. Pp. 288',
    ),
]


@pytest.mark.parametrize('raw,expected_title', REVIEW_HEADERS)
def test_review_header_title_retains_reviewed_work_description(raw, expected_title):
    ref = extract_fields_apa(raw)
    assert ref.title == expected_title
    assert ref.raw_ref == raw


@pytest.mark.parametrize('raw,expected_title', REVIEW_HEADERS)
def test_review_header_reference_is_a_journal_article(raw, expected_title):
    assessment = classify_reference_source_kind(raw, title=expected_title)
    assert assessment.kind == 'journal_article'
    assert assessment.confidence == 'high'


@pytest.mark.parametrize('raw,expected_title', [
    # An ordinary monograph has no extent statement and no journal container.
    ('Bordwell, D. (2006). The way Hollywood tells it: Story and style in '
     'modern movies. University of California Press.',
     'The way Hollywood tells it: Story and style in modern movies'),
    ('Frow, J. (2014). Character and person. Oxford University Press.',
     'Character and person'),
    # A page range is a locator, not a whole-work extent.
    ('Ng, P. (2017). A short note. Ada Perez. Oxford: Oxford University Press, '
     '2015. Pp. 12–34. Some Journal, 4(1), 5–9.',
     'A short note'),
    # Without a journal container the description is an ordinary source element.
    ('Ng, P. (2017). A short note. Ada Perez. Oxford: Oxford University Press, '
     '2015. Pp. xii+200.',
     'A short note'),
    ('Smith, J. (2020). Networks and localities. Regional Studies, 54(3), '
     '301–318.',
     'Networks and localities'),
])
def test_ordinary_reference_titles_are_unchanged(raw, expected_title):
    assert extract_fields_apa(raw).title == expected_title


@pytest.mark.parametrize('locator', ['E276', 'S45', '276'])
def test_letter_prefixed_journal_locator_is_a_journal_structure(locator):
    raw = f'Writer, W. (2020). A study. Modern Philology, 115(4), {locator}.'
    assert classify_reference_source_kind(raw).kind == 'journal_article'


def test_separated_word_after_issue_is_not_a_locator():
    raw = 'Writer, W. (2020). A study. Some Press, 115(4), and other matters.'
    assert classify_reference_source_kind(raw).kind != 'journal_article'


@pytest.mark.parametrize('title,declared,expected', [
    ('Character and Person. John Frow. Oxford: Oxford University Press, 2014. '
     'Pp. xvi+331. | Modern Philology: Vol 115, No 4',
     ['Modern Philology'],
     'Character and Person. John Frow. Oxford: Oxford University Press, 2014. '
     'Pp. xvi+331.'),
    ('Some Article - Nature', ['Nature'], 'Some Article'),
])
def test_declared_site_suffix_is_stripped(title, declared, expected):
    assert _strip_declared_site_suffix(title, declared) == expected


@pytest.mark.parametrize('title,declared', [
    # The name must be one the page declares about itself.
    ('Reading Modern Philology | Elsewhere', ['Modern Philology']),
    # A name inside the title is not a suffix.
    ('Modern Philology and its readers', ['Modern Philology']),
    # Never empty the title.
    ('Modern Philology', ['Modern Philology']),
    # Only an issue label may follow the declared name.
    ('Piece | Nature: a history of the journal', ['Nature']),
    ('Piece | Nature', []),
])
def test_other_titles_keep_their_text(title, declared):
    assert _strip_declared_site_suffix(title, declared) == title


# --- DOI identity is independent of the encoding a reference uses -----------

ENCODED_DOI_PAIRS = [
    # Percent-encoded punctuation copied out of a doi.org URL.
    ('10.21066/CARCL.LIBRI.2015-04%2802%29.0001',
     '10.21066/carcl.libri.2015-04(02).0001'),
    ('https://doi.org/10.1234/ABC%2D123', '10.1234/abc-123'),
]


@pytest.mark.parametrize('submitted,located',
                         [*ENCODED_DOI_PAIRS, ('doi:10.1000/182', '10.1000/182')])
def test_encoded_doi_normalizes_to_the_same_identifier(submitted, located):
    from app.services.reference_discovery import _normalize_doi

    assert _normalize_doi(submitted) == _normalize_doi(located)


@pytest.mark.parametrize('submitted,located', ENCODED_DOI_PAIRS)
def test_encoded_doi_identity_matches(submitted, located):
    # `doi:`-prefixed values are deliberately out of scope here: this resolver
    # helper never stripped that prefix, and the encoding repair did not change
    # which prefixes it recognizes.
    from app.services.source_resolver import _doi_identity_matches

    assert _doi_identity_matches(submitted, located)


@pytest.mark.parametrize('left,right', [
    ('10.21066/carcl.libri.2015-04(02).0001', '10.21066/carcl.libri.2015-04(03).0001'),
    ('10.1234/abc-123', '10.1234/abc-124'),
])
def test_genuinely_different_dois_still_conflict(left, right):
    from app.services.reference_discovery import _normalize_doi
    from app.services.source_resolver import _doi_identity_matches

    assert _normalize_doi(left) != _normalize_doi(right)
    assert not _doi_identity_matches(left, right)


def test_encoded_doi_produces_no_field_conflict():
    """The reported Flegar/Wertag false flag, end to end."""
    from app.services.reference_discovery import (
        ExpectedBibliographicFields,
        build_reference_discovery_candidate,
    )
    from app.services.retrieval.base import RetrievalResult

    located = RetrievalResult(
        source_name='openalex', success=True,
        title='Alice Through the Ages: Childhood and Adaptation',
        authors=['Zeljka Flegar'], year='2016',
        doi='10.21066/carcl.libri.2015-04(02).0001')
    candidate = build_reference_discovery_candidate(
        attempt_id='check', provider='openalex',
        expected=ExpectedBibliographicFields(
            title='Alice Through the Ages: Childhood and Adaptation',
            authors=['Flegar, Z'], year='2016',
            doi='10.21066/CARCL.LIBRI.2015-04%2802%29.0001'),
        result=located)
    doi_comparison = next(
        c for c in candidate.comparisons if c.field_name == 'doi'
    )
    assert doi_comparison.outcome == 'agreement'
    assert doi_comparison.reason_code == 'exact_doi_match'


# --- Registry differences that are not the student's error ------------------

def _reference(source_kind: str):
    from types import SimpleNamespace
    return SimpleNamespace(source_kind=source_kind)


@pytest.mark.parametrize('field,difference,source_kind', [
    # One work can carry an aggregator DOI and a publisher DOI.
    ('doi', {'submitted_value': '10.2307/j.ctt14brqd4',
             'located_value': '10.5117/9789089646392'}, 'journal_article'),
    ('doi', {'submitted_value': '10.1/a', 'located_value': '10.2/b'}, 'monograph'),
    # A journal issue dated one year and registered the next.
    ('year', {'submitted_value': '2015', 'located_value': '2016'}, 'journal_article'),
    ('year', {'submitted_value': '2016', 'located_value': '2015'}, 'journal_article'),
])
def test_registry_artefacts_do_not_become_reference_findings(
    field, difference, source_kind
):
    from app.services.evidence_report import (
        _reference_field_difference_is_reportable as reportable,
    )

    assert not reportable(field, difference, _reference(source_kind))


@pytest.mark.parametrize('field,difference,source_kind', [
    # A real work/edition question survives.
    ('year', {'submitted_value': '2010', 'located_value': '2016'}, 'journal_article'),
    # The one-year allowance is a journal-issue premise, not a book premise.
    ('year', {'submitted_value': '2015', 'located_value': '2016'}, 'monograph'),
    ('year', {'submitted_value': 'n.d.', 'located_value': '2016'}, 'journal_article'),
    ('title', {'submitted_value': 'One work', 'located_value': 'Another'}, 'journal_article'),
    ('author', {'submitted_value': 'Smith', 'located_value': 'Jones'}, 'journal_article'),
])
def test_material_reference_differences_are_still_reported(
    field, difference, source_kind
):
    from app.services.evidence_report import (
        _reference_field_difference_is_reportable as reportable,
    )

    assert reportable(field, difference, _reference(source_kind))


# --- A `doi:` reference address is a resolvable link ------------------------

def test_bare_doi_reference_is_linked_to_its_resolver():
    from app.services.evidence_report import _link_reference_text

    html = _link_reference_text(
        'Rowe, R. (2022). Title. Journal, 50(3). doi:10.1080/01956051.2022.2094868'
    )
    assert 'href="https://doi.org/10.1080/01956051.2022.2094868"' in html
    # The reference keeps the wording the student actually wrote.
    assert '>doi:10.1080/01956051.2022.2094868</a>' in html


def test_existing_url_addresses_are_unchanged():
    from app.services.evidence_report import _link_reference_text

    html = _link_reference_text('A work. https://doi.org/10.1/abc')
    assert html.count('<a ') == 1
    assert 'href="https://doi.org/10.1/abc"' in html


@pytest.mark.parametrize('text', [
    # A bare identifier without the doi: marker stays plain text.
    'Volume 10.1080 of the series, pages 3-9.',
    'Plain text with no address at all.',
])
def test_text_without_an_explicit_address_is_not_linked(text):
    from app.services.evidence_report import _link_reference_text

    assert '<a ' not in _link_reference_text(text)


def test_a_doi_known_to_reach_a_different_work_is_not_linked():
    """A wrong-work identifier stays audit-only, never a route to the source."""
    from app.services.evidence_report import _render_reference_panel_template

    raw = 'Alvarez, A. (2020). Archival Methods. doi:10.1234/coastal'
    finding = dict(
        finding_type='potentially_fabricated_reference', reference_id='ref',
        source=dict(reference_id='ref', raw_reference=raw, author='Alvarez, A.',
                    year='2020', title='Archival Methods'),
        finding='Potentially fabricated reference.',
        records=[dict(observed=dict(title='Different work', authors=['Bishop, B.'],
                                    year='2020', doi='10.1234/coastal'))],
        field_difference=dict(field_name='entry', submitted_value=raw), rectangles=[])
    panel = _render_reference_panel_template(finding, 1)
    assert 'doi.org/10.1234/coastal' not in panel


def test_an_ordinary_reference_doi_is_still_linked_in_its_panel():
    from app.services.evidence_report import _render_reference_panel_template

    raw = 'Rowe, R. (2022). A study. Journal, 50(3). doi:10.1080/01956051.2022.2094868'
    finding = dict(
        finding_type='reference_title_style', reference_id='ref',
        source=dict(reference_id='ref', raw_reference=raw, author='Rowe, R.',
                    year='2022', title='A study'),
        finding='Italicize the reference title.', rectangles=[])
    panel = _render_reference_panel_template(finding, 1)
    assert 'https://doi.org/10.1080/01956051.2022.2094868' in panel


# --- A metadata export is a record about a work, not the work --------------

DUBLIN_CORE_EXPORT = (
    'relation: https://uhra.herts.ac.uk/id/eprint/11980/\n'
    'title: Working Below the Line in the Studio System : Exploring Labour '
    'Processes in the UK Film Industry 1927-1950\n'
    'creator: Atkinson, Will\ncreator: Randle, K.R.\n'
    'description: Drawing on archived interview material from ten participants '
    'in the BECTU Oral History Project this paper gives voice to employees.\n'
    'publisher: Taylor and Francis\ndate: 2014\n'
    'identifier: 10.1080/01439685.2014.879008\n'
)


def test_repository_metadata_export_is_not_source_text():
    from app.services.source_validator import _detect_nonwork_listing

    assert _detect_nonwork_listing(DUBLIN_CORE_EXPORT, 'The Studio System') == (
        'a bibliographic metadata record'
    )


def test_metadata_export_is_detected_after_whitespace_compaction():
    """Callers may supply compacted text, so the rule cannot anchor on lines."""
    import re
    from app.services.source_validator import _detect_nonwork_listing

    compact = re.sub(r'\s+', ' ', DUBLIN_CORE_EXPORT)
    assert _detect_nonwork_listing(compact, 'The Studio System')


def test_metadata_export_is_rejected_as_a_representation():
    from app.services.source_resolver import _verify_text_content_identity

    verdict, reason = _verify_text_content_identity(
        DUBLIN_CORE_EXPORT.encode('utf-8'), None,
        'The Studio System', 'Belton, J', '2013')
    assert verdict == 'rejected'
    assert 'metadata record' in reason


@pytest.mark.parametrize('text', [
    # Ordinary prose that happens to use a couple of these words.
    'The studio system shaped labour. Description: this chapter argues that '
    'the date: 1927 matters for title: a heading in the text.',
    'A discussion of publisher relations and the creator of the format.',
])
def test_ordinary_prose_is_not_a_metadata_record(text):
    from app.services.source_validator import _detect_nonwork_listing

    assert _detect_nonwork_listing(text, 'The Studio System') is None


# --- An unreadable scanned cover title is absent, not conflicting ----------

@pytest.mark.parametrize('observed', [
    "J.·'\\L J4-t o..J-.r:.",   # OCR of a stylised thesis cover
    '### ## ##',
    '   ',
    None,
])
def test_illegible_cover_titles_are_treated_as_absent(observed):
    from app.services.source_resolver import _legible_cover_title

    assert _legible_cover_title(observed) is None


@pytest.mark.parametrize('observed', [
    'Economics',                      # a different title, single word
    'United Kingdom',
    'History of a Different Work',
])
def test_legible_titles_are_kept_so_they_can_still_block(observed):
    """A readable different title must keep blocking a medium-identity copy."""
    from app.services.source_resolver import _legible_cover_title

    assert _legible_cover_title(observed) == observed


# --- A short cited title cannot claim a longer unrelated work --------------

def _identity(text: str, title: str, author: str, year: str):
    from app.services.source_resolver import _verify_text_content_identity

    return _verify_text_content_identity(text.encode('utf-8'), None, title, author, year)


DIFFERENT_WORK = (
    'Working Below the Line in the Studio System: Exploring Labour Processes '
    'in the UK Film Industry 1927-1950. By Will Atkinson and K. R. Randle. '
    'Published 2013. This article studies the studio system and its labour. '
) * 3

OWN_WORK = (
    'The Studio System by John Belton, 2013. Belton argues that the studio '
    'system organised production across the major companies. '
) * 3


def test_short_title_inside_a_different_work_is_not_high_identity():
    """The cited title is a substring of another work's longer title."""
    verdict, reason = _identity(DIFFERENT_WORK, 'The Studio System', 'Belton, J', '2013')
    assert verdict == 'medium'
    assert 'without author agreement' in reason


def test_short_title_with_author_agreement_is_high_identity():
    verdict, _reason = _identity(OWN_WORK, 'The Studio System', 'Belton, J', '2013')
    assert verdict == 'high'


def test_a_distinctive_title_still_needs_only_year_support():
    """Long titles are unaffected: they identify a work on their own."""
    text = (
        'The Disney Dilemma: Modernized Fairy Tales or Modern Disaster? '
        'Published 2002. A discussion of adaptation. '
    ) * 3
    verdict, _reason = _identity(
        text, 'The Disney Dilemma: Modernized Fairy Tales or Modern Disaster?',
        'Nobody, A', '2002')
    assert verdict == 'high'


def test_function_words_do_not_count_as_distinctive():
    from app.services.source_resolver import _TITLE_FUNCTION_WORDS

    assert {'the', 'and', 'study', 'review'} <= _TITLE_FUNCTION_WORDS
