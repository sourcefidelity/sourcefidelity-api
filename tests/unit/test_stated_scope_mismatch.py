"""A source can share a subject and still be about somewhere else.

Ryan & Hearn's abstract is explicitly about Australian next-generation
filmmaking; the citation attributes a claim about postclassical Hollywood and
one country's role in the global market. Ten reports recorded that difference
in the rationale and marked nothing, because the only qualifying ground
required the broad subjects to be incompatible — and at the level of "film
distribution" they are compatible, by construction.
"""
import hashlib

import pytest

from app.services.report_layers import topical_mismatch

ABSTRACT = ("Digital production and distribution technologies may create new opportunities "
            "for filmmaking in Australia.")
CLAIM = ("Distribution in postclassical Hollywood is characterized by global strategies "
         "and multi-platform releases.")


def stated_scope_case(**overrides):
    hashes = {'abstract_sha256': hashlib.sha256(ABSTRACT.encode()).hexdigest(),
              'claim_sha256': hashlib.sha256(CLAIM.encode()).hexdigest()}
    scope = {**hashes, 'status': 'complete', 'relevance': 'apparent_mismatch', 'attention': True,
             'scope_policy_version': 'abstract-topic-v5',
             # The broad-subject test reports exactly what it reported live.
             'topic_relation': 'overlapping', 'broad_subject_relation': 'compatible',
             'plausible_connection': 'present',
             'subject_comparison': 'Both concern film distribution; the source is Australian.',
             'confidence': 'high', 'discrepancy': 'incompatible_stated_scope',
             'stated_scope_conflict': 'present', 'scope_dimension': 'region',
             'source_scope': 'filmmaking in Australia', 'claim_scope': 'postclassical Hollywood',
             'abstract_span': 'filmmaking in Australia', 'claim_span': 'postclassical Hollywood',
             'rationale': 'The abstract states an Australian scope; the claim is about Hollywood.'}
    scope.update(overrides)
    member = {'coverage_level': 'abstract_only', 'best_evidence': {'text': ABSTRACT},
              'reference_identity': {'status': 'confirmed'},
              'abstract_relevance': {**hashes, 'status': 'complete', 'scope_assessment': scope}}
    return member, {'student_text': CLAIM}


def test_a_stated_scope_conflict_marks_despite_an_overlapping_topic():
    member, citation = stated_scope_case()
    assert topical_mismatch(member, citation) is True


@pytest.mark.parametrize('field,value', [
    ('stated_scope_conflict', 'uncertain'),
    ('scope_dimension', None),
    ('source_scope', ''),
    ('claim_scope', ''),
    ('confidence', 'medium'),
    ('discrepancy', None),
    ('broad_subject_relation', 'uncertain'),
    ('abstract_span', 'not in the abstract'),
    ('rationale', ''),
])
def test_every_part_of_the_stated_scope_ground_is_required(field, value):
    member, citation = stated_scope_case(**{field: value})
    assert topical_mismatch(member, citation) is False


def test_the_older_contract_cannot_grant_the_new_ground():
    """A v4 record is read exactly as v4 was written."""
    member, citation = stated_scope_case(scope_policy_version='abstract-topic-v4')
    assert topical_mismatch(member, citation) is False


def test_the_different_subject_ground_still_marks_under_v5():
    member, citation = stated_scope_case(
        topic_relation='disjoint', broad_subject_relation='incompatible',
        plausible_connection='absent', discrepancy='different_subject',
        stated_scope_conflict='absent', scope_dimension=None,
        source_scope='', claim_scope='')
    assert topical_mismatch(member, citation) is True


def test_the_new_scope_fields_are_accepted_by_the_response_schema():
    from app.services.passage_relevance import _AbstractScope
    scope = _AbstractScope.model_validate({
        'relevance': 'apparent_mismatch', 'confidence': 'high',
        'discrepancy': 'incompatible_stated_scope', 'stated_scope_conflict': 'present',
        'scope_dimension': 'jurisdiction', 'source_scope': 'Australia', 'claim_scope': 'Hollywood',
        'abstract_span': 'a', 'claim_span': 'b', 'rationale': 'r', 'subject_comparison': 's'})
    assert scope.scope_dimension == 'jurisdiction'
    assert scope.stated_scope_conflict == 'present'


def identified(member, status='confirmed'):
    return {**member, 'reference_identity': {'status': status}}


@pytest.mark.parametrize('status,marked', [
    ('confirmed', True),
    ('confirmed_with_minor_differences', True),
    ('search_incomplete', False),
    ('unlocated_after_search', False),
    ('possible_match', False),
    ('bibliographic_conflict', False),
    ('insufficient_metadata', False),
    (None, False),
])
def test_a_subject_comparison_requires_an_identified_reference(status, marked):
    """Belton's abstract belonged to a review of another book entirely.

    The app recorded `plausible_identity_match: False` for all 28 candidates
    and left the reference `search_incomplete`, then compared the abstract it
    had fetched by mistake to the citation and marked the citation.
    """
    member, citation = stated_scope_case(
        topic_relation='disjoint', broad_subject_relation='incompatible',
        plausible_connection='absent', discrepancy='different_subject',
        stated_scope_conflict='absent', scope_dimension=None,
        source_scope='', claim_scope='')
    member = {**member, 'reference_identity': {'status': status} if status else {}}
    assert topical_mismatch(member, citation) is marked


def parsed(raw, author, kind='book_section', rid='r1'):
    from app.services.schemas import ParsedReference
    return ParsedReference(reference_id=rid, author=author, year='2013',
                           title='The Studio System', raw_ref=raw, source_kind=kind)


BELTON = ('Belton, J. (2013). The Studio System. In J. Belton (Ed.), '
          'American cinema/American culture (pp. 64-86). New York: McGraw Hill.')


MONOGRAPH = {'r1': {'is_monograph': True, 'title': 'American Cinema/American Culture',
                    'authors': ['John Belton'], 'source_kind': 'monograph',
                    'provider': 'open_library'}}
EDITED = {'r1': {'is_monograph': False, 'title': 'The handbook of things',
                 'source_kind': 'edited_collection', 'provider': 'crossref'}}

EDITOR_WRITTEN_CHAPTER = ('Smith, A. (2019). Introduction. In A. Smith (Ed.), '
                          'The handbook of things (pp. 1-10). Routledge.')


@pytest.mark.parametrize('raw,author,records,expected', [
    # The located work is a single-authored book: the edited-collection form is wrong.
    (BELTON, 'Belton, J', MONOGRAPH, 1),
    # An editor writing the introduction to the collection they edited is correct APA.
    (EDITOR_WRITTEN_CHAPTER, 'Smith, A', EDITED, 0),
    # Without an identified container the app cannot tell those two apart.
    (BELTON, 'Belton, J', {}, 0),
    (EDITOR_WRITTEN_CHAPTER, 'Smith, A', {}, 0),
    # Several editors is an edited collection on its face.
    ('Smith, A. (2019). A chapter. In A. Smith & B. Jones (Eds.), A collection (pp. 1-20). Press.',
     'Smith, A', MONOGRAPH, 0),
])
def test_only_an_identified_monograph_makes_the_chapter_form_wrong(raw, author, records, expected):
    from app.services.reference_formatting import contribution_editor_findings
    findings = contribution_editor_findings([parsed(raw, author)], records)
    assert len(findings) == expected
    if expected:
        assert findings[0]['finding_type'] == 'contribution_author_is_volume_editor'
        assert 'single-authored' in findings[0]['finding']


def test_the_short_title_merge_needs_corroboration():
    """"The Studio System" matched any record containing those two words."""
    from app.services.retrieval.canonical_work import assess_work_identity
    from app.services.retrieval.base import RetrievalResult

    def assess(title, authors=None, year=None):
        return assess_work_identity(
            RetrievalResult(source_name='p', success=True, title=title, authors=authors, year=year),
            expected_doi=None, expected_title='The Studio System',
            expected_author='Belton, J', expected_year='2013')

    assert not assess('A Fine Romance: Adapting Broadway to Hollywood in the Studio System Era').accepted
    assert not assess('Working Below the Line in the Studio System: Labour Processes').accepted
    assert not assess('Creativity in the Korean Studio System').accepted
    assert assess('The Studio System', authors=['John Belton'], year='2013').accepted


def test_an_unidentified_reference_shows_no_abstract():
    from app.services.evidence_report import _withhold_unidentified_abstract
    member = {'coverage_level': 'abstract_only', 'best_evidence': {'text': 'A stranger的 summary.'},
              'abstract_relevance': {'status': 'complete'},
              'reference_identity': {'status': 'search_incomplete'}}
    withheld = _withhold_unidentified_abstract(member)
    assert withheld['coverage_level'] == 'unavailable'
    assert withheld['best_evidence'] is None
    assert withheld['abstract_relevance'] is None
    assert 'could not be identified' in withheld['availability']
    kept = _withhold_unidentified_abstract({**member, 'reference_identity': {'status': 'confirmed'}})
    assert kept['best_evidence'] is not None


@pytest.mark.parametrize('raw,container,pages', [
    ('Belton, J. (2013). The Studio System. In J. Belton (Ed.), '
     'American cinema/American culture (pp. 64-86). New York: McGraw Hill.',
     'American cinema/American culture', '64-86'),
    ('Smith, A. (2019). A chapter. In B. Jones & C. Lee (Eds.), '
     'The handbook of things (pp. 1-20). Routledge.', 'The handbook of things', '1-20'),
    ('Jones, P. (2020). Another chapter. In R. Rowe (Ed.), Studies in media. Polity.',
     'Studies in media', ''),
    # An article names its journal where a chapter names its book. This
    # asserted '' until 2026-09-22, when the journal was not extracted at all.
    ('Khan, L. (2018). The separation of platforms and commerce. '
     'Columbia Law Review, 119(4), 973-1098.', 'Columbia Law Review', '973-1098'),
    ('Khan, L. (2018). The separation of platforms and commerce. '
     '*Harvard Law Review, 131*(5), 100-180.', 'Harvard Law Review', '100-180'),
    ('Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. '
     'Qualitative Research in Psychology, 3(2), 77-101.',
     'Qualitative Research in Psychology', '77-101'),
    ('Grabher, G. (2001). Locating economic action. '
     'Environment and Planning A: Economy and Space, 33(8), 1329-1331.',
     'Environment and Planning A: Economy and Space', '1329-1331'),
    ('Paula, B. D. (2016). Discussing identities through game-making. '
     'Press Start, 3(1).', 'Press Start', ''),
    # A book has no container: its trailing element is a publisher.
    ('Bork, R. (1978). The Antitrust Paradox: A Policy at War with Itself. '
     'Basic Books.', '', ''),
    ('Jenkins, H. (2006). Convergence Culture. New York University Press.', '', ''),
])
def test_the_containing_work_is_kept_from_a_reference(raw, container, pages):
    """The containing work is the part of a miscited reference students get right.

    For a chapter that is the book; for an article it is the journal, which
    is the other half of the title/author/year/journal identity combination.
    """
    from app.services.ref_field_extractor import extract_fields_apa
    parsed_reference = extract_fields_apa(raw)
    assert parsed_reference.container_title == container
    assert parsed_reference.pages == pages


@pytest.mark.parametrize('raw,publisher', [
    (BELTON, 'McGraw Hill'),
    ('Smith, A. (2019). Introduction. In A. Smith (Ed.), The handbook of things (pp. 1-10). Routledge.',
     'Routledge'),
    ('Bork, R. H. (1978). The antitrust paradox: A policy at war with itself. Basic Books.',
     'Basic Books'),
    ('Khan, L. (2018). The separation of platforms. Columbia Law Review, 119(4), 973-1098.', ''),
    ('Smith, A. (2019). An article title.', ''),
    ('Doster, I. V. (2002). The Disney Dilemma. https://trace.tennessee.edu/x', ''),
])
def test_the_publisher_is_kept_as_the_fourth_identity_field(raw, publisher):
    from app.services.ref_field_extractor import extract_fields_apa
    assert extract_fields_apa(raw).publisher == publisher


@pytest.mark.parametrize('authors,year,pub,accepted,agreements', [
    (['John Belton'], '2013', 'McGraw-Hill Education', True, 4),   # all four
    (None, '2013', 'McGraw Hill', True, 3),                        # three of four
    (None, None, 'McGraw Hill', True, 2),                          # two of four
    (None, None, None, False, 1),                                  # title alone, short
])
def test_identity_rests_on_the_combination_not_the_title(authors, year, pub, accepted, agreements):
    """A work agreeing on several fields is unlikely to be a different work."""
    from app.services.retrieval.canonical_work import assess_work_identity
    from app.services.retrieval.base import RetrievalResult
    assessment = assess_work_identity(
        RetrievalResult(source_name='p', success=True, title='American cinema/American culture',
                        authors=authors, year=year,
                        metadata={'publisher': pub} if pub else {}),
        expected_doi=None, expected_title='The Studio System', expected_author='Belton, J',
        expected_year='2013', expected_container_title='American cinema/American culture',
        expected_publisher='McGraw Hill')
    assert assessment.accepted is accepted
    assert assessment.title_basis == 'container'
    assert len(assessment.reason.split(' agree')[0].split('+')) == agreements


@pytest.mark.parametrize('title,authors,year,pub,accepted', [
    # Title is the anchor: it must agree, whatever else does.
    ('American cinema/American culture', ['John Belton'], '2013', 'McGraw Hill', True),
    ('American cinema/American culture', None, None, 'McGraw Hill', True),
    ('A different book entirely', ['John Belton'], '2013', 'McGraw Hill', False),
    ('A different book entirely', ['John Belton'], '2013', None, False),
])
def test_author_year_and_publisher_cannot_identify_a_work_without_the_title(
        title, authors, year, pub, accepted):
    """A prolific author with a regular publisher satisfies three fields twice."""
    from app.services.retrieval.canonical_work import assess_work_identity
    from app.services.retrieval.base import RetrievalResult
    assessment = assess_work_identity(
        RetrievalResult(source_name='p', success=True, title=title, authors=authors,
                        year=year, metadata={'publisher': pub} if pub else {}),
        expected_doi=None, expected_title='The Studio System', expected_author='Belton, J',
        expected_year='2013', expected_container_title='American cinema/American culture',
        expected_publisher='McGraw Hill')
    assert assessment.accepted is accepted
    if not accepted:
        assert 'the title did not agree' in assessment.reason


@pytest.mark.parametrize('observed_title,observed_url,accepted', [
    ('The Disney Dilemma', 'https://trace.tennessee.edu/utk_chanhonoproj/532', True),
    # Scheme, www and a trailing slash are decoration, not a different address.
    ('The Disney Dilemma', 'http://www.trace.tennessee.edu/utk_chanhonoproj/532/', True),
    # The identifier alone cannot carry it: title remains the anchor.
    ('A different thesis entirely', 'https://trace.tennessee.edu/utk_chanhonoproj/532', False),
    # A differing address is not disagreement, but it corroborates nothing.
    ('The Disney Dilemma', 'https://example.org/somewhere-else', False),
])
def test_a_submitted_address_corroborates_identity(observed_title, observed_url, accepted):
    from app.services.retrieval.canonical_work import assess_work_identity
    from app.services.retrieval.base import RetrievalResult
    assessment = assess_work_identity(
        RetrievalResult(source_name='p', success=True, title=observed_title,
                        full_text_url=observed_url),
        expected_doi=None, expected_title='The Disney Dilemma',
        expected_author='Doster, I. V', expected_year='2002',
        expected_url='https://trace.tennessee.edu/utk_chanhonoproj/532')
    assert assessment.accepted is accepted


def test_author_agreement_survives_diacritics():
    """Measured on stored candidates: a dotless i cost a same-work record its author."""
    from app.services.retrieval.canonical_work import assess_work_identity
    from app.services.retrieval.base import RetrievalResult
    assessment = assess_work_identity(
        RetrievalResult(source_name='openalex', success=True,
                        title='An Exploration of Perceptions of Faculty Members',
                        authors=['Elif Kır', 'Aslı Akyüz'], year='2019'),
        expected_doi=None,
        expected_title='An exploration of perceptions of faculty members',
        expected_author='Kir, E., & Akyüz, A', expected_year='2020')
    assert assessment.accepted
    assert 'author' in assessment.reason


DOI = '10.1080/09540253.2019.1634173'


@pytest.mark.parametrize('registered_title,expected_title,accepted,reason_fragment', [
    # The ordinary case is unchanged.
    ('Media concentration and democratic discourse',
     'Media concentration and democratic discourse', True, 'identifier'),
    # A real DOI lifted onto an invented title must not confirm the invention.
    ('Gender and education in the digital classroom',
     'Media concentration and democratic discourse', False,
     'resolves to a different title'),
    # A reference carrying an identifier and no title has nothing to contradict.
    ('Whatever the registry holds', None, True, 'exact DOI match'),
])
def test_a_matching_doi_is_evidence_not_a_bypass(
        registered_title, expected_title, accepted, reason_fragment):
    """LLM-fabricated references often carry a real DOI borrowed from elsewhere."""
    from app.services.retrieval.canonical_work import assess_work_identity
    from app.services.retrieval.base import RetrievalResult
    assessment = assess_work_identity(
        RetrievalResult(source_name='crossref', success=True,
                        title=registered_title, doi=DOI),
        expected_doi=DOI, expected_title=expected_title,
        expected_author='Bennett, S', expected_year='2019')
    assert assessment.accepted is accepted
    assert reason_fragment in assessment.reason


def test_a_borrowed_doi_becomes_a_reference_finding():
    """The discrepancy has to reach the reader, not just the identity record."""
    from app.services.evidence_report import _render_reference_panel_template
    finding = {
        'finding_type': 'doi_registers_a_different_title',
        'reference_id': 'r1',
        'finding': ('The DOI given in this reference is registered to a different title: '
                    '"Gender and education in the digital classroom". Check the identifier '
                    'against the work cited; an identifier that belongs to another work '
                    'does not confirm this reference.'),
        'rule_id': 'doi-title-mismatch-v1',
        'field_difference': {'field_name': 'doi', 'submitted_value': DOI},
        'rectangles': [], 'localization_status': 'not_assessed',
        'source': {'author': 'Bennett, S', 'year': '2019',
                   'title': 'Media concentration and democratic discourse',
                   'raw_reference': 'Bennett, S. (2019). Media concentration and democratic '
                                    f'discourse. Journal of Media Policy. https://doi.org/{DOI}'},
    }
    html = _render_reference_panel_template(finding, 1)
    assert 'registered to a different title' in html
    # Neutral: it reports the discrepancy without asserting invention or intent.
    for accusation in ('fabricat', 'invent', 'false', 'AI-generated'):
        assert accusation not in html.lower()


@pytest.mark.parametrize('raw,volume,issue', [
    ('Khan, L. (2018). The separation of platforms and commerce. '
     '*Harvard Law Review, 131*(5), 100-180.', '131', '5'),
    ('Khan, L. (2019). The separation of platforms and commerce. '
     'Columbia Law Review, 119(4), 973-1098.', '119', '4'),
    ('Paula, B. D. (2016). Discussing identities through game-making. '
     'Press Start, 3(1).', '3', '1'),
    ('Grabher, G. (2001). Locating economic action. '
     'Environment and Planning A: Economy and Space, 33(8), 1329-1331.', '33', '8'),
    # A chapter's container carries no volume, and a book has neither.
    ('Belton, J. (2013). The Studio System. In J. Belton (Ed.), '
     'American cinema/American culture (pp. 64-86). New York: McGraw Hill.', '', ''),
    ('Bork, R. (1978). The Antitrust Paradox. Basic Books.', '', ''),
])
def test_volume_and_issue_are_kept_for_periodicals_only(raw, volume, issue):
    """Volume and issue distinguish an article as a publisher distinguishes a book."""
    from app.services.ref_field_extractor import extract_fields_apa
    parsed = extract_fields_apa(raw)
    assert parsed.volume == volume
    assert parsed.issue == issue


@pytest.mark.parametrize('raw', [
    'Ryan, M. D., & Hearn, G. (2010). Next-generation filmmaking. '
    'Media International Australia, 136(1), 133-145.',
    'Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. '
    'Qualitative Research in Psychology, 3(2), 77-101.',
])
def test_an_article_has_no_publisher(raw):
    """The trailing-element rule read the journal name as an imprint.

    That put a false publisher on 59 of 179 articles and fed bogus publisher
    agreement into the merge score, where publisher is an identity field.
    """
    from app.services.ref_field_extractor import extract_fields_apa
    parsed = extract_fields_apa(raw)
    assert parsed.publisher == ''
    assert parsed.container_title


def test_a_book_keeps_its_publisher():
    from app.services.ref_field_extractor import extract_fields_apa
    parsed = extract_fields_apa(
        'Bork, R. (1978). The Antitrust Paradox: A Policy at War with Itself. Basic Books.')
    assert parsed.publisher == 'Basic Books'
    assert parsed.volume == '' and parsed.container_title == ''


# Title-page statements as Open Library returned them on 2026-09-27.
@pytest.mark.parametrize('statements,surname,expected', [
    (['John Belton.', 'John Belton.', 'John Belton.'], 'Belton', 'single_authored'),
    (['Azar Nafisi.'], 'Nafisi', 'single_authored'),
    (['edited by Patrick Cheney.'], 'Cheney', 'edited'),
    (['[edited by] Gerald Mast, Marshall Cohen, Leo Braudy.'], 'Mast', 'edited'),
    # One edited edition is enough to stay silent.
    (['John Belton.', 'edited by John Belton.'], 'Belton', 'edited'),
    # No statement: the catalogue's author list alone cannot decide.
    ([], 'Neumeyer', 'unresolved'),
    # A second name or a contributor clause does not show a single author.
    (['John Belton and Jane Doe.'], 'Belton', 'unresolved'),
    (['John Belton ; with a foreword by X.'], 'Belton', 'unresolved'),
    (['Someone Else.'], 'Belton', 'unresolved'),
])
def test_the_title_page_statement_decides_single_authorship(statements, surname, expected):
    from app.services.retrieval.open_library import classify_responsibility
    assert classify_responsibility(statements, surname) == expected


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def json(self):
        return self._payload


def test_open_library_reads_the_matching_works_editions(monkeypatch):
    import httpx
    from app.services.retrieval.open_library import OpenLibraryRetriever
    calls = []

    def get(url, params=None, **_kwargs):
        calls.append(url)
        if url.endswith('search.json'):
            return _Response({'docs': [{'key': '/works/OL3933913W', 'title': 'American cinema/American culture',
                                        'author_name': ['John Belton']}]})
        return _Response({'entries': [{'by_statement': 'John Belton.'}, {'publish_date': '2021'}]})

    monkeypatch.setattr(httpx, 'get', get)
    result = OpenLibraryRetriever().statements_of_responsibility('American cinema/American culture', 'Belton, J')
    assert result['classification'] == 'single_authored'
    assert result['statements'] == ['John Belton.'] and calls[-1].endswith('/works/OL3933913W/editions.json')


def test_open_library_does_not_read_a_different_works_editions(monkeypatch):
    import httpx
    from app.services.retrieval.open_library import OpenLibraryRetriever
    monkeypatch.setattr(httpx, 'get', lambda url, **_k: _Response(
        {'docs': [{'key': '/works/OL1W', 'title': 'Study guide', 'author_name': ['John Belton']}]}))
    result = OpenLibraryRetriever().statements_of_responsibility('American cinema/American culture', 'Belton, J')
    assert result == {'classification': 'unresolved', 'reason': 'title_differs'}


@pytest.mark.parametrize('outcome', ['confirmed', 'possible_match', 'search_incomplete'])
@pytest.mark.parametrize('classification,expected', [
    ('single_authored', True), ('edited', False), ('unresolved', False)])
def test_the_container_is_a_monograph_only_on_a_single_author_statement(monkeypatch, classification, expected,
                                                                         outcome):
    from app.services import source_resolver as sr
    from app.services.retrieval.base import RetrievalResult

    class Catalogue:
        def statements_of_responsibility(self, title, author):
            return {'classification': classification}

    resolver = sr.SourceResolver.__new__(sr.SourceResolver)
    resolver._retrieval_sources = [Catalogue()]
    monkeypatch.setattr(resolver, 'resolve_reference', lambda probe, identity_only: RetrievalResult(
        source_name='bibliography_identity', success=False,
        metadata={'reference_discovery': {'outcome': outcome}}), raising=False)
    reference = parsed(BELTON, 'Belton, J').model_copy(
        update={'container_title': 'American cinema/American culture'})
    identity = resolver._identify_container_work(reference)
    assert identity['is_monograph'] is expected
    assert identity['responsibility'] == {'classification': classification}
    # Only a confirmed book is 'identified', which is what blocks Cannot be verified.
    assert identity['status'] == ('identified' if outcome == 'confirmed' else 'not_confirmed')
    from app.services.reference_formatting import contribution_editor_findings
    findings = contribution_editor_findings([reference], {'r1': identity})
    assert len(findings) == int(expected)


@pytest.mark.parametrize('outcome,looked_up', [
    ('search_incomplete', True), ('possible_match', True), ('unlocated_after_search', True),
    ('confirmed', False), ('confirmed_with_minor_differences', False)])
def test_the_containing_book_is_looked_up_whenever_the_part_is_unconfirmed(outcome, looked_up):
    """Belton's rerun ended "search incomplete" on the returning path and skipped the lookup."""
    from app.services import source_resolver as sr
    resolver = sr.SourceResolver.__new__(sr.SourceResolver)
    resolver._identify_container_work = lambda reference: {'status': 'identified'}
    discovery = {'outcome': outcome}
    resolver._attach_container_identity(parsed(BELTON, 'Belton, J'), discovery)
    assert ('container_identity' in discovery) is looked_up



@pytest.mark.parametrize('outcome', ['bibliographic_conflict', 'unlocated_after_search', 'insufficient_metadata'])
def test_no_container_record_when_the_book_search_found_nothing_usable(monkeypatch, outcome):
    from app.services import source_resolver as sr
    from app.services.retrieval.base import RetrievalResult
    resolver = sr.SourceResolver.__new__(sr.SourceResolver)
    resolver._retrieval_sources = []
    monkeypatch.setattr(resolver, 'resolve_reference', lambda probe, identity_only: RetrievalResult(
        source_name='bibliography_identity', success=False,
        metadata={'reference_discovery': {'outcome': outcome}}), raising=False)
    reference = parsed(BELTON, 'Belton, J').model_copy(update={'container_title': 'American cinema/American culture'})
    assert resolver._identify_container_work(reference) is None
