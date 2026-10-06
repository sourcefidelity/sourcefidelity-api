"""reference-verification-v1: "Cannot be verified" only after the searches suited
to the kind of work completed without locating it (owner decision 2026-09-24).

Failures, unattempted routes and possible matches stay neutral; nothing here
claims that a work does not exist.
"""
import hashlib
from types import SimpleNamespace

import pytest

from app.services.reference_verification import (
    FINDING_TYPE, assess_reference_verification, required_adapters,
)

TITLE = 'Synthetic Study of Coastal Archives'


def ref(kind='journal_article'):
    return SimpleNamespace(reference_id='ref-0001-000000000001', title=TITLE, source_kind=kind,
                           raw_ref=f'River, A. (2020). {TITLE}. Journal of Tests.')


def query(qid, provider, text, outcome='no_results', engine=None):
    return {'query_id': qid, 'provider': provider, 'execution_provider': engine or provider,
            'execution_outcome': outcome, 'normalized_query': text,
            'query_sha256': hashlib.sha256(text.encode()).hexdigest()}


def attempt(aid, category, provider, qids, outcome='no_match'):
    done = outcome in {'no_match', 'candidate_found', 'candidates_processed'}
    return {'attempt_id': aid, 'route_category': category, 'provider': provider, 'required': True,
            'permitted': True, 'query_ids': qids, 'outcome': outcome,
            'started_at': '2026-09-24T00:00:00Z', 'completed_at': '2026-09-24T00:00:01Z' if done else None}


def discovery(kind='journal_article', adapters=('crossref', 'openalex'), web=('brave',), outcome='search_incomplete',
              failed=(), doi_only=(), candidates=()):
    title_query = f'title:{TITLE.lower()} author:river, a.'
    queries, attempts = [], []
    for i, provider in enumerate(adapters + tuple(failed) + tuple(doi_only)):
        qid = f'q{i}'
        if provider in doi_only:
            queries.append(query(qid, provider, 'doi:10.1234/other', outcome='results'))
            attempts.append(attempt(f'a{i}', 'academic_adapter', provider, [qid], 'candidate_found'))
        elif provider in failed:
            queries.append(query(qid, provider, title_query, outcome='operational_failure'))
            attempts.append(attempt(f'a{i}', 'academic_adapter', provider, [qid], 'operational_failure'))
        else:
            queries.append(query(qid, provider, title_query))
            attempts.append(attempt(f'a{i}', 'academic_adapter', provider, [qid]))
    for j, engine in enumerate(web):
        qid = f'w{j}'
        queries.append(query(qid, 'web_search', f'"{TITLE}" river', engine=engine))
        attempts.append(attempt(f'b{j}', 'bounded_web', 'web_search', [qid]))
    return {'reference_id': 'ref-0001-000000000001', 'outcome': outcome,
            'expected': {'title': TITLE, 'source_kind': kind}, 'queries': queries,
            'attempts': attempts, 'candidates': list(candidates)}


def candidate(title_outcome, author_outcome, provider='crossref', plausible=False):
    return {'provider': provider, 'plausible_identity_match': plausible,
            'observed': {'title': 'Observed record', 'authors': ['Other, B.']},
            'comparisons': [{'field_name': 'title', 'outcome': title_outcome},
                            {'field_name': 'author', 'outcome': author_outcome}]}


def test_completed_suited_searches_without_a_match_cannot_be_verified():
    result = assess_reference_verification(ref(), discovery())
    assert result['status'] == 'cannot_be_verified'
    [finding] = result['findings']
    assert finding['finding_type'] == FINDING_TYPE and finding['finding'].startswith('Cannot be verified.')
    assert finding['finding'] == 'Cannot be verified. Searches for this source could not locate the reference.'
    assert finding['evidence_explanation'] == 'Searched by title and author: Crossref, OpenAlex, Brave Search.'
    assert finding['field_difference']['field_name'] == 'entry'


@pytest.mark.parametrize('change,reason', [
    (dict(adapters=('crossref',)), 'required_route_incomplete'),                   # never ran
    (dict(adapters=('crossref',), failed=('openalex',)), 'required_route_incomplete'),  # failed
    (dict(web=()), 'title_web_search_incomplete'),
    (dict(adapters=('openalex',), doi_only=('crossref',)), 'required_route_incomplete'),  # DOI lookup only
])
def test_a_failed_or_missing_suited_search_is_neutral(change, reason):
    result = assess_reference_verification(ref(), discovery(**change))
    assert result['status'] == 'search_incomplete' and result['reason_code'] == reason
    assert result['findings'] == []


def test_one_web_provider_is_enough():
    assert assess_reference_verification(ref(), discovery(web=('exa',)))['status'] == 'cannot_be_verified'


@pytest.mark.parametrize('outcome', ['confirmed', 'confirmed_with_minor_differences'])
def test_a_confirmed_work_is_verified(outcome):
    assert assess_reference_verification(ref(), discovery(outcome=outcome))['status'] == 'verified'


def test_possible_matches_are_shown_not_flagged():
    same_title = candidate('agreement', 'material_conflict', plausible=True)
    result = assess_reference_verification(ref(), discovery(candidates=[same_title]))
    assert result['status'] == 'possible_match' and not result['findings']
    assert result['possible_matches'][0]['provider'] == 'crossref'
    assert assess_reference_verification(ref(), discovery(outcome='possible_match'))['status'] == 'possible_match'


def test_a_borrowed_doi_record_or_a_titleless_record_is_not_a_possible_match():
    borrowed = candidate('material_conflict', 'material_conflict', plausible=True)
    empty = candidate('unknown', 'unknown', provider='core', plausible=True)
    result = assess_reference_verification(ref(), discovery(outcome='bibliographic_conflict',
                                                             candidates=[borrowed, empty]))
    assert result['status'] == 'cannot_be_verified'


def test_the_same_doi_author_and_year_is_a_possible_match_whatever_the_title():
    # Sanchez-Lopez and Bakulev (paper 4, 2026-10-02): a parsed title cut at a
    # full stop or carrying volume and pages; DOI, author and year agree.
    same = candidate('material_conflict', 'agreement', plausible=False)
    same['comparisons'] += [{'field_name': 'doi', 'outcome': 'agreement'}, {'field_name': 'year', 'outcome': 'agreement'}]
    result = assess_reference_verification(ref(), discovery(outcome='bibliographic_conflict', candidates=[same]))
    assert result['status'] == 'possible_match' and not result['findings']
    other_year = candidate('material_conflict', 'agreement')
    other_year['comparisons'] += [{'field_name': 'doi', 'outcome': 'agreement'},
                                  {'field_name': 'year', 'outcome': 'material_conflict'}]
    result = assess_reference_verification(ref(), discovery(outcome='bibliographic_conflict', candidates=[other_year]))
    assert result['status'] == 'cannot_be_verified'


@pytest.mark.parametrize('kind', ['webpage', 'news_article', 'video', 'archival_source', 'dataset', 'software'])
def test_kinds_academic_indexes_do_not_hold_are_not_assessed(kind):
    result = assess_reference_verification(ref(kind), discovery(kind=kind))
    assert result['status'] == 'not_assessed' and result['reason_code'] == 'not_held_by_academic_indexes'


def test_insufficient_metadata_or_no_record_is_not_assessed():
    assert assess_reference_verification(ref(), discovery(outcome='insufficient_metadata'))['status'] == 'not_assessed'
    assert assess_reference_verification(ref(), None)['reason_code'] == 'discovery_unavailable'


@pytest.mark.parametrize('kind,routes', [
    ('journal_article', ('crossref', 'openalex')),
    ('monograph', ('google_books', 'open_library')),
    ('book_section', ('google_books', 'open_library', 'crossref', 'openalex')),
    ('report', ('openalex', 'crossref', 'datacite')),
    ('unknown', ('openalex', 'crossref', 'datacite')),
])
def test_required_routes_follow_the_kind(kind, routes):
    assert required_adapters(kind) == routes
    assert assess_reference_verification(ref(kind), discovery(kind=kind, adapters=routes))['status'] == 'cannot_be_verified'
    for missing in routes:
        partial = tuple(p for p in routes if p != missing)
        assert assess_reference_verification(ref(kind), discovery(kind=kind, adapters=partial))['status'] == 'search_incomplete'


def test_a_book_is_not_held_to_article_indexes():
    result = assess_reference_verification(ref('monograph'), discovery(kind='monograph', adapters=('google_books', 'open_library')))
    assert result['status'] == 'cannot_be_verified'
    assert 'Crossref' not in result['findings'][0]['evidence_explanation']


class _Source:
    def __init__(self, name, capabilities=frozenset({'title_author', 'doi'})):
        self.name, self.capabilities = name, capabilities


def _resolver(outcome, calls):
    from app.services.source_resolver import SourceResolver
    from app.services.retrieval.base import RetrievalResult
    resolver = object.__new__(SourceResolver)
    resolver._discovery_artifacts = lambda: (None, {'outcome': outcome})
    resolver._lookup_structured_sources = lambda sources, doi, *rest: calls.append(
        ([s.name for s in sources], doi)) or [(s, RetrievalResult(source_name=s.name, success=False)) for s in sources]
    resolver._record_discovery_attempt = lambda **kwargs: calls.append(('recorded', kwargs['provider']))
    return resolver


def test_an_unconfirming_doi_leads_to_a_title_search_of_the_required_indexes():
    calls = []
    sources = [_Source('crossref'), _Source('openalex'), _Source('core'), _Source('doi_only', frozenset({'doi'}))]
    _resolver('bibliographic_conflict', calls)._title_search_when_doi_unconfirmed(
        sources, '10.1234/other', TITLE, 'River', '2020', 'journal_article')
    assert calls[0] == (['crossref', 'openalex'], None)          # title only, no DOI
    assert ('recorded', 'crossref') in calls and ('recorded', 'openalex') in calls


@pytest.mark.parametrize('outcome,doi', [('confirmed', '10.1234/x'), ('search_incomplete', None)])
def test_no_title_search_when_the_doi_confirmed_or_none_was_given(outcome, doi):
    calls = []
    _resolver(outcome, calls)._title_search_when_doi_unconfirmed(
        [_Source('crossref')], doi, TITLE, 'River', '2020', 'journal_article')
    assert calls == []


def test_unsuited_indexes_are_declined_not_failed():
    from app.services.source_resolver import SourceResolver
    from app.services.retrieval.core import PROVIDER_SKIPPED_ERROR
    [(source, result)] = list(SourceResolver._declined_sources([_Source('core')], 'not suited to this kind of work'))
    assert result.error.startswith(PROVIDER_SKIPPED_ERROR) and result.metadata['lookup_applicable'] is False


def test_a_confirmed_containing_book_is_never_cannot_be_verified():
    # Owner decision 2026-09-21 (Belton): a section cited as a chapter of an
    # edited collection still names a real book; locating the book settles it.
    record = discovery(kind='book_section', adapters=('google_books', 'open_library', 'crossref', 'openalex'))
    record['container_identity'] = {'status': 'identified', 'outcome': 'confirmed', 'title': 'The containing book',
                                    'authors': [], 'year': None, 'provider': 'bibliography_identity'}
    result = assess_reference_verification(ref('book_section'), record)
    assert result['status'] == 'container_located' and result['findings'] == []
    assert result['containing_work']['title'] == 'The containing book'


def test_an_unconfirmed_containing_book_does_not_change_the_outcome():
    record = discovery(kind='book_section', adapters=('google_books', 'open_library', 'crossref', 'openalex'))
    record['container_identity'] = {'status': 'identified', 'outcome': 'search_incomplete'}
    assert assess_reference_verification(ref('book_section'), record)['status'] == 'cannot_be_verified'


def test_a_possible_match_containing_book_is_never_cannot_be_verified():
    record = discovery(kind='book_section', adapters=('google_books', 'open_library', 'crossref', 'openalex'))
    record['container_identity'] = {'status': 'not_confirmed', 'outcome': 'possible_match', 'is_monograph': True,
                                    'title': 'The containing book'}
    result = assess_reference_verification(ref('book_section'), record)
    assert result['status'] == 'possible_match' and result['findings'] == []
    assert result['reason_code'] == 'containing_work_possible_match'


# reference-verification-v2 (2026-09-29): the Johnson shape. The supplied DOI
# registers a different work; the only Crossref title query recorded was a
# journal-scoped structured request, and OpenAlex was asked by DOI alone.
import json as _json


def _johnson(*, unscoped_title_searches, structured=False):
    record = discovery(adapters=(), doi_only=('openalex', 'crossref'), outcome='bibliographic_conflict',
                       candidates=[candidate('material_conflict', 'material_conflict', plausible=True)])
    scoped = _json.dumps({'path': '/journals/0000-0000/works',
                          'params': {'query.title': TITLE.lower(), 'rows': 5}}, sort_keys=True)
    record['queries'].append(query('j0', 'crossref', scoped, outcome='results'))
    record['attempts'].append(attempt('aj0', 'academic_adapter', 'crossref', ['j0'], 'candidate_found'))
    if unscoped_title_searches:
        for i, provider in enumerate(('openalex', 'crossref')):
            text = (_json.dumps({'path': '/works', 'params': {'query.title': TITLE.lower(), 'rows': 5}})
                    if structured and provider == 'crossref'
                    else f'title:{TITLE.lower()} author:river, a.')
            record['queries'].append(query(f't{i}', provider, text))
            record['attempts'].append(attempt(f'at{i}', 'academic_adapter', provider, [f't{i}']))
    return record


def test_doi_naming_a_different_work_then_unscoped_title_searches_without_match_cannot_be_verified():
    result = assess_reference_verification(ref(), _johnson(unscoped_title_searches=True))
    assert result['status'] == 'cannot_be_verified'
    assert result['policy_version'] == 'reference-verification-v2'


def test_a_structured_unscoped_title_query_counts_as_title_led():
    result = assess_reference_verification(ref(), _johnson(unscoped_title_searches=True, structured=True))
    assert result['status'] == 'cannot_be_verified'


def test_a_journal_scoped_title_query_alone_stays_incomplete():
    result = assess_reference_verification(ref(), _johnson(unscoped_title_searches=False))
    assert result['status'] == 'search_incomplete'
    assert result['reason_code'] == 'required_route_incomplete'
    assert set(result['incomplete_routes']) == {'crossref', 'openalex'}


@pytest.mark.parametrize('text,led', [
    ('title:a study author:river', True),
    ('doi:10.1234/x', False),
    ('{"params": {"query.title": "a study", "rows": 5}, "path": "/works"}', True),
    ('{"params": {"query.title": "a study", "rows": 5}, "path": "/journals/0000-0000/works"}', False),
    ('{"params": {"filter": "title.search:a study"}, "path": "/works"}', True),
    ('{"params": {"filter": "title.search:a study,primary_location.source.id:s1"}, "path": "/works"}', False),
    ('{"params": {"filter": "from-pub-date:2020-01-01", "rows": 50}, "path": "/works"}', False),
    ('query.title=a+study&rows=1', True),
    ('query.title=a+study&filter=issn:0000-0000', False),
    # Google Books' title search, counted since 2026-09-30.
    ('intitle:a study inauthor:river', True),
    ('inauthor:river intitle:a study', True),
    ('inauthor:river', False),
])
def test_title_led_recognises_structured_title_parameters(text, led):
    from app.services.reference_verification import _title_led
    assert _title_led({'normalized_query': text}) is led


@pytest.mark.parametrize('cited,record,match', [
    # Replay 2026-09-30: genuine books a strict title comparison rejected.
    ({'title': 'Ethical journalism in apopulistage: The engaged reporter'},
     'Ethical Journalism in a Populist Age: The Engaged Reporter', True),
    ({'title': 'A narrative and stylistic study (2nd ed.)'}, 'A Narrative and Stylistic Study', True),
    ({'title': 'Film makers: Visual poets'}, 'Film Makers: Visual Poets 1928-1999', True),
    ({'title': 'A chapter', 'container_title': 'The Business of Things: Goods and the public'},
     'The Business of Things: Goods and the Public', True),
    # Shared words or a short shared start are not the same title.
    ({'title': 'Telecommunications and the Future'}, 'The Future of Telecommunications Industries', False),
    ({'title': 'Media studies'}, 'Media Studies in Practice', False),
])
def test_book_catalogue_titles_are_read_tolerantly(cited, record, match):
    from app.services.reference_verification import _book_title_matches, _possible_match
    assert _book_title_matches(cited, record) is match
    candidate = {'provider': 'google_books', 'observed': {'title': record},
                 'comparisons': [{'field_name': 'title', 'outcome': 'material_conflict'}]}
    assert _possible_match(candidate, cited) is match
    # Only a book catalogue's record counts this way.
    assert _possible_match({**candidate, 'provider': 'crossref'}, cited) is False


@pytest.mark.parametrize('title,match', [
    # 2026-09-30: pages under the exact cited title plus a site suffix.
    ('A Study of Rivers in the Northern Plains | Journal Site', True),
    ('A Study of Rivers in the Northern Plains', True),
    ('A Study of Rivers in the Northern Plains, de A. Writer - Artigo - Scribd', True),
    ('A Study of Rivers in the Northern Plains - PDF', True),
    # More title words, a truncated snippet or a different title are not the work.
    ('A Study of Rivers in the Northern Plains and Their Deltas', False),
    ('A Study of Rivers in the ...', False),
    ('Rivers of the Northern Plains', False),
])
def test_web_pages_under_the_cited_title_are_possible_matches(title, match):
    from app.services.reference_verification import _page_title_matches, _possible_match
    cited = {'title': 'A study of rivers in the northern plains'}
    assert _page_title_matches(cited, title) is match
    candidate = {'provider': 'web_search', 'observed': {'title': title},
                 'comparisons': [{'field_name': 'title', 'outcome': 'material_conflict'}]}
    assert _possible_match(candidate, cited) is match
    assert _page_title_matches({'title': 'Short title'}, 'Short title | Site') is False


def _record(*rows):
    return {'candidates': [{'provider': provider, 'candidate_id': f'c{i}',
                            'observed': {'title': title, 'authors': authors}}
                           for i, (provider, title, authors) in enumerate(rows)]}


def test_a_same_titled_work_by_someone_else_is_an_author_difference():
    # Owner decision 2026-09-30 (Singer, Mather, Kozlovic).
    from types import SimpleNamespace
    from app.services.reference_verification import same_title_author_difference
    ref = SimpleNamespace(title='A study of rivers in the northern plains (2nd ed.)', author='Stone, B')
    found = same_title_author_difference(ref, _record(
        ('google_books', 'A Study of Rivers in the Northern Plains', ['Mary River']),
        ('google_books', 'A Study of Rivers in the Northern Plains', [', Mary River'])))
    assert found['submitted_value'] == 'Stone, B' and found['located_value'] == 'Mary River'
    # No difference when any same-titled record names the cited author,
    # when the records disagree, or when no exact title or author is observed.
    for rows in (
        [('google_books', 'A Study of Rivers in the Northern Plains', ['Mary River']),
         ('open_library', 'A study of rivers in the northern plains', ['B. Stone'])],
        [('google_books', 'A Study of Rivers in the Northern Plains', ['Mary River']),
         ('web_search', 'A Study of Rivers in the Northern Plains', ['Tom Lake'])],
        [('google_books', 'A Study of Rivers in the Northern Plains | Site', ['Mary River'])],
        [('web_search', 'A Study of Rivers in the Northern Plains and Deltas | Site', ['Mary River'])],
        [('web_search', 'A Study of Rivers in the Northern Plains', [])],
    ):
        assert same_title_author_difference(ref, _record(*rows)) is None
    # A web page may add a site suffix to the exact title (Decent Films).
    page = same_title_author_difference(ref, _record(
        ('web_search', 'A Study of Rivers in the Northern Plains | Film Site - Reviews', ['Mary River'])))
    assert page['located_value'] == 'Mary River'
    # A page splitting the name into two items names the same author, and the
    # catalogue record is the one shown (paper 2's Falsetto, 2026-10-03).
    split = same_title_author_difference(ref, _record(
        ('web_search', 'A Study of Rivers in the Northern Plains - Softcover', ['River', 'Mary']),
        ('web_search', 'A Study of Rivers in the Northern Plains', ['Mary River']),
        ('google_books', 'A Study of Rivers in the Northern Plains', ['Mary River'])))
    assert split['provider'] == 'google_books' and split['located_value'] == 'Mary River'


def test_a_same_titled_record_with_another_publisher_names_the_publisher_too():
    # Singer, 2026-09-30: Palgrave Macmillan against the record's Greenwood.
    from types import SimpleNamespace
    from app.services.reference_verification import same_title_author_difference
    ref = SimpleNamespace(title='A study of rivers in the northern plains', author='Stone, B', publisher='',
                          raw_ref='Stone, B. (2014). A study of rivers in the northern plains. Palgrave Macmillan.',
                          source_kind='monograph')
    record = lambda publisher: {'candidates': [{'provider': 'google_books', 'candidate_id': 'c1',
                                               'observed': {'title': 'A Study of Rivers in the Northern Plains',
                                                            'authors': ['Mary River']},
                                               'edition_metadata': {'publisher': publisher}}]}
    found = same_title_author_difference(ref, record('Greenwood'))
    assert found['publisher_difference'] == {'field_name': 'publisher', 'submitted_value': 'Palgrave Macmillan',
                                             'located_value': 'Greenwood'}
    assert 'publisher_difference' not in same_title_author_difference(ref, record('Palgrave'))
    assert 'publisher_difference' not in same_title_author_difference(ref, record(''))


def _with_web_query(found, engine, outcome):
    qid = f'x-{engine}-{outcome}'
    found['queries'].append(query(qid, 'web_search', f'"{TITLE}" river', outcome=outcome, engine=engine))
    found['attempts'].append(attempt(f'y-{engine}', 'bounded_web', 'web_search', [qid]))
    return found


def test_a_web_provider_skipped_by_our_own_budget_leaves_the_search_incomplete():
    """Owner decision 2026-10-01: Exa skipped by our budget while Brave completed is not complete coverage."""
    result = assess_reference_verification(ref(), _with_web_query(discovery(web=('brave',)), 'exa', 'budget_skipped'))
    assert result['status'] == 'search_incomplete' and result['reason_code'] == 'web_search_skipped_by_budget'
    assert result['incomplete_routes'] == ['Exa'] and result['findings'] == []


def test_a_provider_failure_or_a_completed_retry_still_counts_as_before():
    # A provider-side failure keeps the quorum: Brave alone completes the web search.
    failed = assess_reference_verification(ref(), _with_web_query(discovery(web=('brave',)), 'exa', 'timeout'))
    assert failed['status'] == 'cannot_be_verified'
    # A skipped query does not matter when the same provider completed another title search.
    retried = assess_reference_verification(ref(), _with_web_query(discovery(web=('brave', 'exa')), 'exa', 'budget_skipped'))
    assert retried['status'] == 'cannot_be_verified'


def test_chapter_containers_with_common_student_variants():
    # P5 run 2026-10-04: "(Eds)" without a full stop, a full stop after "(Ed.)",
    # an edition before the pages, and commas inside the book title.
    from app.services.ref_field_extractor import extract_fields_apa
    cases = {
        'Stone, T. (1993). The new cinema. In J. Reed, & A. Lowe (Eds), Theory goes to the movies (pp. 8-36). Routledge.':
            ('Theory goes to the movies', '8-36'),
        'Gray, R. (2005). The transnationals. In Cooper-Lane, A. (Ed.). Global media: Content, audiences, issues (pp. 17-35). Erlbaum.':
            ('Global media: Content, audiences, issues', '17-35'),
        'Elm, T. (1998). Specularity and engulfment. In S. Neale, & M. Smith (Ed.), Contemporary cinema (1st ed.) (pp.191-206). Routledge.':
            ('Contemporary cinema', '191-206'),
    }
    for raw, (container, pages) in cases.items():
        ref = extract_fields_apa(raw)
        assert (ref.source_kind, ref.container_title, ref.pages) == ('book_section', container, pages), raw


def test_the_same_authors_book_under_another_subtitle_is_a_possible_match():
    from app.services.reference_verification import _possible_match
    candidate = {'provider': 'google_books', 'observed': {'title': 'The Curse of Bigness: How Giants Came to Rule'},
                 'comparisons': [{'field_name': 'title', 'outcome': 'material_conflict'},
                                 {'field_name': 'author', 'outcome': 'agreement'}]}
    cited = {'title': 'The curse of bigness: Antitrust in the new gilded age (Vol. 15)'}
    assert _possible_match(candidate, cited)
    other_author = {**candidate, 'comparisons': [{'field_name': 'title', 'outcome': 'material_conflict'},
                                                 {'field_name': 'author', 'outcome': 'material_conflict'}]}
    assert not _possible_match(other_author, cited)


def test_an_undated_entry_after_a_doi_is_its_own_reference():
    # Regulation 2, 2026-10-04: a regulation's title after a DOI-ended entry.
    from app.services.parsers.apa_parser import ApaParser
    section = ("Humphreys, P. (2006). Policy transfer. Journal of Policy, 29(4), 305–334. https://doi.org/10.1080/0190069\n\n"
               "Measures for the Administration of Receiving Facilities 1990.\n\nhttp://www.example.test/art/1.html\n\n"
               "Parc, J. (2022). Protectionism. Journal of Media Economics, 34(2), 117–133.")
    refs = ApaParser.split_references(section)
    assert len(refs) == 3 and refs[1].startswith("Measures for") and refs[1].endswith("art/1.html")


def test_a_chapters_book_is_looked_up_under_its_editors():
    from app.services.source_resolver import chapter_editors
    assert chapter_editors("Gray, R. (2005). Chapter. In Cooper-Lane, A. (Ed.). Global media (pp. 17-35). Erlbaum.") == "Cooper-Lane, A"
    assert chapter_editors("Elm, T. (1998). Chapter. In S. Neale, & M. Smith (Ed.), Cinema (pp.191-206).") == "S. Neale, & M. Smith"
    assert chapter_editors("Stone, T. (2010). A whole book. Routledge.") == ""
