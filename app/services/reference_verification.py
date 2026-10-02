"""Whether the application could confirm that a cited work exists.

Owner-approved 2026-09-24 as `reference-verification-v1`; the current policy is
v2 (2026-09-29/30, see POLICY). This replaces the
former "potentially fabricated" review flag. The application cannot decide
fabrication; it can say, truthfully, that the searches suited to the kind of
work did not locate it. That statement is an Evidence finding, not an Academic
Practice one, and it never asserts that the work does not exist.

The rule reads the stored discovery record only. It adds no requests and is a
pure function of what was searched and what came back:

* a confirmed work is verified;
* a similar record that may be the work is a possible match, shown to the
  reader but never flagged;
* a work whose kind academic indexes do not hold (web pages, media, archival
  items, data, software), or a reference too thin to search, is not assessed;
* a search the reference needed that failed, timed out or never ran leaves it
  incomplete, which is neutral and never flagged;
* only when every search suited to the kind completed and none located the
  work is the reference marked "Cannot be verified".
"""
from __future__ import annotations

import hashlib
import json
import re

from app.services.reference_review_scope import text_key
from app.services.source_type import is_bibliographically_searchable, normalize_source_kind

# v2 (2026-09-29): structured title queries (`query.title`, `title.search`)
# count as title-led unless confined to one journal. Amended 2026-09-30 (Google
# Books title searches) and 2026-10-01 (a required web provider skipped by our
# own budget leaves the search incomplete). v1 values stay readable; nothing
# parses this string back.
POLICY = 'reference-verification-v2'
POLICY_VERSIONS = ('reference-verification-v1', POLICY)
FINDING_TYPE = 'unverified_reference'
LABEL = 'Cannot be verified'
# Owner wording 2026-09-30.
FINDING_TEXT = 'Cannot be verified. Searches for this source could not locate the reference.'

ARTICLE_KINDS = frozenset({'journal_article', 'conference_paper', 'book_review'})
BOOK_KINDS = frozenset({'monograph', 'edited_collection'})

# The metadata routes that must complete, by kind. Each is an index that
# holds that kind of work; an index that cannot hold it is neither called nor
# required. An unclassified reference gets the broad scholarly set rather than
# a guess at one kind.
REQUIRED_ADAPTERS = {
    **{kind: ('crossref', 'openalex') for kind in ARTICLE_KINDS},
    **{kind: ('google_books', 'open_library') for kind in BOOK_KINDS},
    'book_section': ('google_books', 'open_library', 'crossref', 'openalex'),
    'report': ('openalex', 'crossref', 'datacite'),
    'thesis': ('openalex', 'crossref', 'datacite'),
    'unknown': ('openalex', 'crossref', 'datacite'),
}
WEB_PROVIDERS = frozenset({'brave', 'exa'})
BOOK_CATALOGUES = frozenset({'google_books', 'open_library', 'internet_archive'})
COMPLETED_ROUTE = frozenset({'no_match', 'candidate_found', 'candidates_processed'})
COMPLETED_QUERY = frozenset({'results', 'no_results'})
# Skipped by this application's own search or elapsed budget, not by the
# provider. A required web provider skipped this way leaves the web search
# incomplete even when the other one completed (owner decision 2026-10-01:
# both genuine regional articles flagged on the owner's paper had Exa skipped).
SELF_SKIPPED_QUERY = frozenset({'budget_skipped'})
CONFIRMED = frozenset({'confirmed', 'confirmed_with_minor_differences'})
PROVIDER_NAMES = {
    'crossref': 'Crossref', 'openalex': 'OpenAlex', 'datacite': 'DataCite',
    'google_books': 'Google Books', 'open_library': 'Open Library',
    'brave': 'Brave Search', 'exa': 'Exa',
}


def required_adapters(kind: str | None) -> tuple[str, ...]:
    return REQUIRED_ADAPTERS.get(normalize_source_kind(kind), REQUIRED_ADAPTERS['unknown'])


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _outcomes(candidate: dict) -> dict:
    return {c.get('field_name'): c.get('outcome') for c in candidate.get('comparisons') or []}


_EDITION = re.compile(r'\(\s*(?:\d+(?:st|nd|rd|th)|rev(?:ised)?\.?|new|updated)\s*ed(?:ition|\.)?\s*\)',
                      re.IGNORECASE)


def _title_forms(value) -> tuple[str, str]:
    """The title as words, and without spaces, edition text removed."""
    words = text_key(_EDITION.sub(' ', str(value or '')))
    return words, words.replace(' ', '')


def _book_title_matches(expected: dict, observed_title) -> bool:
    """A book record under the cited title, read tolerantly (2026-09-30).

    Replay on 21 stored papers: genuine books were missed because the paper's
    text lost spaces ("apopulistage"), the title carried "(2nd ed.)", the
    student gave only the start of a longer title, or the reference named a
    chapter whose book the catalogue returned. Each is still a record that a
    work by that title exists, so it is a possible match, never a flag.
    """
    seen, seen_joined = _title_forms(observed_title)
    if not seen:
        return False
    for cited in (expected.get('title'), expected.get('container_title')):
        words, joined = _title_forms(cited)
        if not words:
            continue
        if joined == seen_joined:
            return True
        # The cited title is the start of the record's title, at word
        # boundaries and at least four words long.
        if len(words.split()) >= 4 and (seen + ' ').startswith(words + ' '):
            return True
    return False


def _page_title_matches(expected: dict, observed_title) -> bool:
    """A web page titled with the cited title and then only a site suffix.

    2026-09-30: Manh's article ("… in Vietnam" then the site name) and
    Schatz's chapter ("Film Genre and The Genre Film, de Thomas Schatz -
    Scribd") were flagged although the web search returned pages under the
    exact cited title. The cited title (four words or more) must open the page
    title and be followed by nothing or a separator, never by more title words.
    """
    words = text_key(_EDITION.sub(' ', str(expected.get('title') or ''))).split()
    title = str(observed_title or '').strip()
    if len(words) < 4 or not title:
        return False
    pattern = r'\W*'.join(re.escape(w) for w in words)
    return re.match(rf'\W*{pattern}\W*(?:$|[|\-–—:,(\[•·]|\s(?:by|de|von|par)\s)', title,
                    re.IGNORECASE) is not None


def _possible_match(candidate: dict, expected: dict | None = None) -> bool:
    """A record that may be the cited work: its title agrees or nearly agrees.

    Author disagreement does not remove it, because a record under the same
    title is evidence that a work by that title exists. A record whose title
    conflicts is not one, including the record a borrowed DOI resolves to, and
    neither is a record that observed no title at all.
    """
    outcomes = _outcomes(candidate)
    if outcomes.get('title') in {'agreement', 'minor_difference'}:
        return True
    # The same DOI with the same author and year is the cited work whatever
    # its title: a title cut at a full stop or carrying the volume and pages
    # (Sanchez-Lopez, Bakulev; paper 4, 2026-10-02) is a parsing difference.
    if (outcomes.get('doi') == 'agreement' and outcomes.get('author') in {'agreement', 'minor_difference'}
            and outcomes.get('year') == 'agreement'):
        return True
    if not expected:
        return False
    observed = (candidate.get('observed') or {}).get('title')
    if candidate.get('provider') in BOOK_CATALOGUES:
        return _book_title_matches(expected, observed)
    return candidate.get('provider') == 'web_search' and _page_title_matches(expected, observed)


# Structured title parameters: Crossref `query.title`, OpenAlex `title.search`
# (as a parameter or inside `filter`).
_TITLE_PARAMETERS = ('query.title', 'title.search')
# A parameter or path that confines the search to one journal or source. A
# title query inside one journal cannot show that the work is absent from the
# index, so it never satisfies the title-led requirement by itself.
_SCOPE_MARKERS = ('/journals/', 'issn:', 'container-title', 'primary_location.source', 'locations.source',
                  'host_venue', 'source.id', 'journal')


def _structured_query(text: str) -> dict | None:
    """Parse a recorded structured request: JSON ``{path, params}`` or ``k=v&...``."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        params = parsed.get('params') if isinstance(parsed.get('params'), dict) else parsed
        return {'path': str(parsed.get('path') or ''), 'params': {str(k): str(v) for k, v in params.items()}}
    if '=' in text and not text.startswith(('title:', 'doi:', 'isbn:')):
        from urllib.parse import parse_qsl, urlsplit
        split = urlsplit(text)
        return {'path': split.path, 'params': dict(parse_qsl(split.query or text))}
    return None


def _title_led(query: dict) -> bool:
    """A query that searched the index by the cited title, not only an identifier.

    The resolver records `title:<title> author:<author>`; journal discovery and
    other structured requests record their parameters instead, so
    `query.title` / `title.search` count too (reference-verification-v2). A
    structured title query confined to one journal or source does not count.
    """
    text = str(query.get('normalized_query') or '')
    # Google Books records its title search as `intitle:<title>`. Counted
    # since 2026-09-30, together with the title-only fallback search and the
    # tolerant book-title match that keep genuine books unflagged.
    if text.startswith(('title:', 'intitle:')) or ' title:' in text or ' intitle:' in text:
        return True
    structured = _structured_query(text)
    if structured is None:
        return False
    params = structured['params']
    titled = any(key in params for key in _TITLE_PARAMETERS) or any(
        f'{key}:' in params.get('filter', '') for key in _TITLE_PARAMETERS)
    scoped = any(marker in structured['path'] for marker in _SCOPE_MARKERS) or any(
        marker in f'{key}={value}' for key, value in params.items()
        if key not in _TITLE_PARAMETERS for marker in _SCOPE_MARKERS)
    return titled and not scoped


def assess_reference_verification(reference, discovery: dict | None) -> dict:
    """Return the verification status and, when flagged, one finding."""
    result = dict(policy_version=POLICY, reference_id=getattr(reference, 'reference_id', None),
                  status='not_assessed', reason_code='discovery_unavailable', findings=[])
    if not isinstance(discovery, dict) or not discovery.get('attempts') and not discovery.get('outcome'):
        return result
    expected = discovery.get('expected') or {}
    kind = normalize_source_kind(expected.get('source_kind') or getattr(reference, 'source_kind', None))
    result['source_kind'] = kind
    if not is_bibliographically_searchable(kind):
        result['reason_code'] = 'not_held_by_academic_indexes'
        return result
    outcome = discovery.get('outcome')
    if outcome == 'insufficient_metadata':
        result['reason_code'] = 'insufficient_metadata'
        return result
    if outcome in CONFIRMED:
        result.update(status='verified', reason_code=outcome)
        return result
    container = discovery.get('container_identity') or {}
    if container.get('status') == 'identified' and container.get('outcome') in CONFIRMED:
        # Owner decision 2026-09-21 (Belton): a part cited with the wrong
        # framing still names a real book. The containing work was located, so
        # the reference cannot be "unverifiable"; the part itself stays
        # unconfirmed and any miscitation is a formatting finding, not this one.
        result.update(status='container_located', reason_code='containing_work_located',
                      containing_work={k: container.get(k) for k in ('title', 'authors', 'year', 'provider')})
        return result
    if container.get('outcome') == 'possible_match':
        # A possible match is shown but never flagged; that holds for the book
        # a part names as much as for the part (Belton's 2026-09-27 rerun).
        result.update(status='possible_match', reason_code='containing_work_possible_match',
                      containing_work={k: container.get(k) for k in ('title', 'authors', 'year', 'provider')})
        return result
    candidates = discovery.get('candidates') or []
    matches = [c for c in candidates if _possible_match(c, expected)]
    if outcome == 'possible_match' or matches:
        result.update(status='possible_match', reason_code='possible_match_not_confirmed',
                      possible_matches=[dict(provider=c.get('provider'),
                                             observed={k: (c.get('observed') or {}).get(k)
                                                       for k in ('title', 'authors', 'year', 'container_title', 'doi')})
                                        for c in matches[:3]])
        return result

    queries = {q.get('query_id'): q for q in discovery.get('queries') or []}
    title = text_key(expected.get('title') or getattr(reference, 'title', '') or '')
    completed, failed, web, self_skipped = set(), set(), set(), set()
    for attempt in discovery.get('attempts') or []:
        category, provider = attempt.get('route_category'), attempt.get('provider')
        attempted = [queries[q] for q in attempt.get('query_ids') or [] if q in queries]
        route_done = attempt.get('permitted') and attempt.get('completed_at') and attempt.get('outcome') in COMPLETED_ROUTE
        if category == 'academic_adapter':
            done = [q for q in attempted if q.get('execution_outcome') in COMPLETED_QUERY and _title_led(q)]
            if route_done and done:
                completed.add(provider)
            elif (attempt.get('outcome') in {'operational_failure', 'unavailable', 'access_restricted'}
                    and attempt.get('reason_code') != 'route_not_applicable'):   # declined, not failed
                failed.add(provider)
        elif category == 'bounded_web':
            for q in attempted:
                engine = q.get('execution_provider')
                if engine not in WEB_PROVIDERS:
                    continue
                if (route_done and q.get('execution_outcome') in COMPLETED_QUERY
                        and title and title in text_key(q.get('normalized_query'))):
                    web.add(engine)
                elif q.get('execution_outcome') not in COMPLETED_QUERY:
                    failed.add(engine)
                    if q.get('execution_outcome') in SELF_SKIPPED_QUERY:
                        self_skipped.add(engine)
    required = required_adapters(kind)
    missing = [p for p in required if p not in completed]
    result.update(required_routes=list(required), completed_routes=sorted(completed & set(required)),
                  web_routes=sorted(web))
    if missing or not web:
        result.update(status='search_incomplete',
                      reason_code='required_route_incomplete' if missing else 'title_web_search_incomplete',
                      incomplete_routes=missing + ([] if web else ['web search']),
                      failed_routes=sorted(failed))
        return result
    budget_skipped = sorted(self_skipped - web)
    if budget_skipped:
        result.update(status='search_incomplete', reason_code='web_search_skipped_by_budget',
                      incomplete_routes=[PROVIDER_NAMES[p] for p in budget_skipped],
                      failed_routes=sorted(failed))
        return result
    searched = [PROVIDER_NAMES.get(p, p) for p in required] + [PROVIDER_NAMES[p] for p in sorted(web)]
    finding = dict(
        finding_type=FINDING_TYPE, reference_id=result['reference_id'], policy_version=POLICY,
        finding=FINDING_TEXT,
        evidence_explanation='Searched by title and author: ' + ', '.join(searched) + '.',
        searched_routes=searched, source_kind=kind, discovery_outcome=outcome,
        limitations=['Search coverage is bounded, not an exhaustive catalog of published works.'],
        field_difference=dict(field_name='entry', submitted_value=getattr(reference, 'raw_ref', '')),
        rectangles=[], localization_status='not_assessed', discovery_sha256=_digest(discovery))
    result.update(status='cannot_be_verified', reason_code='suited_searches_completed_without_match',
                  findings=[finding])
    return result


def same_title_author_difference(reference, discovery: dict | None) -> dict | None:
    """The author a same-titled record names, when it is not the cited one.

    Owner decision 2026-09-30: a real work under the cited title credited to
    someone else (Singer's reference to Falsetto's book, Mather's to Duncan's,
    Kozlovic's to Greydanus's essay) is shown as the existing Source Record
    Conflict on the author, not left as a silent possible match. Only an
    exact title (case, spacing and edition text aside) and a record that names
    its author qualify (a web page may add only a site suffix to the title);
    any same-titled record naming the cited author, or
    same-titled records that disagree about the author, withhold it.
    """
    from app.services.relevance import extract_surnames

    if not isinstance(discovery, dict):
        return None
    cited_title = _title_forms(getattr(reference, 'title', '') or '')[1]
    cited_author = str(getattr(reference, 'author', '') or '').strip()
    cited_surnames = {s.casefold() for s in extract_surnames(cited_author)}
    if len(cited_title) < 12 or not cited_surnames:
        return None
    located: dict[frozenset, dict] = {}
    for candidate in discovery.get('candidates') or []:
        observed = candidate.get('observed') or {}
        if _title_forms(observed.get('title'))[1] != cited_title and not (
                candidate.get('provider') == 'web_search'
                and _page_title_matches({'title': getattr(reference, 'title', '')}, observed.get('title'))):
            continue
        authors = [re.sub(r'^[\W_]+', '', str(a)).strip() for a in observed.get('authors') or []]
        authors = [a for a in authors if a]
        surnames = frozenset(s.casefold() for a in authors for s in extract_surnames(a))
        if not surnames:
            continue
        if surnames & cited_surnames:
            return None
        publisher = str(observed.get('publisher') or (candidate.get('edition_metadata') or {}).get('publisher') or '')
        located.setdefault(surnames, dict(field_name='author', submitted_value=cited_author,
                                          located_value=', '.join(authors),
                                          provider=candidate.get('provider'),
                                          candidate_id=candidate.get('candidate_id'),
                                          located_title=str(observed.get('title') or ''),
                                          located_publisher=publisher.strip()))
    if len(located) != 1:
        return None
    found = next(iter(located.values()))
    # The same record's publisher, when it shares no name with the reference's
    # (Singer: Palgrave Macmillan against Falsetto's Greenwood, 2026-09-30).
    cited_publisher = str(getattr(reference, 'publisher', '') or '')
    if not cited_publisher:
        from app.services.ref_field_extractor import _publisher
        raw = str(getattr(reference, 'raw_ref', '') or '').rstrip()
        # Stored entries can lose their final full stop, which the reader needs.
        cited_publisher = _publisher(raw if raw.endswith('.') else raw + '.',
                                     str(getattr(reference, 'source_kind', '') or ''))
    generic = {'press', 'publishing', 'publishers', 'publisher', 'books', 'inc', 'ltd', 'llc', 'group', 'the', 'and', 'co'}
    names = lambda value: {w for w in re.findall(r'[a-z]+', value.casefold()) if w not in generic and len(w) > 2}
    if (found['located_publisher'] and cited_publisher
            and names(found['located_publisher']) and names(cited_publisher)
            and not names(found['located_publisher']) & names(cited_publisher)):
        found['publisher_difference'] = dict(field_name='publisher', submitted_value=cited_publisher.strip(),
                                             located_value=found['located_publisher'])
    return found
