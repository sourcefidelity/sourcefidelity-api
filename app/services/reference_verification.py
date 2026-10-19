"""Whether the application could confirm that a cited work exists.

Owner-approved 2026-09-24 as `reference-verification-v1`; the current policy is
v3 (2026-10-12, see POLICY). This replaces the
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

from app.services.search.policy import REQUIRED_WEB_PROVIDER_NAMES
from app.services.reference_review_scope import text_key
from app.services.source_type import is_bibliographically_searchable, normalize_source_kind

# v2 (2026-09-29): structured title queries (`query.title`, `title.search`)
# count as title-led unless confined to one journal. Amended 2026-09-30 (Google
# Books title searches) and 2026-10-01 (a required web provider skipped by our
# own budget leaves the search incomplete). v1 values stay readable; nothing
# parses this string back.
# v3 (owner decision 2026-10-12): a likely other-language work, a candidate
# whose title could not be compared across writing systems and a supplied link
# that answered with unreadable text stay neutral, and a web page confirmed by
# discovery is verified whatever its link check says. A likely other-language
# work can be found unverifiable only after its native-script route
# (native-script-route-v1) completed; a failed or unattempted route stays
# neutral, and that route's searches neither complete nor fail the ordinary
# required routes.
POLICY = 'reference-verification-v4'
POLICY_VERSIONS = ('reference-verification-v1', 'reference-verification-v2', 'reference-verification-v3', POLICY)
FINDING_TYPE = 'unverified_reference'
LABEL = 'Cannot be verified'
# Owner wording 2026-09-30.
FINDING_TEXT = 'Cannot be verified. Searches for this source could not locate the reference.'

ARTICLE_KINDS = frozenset({'journal_article', 'conference_paper', 'book_review'})
BOOK_KINDS = frozenset({'monograph', 'edited_collection'})
WEB_PAGE_KINDS = frozenset({'webpage', 'blog_post', 'news_article'})

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
WEB_PROVIDERS = REQUIRED_WEB_PROVIDER_NAMES
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
    'brave': 'Brave Search', 'exa': 'Exa', 'deepseek': 'DeepSeek Search',
}


def required_adapters(kind: str | None) -> tuple[str, ...]:
    return REQUIRED_ADAPTERS.get(normalize_source_kind(kind), REQUIRED_ADAPTERS['unknown'])


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


_REDIRECT = {301, 302, 303, 307, 308}


def registered_unread_doi(reference, discovery: dict | None, link_observations) -> str | None:
    """How a cited DOI that doi.org registers, but no index could read, agrees.

    Owner decision 2026-10-06 (Yang): doi.org redirecting the DOI shows it is
    registered, here by an agency whose metadata the indexes do not carry, so
    the reference is never "Cannot be verified". 'agrees' when the DOI's own
    suffix carries the cited year (and issue, where both give one), else
    'registered'; None when the DOI was not registered or an index read it.
    """
    doi = str(getattr(reference, 'doi', '') or '').strip().casefold()
    if not doi:
        return None
    for candidate in (discovery or {}).get('candidates') or []:
        if str((candidate.get('observed') or {}).get('doi') or '').strip().casefold() == doi:
            return None
    registered = False
    for observation in link_observations or []:
        if observation.get('kind') != 'doi':
            continue
        for request in observation.get('requests') or []:
            hops = request.get('hops') or []
            if (hops and str(hops[0].get('destination_origin') or '').split('://')[-1] == 'doi.org'
                    and hops[0].get('http_status') in _REDIRECT and hops[0].get('location_sha256')):
                registered = True
    if not registered:
        return None
    suffix = doi.split('/', 1)[-1]
    year = str(getattr(reference, 'year', '') or '')[:4]
    found = re.search(rf'(?<!\d){year}(?!\d)(?:\.(\d{{1,2}})(?!\d))?', suffix) if year.isdigit() else None
    if not found:
        return 'registered'
    issue = str(getattr(reference, 'issue', '') or '').strip()
    if found.group(1) and issue.isdigit() and int(found.group(1)) != int(issue):
        return 'registered'
    return 'agrees'


# Reasons that mean every search suited to the kind completed without
# locating the work; the neutral ones withhold the finding for a reason the
# searches could not overcome (reference-verification-v3).
COMPLETED_WITHOUT_MATCH_REASONS = frozenset({
    'suited_searches_completed_without_match', 'cross_script_candidate_unresolved',
    'other_language_route_not_searched', 'submitted_link_unreadable',
})

# Moved to app/services/other_language.py so discovery can trigger the
# native-script route (native-script-route-v1); re-exported here.
from app.services.other_language import (  # noqa: E402
    CJK as _CJK, other_language_signals, split_cited_title,
)


NATIVE_SCRIPT_ROUTE_POLICIES = ('native-script-route-v1',)


def _native_script_route_state(discovery: dict) -> str:
    """'completed', 'incomplete' or 'not_searched' for the work's own script.

    A recorded native-script route (`native-script-route-v1`) decides: only a
    route whose adapter and web searches completed counts; a route that failed,
    was skipped by a budget or had no query text stays incomplete. Without a
    route record, a reference cited in its own script was searched in that
    script by the ordinary route when a completed query carries Han, kana or
    Hangul.
    """
    route = discovery.get('native_script_route')
    if isinstance(route, dict) and route.get('policy_version') in NATIVE_SCRIPT_ROUTE_POLICIES:
        return 'completed' if route.get('status') == 'completed' else 'incomplete'
    if any(q.get('execution_outcome') in COMPLETED_QUERY and _CJK.search(str(q.get('normalized_query') or ''))
           for q in discovery.get('queries') or []):
        return 'completed'
    return 'not_searched'


def _link_unreadable(discovery: dict, link_observations) -> bool:
    """A supplied link answered, but no readable text could be taken from it."""
    if any(a.get('reason_code') == 'link_text_unreadable' for a in discovery.get('attempts') or []):
        return True
    return any(isinstance(request, dict) and request.get('outcome') == 'response'
               and isinstance(request.get('http_status'), int) and 200 <= request['http_status'] < 300
               and request.get('page_observation') == 'readable_text_unavailable'
               for row in link_observations or [] if isinstance(row, dict)
               for request in row.get('requests') or [])


def _outcomes(candidate: dict) -> dict:
    return {c.get('field_name'): c.get('outcome') for c in candidate.get('comparisons') or []}


_EDITION = re.compile(r'\(\s*(?:(?:\d+(?:st|nd|rd|th)|rev(?:ised)?\.?|new|updated)\s*ed(?:ition|\.)?'
                      # A series volume carried in the title, "(Vol. 15)" (Wu, 2026-10-04).
                      r'|vol(?:ume|\.)?\s*\d+)\s*\)',
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


def _same_main_title(cited, observed) -> bool:
    """The titles before their subtitles agree, three words or more."""
    cited_main, _ = _title_forms(str(cited or '').split(':', 1)[0])
    seen_main, _ = _title_forms(str(observed or '').split(':', 1)[0])
    return bool(cited_main) and len(cited_main.split()) >= 3 and cited_main == seen_main


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
    cited_words = re.findall(r'\w+', _title_forms(expected.get('title'))[0])
    observed_words = re.findall(r'\w+', _title_forms(observed)[0])
    if (outcomes.get('doi') == 'agreement' and outcomes.get('year') == 'agreement'
            and len(cited_words) >= 2 and observed_words[:len(cited_words)] == cited_words):
        # The same DOI and year, the cited title the start of the record's: a
        # title cut short by parsing ("Party Dominance vs", Hao, 2026-10-06),
        # whatever the author's name order.
        return True
    if (outcomes.get('author') in {'agreement', 'minor_difference'} and outcomes.get('year') == 'agreement'
            and len(cited_words) >= 4 and observed_words[:len(cited_words)] == cited_words):
        # The same author and year, the cited title (four words or more, read
        # word by word so an unspaced dash separates words) the start of the
        # record's: a title cut before its part or subtitle ("English in higher
        # education–English medium" for "… English medium. Part 1: Literature
        # review", Curle 2020, the long article, 2026-10-08).
        return True
    if (outcomes.get('author') in {'agreement', 'minor_difference'}
            and candidate.get('provider') in BOOK_CATALOGUES | {'crossref', 'openalex'}
            and _same_main_title(expected.get('title'), observed)):
        # The same author's book under the same main title with another
        # subtitle, as editions retitle (Wu's "The Curse of Bigness",
        # 2026-10-04): a record that the work exists, never a flag.
        return True
    if candidate.get('provider') in BOOK_CATALOGUES:
        return _book_title_matches(expected, observed)
    return candidate.get('provider') == 'web_search' and _page_title_matches(expected, observed)


from app.services.search_completion import _structured_query, _title_led  # noqa: E402,F401


def assess_reference_verification(reference, discovery: dict | None, web_page_check: dict | None = None,
                                  link_observations: list | None = None) -> dict:
    """Return the verification status and, when flagged, one finding.

    A web page is assessed under `webpage-verification-v1` from
    ``web_page_check`` (its link, an archived capture) plus the web searches;
    see app/services/web_page_verification.py."""
    result = dict(policy_version=POLICY, reference_id=getattr(reference, 'reference_id', None),
                  status='not_assessed', reason_code='discovery_unavailable', findings=[])
    web_rule = False
    kind = normalize_source_kind(getattr(reference, 'source_kind', None))
    if web_page_check and kind in WEB_PAGE_KINDS:
        result.update(source_kind=kind, web_page_check=dict(web_page_check))
        if web_page_check.get('link') == 'confirmed':
            result.update(status='verified', reason_code='submitted_page_confirmed')
            return result
        if web_page_check.get('archive') == 'matched':
            result.update(status='verified', reason_code='archived_page_confirmed')
            return result
        if isinstance(discovery, dict) and discovery.get('outcome') in CONFIRMED:
            # Discovery confirmed the work (a student link named it across
            # writing systems, 2026-10-12): the link check cannot undo that.
            result.update(status='verified', reason_code=discovery['outcome'])
            return result
        if web_page_check.get('link') == 'near':
            # The linked page carries nearly the cited title: shown, never flagged.
            result.update(status='possible_match', reason_code='submitted_page_near_title')
            return result
        if web_page_check.get('link') != 'telling':
            result['reason_code'] = 'web_page_link_not_telling'
            return result
        web_rule = True
    if not isinstance(discovery, dict) or not discovery.get('attempts') and not discovery.get('outcome'):
        return result
    expected = discovery.get('expected') or {}
    kind = normalize_source_kind(expected.get('source_kind') or getattr(reference, 'source_kind', None))
    result['source_kind'] = kind
    archive_page = discovery.get('periodical_archive_match') or {}
    if archive_page.get('page_id'):
        # A digitised magazine page shows the article (media-history-archive-v1):
        # shown as a possible match and never flagged (2026-10-16).
        result.update(status='possible_match', reason_code='periodical_archive_page_located',
                      archive_page={k: archive_page.get(k) for k in
                                    ('page_id', 'publication', 'volume_dates', 'read_url', 'excerpt')})
        return result
    if not web_rule and not is_bibliographically_searchable(kind):
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
    responsibility = container.get('responsibility') or {}
    title_page_found = bool(responsibility.get('statements')
                            and responsibility.get('classification') in {'single_authored', 'edited'})
    if container.get('outcome') == 'possible_match' or title_page_found:
        # A possible match is shown but never flagged; that holds for the book
        # a part names as much as for the part (Belton's 2026-09-27 rerun). A
        # catalogue's title page for the book under its main title and the
        # cited author also shows the book exists, though the search under the
        # entry's own garbled title found nothing (Synthetic Embrace reference
        # 10, owner decision 2026-10-18).
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

    if any(c.get('reason_code') == 'cross_script_model_unsure'
           for candidate in candidates for c in candidate.get('comparisons') or []):
        # The student's link names a work in another writing system that the
        # model could not tell apart from the cited one (2026-10-06).
        result['reason_code'] = 'cross_script_page_unresolved'
        return result

    # One answer for every use (search-completion-v1, owner decisions 2026-10-19).
    # The native-script route has its own completion; its searches neither
    # complete nor fail the ordinary required routes.
    from app.services.search_completion import assess_search_completion
    native_attempts = set((discovery.get('native_script_route') or {}).get('attempt_ids') or [])
    completion = assess_search_completion(
        discovery, kind=kind, title=expected.get('title') or getattr(reference, 'title', '') or '',
        web_rule=web_rule, exclude_attempt_ids=native_attempts)
    result.update(required_routes=list(completion.required_routes),
                  completed_routes=list(completion.completed_routes), web_routes=list(completion.web_routes))
    if not completion.complete:
        result.update(status='search_incomplete', reason_code=completion.reason_code,
                      incomplete_routes=list(completion.incomplete_routes),
                      failed_routes=list(completion.failed_routes))
        return result
    required, web = completion.required_routes, set(completion.web_routes)
    doi_state = registered_unread_doi(reference, discovery, link_observations)
    if doi_state == 'agrees':
        result.update(status='possible_match', reason_code='registered_doi_agrees')
        return result
    if doi_state:
        result['reason_code'] = 'registered_doi_unread'
        return result
    # Owner decision 2026-10-12: searches that could not reach the work are
    # no sign that it is missing. A candidate titled in another writing
    # system that nothing compared, a work in another language that no search
    # looked for in its own script, and a supplied link that answered with
    # text that could not be read all leave the reference not assessed.
    if any(c.get('field_name') == 'title' and c.get('reason_code') == 'cross_script_title_unresolved'
           for candidate in candidates for c in candidate.get('comparisons') or []):
        result['reason_code'] = 'cross_script_candidate_unresolved'
        return result
    signals = other_language_signals(reference)
    native_state = _native_script_route_state(discovery) if signals else 'completed'
    if native_state != 'completed':
        # Only a completed search in the work's own script can stand behind
        # the finding; a failed or unattempted one stays neutral (2026-10-12).
        result.update(reason_code=('other_language_route_incomplete' if native_state == 'incomplete'
                                   else 'other_language_route_not_searched'),
                      other_language_signals=signals)
        return result
    if _link_unreadable(discovery, link_observations):
        result['reason_code'] = 'submitted_link_unreadable'
        return result
    searched = [PROVIDER_NAMES.get(p, p) for p in required] + [PROVIDER_NAMES[p] for p in sorted(web)]
    finding = dict(
        finding_type=FINDING_TYPE, reference_id=result['reference_id'], policy_version=POLICY,
        finding=FINDING_TEXT,
        evidence_explanation='Searched by title and author: ' + ', '.join(searched) + '.',
        searched_routes=searched, source_kind=kind, discovery_outcome=outcome,
        **({'web_page_check': dict(web_page_check)} if web_rule else {}),
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
    located: list[tuple[frozenset, dict]] = []
    cited_container = _title_forms(getattr(reference, 'container_title', '') or '')[1]
    for candidate in discovery.get('candidates') or []:
        observed = candidate.get('observed') or {}
        # An article in a named journal or newspaper is not the same work as a
        # same-titled record outside it (Nuhn's newspaper article against an
        # Amazon book listing by another author, 2026-10-07).
        if cited_container and _title_forms(observed.get('container_title') or '')[1] != cited_container:
            continue
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
        located.append((surnames, dict(field_name='author', submitted_value=cited_author,
                                       located_value=', '.join(authors),
                                       provider=candidate.get('provider'),
                                       candidate_id=candidate.get('candidate_id'),
                                       located_title=str(observed.get('title') or ''),
                                       located_publisher=publisher.strip())))
    # Records sharing a surname name the same author, however a page splits the
    # name ("Falsetto", "Mario" against "Mario Falsetto", 2026-10-03).
    if not located or any(not a & b for a, _ in located for b, _ in located):
        return None
    # A catalogue record naming its publisher is shown before a web page's.
    found = min((record for _, record in located),
                key=lambda r: (r['provider'] == 'web_search', not r['located_publisher']))
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
