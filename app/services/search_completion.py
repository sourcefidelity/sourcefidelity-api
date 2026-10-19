"""Search completion: did every search a reference requires finish? (CONTEXT: Search completion)

One answer for every use (owner decisions 2026-10-19): the Cannot be verified
finding, the discovery outcome, automatic retries, re-run reuse and the
own-script route all read it. A search that did not finish never counts as
evidence that a work is missing.

Rules, applied to the attempts considered:
* an attempt counts only when permitted, finished (`completed_at`) and with a
  completed route outcome; a started-but-unfinished attempt does not;
* an index counts only through a completed title-led query, and a web provider
  only through a completed query containing the title searched for;
* every index the kind requires must count, and at least one required web
  provider must; a provider-side failure of the other web provider leaves the
  one that finished sufficient;
* a required web provider we skipped ourselves (our budget or an outage
  pause) leaves the search incomplete unless it finished another title query;
* a student's own link that failed on our side, with no attempt on it
  answering, leaves the search incomplete; a link that answered "gone" does not;
* a required route outside the indexes and web search that failed leaves it
  incomplete.
A stored record is judged by the web providers it names.
Not rules here (owner decision 2026-10-20, after the replay): a filtered
metadata answer and an unresolved web-search lead do not make a search
unfinished; each removed most correct Cannot be verified findings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.services.reference_review_scope import text_key
from app.services.search.policy import REQUIRED_WEB_PROVIDER_NAMES, is_required_web_provider_set, required_api_providers
from app.services.source_type import normalize_source_kind

VERSION = "search-completion-v1"

ARTICLE_KINDS = frozenset({"journal_article", "conference_paper", "book_review"})
BOOK_KINDS = frozenset({"monograph", "edited_collection"})
REQUIRED_ADAPTERS = {
    **{kind: ("crossref", "openalex") for kind in ARTICLE_KINDS},
    **{kind: ("google_books", "open_library") for kind in BOOK_KINDS},
    "book_section": ("google_books", "open_library", "crossref", "openalex"),
    "report": ("openalex", "crossref", "datacite"),
    "thesis": ("openalex", "crossref", "datacite"),
    "unknown": ("openalex", "crossref", "datacite"),
}
COMPLETED_ROUTE = frozenset({"no_match", "candidate_found", "candidates_processed"})
# A metadata lookup that answered, but whose records were filtered or not
# qualified before they were checked: not a certified empty result.
UNCERTIFIED_METADATA_REASONS = frozenset({"metadata_candidates_filtered", "metadata_empty_response_unqualified"})
COMPLETED_QUERY = frozenset({"results", "no_results"})
# Our own budget, an outage pause we set, or our own code failing.
SKIPPED_BY_US = frozenset({"budget_skipped", "cooldown_skipped", "internal_error"})
ROUTE_FAILURE = frozenset({"operational_failure", "access_restricted", "unavailable", "not_permitted"})
# Query outcomes a provider recovering could plausibly cure (our own budget,
# our own code and unknowns excluded).
PROVIDER_CURABLE = frozenset({"timeout", "captcha", "operational_failure", "access_restricted",
                              "rate_limited", "response_invalid", "cooldown_skipped",
                              "recovery_probe_in_progress"})
PROVIDER_NAMES = {
    "crossref": "Crossref", "openalex": "OpenAlex", "datacite": "DataCite",
    "google_books": "Google Books", "open_library": "Open Library",
    "brave": "Brave Search", "exa": "Exa", "deepseek": "DeepSeek Search",
}

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


@dataclass(frozen=True)
class SearchCompletion:
    complete: bool
    reason_code: str | None = None
    required_routes: tuple[str, ...] = ()
    completed_routes: tuple[str, ...] = ()
    completed_adapters: tuple[str, ...] = ()   # every index that finished, required or not
    web_routes: tuple[str, ...] = ()
    incomplete_routes: tuple[str, ...] = ()
    failed_routes: tuple[str, ...] = ()
    blocking_providers: tuple[str, ...] = field(default=())


def required_adapters(kind: str | None) -> tuple[str, ...]:
    return REQUIRED_ADAPTERS.get(normalize_source_kind(kind), REQUIRED_ADAPTERS["unknown"])


def _get(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def title_led(query) -> bool:
    """`_title_led` for a query given as a dict or a model."""
    return _title_led(query if isinstance(query, dict) else query.model_dump(mode="json"))


def _finished(attempt) -> bool:
    return bool(_get(attempt, "permitted") and _get(attempt, "completed_at")
                and _get(attempt, "outcome") in COMPLETED_ROUTE)


def assess_search_completion(
    discovery,
    *,
    kind: str | None = None,
    title: str | None = None,
    web_rule: bool = False,
    attempt_ids=None,
    exclude_attempt_ids=(),
    any_adapter: bool = False,
    any_web_provider: bool = False,
    require_web: bool = True,
    completed_negative: bool = False,
) -> SearchCompletion:
    """Whether the considered attempts of a discovery record (dict or model) finished.

    `kind` and `title` default to the record's expected fields. `web_rule`: the
    reference is a web page held by web search alone, so no index is required.
    `attempt_ids` restricts to a route's own attempts (the own-script route);
    `exclude_attempt_ids` leaves them out of the ordinary routes. With
    `any_adapter`, one finished index suffices instead of the kind's set; with
    `any_web_provider` (records under the legacy configured-search policy), any
    web engine the record ran counts. `completed_negative` adds the stricter
    checks the "unlocated after search" label and the credibility audit keep:
    an unresolved web-search lead and an uncertified metadata answer also leave
    the search unfinished (not used for Cannot be verified, owner decision
    2026-10-20).
    """
    expected = _get(discovery, "expected") or {}
    kind = kind if kind is not None else _get(expected, "source_kind")
    title_key = text_key(title if title is not None else (_get(expected, "title") or ""))
    queries = {_get(q, "query_id"): q for q in (_get(discovery, "queries") or [])}
    attempts = [a for a in (_get(discovery, "attempts") or [])
                if (attempt_ids is None or _get(a, "attempt_id") in set(attempt_ids))
                and _get(a, "attempt_id") not in set(exclude_attempt_ids)]
    named = _get(discovery, "required_web_providers")
    web_providers = (frozenset(named) if is_required_web_provider_set(named)
                     else frozenset(required_api_providers()) if not named
                     else REQUIRED_WEB_PROVIDER_NAMES)

    completed, failed, web, skipped, engines = set(), set(), set(), set(), set()
    blocking: set[str] = set()
    other_failed: list[str] = []
    student_attempts = []
    for attempt in attempts:
        category, provider = _get(attempt, "route_category"), _get(attempt, "provider")
        attempted = [queries[q] for q in (_get(attempt, "query_ids") or []) if q in queries]
        done_route = _finished(attempt)
        if category == "academic_adapter":
            # A filtered metadata answer still counts here (owner decision
            # 2026-10-20: counting it as unfinished removed 52 of 71 correct
            # findings); the discovery record keeps its reason code.
            done = [q for q in attempted if _get(q, "execution_outcome") in COMPLETED_QUERY and title_led(q)
                    and not (completed_negative and _get(q, "reason_code") in UNCERTIFIED_METADATA_REASONS)]
            if done_route and done:
                completed.add(provider)
            elif (_get(attempt, "outcome") in ROUTE_FAILURE
                    and _get(attempt, "reason_code") != "route_not_applicable"):
                failed.add(provider)
                if (_get(attempt, "reason_code") != "route_elapsed_budget_exhausted"
                        or any(_get(q, "execution_outcome") in PROVIDER_CURABLE for q in attempted)):
                    blocking.add(provider)
            for q in attempted:
                if _get(q, "execution_outcome") in PROVIDER_CURABLE:
                    blocking.add(provider)
        elif category == "bounded_web":
            for q in attempted:
                engine = _get(q, "execution_provider")
                if engine not in web_providers and not (any_web_provider and engine):
                    continue
                engines.add(engine)
                outcome = _get(q, "execution_outcome")
                if (done_route and outcome in COMPLETED_QUERY
                        and title_key and title_key in text_key(_get(q, "normalized_query") or "")):
                    web.add(engine)
                elif outcome not in COMPLETED_QUERY:
                    failed.add(engine)
                    if outcome in SKIPPED_BY_US:
                        skipped.add(engine)
                    if outcome in PROVIDER_CURABLE:
                        blocking.add(engine)
        elif category == "student_url":
            if str(provider or "").startswith("student_url"):
                student_attempts.append(attempt)
        elif _get(attempt, "required") and _get(attempt, "outcome") in ROUTE_FAILURE:
            other_failed.append(provider)
            blocking.add(provider)
        if completed_negative:
            for audit in (_get(attempt, "transient_search_audits") or []):
                if _get(audit, "unresolved") or _get(audit, "not_attempted") or _get(audit, "identity_established"):
                    return SearchCompletion(complete=False, reason_code="web_search_lead_unresolved")

    required = () if web_rule else required_adapters(kind)
    if any_adapter:
        missing = [] if completed else ["academic index"]
    else:
        missing = [p for p in required if p not in completed]
    for provider in ([] if any_web_provider else web_providers):
        if provider not in web and provider not in failed:
            # A required web provider that never ran at all could still finish.
            blocking.add(provider)
    common = dict(required_routes=tuple(required), completed_routes=tuple(sorted(completed & set(required))),
                  completed_adapters=tuple(sorted(completed)),
                  web_routes=tuple(sorted(web)), failed_routes=tuple(sorted(failed)))

    def incomplete(reason, routes):
        web_candidates = engines if any_web_provider else set(web_providers)
        relevant = {p for p in blocking if p in set(missing) | (web_candidates - web) | set(other_failed)}
        return SearchCompletion(complete=False, reason_code=reason, incomplete_routes=tuple(routes),
                                blocking_providers=tuple(sorted(relevant)), **common)

    if other_failed:
        return incomplete("required_route_incomplete", other_failed)
    if missing or (require_web and not web):
        return incomplete("required_route_incomplete" if missing else "title_web_search_incomplete",
                          missing + ([] if web or not require_web else ["web search"]))
    skipped_only = sorted(skipped - web)
    if skipped_only:
        return incomplete("web_search_skipped_by_us", [PROVIDER_NAMES.get(p, p) for p in skipped_only])
    if student_attempts and all(_get(a, "outcome") == "operational_failure" for a in student_attempts):
        return SearchCompletion(complete=False, reason_code="student_link_failed",
                                incomplete_routes=("student link",), **common)
    return SearchCompletion(complete=True, **common)


def assess_record(record, *, completed_negative: bool = False) -> SearchCompletion:
    """The completion of a whole discovery record or trace (dict or model): the
    kind's indexes unless the record required none (a web page), and the
    record's own web providers."""
    from app.services.search.policy import API_FIRST_SEARCH_POLICY
    categories = set(_get(record, "required_route_categories") or ())
    return assess_search_completion(
        record, web_rule="academic_adapter" not in categories,
        any_web_provider=_get(record, "search_policy_version") != API_FIRST_SEARCH_POLICY,
        require_web="bounded_web" in categories, completed_negative=completed_negative)
