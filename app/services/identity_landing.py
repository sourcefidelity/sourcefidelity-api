"""Bounded HTML bibliographic inspection; never acquires source evidence."""
import hashlib
from contextvars import ContextVar
from functools import wraps
import httpx

from app.services.candidate_budget import require_source_candidate, CandidateBudgetExceeded
from app.services.reference_review_scope import scope
from app.services.retrieval.base import RetrievalResult, RepresentationKind
from app.services.reference_discovery import build_reference_discovery_candidate
from app.services.safe_fetch import safe_request, UnsafeUrlError, ResponseTooLargeError, ResponseMediaTypeError
from app.services.web_source_metadata import extract_web_source_metadata

POLICY = "identity-only-landing-metadata-v1"
MAX_PAGES = 3
_ALLOWANCE = ContextVar('identity_landing_allowance', default=None)


def remaining_inspections():
    allowance = _ALLOWANCE.get()
    return MAX_PAGES if allowance is None else max(0, MAX_PAGES - allowance[0])


def bounded_landing_inspection(function):
    @wraps(function)
    def run(*args, **kwargs):
        if _ALLOWANCE.get() is not None:
            return function(*args, **kwargs)
        token = _ALLOWANCE.set([0])
        try:
            return function(*args, **kwargs)
        finally:
            _ALLOWANCE.reset(token)
    return run


def inspect(location, expected):
    attempt = dict(url=location.url, outcome="not_attempted",
                   reason_code="identity_landing_not_attempted")
    if location.representation_kind not in {None, RepresentationKind.HTML}:
        attempt['reason_code'] = 'identity_landing_non_html'
        return attempt
    allowance = _ALLOWANCE.get()
    if allowance is not None:
        if allowance[0] >= MAX_PAGES:
            attempt['reason_code'] = 'identity_landing_limit'
            return attempt
        allowance[0] += 1
    try:
        require_source_candidate(location.url)
        response = safe_request(location.url, usage_label="landing page", timeout=10, max_bytes=512 * 1024,
            headers={"Accept": "text/html,application/xhtml+xml"},
            allowed_media_types=frozenset({"text/html", "application/xhtml+xml"}))
        if not 200 <= response.status_code < 300:
            raise ValueError('identity_landing_http_status')
        observed = extract_web_source_metadata(response.text, str(response.url))
        # A generic page title alone cannot establish a bibliographic object.
        if not observed.get('title') or not observed.get('authors'):
            attempt.update(outcome='identity_unconfirmed', reason_code='landing_page_bibliography_unavailable')
            return attempt
        fields = {key: observed.get(key) for key in ('title', 'authors', 'year', 'doi')}
        fields['metadata'] = {key: observed[key]
                              for key in ('container_title', 'volume', 'issue', 'pages')
                              if observed.get(key)}
        candidate = build_reference_discovery_candidate(attempt_id='landing-inspection',
            provider='independent_landing_metadata', expected=expected,
            result=RetrievalResult(source_name='independent_landing_metadata', success=True, **fields))
        title_conflict = any(c.field_name == 'title' and c.outcome == 'material_conflict'
                             for c in candidate.comparisons)
        different = title_conflict and scope(expected.title, fields['title']) == 'outside_bound'
        confirmed = (not expected.reference_parse_review and candidate.plausible_identity_match
            and not candidate.has_material_conflict and not candidate.has_unresolved_supplied_identity_fields
            and (candidate.authoritative_identifier_match or sum(c.outcome in {'agreement', 'minor_difference'}
                 for c in candidate.comparisons if c.field_name in {'title', 'author', 'year', 'doi', 'isbn'}) >= 2))
        observation = dict(observed=fields, content_sha256=hashlib.sha256(response.content).hexdigest(),
                           source_url=str(response.url))
        attempt.update(outcome='unavailable' if confirmed else 'identity_rejected' if different else 'identity_unconfirmed',
            reason_code='identity_landing_metadata_observed',
            **{'landing_metadata_identity' if confirmed else 'landing_metadata_observation': observation})
    except CandidateBudgetExceeded:
        attempt['reason_code'] = 'candidate_budget_exhausted'
    except ResponseMediaTypeError:
        attempt.update(outcome='unavailable', reason_code='identity_landing_non_html_response')
    except ResponseTooLargeError:
        attempt.update(outcome='unavailable', reason_code='identity_landing_response_too_large')
    except UnsafeUrlError:
        attempt.update(outcome='transport_failure', reason_code='identity_landing_unsafe_url')
    except (httpx.TimeoutException, TimeoutError):
        attempt.update(outcome='transport_failure', reason_code='identity_landing_timeout')
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        attempt.update(outcome='access_restricted' if status in {401,403} else 'transport_failure',
                       reason_code=f'identity_landing_http_{status}')
    except Exception:
        # No URLs, response bodies or provider exception strings in diagnostics.
        attempt.update(outcome='transport_failure', reason_code='identity_landing_inspection_failed')
    return attempt
