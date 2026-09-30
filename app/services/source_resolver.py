"""Source resolution orchestrator.

Implements the resolution priority chain:
    Local S3 cache -> Student URL -> Retrieval Sources -> Fail
"""

import hashlib
import html
import logging
import re
import unicodedata
import time
from copy import deepcopy
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
from app.services.retrieval_deadline import bounded_reference, expired, POLICY as ELAPSED_POLICY

# Reference-local fallback only: never an accepted location or reusable cache hit.
_PROVISIONAL_CANDIDATES: ContextVar = ContextVar("provisional_candidates", default=None)


def _timed_retrieval(operation):
    """Measure actual operation boundaries; nested spans are not additive."""
    @wraps(operation)
    def measured(*args, **kwargs):
        started = datetime.now(timezone.utc)
        tick = time.perf_counter()
        if expired() and operation.__name__ not in {"_download_and_cache", "_check_local_cache"}:
            result = RetrievalResult(source_name=operation.__name__, success=False,
                                     error="reference_elapsed_budget_timeout")
        else:
            result = operation(*args, **kwargs)
        if isinstance(result, RetrievalResult):
            observation = {
                'version': 'retrieval-operation-timing-v1',
                'operation': operation.__name__,
                'started_at': started.isoformat(),
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': max(0.0, time.perf_counter() - tick),
            }
            result = replace(result, metadata={
                **(result.metadata or {}), 'operation_timing': observation,
                'operation_timings': [*(result.metadata or {}).get('operation_timings', []), observation],
            })
        return result
    return measured
from app.services.web_fetch_diagnostics import WebFetchDiagnostic
from app.services.submitted_links import observe_reference, identity_observed, page_observed, link_not_visited, validated_response_identity, source_validation_observed
from app.services.bibliographic_scripts import cross_script_comparison_unresolved

from app.log_safety import private_value_id, safe_exception_code

from app.config import settings
from app.database import SessionLocal
from app.services.file_safety import (
    FileSafetyUnavailable,
    SafetyVerdict,
    inspect_uploaded_pdf,
)
from app.services.storage import get_storage_backend
from app.services.retrieval import (
    AcquisitionLocation,
    CanonicalWorkGraph,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
    SourceRepresentation,
    get_retrieval_sources,
)
from app.services.retrieval.landing_page import discover_scholarly_locations
from app.services.publisher_urls import (
    MDPI_BLOCKED_STATUSES, MDPI_FILE_SERVER_ROUTE, is_mdpi_doi, is_mdpi_location,
    mdpi_file_server_urls, mdpi_request_headers,
)
from app.services.retrieval.core import PROVIDER_SKIPPED_ERROR
from app.services.retrieval.web_search import reference_search_scoped
from app.services.candidate_budget import (
    ACTIVE_CANDIDATE_BUDGET, CandidateBudgetExceeded,
    require_source_candidate, reserve_metadata_candidate,
)
from contextvars import copy_context
from app.services.retrieval_lookup_cache import RetrievalLookupCache
from app.services.source_type import (
    SourceKindAssessment,
    classify_content_source_kind,
    classify_html_source_kind,
    classify_provider_source_kind,
    classify_reference_source_kind,
    compare_source_kinds,
    document_kind_for_source_kind,
    is_archive_source,
    is_library_locator_url,
    is_traditional_media,
    normalize_source_kind,
)
from app.services.source_validator import _detect_nonwork_listing, validate_retrieved_pdf
from app.services.web_source_metadata import extract_web_source_metadata
from app.services.identity_landing import bounded_landing_inspection
from app.services.identity_pdf import bounded_pdf_inspection
from app.services.search.candidate_ranking import identity_candidate_ranking
from app.services.pure_scan_ocr import prepare_pure_scan_ocr
from app.services.reference_discovery import (
    INCOMPLETE_METADATA_SEARCH_REASONS,
    CandidateAcquisitionOutcome,
    ExpectedBibliographicFields,
    ReferenceDiscoveryTrace,
    ReferenceRouteAttempt,
    ReferenceSearchQuery,
    assess_reference_discovery_trace,
    build_reference_discovery_candidate,
    required_reference_discovery_routes,
    is_edition_sensitive_reference,
)
from app.services.source_repository import (
    LICENSE_CLASSES,
    AdmissionError,
    AdmissionRequest,
    WorkIdentity,
    admit_derived_representation_pair,
    admit_representation,
    commit_source_admissions,
    find_accepted_representation,
    rollback_source_admissions,
)
from app.services.safe_fetch import (
    UnsafeUrlError,
    ResponseTooLargeError,
    safe_request,
    safe_fetch_bytes,
    trusted_url_matches_prefix,
)

logger = logging.getLogger(__name__)


def _append_unique_candidate(trace: dict, candidate) -> None:
    """Append a discovery candidate unless its exact record is already present.

    A candidate's id is derived from its attempt, provider and record key, so a
    repeat means the provider returned the same record twice in one attempt —
    Google Books did exactly that for a paper-11 reference. A duplicate id makes
    the whole trace invalid, and the reference then cannot be assessed at all,
    so one repeated row silently costs a real finding.
    """
    if any(existing.candidate_id == candidate.candidate_id
           for existing in trace["candidates"]):
        return
    trace["candidates"].append(candidate)


def _search_execution_outcome(result: RetrievalResult) -> str:
    if (result.metadata or {}).get("candidate_budget_skipped"):
        return "budget_skipped"
    if result.success:
        return "results"
    error = (result.error or "").casefold().strip()
    if not error or error in {"not found", "no results", "no search results", "no match"} or error.startswith("no relevant match"):
        return "no_results"
    if any(value in error for value in ("401", "403", "451", "paywall", "access restricted")):
        return "access_restricted"
    # A declined call is not an outcome of a request. Classify it before the
    # timeout test: a circuit opened by earlier timeouts used to describe itself
    # with the word "timeout", so every skipped call was recorded as one.
    if "provider_call_skipped" in error:
        return "cooldown_skipped"
    if "timeout" in error:
        return "timeout"
    if "429" in error or "rate limit" in error:
        return "rate_limited"
    return "operational_failure"

_HTTP_STATUS_IN_ERROR = re.compile(r"\bhttp[_\s]?(\d{3})\b", re.IGNORECASE)


def _route_error_code(result) -> str | None:
    """Classify a failed route against a fixed vocabulary.

    Nothing from the error text is carried through; the text is only matched
    against known markers, so a provider message, URL or document fragment can
    never reach the trace. The point is to separate a local network loss from a
    provider fault: a sleeping machine once produced 156 failures across every
    provider, and the trace recorded only `route_unclassified_failure`, which
    reads as a provider incident.

    An HTTP status is read only from the two shapes the adapters produce -
    `http_503` from `safe_exception_code` and `... HTTP 503` from the adapter
    messages - never from bare digits, which occur incidentally in any text.
    Timeouts are tested before connection errors because `ConnectTimeout`
    contains both words and is a timeout.
    """
    error = (getattr(result, "error", None) or "").casefold()
    if not error:
        return None
    if "provider_call_skipped" in error or "circuit" in error:
        return "circuit_open"
    status = _HTTP_STATUS_IN_ERROR.search(error)
    if status:
        code = int(status.group(1))
        if code == 429:
            return "rate_limited"
        if code in {401, 403, 451}:
            return "access_restricted"
        if 500 <= code <= 599:
            return "http_server_error"
        if 400 <= code <= 499:
            return "http_client_error"
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if "connect" in error:
        return "connect_error"
    if any(marker in error for marker in ("dns", "network", "unreachable", "resolve")):
        return "network_error"
    if "rate limit" in error:
        return "rate_limited"
    if any(marker in error for marker in ("paywall", "access restricted", "unauthorized", "forbidden")):
        return "access_restricted"
    if any(marker in error for marker in ("not configured", "no api_key", "unavailable")):
        return "provider_unavailable"
    if any(marker in error for marker in ("invalid", "json", "parse", "decod")):
        return "invalid_response"
    return "unclassified"


_ACTIVE_DISCOVERY_TRACE: ContextVar[dict | None] = ContextVar(
    "sourcefidelity_reference_discovery_trace", default=None
)

# Magic-byte check for PDF
PDF_MAGIC = b"%PDF-"
_MAX_LOCATION_ATTEMPTS = 5
_MIN_ROOT_LOCATION_ATTEMPTS = 3
_MAX_STRUCTURED_PROVIDER_WORKERS = 5


def _route_blocks_completion(source) -> bool:
    """Whether this route's failure blocks saying a reference was not found.

    The adapter declares it. Previously this was derived from `deferred`,
    which means "batches DOI prefetch" and has nothing to do with evidence;
    that derivation survives only as the fallback for a source declaring
    neither, so a duck-typed source behaves exactly as before.
    """
    blocks = getattr(source, "blocks_search_completion", None)
    if callable(blocks):
        declared = blocks()
        # Only a real boolean is a declaration. Anything else -- a stub, a
        # Mock, a misconfigured adapter -- must not become a truthy "required"
        # by accident, because that silently changes whether a route's failure
        # blocks a finding.
        if isinstance(declared, bool):
            return declared
    return not getattr(source, "deferred", False)


def _any_source_kind(_kind: str | None) -> bool:
    """An adapter that declares no kind restriction is consulted for every kind.

    Read through `getattr`, as `capabilities` already is, so a source that does
    not derive from `RetrievalSource` keeps working unchanged.
    """
    return True


def _expected_source_kind_assessment(
    *,
    source_kind: str | None,
    source_kind_confidence: str,
    source_kind_evidence: tuple[str, ...],
    raw_ref: str | None,
    title: str | None,
    url: str | None,
) -> SourceKindAssessment:
    normalized = normalize_source_kind(source_kind)
    if normalized != "unknown":
        confidence = (
            source_kind_confidence
            if source_kind_confidence in {"high", "medium", "low"}
            else "medium"
        )
        return SourceKindAssessment(normalized, confidence, source_kind_evidence[:4])
    return classify_reference_source_kind(raw_ref, title=title, url=url)


class _CanonicalGraphSource(RetrievalSource):
    """Shared acquisition marker for an already-merged provider graph."""

    name = "canonical_work_graph"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(source_name=self.name, success=False)

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        return RetrievalResult(source_name=self.name, success=False)


_CANONICAL_GRAPH_SOURCE = _CanonicalGraphSource()


class _MdpiFileServerSource(_CanonicalGraphSource):
    """Acquisition marker for a PDF constructed from an MDPI DOI."""

    name = "mdpi_file_server"


_MDPI_FILE_SERVER_SOURCE = _MdpiFileServerSource()


def _url_prefers_pdf(url: str) -> bool:
    """Return True only when the cited URL explicitly looks like a PDF route."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    path = parsed.path.lower()
    if path.endswith(".pdf") or ".pdf/" in path:
        return True
    query = {k.lower(): [v.lower() for v in values]
             for k, values in parse_qs(parsed.query).items()}
    return any(
        value in ("pdf", "application/pdf")
        for key in ("format", "type", "download", "output")
        for value in query.get(key, [])
    )


def _requires_confirmed_non_web_source_kind(
    assessment: SourceKindAssessment,
) -> bool:
    """Require positive work-type evidence before HTML can represent a source."""
    return bool(
        assessment.confidence == "high"
        and assessment.kind
        not in {"unknown", "webpage", "news_article", "blog_post"}
    )


def _normalize_cited_url(url: str) -> str:
    """Normalize the common scheme-less ``www.`` citation form only."""
    normalized = url.strip()
    if normalized.lower().startswith("www."):
        return f"https://{normalized}"
    return normalized


def _normalize_cache_author(value):
    return ' '.join(str(value or '').casefold().replace('.', '').split())


def _accepted_copy_has_edition_label(data, title):
    """Compare an explicit edition label on an already admitted book, not work equivalence."""
    import fitz
    match = re.search(r'\((\d+)(?:st|nd|rd|th)\s+ed\.?\)\s*$', title or '', re.I)
    if not match:
        return False
    names = dict(zip(('first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth'),range(1,11)))
    labels = set()
    try:
        with fitz.open(stream=data, filetype='pdf') as doc:
            for index in range(min(6, len(doc))):
                page = doc[index]
                for line in page.get_text().splitlines():
                    label = re.fullmatch(r'\s*(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\s+edition\s*',line,re.I)
                    if label:
                        token=label[1].casefold()
                        labels.add(names[token] if token in names else int(re.match(r'\d+',token)[0]))
    except (ValueError, RuntimeError):
        return False
    return labels == {int(match[1])}


def _web_link_differences(comparisons, result, *, title, author, year, doi):
    expected = dict(title=title, author=author, year=year, doi=doi)
    observed = dict(title=result.title, author='; '.join(result.authors or []), year=result.year, doi=result.doi)
    return [dict(field=c.field_name, submitted=str(expected[c.field_name])[:2000],
                 destination=str(observed[c.field_name])[:2000])
            for c in comparisons if c.outcome == 'material_conflict'
            and expected.get(c.field_name) and observed.get(c.field_name)]


def _web_identity_comparison(result, *, title=None, author=None, year=None, doi=None):
    """Shared existing identity standard for readable and metadata-only pages."""
    identity = build_reference_discovery_candidate(
        attempt_id='web-identity', provider='web_fetch',
        expected=ExpectedBibliographicFields(title=title or '', authors=[author] if author else [],
            year=year or '', doi=doi or ''), result=result)
    comparisons = [c for c in identity.comparisons if c.field_name in {'title', 'author', 'year', 'doi'}]
    confirmed = bool(identity.plausible_identity_match and not identity.has_material_conflict
        and all(c.outcome in {'agreement', 'minor_difference'} for c in comparisons)
        and (len(comparisons) >= 2 or identity.authoritative_identifier_match))
    return identity, comparisons, confirmed


def _doi_identity_matches(expected: str | None, observed: str | None) -> bool:
    if not expected or not observed:
        return False
    def clean(value: str) -> str:
        # Percent-encoded punctuation survives a DOI copied from a URL; the
        # decoded form is the same identifier.
        from urllib.parse import unquote

        return unquote(re.sub(
            r"^https?://(?:dx\.)?doi\.org/", "", value.strip().casefold()
        ))

    return clean(expected) == clean(observed)


def _location_rank(location: AcquisitionLocation) -> tuple[int, int, int]:
    """Keep work-aware web order; structured providers retain best/version hints."""
    if location.provider == "web_search" or location.metadata.get("search_provider"):
        # Stable sort preserves the complete in-memory search order, including
        # transient Brave locations, without persisting a result-derived rank.
        return (0, 0, 0)
    version_rank = {
        "publishedVersion": 0,
        "acceptedVersion": 1,
        "submittedVersion": 2,
    }.get(location.version, 3)
    return (
        0 if location.is_best else 1,
        version_rank,
        0 if location.representation_kind is not None else 1,
    )


_ACQUIRED_LOCATION_OUTCOMES = frozenset({"acquired", "acquired_fallback"})
# Keys of a location attempt that carry an address or a search listing's text.
_SEARCH_LOCATION_URL_KEYS = ("url", "independent_source_url", "discovered_urls", "candidate_title")


def _search_derived_location(attempt: dict) -> bool:
    """A location that a web-search provider returned, or one reached from it."""
    return bool(attempt.get("discovery_provider")) or attempt.get("provider") == "web_search"


def _withhold_search_result_urls(location_attempts: list) -> list[dict]:
    """Drop search-result addresses before location attempts are stored.

    Owner decision 2026-09-29: the deployed application keeps no search-result
    links. The one exception is the location that was acquired and admitted:
    its address is the stored source's own provenance, not a search listing.
    Outcomes, reason codes and hashes stay, so the disposition of every lead
    remains inspectable. The development candidate audit
    (SEARCH_CANDIDATE_AUDIT_URLS) is recorded separately on the discovery trace.
    """
    kept: list[dict] = []
    for attempt in location_attempts or []:
        if not isinstance(attempt, dict):
            continue
        if not _search_derived_location(attempt) or attempt.get("outcome") in _ACQUIRED_LOCATION_OUTCOMES:
            kept.append(attempt)
            continue
        withheld = {key: value for key, value in attempt.items() if key not in _SEARCH_LOCATION_URL_KEYS}
        for nested in ("landing_metadata_identity", "landing_metadata_observation"):
            if isinstance(withheld.get(nested), dict):
                withheld[nested] = {k: v for k, v in withheld[nested].items() if k != "source_url"}
        withheld["url_withheld"] = "search_result"
        kept.append(withheld)
    return kept


# Marks page text kept because the page's own title, author and year confirm
# the cited work; such a page is the work republished, whatever its site type.
REPUBLISHED_WORK_EVIDENCE = "page bibliography confirms the cited work"
STATED_COMPLETE_EVIDENCE = "whole stated work on the page"


def _registration_bound_doi(record, supplied_doi: str | None) -> str | None:
    """The stored work's DOI, only when a Crossref record confirmed it at
    admission or the reference itself supplied it (2026-09-30): a reused source
    must not pass on a DOI that no registration record bound."""
    doi = getattr(getattr(record, "canonical_work", None), "doi", None)
    if not doi:
        return None
    key = lambda value: re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(value or "").strip(), flags=re.I).casefold()
    if supplied_doi and key(supplied_doi) == key(doi):
        return doi
    evidence = ((getattr(record, "validation_evidence", None) or {}).get("canonical_work") or {}).get("identity_evidence") or []
    if any(isinstance(item, dict) and item.get("provider") == "crossref" and key(item.get("doi")) == key(doi)
           for item in evidence):
        return doi
    return None


def _stated_page_marker(observed_kind, text: str, expected_kind) -> dict:
    """Record a page accepted as complete under rule A, for review (2026-09-30)."""
    if STATED_COMPLETE_EVIDENCE not in getattr(observed_kind, "evidence", ()):
        return {}
    return {"stated_page_completeness": {"rule": "stated-page-coverage-v1",
                                         "words": len(re.findall(r"\w+", text or "")),
                                         "cited_kind": getattr(expected_kind, "kind", None)}}


def _page_states_cited_work(observation: dict | None, title, author, year) -> bool:
    """The page's own title, author surname and year all name the cited work.

    Journal, volume and page details a page does not state leave full identity
    unresolved, but they cannot make a page stating this title, author and
    year a different work (Hess, 2026-09-30).
    """
    from app.services.reference_review_scope import text_key
    from app.services.relevance import extract_surnames
    observed = (observation or {}).get("observed") or {}
    if not (title and author and year and observed.get("title") and observed.get("year")):
        return False
    cited = {s.casefold() for s in extract_surnames(str(author))}
    stated = {s.casefold() for a in observed.get("authors") or [] for s in extract_surnames(str(a))}
    return (text_key(observed["title"]) == text_key(title) and bool(cited & stated)
            and str(observed["year"])[:4] == str(year)[:4])


def _url_like(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return text.startswith(("http://", "https://", "www."))


def _blocked_status(exc: BaseException) -> int | None:
    """The HTTP status behind a failed fetch, including one wrapped by _safe_download."""
    for error in (exc, getattr(exc, "__cause__", None)):
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
    match = re.search(r"status=(\d{3})\b", str(exc))
    return int(match.group(1)) if match else None


def _mdpi_file_server_locations(location, exc: BaseException, doi: str | None) -> list:
    """File-server locations to try after an MDPI location was blocked."""
    if location.metadata.get("constructed_route") == MDPI_FILE_SERVER_ROUTE:
        return []
    if _blocked_status(exc) not in MDPI_BLOCKED_STATUSES or not is_mdpi_location(location.url):
        return []
    if not is_mdpi_doi(doi):
        parsed = urlparse(location.url)
        doi = parsed.path.lstrip("/") if (parsed.hostname or "").casefold() in {"doi.org", "dx.doi.org"} else None
    inherited = {key: location.metadata[key] for key in (
        "search_provider", "search_engine_group", "search_title") if location.metadata.get(key)}
    return [
        AcquisitionLocation(
            url=url,
            provider=location.provider,
            media_type="application/pdf",
            representation_kind=RepresentationKind.PDF,
            metadata={
                # Reached from a search result, it keeps that provenance so its
                # candidate stays bound to the search that led to it.
                **inherited,
                "constructed_route": MDPI_FILE_SERVER_ROUTE,
                "discovered_from_location_sha256": hashlib.sha256(location.url.encode("utf-8")).hexdigest(),
            },
        )
        for url in mdpi_file_server_urls(doi)
    ]


def _location_exception_disposition(exc: Exception) -> tuple[str, str]:
    """Classify transport/access failures without retaining sensitive URLs."""
    if isinstance(exc, CandidateBudgetExceeded):
        return "not_attempted", "candidate_budget_exhausted"
    status = None
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
    message = str(exc).casefold()
    if status in {401, 403, 407, 451} or any(
        marker in message
        for marker in ("access denied", "access restricted", "paywall", "forbidden")
    ):
        return "access_restricted", "access_restricted"
    if isinstance(exc, (httpx.TransportError, TimeoutError, OSError)):
        return "transport_failure", "transport_or_download_failure"
    if "landing page bibliography unavailable" in message:
        return "identity_unconfirmed", "landing_page_bibliography_unavailable"
    if "landing page title does not match" in message:
        return "identity_rejected", "landing_page_identity_rejected"
    if "representation too short" in message:
        return "completeness_rejected", "representation_too_short"
    if "landing page yielded" in message:
        return "unavailable", "landing_page_no_full_text_location"
    return "unavailable", "location_unavailable"


def _extract_text_representation(payload: bytes, kind: RepresentationKind) -> str:
    """Normalize XML or plain bytes without claiming that metadata is complete.

    Refuses a payload that is not a document's prose. Until 2026-09-23 the only
    check on retrieved text was a 500-character minimum, so a cookie-consent
    JavaScript bundle and a WordPress RSS feed were both stored as full-text
    representations of works they have nothing to do with.
    """
    from app.services.source_validator import detect_nonprose_payload
    text = payload.decode("utf-8", errors="replace")
    if kind is RepresentationKind.XML:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(text, "xml").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    nonprose = detect_nonprose_payload(text)
    if nonprose:
        raise ValueError(f"retrieved representation is {nonprose}")
    return text


# Title words that carry no distinguishing power. A cited title reduced to
# these plus one or two content words cannot identify a work by overlap alone.
_TITLE_FUNCTION_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "its", "was", "are",
    "into", "onto", "over", "under", "between", "among", "about", "against",
    "through", "during", "before", "after", "above", "below", "than", "then",
    "their", "there", "these", "those", "have", "has", "had", "not", "but",
    "how", "why", "what", "when", "where", "who", "whom", "which", "some",
    "such", "only", "also", "more", "most", "other", "another", "new",
    "case", "study", "studies", "analysis", "review", "introduction",
    "chapter", "part", "volume", "edition", "essay", "essays", "paper",
    "notes", "report", "research", "towards", "toward",
})
# At or below this many distinctive words, a title is too generic for token
# overlap to identify a work without author agreement.
_SHORT_TITLE_TOKENS = 3


def _verify_text_content_identity(
    content: bytes,
    expected_doi: str | None,
    expected_title: str | None,
    expected_author: str | None,
    expected_year: str | None,
) -> tuple[str, str]:
    """Score identity evidence inside normalized non-PDF content itself."""
    text = content.decode("utf-8", errors="replace").lower()
    compact_text = re.sub(r"\s+", " ", text)
    purpose_conflict = _detect_nonwork_listing(compact_text, expected_title)
    if purpose_conflict:
        return "rejected", f"representation is {purpose_conflict}, not the cited work"
    if expected_doi and expected_doi.lower() in compact_text:
        return "high", "exact DOI appears in acquired text"

    title_overlap = 0.0
    distinctive: set[str] = set()
    if expected_title:
        title_tokens = {
            token for token in re.findall(r"[a-z0-9]+", expected_title.lower())
            if len(token) >= 3
        }
        # Function words appear in every document, so they inflate overlap
        # without distinguishing one work from another.
        distinctive = (title_tokens - _TITLE_FUNCTION_WORDS) or title_tokens
        content_tokens = set(re.findall(r"[a-z0-9]+", compact_text))
        if distinctive:
            title_overlap = len(distinctive & content_tokens) / len(distinctive)

    author_support = False
    if expected_author:
        surname = expected_author.split(",", 1)[0].strip().lower()
        author_support = len(surname) >= 3 and surname in compact_text
    year_support = bool(expected_year and expected_year in compact_text)
    support = int(author_support) + int(year_support)

    # A short cited title can sit inside an unrelated longer title, so full
    # token overlap is weak evidence on its own and a shared publication year
    # is not distinguishing. Require the author for the strongest verdict.
    short_title = len(distinctive) <= _SHORT_TITLE_TOKENS
    if title_overlap >= 0.8 and support and not (short_title and not author_support):
        return "high", f"title token overlap={title_overlap:.2f} with author/year support"
    if title_overlap >= 0.8 and short_title and not author_support:
        return "medium", (
            f"title token overlap={title_overlap:.2f} on only {len(distinctive)} "
            "distinctive word(s) without author agreement"
        )
    if title_overlap >= 0.6:
        return "medium", f"title token overlap={title_overlap:.2f} in acquired text"
    return "low", f"insufficient content-level identity evidence (title overlap={title_overlap:.2f})"


# Why no title could be read off the front page, in words a reader can act on.
# "We could not read the cover" and "the cover carries no title" lead to
# different next steps, and the report used to say the same thing for both.
_COVER_TITLE_EXPLANATIONS = {
    'no_text_layer': 'The opening page is an image with no readable text, so the '
                     'title could not be checked against this copy.',
    'page_unreadable': 'The opening page could not be read, so the title could not '
                       'be checked against this copy.',
    'illegible_title': 'The opening page is a scan whose title text is not legible, '
                       'so the title could not be checked against this copy.',
    'front_matter_label_only': 'The opening page shows a section heading rather than '
                               'the work\'s title, so the title could not be checked '
                               'against this copy.',
    'uniform_typography': 'This copy sets every line in the same type, so no title '
                          'could be identified on the opening page.',
}


def _distinctive_title_tokens(title: str | None) -> set[str]:
    """Words in a cited title that can distinguish one work from another."""
    tokens = {
        token for token in re.findall(r"[a-z0-9]+", str(title or "").lower())
        if len(token) >= 3
    }
    return (tokens - _TITLE_FUNCTION_WORDS) or tokens


# How far into a document the work's own title should still be visible. A title
# page, an article opener and a publisher preview all name the work well inside
# this window; a body page that merely shares the subject does not. The window
# spans three pages because a thesis or report often puts a scanned approval or
# cover sheet ahead of its typed title page.
_OPENING_TITLE_PAGES = 3
_OPENING_TITLE_WINDOW_WORDS = 300
_OPENING_TITLE_MIN_OVERLAP = 0.6


def _opening_pages_present_title(content: bytes, expected_title: str | None) -> bool:
    """Do the opening pages of this PDF name the work the reference cites?

    The visible-title comparison in the provisional retainer only runs when a
    title can be parsed off the first page by font size. That parse fails on any
    uniformly typeset document — a working paper, a court opinion — and a
    skipped comparison used to mean the candidate was kept. A different work by
    the same author and an unrelated legal opinion were both retained that way
    and then quoted in the report as the cited source.

    Provisional retention is a concession, so it should require affirmative
    evidence rather than merely the absence of contrary evidence. Read the front
    of the document and require the cited title's distinguishing words to be
    there. Front matter that carries too little text to judge does not qualify.
    """
    distinctive = _distinctive_title_tokens(expected_title)
    if not distinctive:
        return False
    try:
        import fitz

        document = fitz.open(stream=content, filetype="pdf")
        try:
            opening = " ".join(
                document[index].get_text()
                for index in range(min(_OPENING_TITLE_PAGES, len(document)))
            )
        finally:
            document.close()
    except Exception:
        return False
    words = re.findall(r"[a-z0-9]+", opening.lower())
    if len(set(words)) < 40:
        return False
    window = set(words[:_OPENING_TITLE_WINDOW_WORDS])
    return len(distinctive & window) / len(distinctive) >= _OPENING_TITLE_MIN_OVERLAP


def _legible_cover_title(value: str | None) -> str | None:
    """Treat an unreadable scanned cover title as absent, not as a conflict.

    The visible-title guard exists so that a copy showing a different title
    cannot be rescued by incidental body-text mentions. OCR of a scanned or
    stylised cover can instead return characters that are not words at all,
    which is missing evidence rather than a differing title. The test is
    legibility alone: text that is mostly non-alphabetic is not a readable
    title. A genuinely different title still blocks the candidate even when it
    is a single word, because it is legible.
    """
    text = (value or "").strip()
    if not text:
        return None
    letters = sum(character.isalpha() for character in text)
    if letters < 0.6 * len(text):
        return None
    return text


def _combine_identity_confidence(provider: str, content: str) -> str:
    """Require both record identity and representation identity for high confidence."""
    if "rejected" in {provider, content}:
        return "rejected"
    rank = {"low": 0, "medium": 1, "high": 2}
    provider_rank = rank.get(provider, 0)
    content_rank = rank.get(content, 0)
    if provider_rank == 2 and content_rank == 2:
        return "high"
    if provider_rank >= 1 and content_rank >= 1:
        return "medium"
    if max(provider_rank, content_rank) == 2:
        return "medium"
    return "low"


def _extract_scholarly_html(page_html: str, cited_title: str | None) -> str | None:
    """Extract article text from an explicitly advertised full-text HTML page."""
    if cited_title:
        page_titles = _extract_html_titles(page_html)
        if page_titles and not _html_title_matches(cited_title, page_titles):
            return None
    import trafilatura

    text = trafilatura.extract(
        page_html,
        include_links=False,
        include_tables=True,
        favor_recall=True,
    )
    return text if text and len(text) >= 500 else None


class SourceResolutionError(Exception):
    """Raised when a source cannot be found, with bounded attempt provenance."""

    def __init__(self, message: str, *, retrieval_trace: list[dict] | None = None):
        super().__init__(message)
        self.retrieval_trace = retrieval_trace or []
        self.reference_discovery_trace: dict | None = None
        self.reference_discovery: dict | None = None


# ── HTML source-identity helpers (REVIEW §2b #18) ────────────────────────
# Web pages are fetched when a student URL points to HTML (not a PDF). The
# page was previously trusted unconditionally — whatever trafilatura extracted
# became the "source text" for verification, with no check that the page IS
# the cited source. These helpers compare the page's own titles (<title>,
# og:title, citation_title) against the cited reference title, the same
# triangulation intent as the PDF identity check (but easier — HTML has
# structured metadata).

_HTML_TITLE_PATTERNS = (
    # Academic pages expose the publication title verbatim.
    re.compile(r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    # Open Graph title (news, blogs).
    re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    # Fallback: the document <title>.
    re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE),
)


def _extract_html_titles(page_html: str) -> list[str]:
    """Pull candidate titles out of an HTML page (citation_title, og:title, <title>)."""
    titles: list[str] = []
    for pattern in _HTML_TITLE_PATTERNS:
        m = pattern.search(page_html)
        if m:
            t = html.unescape(m.group(1)).strip()
            if t:
                titles.append(t)
    return titles


def _html_title_matches(cited_title: str, page_titles: list[str]) -> bool:
    """True if any page title shares ≥50% of the cited title's significant tokens.

    Tokens < 3 chars are ignored (stopwords like "the", "of"). A page that is
    actually the cited source almost always carries its headline in <title> or
    og:title; a login page, error page, or unrelated article will not.
    """
    cited = {tok for tok in re.split(r"[^A-Za-z0-9]+", cited_title.lower()) if len(tok) >= 3}
    if not cited:
        return True  # nothing to compare against — don't reject
    for pt in page_titles:
        pt_tokens = {tok for tok in re.split(r"[^A-Za-z0-9]+", pt.lower()) if len(tok) >= 3}
        if not pt_tokens:
            continue
        if len(cited & pt_tokens) / len(cited) >= 0.5:
            return True
    return False



_TITLE_NOISE_TOKENS = frozenset({
    "the", "and", "for", "with", "from", "into", "its", "their", "this",
    "that", "was", "were", "are", "new",
})


def _distinctive_title_tokens(title: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9]+", title.lower())
            if len(token) >= 3 and token not in _TITLE_NOISE_TOKENS}


def _landing_title_confirms(cited_title: str, page_titles: list[str]) -> bool:
    """Grant identity from a landing page only on an aligned title.

    `_html_title_matches` answers a different question: is this page obviously
    not the cited source? Half the cited tokens is a fair bar for rejecting a
    login or error page. Granting identity is the opposite question, and the
    same bar fails on short titles — "The Studio System" shares every token it
    has with "Working Below the Line in the Studio System", a different
    author's thesis. A cited title carrying fewer than four distinctive words
    must therefore align with the page title, allowing only a subtitle or a
    site suffix, rather than merely appear somewhere inside it.
    """
    from app.services.pdf_verifier import _aligned_title_containment, _normalize
    if not _html_title_matches(cited_title, page_titles):
        return False
    if len(_distinctive_title_tokens(cited_title)) >= 4:
        return True
    cited = _normalize(cited_title)
    return any(_aligned_title_containment(cited, _normalize(page_title))
               for page_title in page_titles)


def _trusted_graph_doi(graph: CanonicalWorkGraph) -> str | None:
    """Return a provider DOI only when identity evidence safely supports it."""
    observed: dict[str, list[dict]] = {}
    for evidence in graph.identity_evidence:
        value = str(evidence.get("doi") or "").strip()
        normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I).casefold()
        if normalized:
            observed.setdefault(normalized, []).append(evidence)
    for doi, evidence_items in observed.items():
        if any(item.get("confidence") == "high" for item in evidence_items):
            return doi
        if len({item.get("provider") for item in evidence_items}) >= 2:
            return doi
    return None


STUDENT_LINK_RETRY_DELAY_SECONDS = 2.0
_NETWORK_FAILURES = ("connect_error", "connect_timeout", "read_timeout", "read_error", "write_error",
                     "pool_timeout", "remote_protocol_error", "http_502", "http_503", "http_504",
                     "ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError", "WriteError",
                     "PoolTimeout", "RemoteProtocolError")


def _network_failure(error: str | None) -> str | None:
    """The network-level failure code in a route error, or None."""
    return next((code for code in _NETWORK_FAILURES if code in (error or "")), None)


class SourceResolver:
    """Resolves a reference to a source document."""

    def __init__(self) -> None:
        # None enables every configured acquisition capability. Evaluation
        # harnesses may set a bounded capability set to measure adapters, URLs,
        # publisher construction, and institutional routes independently.
        self._acquisition_capabilities: set[str] | None = None
        # Storage backend is optional — if S3/MinIO is unavailable, the
        # resolver still works (just without caching/persistence). This makes
        # the resolver robust to S3 outages and simplifies testing without Docker.
        try:
            self._backend = get_storage_backend()
        except Exception as e:
            logger.warning(
                "Storage backend unavailable — caching disabled (type=%s)",
                type(e).__name__,
            )
            self._backend = None
        self._repository_session_factory = (
            SessionLocal if settings.SOURCE_REPOSITORY_ENABLED else None
        )
        self._retrieval_sources = get_retrieval_sources()
        self._lookup_cache = RetrievalLookupCache()

    # ── Public API ────────────────────────────────────────

    @staticmethod
    def _require_discovery_route(category: str) -> None:
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is not None:
            trace["required"].add(category)

    @staticmethod
    def _record_discovery_attempt(
        *,
        category: str,
        provider: str,
        result: RetrievalResult,
        required: bool,
    ) -> None:
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is None:
            return
        applicable = (result.metadata or {}).get("lookup_applicable", True)
        if not applicable:
            required = False
        if required:
            trace["required"].add(category)
        expected: ExpectedBibliographicFields = trace["expected"]
        if expected.doi:
            query_text = f"doi:{expected.doi}"
        elif expected.isbn:
            query_text = f"isbn:{expected.isbn}"
        else:
            query_text = " ".join(
                value
                for value in (
                    f"title:{expected.title}" if expected.title else "",
                    f"author:{expected.authors[0]}" if expected.authors else "",
                    f"year:{expected.year}" if expected.year else "",
                )
                if value
            )
        ordinal = len(trace["attempts"]) + 1
        query_inputs = []
        for search_attempt in (result.metadata or {}).get("structured_search_attempts", []):
            query_inputs.append((
                search_attempt["query"], provider, search_attempt["outcome"], search_attempt.get("result_count"),
                {key: search_attempt[key] for key in ("reason_code", "bounded_review_screen") if key in search_attempt},
            ))
        if category == "bounded_web":
            seen_query_inputs: set[tuple[str, str | None, str, int | None]] = set()
            for phase in (result.metadata or {}).get("retrieval_trace", []):
                for search_attempt in phase.get("search_attempts", []):
                    raw_query = str(search_attempt.get("query") or "").strip()
                    if not raw_query:
                        continue
                    item = (
                        raw_query,
                        str(search_attempt.get("provider") or "unknown").lower(),
                        str(search_attempt.get("outcome") or "unknown"),
                        (
                            int(search_attempt["result_count"])
                            if isinstance(search_attempt.get("result_count"), int)
                            else None
                        ),
                    )
                    if trace.get("search_policy_version") == "api-first-search-v2" or item not in seen_query_inputs:
                        seen_query_inputs.add(item)
                        query_inputs.append((*item, search_attempt))
        if not query_inputs:
            query_inputs = [(query_text, None, "unknown", None, {})]
        queries: list[ReferenceSearchQuery] = []
        for query_index, (
            raw_query,
            execution_provider,
            execution_outcome,
            result_count,
            telemetry,
        ) in enumerate(
            query_inputs, start=1
        ):
            normalized_query = re.sub(
                r"\s+",
                " ",
                unicodedata.normalize("NFKC", raw_query).casefold(),
            ).strip() or "insufficient-metadata"
            query_seed = (
                f"{trace['reference_id']}:{category}:{provider}:{ordinal}:"
                f"{query_index}:{execution_provider or ''}:{execution_outcome}:"
                f"{result_count if result_count is not None else ''}:{normalized_query}"
            )
            query_digest = hashlib.sha256(query_seed.encode("utf-8")).hexdigest()
            queries.append(
                ReferenceSearchQuery(
                    **({'bounded_review_screen': telemetry['bounded_review_screen']}
                       if 'bounded_review_screen' in telemetry else {}),
                    **{key: telemetry[key] for key in (
                        "required", "provider_calls", "latency_seconds", "cost_usd",
                        "credits_remaining", "cache_hit", "reason_code",
                    ) if key in telemetry},
                    execution_engine_group=telemetry.get("engines"),
                    query_id=f"query-{query_digest[:24]}",
                    route_category=category,
                    provider=provider,
                    execution_provider=execution_provider,
                    execution_outcome=(
                        execution_outcome
                        if execution_outcome
                        in {
                            "results",
                            "no_results",
                            "timeout",
                            "captcha",
                            "operational_failure",
                            "access_restricted",
                            "rate_limited",
                            "response_invalid",
                            "budget_skipped",
                            "cooldown_skipped",
                            "declined",
                            "recovery_probe_in_progress",
                        }
                        else "unknown"
                    ),
                    result_count=result_count,
                    normalized_query=normalized_query,
                    query_sha256=hashlib.sha256(
                        normalized_query.encode("utf-8")
                    ).hexdigest(),
                )
            )
        seed = ":".join(
            (
                trace["reference_id"],
                category,
                provider,
                str(ordinal),
                *[query.query_id for query in queries],
            )
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        error = (result.error or "").casefold()
        transient_audits = [audit for phase in (result.metadata or {}).get("retrieval_trace", [])
                            for audit in phase.get("transient_search_audits", [])]
        bounded_candidate_returned = bool(
            category == "bounded_web"
            and any(
                phase.get("candidate_locations") or phase.get("location_attempts")
                for phase in (result.metadata or {}).get("retrieval_trace", [])
            )
        )
        observed_web_candidate = bool(
            category == "student_url" and (result.metadata or {}).get("web_identity")
        )
        if not applicable:
            outcome = "unavailable"
            reason_code = "route_not_applicable"
        elif category == "bounded_web" and transient_audits and not bounded_candidate_returned:
            outcome = "candidates_processed"
            reason_code = "transient_candidate_accounting"
        elif result.success or bounded_candidate_returned or observed_web_candidate or (result.metadata or {}).get("candidate_budget_skipped"):
            outcome = "candidate_found"
            reason_code = "candidate_returned"
        elif any(value in error for value in ("401", "403", "451", "paywall", "access restricted")):
            outcome = "access_restricted"
            reason_code = "route_access_restricted"
        elif "reference_elapsed_budget" in error:
            # The reference's own time budget ran out before this provider was
            # contacted. Recording it as a provider failure blamed adapters for
            # calls that never left the process and made every provider
            # reliability figure drawn from the trace wrong. The outcome stays
            # in the incomplete family, so nothing here can read as a completed
            # search that found nothing.
            outcome = "unavailable"
            reason_code = "route_elapsed_budget_exhausted"
        elif any(value in error for value in ("not configured", "no api_key", "unavailable")):
            outcome = "unavailable"
            reason_code = "route_unavailable"
        elif any(
            value in error
            for value in ("failed", "timeout", "rate limit", "circuit", "network")
        ):
            outcome = "operational_failure"
            reason_code = "route_operational_failure"
        elif _search_execution_outcome(result) == "no_results":
            outcome = "no_match"
            reason_code = "no_candidate_returned"
        else:
            # Redacted exception codes (e.g. http_500, connect_error) and
            # unknown failures must never fall through to completed no-match.
            outcome = "operational_failure"
            reason_code = "route_unclassified_failure"
        if category != "bounded_web" and not (result.metadata or {}).get("structured_search_attempts"):
            # The reference's own elapsed budget ran out before this provider
            # was called. The attempt already says so; the query must too, or
            # it records `operational_failure` and names the provider as the
            # cause -- which is exactly what the provider-recovery trigger reads.
            budget_exhausted = (
                (result.metadata or {}).get("candidate_budget_skipped")
                or reason_code == "route_elapsed_budget_exhausted"
            )
            # A route the application declined was never called; recording it
            # as `operational_failure` made a skipped CORE lookup read as a
            # CORE fault in every audit (289 such records by 2026-09-24).
            execution = "budget_skipped" if budget_exhausted else "declined" \
                if reason_code == "route_not_applicable" else {
                "candidate_found": "results", "no_match": "no_results",
                "access_restricted": "access_restricted",
            }.get(outcome, "operational_failure")
            queries = [query.model_copy(update={
                "execution_provider": provider, "execution_outcome": execution,
            }) for query in queries]
        now = datetime.now(timezone.utc)
        timing = (result.metadata or {}).get('operation_timing') or {}
        candidate_audit = None
        if category == "bounded_web":
            from app.services.search.candidate_audit import build_candidate_audit
            audit_query_ids: dict[tuple[str, str], str] = {}
            for (raw_query, execution_provider, *_rest), query in zip(query_inputs, queries):
                audit_query_ids.setdefault(
                    (str(raw_query).strip(), str(execution_provider or "").lower()), query.query_id)
            # None unless SEARCH_CANDIDATE_AUDIT_URLS is on; never Brave.
            candidate_audit = build_candidate_audit(
                (result.metadata or {}).get("retrieval_trace", []), audit_query_ids)
        attempt = ReferenceRouteAttempt(
            **({"candidate_audit": candidate_audit} if candidate_audit is not None else {}),
            transient_search_audits=transient_audits,
            attempt_id=f"attempt-{digest[:24]}",
            route_category=category,
            provider=provider,
            required=required,
            permitted=True,
            query_ids=[query.query_id for query in queries],
            outcome=outcome,
            reason_code=reason_code,
            error_code=_route_error_code(result) if outcome in {
                "operational_failure", "timeout", "rate_limited", "unavailable",
                "access_restricted",
            } else None,
            started_at=datetime.fromisoformat(timing['started_at']) if timing else now,
            completed_at=datetime.fromisoformat(timing['completed_at']) if timing else now,
            timing_version=timing.get('version'),
            elapsed_seconds=timing.get('elapsed_seconds'),
        )
        trace["queries"].extend(queries)
        trace["attempts"].append(attempt)
        location_records: dict[str, dict] = {}
        for phase in (result.metadata or {}).get("retrieval_trace", []):
            for candidate_location in phase.get("candidate_locations", []):
                location_url = str(candidate_location.get("url") or "")
                if location_url:
                    location_records.setdefault(location_url, dict(candidate_location))
            for location_attempt in phase.get("location_attempts", []):
                location_url = str(location_attempt.get("url") or "")
                if location_url:
                    location_records.setdefault(location_url, {}).update(
                        location_attempt
                    )
        location_attempts = list(location_records.values())
        if category == "bounded_web" and location_attempts:
            seen_location_hashes: set[str] = set()
            for location_attempt in location_attempts:
                location_url = str(location_attempt.get("url") or "")
                if not location_url:
                    continue
                location_hash = hashlib.sha256(location_url.encode("utf-8")).hexdigest()
                if location_hash in seen_location_hashes:
                    continue
                seen_location_hashes.add(location_hash)
                raw_outcome = str(location_attempt.get("outcome") or "unknown")
                acquisition_outcome: CandidateAcquisitionOutcome = (
                    raw_outcome
                    if raw_outcome
                    in {
                        "metadata_only",
                        "acquired",
                        "acquired_fallback",
                        "identity_rejected",
                        "identity_unconfirmed",
                        "completeness_rejected",
                        "type_rejected",
                        "type_unconfirmed",
                        "transport_failure",
                        "access_restricted",
                        "unavailable",
                        "not_attempted",
                    }
                    else "unknown"
                )
                landing = ((location_attempt.get("landing_metadata_identity") if raw_outcome == "unavailable" else None)
                           or location_attempt.get("landing_metadata_observation"))
                # The page's own title and byline, observed only: it supplies
                # the observed fields but never identity or provenance.
                page_only = None if landing else location_attempt.get("landing_page_observation")
                observed_fields = (landing or page_only or {}).get("observed") or {}
                observed = RetrievalResult(
                    source_name=provider,
                    success=True,
                    title=(observed_fields.get("title") if (landing or page_only) else
                           # A result "title" that is an address is a search
                           # link, which the application does not keep.
                           (None if _url_like(location_attempt.get("candidate_title"))
                            else str(location_attempt.get("candidate_title") or "") or None)),
                    authors=observed_fields.get("authors", []),
                    year=observed_fields.get("year"),
                    doi=observed_fields.get("doi"),
                    metadata={
                        **(observed_fields.get("metadata") or {}),
                        "observed_source_kind": location_attempt.get(
                            "observed_source_kind", "unknown"
                        ),
                        "source_kind_verdict": location_attempt.get(
                            "source_kind_verdict"
                        ),
                        # What the fetched page showed, or that nothing beyond a
                        # search snippet was ever observed for this location.
                        "title_reason": (
                            (observed_fields.get("metadata") or {}).get("title_reason")
                            or location_attempt.get("title_reason")
                            or ("search_snippet_only" if not (landing or page_only) else None)
                        ),
                    },
                )
                trace["candidates"].append(
                    build_reference_discovery_candidate(
                        attempt_id=attempt.attempt_id,
                        provider=provider,
                        expected=expected,
                        result=observed,
                        candidate_key=location_hash,
                        location_url=(landing.get("source_url") or location_url) if landing else location_url,
                        acquisition_outcome=acquisition_outcome,
                        validation_reason=str(location_attempt.get("reason") or "")
                        or None,
                        access_restricted=(
                            raw_outcome == "access_restricted"
                            or (
                                raw_outcome == "unavailable"
                                and any(
                                    marker
                                    in str(location_attempt.get("reason") or "").casefold()
                                    for marker in ("403", "paywall", "access restricted")
                                )
                            )
                        ),
                        discovery_provider=str(
                            location_attempt.get("discovery_provider") or "unknown"
                        ).lower(),
                        location_rank=(
                            int(location_attempt["rank"])
                            if isinstance(location_attempt.get("rank"), int)
                            else None
                        ),
                        origin_providers=[
                            str(value)
                            for value in (
                                location_attempt.get("origin_providers") or [provider]
                            )
                        ],
                        disposition_reason_code=(
                            str(location_attempt.get("reason_code") or "") or None
                        ),
                    ).model_copy(update={
                        "validated_identity_content_sha256": landing["content_sha256"] if landing else location_attempt.get("validated_identity_content_sha256"),
                        "location_provenance": "independently_acquired_content" if landing else location_attempt.get("provenance_basis", "discovery_result"),
                        "identity_evidence_kind": landing.get("identity_evidence_kind") or ("landing_page_observation" if location_attempt.get("landing_metadata_observation") else "landing_page_metadata") if landing else "source_representation" if location_attempt.get("validated_identity_content_sha256") else None,
                    })
                )
        elif "journal_metadata_candidates" in (result.metadata or {}):
            for item in result.metadata["journal_metadata_candidates"]:
                candidate_result = reserve_metadata_candidate(provider, RetrievalResult(**item))
                _append_unique_candidate(trace, build_reference_discovery_candidate(
                    attempt_id=attempt.attempt_id, provider=provider,
                    expected=expected, result=candidate_result,
                    candidate_key=candidate_result.doi or candidate_result.title,
                    acquisition_outcome="metadata_only",
                    disposition_reason_code="journal_registration_only",
                ))
        elif "book_metadata_candidates" in (result.metadata or {}):
            for item in result.metadata["book_metadata_candidates"]:
                candidate_result = RetrievalResult(**item)
                candidate_result = reserve_metadata_candidate(provider, candidate_result)
                _append_unique_candidate(trace, build_reference_discovery_candidate(
                    attempt_id=attempt.attempt_id, provider=provider,
                    expected=expected, result=candidate_result,
                    candidate_key=candidate_result.metadata["book_edition_metadata"]["volume_id"],
                    acquisition_outcome=("identity_rejected" if candidate_result.metadata.get("identity_confidence") == "rejected" else "metadata_only"),
                    disposition_reason_code=candidate_result.metadata.get("identity_limitation") or "edition_metadata_only",
                ))
        elif (result.success or observed_web_candidate or (result.metadata or {}).get("candidate_budget_skipped")) and category != "bounded_web":
            if result.locations:
                for rank, location in enumerate(
                    sorted(result.locations, key=_location_rank), start=1
                ):
                    location_hash = hashlib.sha256(
                        location.url.encode("utf-8")
                    ).hexdigest()
                    # Deduplicate repeated locations within this attempt,
                    # not across title and DOI lookup attempts. Each successful
                    # attempt needs its own observed candidate binding; physical
                    # acquisition is deduplicated separately by canonical URL.
                    existing = next(
                        (
                            candidate
                            for candidate in trace["candidates"]
                            if candidate.provider == provider
                            and candidate.location_sha256 == location_hash
                            and candidate.attempt_id == attempt.attempt_id
                        ),
                        None,
                    )
                    if existing is not None:
                        if existing.location_rank is None or rank < existing.location_rank:
                            existing.location_rank = rank
                        continue
                    trace["candidates"].append(
                        build_reference_discovery_candidate(
                            attempt_id=attempt.attempt_id,
                            provider=provider,
                            expected=expected,
                            result=result,
                            candidate_key=location_hash,
                            location_url=location.url,
                            discovery_provider=location.provider,
                            location_rank=rank,
                            origin_providers=[
                                str(value)
                                for value in (
                                    location.metadata.get("providers")
                                    or [location.provider]
                                )
                            ],
                            acquisition_outcome=(
                                "acquired" if (result.metadata or {}).get("accepted_representation_sha256")
                                else "identity_rejected" if (result.metadata or {}).get("identity_confidence") == "rejected"
                                else "identity_unconfirmed" if observed_web_candidate
                                else "metadata_only"
                            ),
                        )
                    )
            else:
                trace["candidates"].append(
                    build_reference_discovery_candidate(
                        attempt_id=attempt.attempt_id,
                        provider=provider,
                        expected=expected,
                        result=result,
                    )
                )

    @staticmethod
    def _record_canonical_location_outcomes(result: RetrievalResult) -> None:
        """Attach shared acquisition outcomes to their structured-provider URLs."""
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is None:
            return
        attempts = list((result.metadata or {}).get("location_attempts") or [])
        by_hash: dict[str, list] = {}
        for candidate in trace["candidates"]:
            if candidate.location_sha256:
                by_hash.setdefault(candidate.location_sha256, []).append(candidate)
        for item in attempts:
            url = str(item.get("url") or "")
            if not url:
                continue
            candidates = by_hash.get(hashlib.sha256(url.encode("utf-8")).hexdigest())
            if not candidates:
                continue
            raw_outcome = str(item.get("outcome") or "unknown")
            if raw_outcome in {
                "metadata_only", "acquired", "acquired_fallback",
                "identity_rejected", "identity_unconfirmed",
                "completeness_rejected", "type_rejected", "type_unconfirmed",
                "transport_failure", "access_restricted", "unavailable",
                "not_attempted",
            }:
                for candidate in candidates:
                    candidate.acquisition_outcome = raw_outcome
                    if isinstance(item.get("rank"), int):
                        candidate.location_rank = int(item["rank"])
                    if item.get("origin_providers"):
                        candidate.origin_providers = list(
                            dict.fromkeys(
                                str(value) for value in item["origin_providers"]
                            )
                        )
                    candidate.disposition_reason_code = (
                        str(item.get("reason_code") or "") or None
                    )
            reason = str(item.get("reason") or "")
            if reason:
                reason_hash = hashlib.sha256(reason.encode("utf-8")).hexdigest()
                for candidate in candidates:
                    candidate.validation_reason_sha256 = reason_hash

    @staticmethod
    def _discovery_artifacts() -> tuple[dict | None, dict | None]:
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is None:
            return None, None
        if expired():
            limitation = f"{ELAPSED_POLICY}: reference elapsed allowance exhausted; unfinished operations are not negative search evidence."
            if limitation not in trace.setdefault("limitations", []):
                trace["limitations"].append(limitation)
        budget = ACTIVE_CANDIDATE_BUDGET.get()
        if budget is not None:
            counts = budget.snapshot()
            limitation = (f"Operation candidate allowance: {counts['attempted']} attempted; "
                          f"{counts['not_attempted']} not attempted; limit {counts['limit']}.")
            if limitation not in trace.setdefault("limitations", []):
                trace["limitations"].append(limitation)
        precursor = ReferenceDiscoveryTrace(
            candidate_ranking_policy_version="candidate-purpose-ranking-v1",
            search_capacity_policy_version="search-inspection-capacity-v4",
            metadata_screen_resolution_policy_version="screened-metadata-work-binding-v1",
            bounded_review_policy_version="bounded-reference-review-v9",
            metadata_identity_policy_version="corroborated-metadata-identity-v1",
            credibility_policy_version="reference-credibility-v4",
            credibility_reference_sha256=hashlib.sha256(trace["submitted_reference"].encode()).hexdigest()
                if trace.get("submitted_reference") else None,
            **({"journal_checks": trace["journal_checks"]} if "journal_checks" in trace else {}),
            search_retention_policy=trace.get("search_retention_policy"),
            search_policy_version=trace.get("search_policy_version", "configured-search-v1"),
            required_web_providers=trace.get("required_web_providers", []),
            reference_id=trace["reference_id"],
            expected=trace["expected"],
            required_route_categories=sorted(trace["required"]),
            queries=trace["queries"],
            attempts=trace["attempts"],
            candidates=trace["candidates"],
            limitations=list(trace.get("limitations") or [])
            + [
                "The trace is evaluated by the fail-closed completion assessor before any live outcome is attached."
            ],
        )
        completion = assess_reference_discovery_trace(precursor)
        # Read by the completed-search memo (search-reuse-memo-v1), including
        # the suppressed completed no-match outcome below.
        trace["derived_outcome"] = (
            completion.record.outcome if completion.ready and completion.record is not None else None)
        accepted_record = None
        limitations = list(precursor.limitations)
        if completion.ready and completion.record is not None:
            if completion.record.outcome == "unlocated_after_search":
                limitations.append(
                    "A completed no-match outcome is suppressed pending its real acceptance control."
                )
            else:
                accepted_record = completion.record.model_dump(mode="json")
        elif completion.blocker_codes:
            limitations.append(
                "Completion blockers: " + ", ".join(completion.blocker_codes)
            )
        completed_trace = precursor.model_copy(
            update={
                "candidates_complete": completion.ready,
                "outcome_derived": accepted_record is not None,
                "limitations": limitations,
            }
        )
        return completed_trace.model_dump(mode="json"), accepted_record

    @observe_reference
    @bounded_reference
    # Per-reference required-API call floor (WEB_SEARCH_PER_REFERENCE_FLOOR).
    @reference_search_scoped
    def resolve_reference(
        self,
        reference,
        *,
        isbn: str | None = None,
        identity_only: bool = False,
    ) -> RetrievalResult:
        """Resolve a ParsedReference without dropping its typed identity data."""
        if identity_only:
            from app.services.bibliography_identity import eligible_identity_only
            if not eligible_identity_only(reference):
                raise ValueError("Reference is not eligible for bibliography identity search")
        reference_id = getattr(reference, "reference_id", None) or "unassigned-reference"
        expected = ExpectedBibliographicFields(
            author_normalization_policy_version='author-etal-comparison-v1',
            title=getattr(reference, "title", None) or "",
            reference_parse_review=bool(getattr(reference, 'needs_review', False)),
            container_title=getattr(reference, 'container_title', '') or '',
            publisher=getattr(reference, 'publisher', '') or '',
            volume=getattr(reference, 'volume', '') or '',
            issue=getattr(reference, 'issue', '') or '',
            pages=getattr(reference, 'pages', '') or '',
            authors=[getattr(reference, "author", None)]
            if getattr(reference, "author", None)
            else [],
            year=getattr(reference, "year", None) or "",
            doi=getattr(reference, "doi", None) or "",
            isbn=isbn or "",
            source_kind=getattr(reference, "source_kind", None) or "unknown",
        )
        expected.edition_sensitive = is_edition_sensitive_reference(
            expected, getattr(reference, "raw_ref", "") or ""
        )
        token = _ACTIVE_DISCOVERY_TRACE.set(
            {
                "reference_id": reference_id,
                "submitted_reference": getattr(reference, 'raw_ref', '') or '',
                "search_policy_version": settings.SEARCH_POLICY_VERSION,
                "search_retention_policy": "brave-operational-transient-v1" if settings.BRAVE_SEARCH_TRANSIENT_ENABLED else None,
                "required_web_providers": ["brave", "exa"] if settings.SEARCH_POLICY_VERSION == "api-first-search-v2" else [],
                "expected": expected,
                "required": set(
                    required_reference_discovery_routes(
                        expected,
                        library_metadata_enabled=bool(
                            expected.doi and settings.DOI_RESOLVER_URL
                        ),
                    )
                ),
                "queries": [],
                "attempts": [],
                "candidates": [],
                "limitations": [],
            }
        )
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is not None and not trace["required"]:
            trace["limitations"].append(
                "No completed-search route policy exists for this source kind."
            )
        provisional_token = _PROVISIONAL_CANDIDATES.set([])
        try:
            if identity_only:
                result = self._resolve_identity_only(reference, expected)
            else:
                result = self.resolve(
                    doi=expected.doi or None,
                    isbn=expected.isbn or None,
                    title=expected.title or None,
                    author=expected.authors[0] if expected.authors else None,
                    year=expected.year or None,
                    student_url=getattr(reference, "url", None) or None,
                    raw_ref=getattr(reference, "raw_ref", None) or None,
                    source_kind=expected.source_kind,
                    source_kind_confidence=(
                        getattr(reference, "source_kind_confidence", "unknown")
                    ),
                    source_kind_evidence=tuple(
                        getattr(reference, "source_kind_evidence", ()) or ()
                    ),
                    container_title=expected.container_title or "",
                    publisher=expected.publisher or "",
                )
        except SourceResolutionError as exc:
            if not identity_only:
                self._enrich_journal_registration(reference)
                self._enrich_book_editions()
            (
                exc.reference_discovery_trace,
                exc.reference_discovery,
            ) = self._discovery_artifacts()
            if not identity_only and isinstance(exc.reference_discovery, dict):
                self._attach_container_identity(reference, exc.reference_discovery)
                mismatch = self._doi_title_mismatch()
                if mismatch:
                    exc.reference_discovery["doi_title_mismatch"] = mismatch
            candidates = _PROVISIONAL_CANDIDATES.get()
            if not identity_only:
                self._update_search_memo(candidates[0] if candidates else None,
                                         exc.reference_discovery_trace)
            if candidates:
                fallback = candidates[0]
                fallback.metadata['reference_discovery_trace'] = exc.reference_discovery_trace
                fallback.metadata['reference_discovery'] = exc.reference_discovery
                return fallback
            raise
        else:
            candidates = _PROVISIONAL_CANDIDATES.get()
            if candidates and not result.full_text:
                result = candidates[0]
            if not identity_only:
                self._enrich_journal_registration(reference)
                self._enrich_book_editions()
            result.metadata = result.metadata or {}
            trace_payload, discovery_payload = self._discovery_artifacts()
            if not identity_only:
                self._update_search_memo(result, trace_payload)
            result.metadata["reference_discovery_trace"] = trace_payload
            if discovery_payload is not None:
                result.metadata["reference_discovery"] = discovery_payload
                if not identity_only:
                    self._attach_container_identity(reference, discovery_payload)
                mismatch = self._doi_title_mismatch()
                if mismatch:
                    discovery_payload["doi_title_mismatch"] = mismatch
            return result
        finally:
            _PROVISIONAL_CANDIDATES.reset(provisional_token)
            _ACTIVE_DISCOVERY_TRACE.reset(token)

    @staticmethod
    def _search_memo_key(trace: dict) -> str | None:
        from app.services.search.search_memo import reference_key
        expected: ExpectedBibliographicFields = trace["expected"]
        return reference_key(expected.doi, expected.title,
                             expected.authors[0] if expected.authors else None, expected.year)

    def _reuse_search_memo(self) -> bool:
        """Reuse this scope's completed-search memo instead of the paid web tier.

        `search-reuse-memo-v1`. Only inside a traced reference resolution with
        an authorization scope; the reused attempt is recorded on the trace as
        provider `search_memo` with the original search's date and memo id.
        """
        from app.services.search import search_memo
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        context = search_memo.active_context()
        if trace is None or context is None or "bounded_web" not in trace["required"]:
            return False
        key = self._search_memo_key(trace)
        if key is None:
            return False
        record = search_memo.reusable_memo(context, key, source_kind=trace["expected"].source_kind)
        if record is None:
            return False
        try:
            rebuilt = search_memo.reuse_attempts(
                record, reference_id=trace["reference_id"],
                expected_required_providers=list(trace.get("required_web_providers") or []))
        except (KeyError, TypeError, ValueError):
            rebuilt = None
        if rebuilt is None:
            return False
        queries, attempts = rebuilt
        trace["queries"].extend(queries)
        trace["attempts"].extend(attempts)
        trace["search_memo_reused"] = True
        return True

    def _update_search_memo(self, result: RetrievalResult | None, trace_payload: dict | None) -> None:
        """Record a completed no-source web search, or clear a memo once a source is found."""
        from app.services.search import search_memo
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        context = search_memo.active_context()
        if trace is None or context is None:
            return
        key = self._search_memo_key(trace)
        if key is None:
            return
        metadata = (result.metadata or {}) if result is not None else {}
        if metadata.get("provisional_source"):
            # A provisional possible match is neither a found source nor none.
            return
        if result is not None and (result.full_text or (
                result.representation is not None and result.representation.content)):
            search_memo.clear_memo(context, key)
            return
        if trace.get("search_memo_reused"):
            # Reuse never extends a memo; the next search after it expires does.
            return
        summary = search_memo.completed_search_summary(trace_payload, trace.get("derived_outcome"))
        if summary is not None:
            search_memo.record_completed_search(context, key, summary)

    @bounded_landing_inspection
    @bounded_pdf_inspection
    @identity_candidate_ranking
    def _resolve_identity_only(self, reference, expected):
        """Use metadata/search adapters without entering any acquisition path.

        Bounded HTML metadata inspection can resolve unclear web leads,
        but material/unknown leads still block negative findings. Neither snippets
        nor a successful API response establish source identity.
        """
        from app.services.bibliography_identity import POLICY
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        trace["limitations"].append(
            f"{POLICY}: source-byte reuse, direct URL and full-text acquisition "
            "are not attempted; identity-only-landing-metadata-v1 permits bounded HTML metadata inspection; "
            "identity-only-pdf-metadata-v1 permits temporary safety-gated PDF front-matter inspection."
        )
        structured = [
            source for source in self._retrieval_sources
            if "metadata_only_search" in getattr(source, "capabilities", ())
            # A catalogue that cannot hold this kind of work is not consulted.
            # Declared on the adapter class rather than written here, so the
            # routing is checkable; an unknown expected kind always passes.
            and getattr(source, "handles_source_kind", _any_source_kind)(
                expected.source_kind)
        ]
        for source, result in self._lookup_structured_sources(
                structured, expected.doi or None, expected.title,
                reference.author, expected.year or None):
            self._record_discovery_attempt(category="academic_adapter",
                provider=source.name, result=result,
                required=_route_blocks_completion(source))
        self._title_search_when_doi_unconfirmed(
            structured, expected.doi or None, expected.title, reference.author,
            expected.year or None, expected.source_kind)
        self._enrich_journal_registration(reference)
        self._enrich_book_editions()
        _, discovery = self._discovery_artifacts()
        if (discovery or {}).get("outcome") not in {"confirmed", "confirmed_with_minor_differences"}:
            for source in self._retrieval_sources:
                if "web_discovery" not in getattr(source, "capabilities", ()):
                    continue
                result = self._try_source(source, expected.doi or None,
                    expected.title, reference.author, expected.year or None,
                    identity_only=True)
                self._record_discovery_attempt(category="bounded_web",
                    provider=source.name, result=result, required=True)
        # No provider result (including incidental abstracts) leaves this route.
        return RetrievalResult(source_name="bibliography_identity", success=False,
            metadata={"identity_only_policy": POLICY})

    def _doi_title_mismatch(self):
        """Report a supplied identifier that registers a different work.

        An LLM composing a reference often appends a real DOI to an invented
        title. Resolving it and accepting whatever it names would confirm the
        invention, so the graph refuses the merge; the discrepancy itself is
        what an instructor needs to see.
        """
        graph = getattr(self, "_last_graph", None)
        for candidate in getattr(graph, "rejected_candidates", None) or []:
            if candidate.get("rejection_code") == "doi_registers_a_different_title":
                return {
                    "policy_version": "doi-title-mismatch-v1",
                    "doi": candidate.get("doi") or "",
                    "registered_title": candidate.get("title") or "",
                    "provider": candidate.get("provider") or "",
                }
        return None

    def _attach_container_identity(self, reference, discovery: dict) -> None:
        """Identify the containing book whenever the cited part is not confirmed.

        The part search can end unconfirmed on either path out of
        `resolve_reference`: raised as unresolved, or returned with an
        unchecked lead (Belton's 2026-09-27 rerun: five Brave leads unchecked,
        so "search incomplete"). The book is identifiable either way.
        """
        if discovery.get("outcome") in _CONFIRMED_DISCOVERY_OUTCOMES:
            return
        container_identity = self._identify_container_work(reference)
        if container_identity:
            discovery["container_identity"] = container_identity

    def _identify_container_work(self, reference):
        """Identify the book a miscited chapter names, without claiming the part.

        A student who writes a section of a monograph up as a chapter in an
        edited collection still names the book correctly: Belton's reference
        carries "American cinema/American culture", the author, the year, the
        publisher and pp. 64-86. Only the part-level framing is wrong, and
        searching the part title alone finds a chapter that does not exist.
        This runs only after the ordinary routes fail, identifies the
        containing work on its own fields, and never asserts that the cited
        part was located within it.
        """
        container = (getattr(reference, "container_title", "") or "").strip()
        if not container:
            return None
        if getattr(reference, "source_kind", "") not in {"book_section", "edited_collection"}:
            return None
        try:
            probe = reference.model_copy(update={
                "title": container,
                "container_title": "",
                "source_kind": "monograph",
                "source_kind_confidence": "medium",
                "source_kind_evidence": ["container title of a cited part"],
                "needs_review": False,
            })
            result = self.resolve_reference(probe, identity_only=True)
        except (SourceResolutionError, ValueError, RuntimeError):
            return None
        discovery = (result.metadata or {}).get("reference_discovery") or {}
        confirmed = discovery.get("outcome") in _CONFIRMED_DISCOVERY_OUTCOMES
        # A book usually stays a possible match because the edition year is
        # not bound by an ISBN. The title-page lookup below matches title and
        # author itself, so it may still speak; only a confirmed book counts
        # as "identified" for the Cannot be verified rule.
        if not confirmed and discovery.get("outcome") not in {"possible_match", "search_incomplete"}:
            return None
        authors = [a for a in (result.authors or []) if a]
        # Only a record that positively types itself as a single-authored book
        # can show that the edited-collection form is wrong. An editor writing
        # a chapter of their own collection is ordinary and must not be flagged.
        located_kind = classify_provider_source_kind(result.metadata or {})
        is_monograph = (confirmed and located_kind.kind == "monograph"
                        and located_kind.confidence in {"high", "medium"})
        kind, kind_confidence = located_kind.kind, located_kind.confidence
        # An identity-only lookup returns no provider record to type, so the
        # title page decides: see OpenLibraryRetriever.statements_of_responsibility.
        responsibility = self._container_responsibility(container, getattr(reference, "author", "") or "")
        if responsibility.get("classification") == "edited":
            is_monograph = False
        elif responsibility.get("classification") == "single_authored" and not is_monograph:
            is_monograph, kind, kind_confidence = True, "monograph", "medium"
        return {
            "located_source_kind": kind,
            "located_source_kind_confidence": kind_confidence,
            "is_monograph": is_monograph,
            "responsibility": responsibility,
            "status": "identified" if confirmed else "not_confirmed",
            "policy_version": "container-work-identity-v2",
            "searched_title": container,
            "title": result.title or container,
            "authors": authors,
            "year": result.year,
            "provider": result.source_name,
            "outcome": discovery.get("outcome"),
            "pages_cited": (getattr(reference, "pages", "") or ""),
            "part_title_unconfirmed": getattr(reference, "title", "") or "",
        }

    def _container_responsibility(self, title: str, author: str) -> dict:
        source = next((s for s in self._retrieval_sources
                       if hasattr(s, "statements_of_responsibility")), None)
        if source is None or not author:
            return {"classification": "unresolved", "reason": "catalogue_not_configured" if source is None
                    else "no_cited_author"}
        try:
            return source.statements_of_responsibility(title, author)
        except Exception as exc:   # an optional check never fails the reference
            return {"classification": "unresolved", "reason": f"lookup_failed:{type(exc).__name__}"}

    def _identity_settled(self, graph) -> bool:
        """Is the work confirmed AND already reachable?

        Both halves matter. A confirmed identity with no route to the text
        still needs the remaining adapters, because an open-access link is the
        one thing they can still contribute — measured across every stored job,
        they supplied the acquired representation once in seventy-five.
        """
        _, discovery = self._discovery_artifacts()
        if (discovery or {}).get("outcome") not in _CONFIRMED_DISCOVERY_OUTCOMES:
            return False
        try:
            merged = graph.to_result()
        except Exception:
            return False
        return bool(merged.success and (merged.locations or merged.full_text_url))

    def _title_search_when_doi_unconfirmed(self, sources, doi, title, author, year, kind, graph=None):
        """Search the required indexes by title when the DOI settled nothing.

        A lookup stops at the first answer, so a supplied DOI that resolves to
        another work answers for the reference and its title is never searched.
        A real work carrying a wrong DOI would then go unlocated, and whether a
        reference can be verified must not turn on the identifier alone.
        """
        if not doi or not title:
            return
        _, discovery = self._discovery_artifacts()
        if (discovery or {}).get("outcome") in _CONFIRMED_DISCOVERY_OUTCOMES:
            return
        from app.services.reference_verification import required_adapters
        wanted = set(required_adapters(kind))
        title_sources = [s for s in sources if s.name in wanted and (
            not getattr(s, "capabilities", None)
            or {"title_author", "title_search"} & getattr(s, "capabilities", frozenset()))]
        for source, provider_result in self._lookup_structured_sources(
                title_sources, None, title, author, year):
            self._record_discovery_attempt(
                category="academic_adapter", provider=source.name,
                result=provider_result, required=_route_blocks_completion(source))
            if graph is not None and provider_result.success:
                graph.add(provider_result)

    @staticmethod
    def _declined_sources(sources, reason):
        """Record routes declined for a stated reason, never as failures."""
        for source in sources:
            yield source, RetrievalResult(
                source_name=source.name, success=False,
                error=f"{PROVIDER_SKIPPED_ERROR}: {reason}",
                metadata={"lookup_applicable": False},
            )

    @staticmethod
    def _skipped_sources(sources):
        """Record a route the application declined to run, as exactly that.

        Never a timeout, a failure or an absence. A skipped call that described
        itself in the vocabulary of a completed one has already cost this
        project two misdiagnoses.
        """
        for source in sources:
            yield source, RetrievalResult(
                source_name=source.name,
                success=False,
                error=f"{PROVIDER_SKIPPED_ERROR}: identity confirmed and source reachable",
                metadata={"lookup_applicable": False},
            )

    @staticmethod
    def _front_matter_corroborates(content, author, year):
        """Does this copy's own front matter support the cited author or year?

        A title that differs from the cited one, even by only a subtitle,
        cannot carry an identity alone: different authors publish different
        works under the same main title, and a possible match offered on title
        resemblance alone sends the reader to a stranger's work. Require the
        copy to name the cited author or carry the cited year.
        """
        if not author and not year:
            return False
        from app.services.pdf_verifier import _author_matches, extract_metadata_from_pdf
        try:
            metadata = extract_metadata_from_pdf(content)
        except Exception:
            return False
        if author and _author_matches(author, metadata):
            return True
        observed_year = str(metadata.get('year') or '').strip()
        return bool(year and observed_year and str(year).strip() == observed_year)

    def _retain_provisional_candidate(self, result, *, confidence, reason,
                                      completeness, text_quality, kind_verdict,
                                      expected_title=None, expected_author=None,
                                      expected_year=None):
        """Keep one clean readable possible match until this reference finishes.

        Discovery continues normally. This never grants identity, a cache write,
        or source-library admission, and cannot run outside resolve_reference.
        """
        candidates = _PROVISIONAL_CANDIDATES.get()
        representation = result.representation
        if (candidates is None or candidates or confidence != 'medium'
                or representation is None or representation.kind is not RepresentationKind.PDF
                or text_quality not in {'digital', 'born_digital', 'scan_ocr'}
                or kind_verdict == 'incompatible'):
            return
        inspection = (result.metadata or {}).get('source_inspection') or {}
        if re.search(r'\b(translation|abridg|different edition|wrong work|identifier conflict)', reason, re.I):
            return
        if (inspection.get('identity') == 'different_work'
                or inspection.get('representation_role') in {'catalog_or_listing', 'review'}):
            return
        if any(item.get('field') in {'edition', 'identifier'}
               for item in inspection.get('differences', []) if isinstance(item, dict)):
            return
        try:
            safety = inspect_uploaded_pdf(representation.content)
        except FileSafetyUnavailable:
            return
        if safety.verdict is not SafetyVerdict.CLEAN:
            return
        # Reuse the upload verifier's title observation/comparison. A visible
        # different title cannot be rescued by incidental body-text mentions.
        from app.services.pdf_verifier import describe_first_page_title, _title_matches
        cover = describe_first_page_title(representation.content)
        observed_title = _legible_cover_title(cover.title)
        if (expected_title and observed_title
                and not _title_matches(expected_title, {'title': observed_title})):
            return
        if (expected_title and not observed_title
                and not _opening_pages_present_title(
                    representation.content, expected_title)):
            return
        # A visible title that only extends or abbreviates the cited one is not
        # identity on its own. Belton's chapter "The Studio System" reached the
        # report as a possible match for an unrelated Hertfordshire thesis that
        # merely carried the phrase, with a different author, year and
        # publication history, so require this copy to corroborate one of them.
        if (expected_title and observed_title
                and expected_title.casefold() != observed_title.casefold()
                and not self._front_matter_corroborates(
                    representation.content, expected_author, expected_year)):
            return
        explanation = (
            f'The reference gives “{expected_title}”; the visible title is “{observed_title}”. '
            'An exact identity match has not been confirmed.'
            if expected_title and observed_title and expected_title.casefold() != observed_title.casefold()
            else _COVER_TITLE_EXPLANATIONS.get(
                cover.reason,
                'The title, author and publication year could not all be confirmed from this copy.',
            )
        )
        candidate = replace(result, success=True, full_text=representation.content,
            full_text_url=representation.source_url,
            representation=deepcopy(representation), locations=[], metadata={
            'identity_confidence': 'medium', 'identity_reason': reason,
            'completeness': completeness, 'text_quality': text_quality,
            'provisional_source': {
                'policy_version': 'submission-possible-match-v1',
                'content_sha256': hashlib.sha256(representation.content).hexdigest(),
                'identity_status': 'possible_match', 'reason': explanation,
                'validation_reason': reason,
                'source_url': representation.source_url,
                'observed_title': observed_title,
                'source_inspection': inspection or None,
                'library_admission': False,
            },
        })
        from app.services.search.transient import is_transient_brave, BRAVE_TRANSIENT_POLICY
        if is_transient_brave(result):
            # This copy is not an independently confirmed source identity.
            # Scrub before it can escape the active acquisition operation;
            # finalizing the original result cannot clean a detached copy.
            # Call/disposition accounting remains on the original tier trace,
            # not a second synthetic attempt attached to the fallback.
            candidate.full_text_url = None
            candidate.error = None
            candidate.abstract = None
            candidate.locations.clear()  # __post_init__ recreates a URL location.
            candidate.representation.source_url = None
            candidate.representation.metadata = {}
            candidate.parent_representation = None
            candidate.metadata['provisional_source']['source_url'] = None
            candidate.metadata['search_retention_policy'] = BRAVE_TRANSIENT_POLICY
            candidate.metadata['transient_details_discarded'] = True
        candidates.append(candidate)

    def _enrich_journal_registration(self, reference) -> None:
        """Optional bounded metadata escalation, never an exhaustive archive check."""
        if expired():
            return
        from app.services.journal_discovery import journal_claim, registration_checks
        from app.services.retrieval.crossref import CrossrefRetriever

        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is None:
            return
        claim = journal_claim(reference)
        if claim is None:
            return
        capabilities = getattr(self, "_acquisition_capabilities", None)
        if capabilities is not None and "academic_adapters" not in capabilities:
            return
        # Do not spend additional requests on already established identities.
        if any(c.is_credible and not c.has_material_conflict and
               sum(v.outcome == "agreement" for v in c.comparisons) >= 2
               for c in trace["candidates"]):
            return
        adapter = next((s for s in getattr(self, "_retrieval_sources", [])
                        if isinstance(s, CrossrefRetriever)), None)
        if adapter is None:
            return
        # Reserve structural room for every bounded result before any request.
        if len(trace["candidates"]) > 73 or len(trace["queries"]) > 62 or len(trace["attempts"]) > 126:
            return
        trace["journal_checks"] = []
        for check, rows in registration_checks(adapter, claim, reference.title):
            trace["journal_checks"].append(check)
            if check.stage == "journal":
                continue  # ISSN resolution is not an article identity search.
            success = check.outcome in {"candidates_found", "no_registered_candidates"}
            self._record_discovery_attempt(
                category="academic_adapter", provider="crossref", required=False,
                result=RetrievalResult(source_name="crossref", success=bool(rows),
                    error=None if rows else "No results" if success else check.outcome,
                    metadata={
                        "structured_search_attempts": [{"query": check.query,
                            "outcome": "results" if rows else "no_results" if success else "operational_failure"}],
                        "journal_metadata_candidates": [vars(r) for r in rows],
                    }),
            )

    def _enrich_book_editions(self) -> None:
        """One optional metadata query, separate from source acquisition/search completion."""
        if expired():
            return
        from app.services.book_metadata import isbn10_to_isbn13
        from app.services.retrieval.google_books import GoogleBooksRetriever

        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is None or not trace["expected"].edition_sensitive:
            return
        expected = trace["expected"]
        if not expected.supports_identity_search():
            return
        capabilities = getattr(self, "_acquisition_capabilities", None)
        if capabilities is not None and "academic_adapters" not in capabilities:
            trace["limitations"].append("Book edition metadata was not requested under this acquisition profile.")
            return
        from app.services.reference_verification import _EDITION, _book_title_matches
        # "(2nd ed.)" is not part of a catalogue title (Singer, 2026-09-30).
        title = re.sub(r"\s+", " ", _EDITION.sub(" ", expected.title or "")).strip() or None
        search = GoogleBooksRetriever().search_metadata_result(
            isbn=expected.isbn or None, title=title,
            author=expected.authors[0].split(",", 1)[0] if expected.authors else None,
            year=expected.year or None, max_results=5,
        )
        self._record_book_search(expected, search)
        cited = {"title": expected.title, "container_title": expected.container_title}
        if (title and expected.authors and not expected.isbn and not expired()
                and search.outcome in {"results", "no_results"}
                and not any(_book_title_matches(cited, ": ".join(v for v in (b.title, b.subtitle) if v))
                            for b in search.candidates)):
            # A genuine book cited under the wrong author is found by its title
            # alone (Singer, Mather 2026-09-30); a record under that title is a
            # possible match, never a flag. Recorded as a second title search.
            self._record_book_search(expected, GoogleBooksRetriever().search_metadata_result(
                title=title, year=expected.year or None, max_results=5), fallback=True)

    def _record_book_search(self, expected, search, *, fallback: bool = False) -> None:
        """Record one Google Books metadata search as a discovery attempt."""
        from app.services.book_metadata import isbn10_to_isbn13
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        candidates = []
        for book in search.candidates:
            # Keep the observed record even if unrelated, but never supply a
            # description as an abstract or a preview URL as acquired content.
            canonical = {isbn10_to_isbn13(value) for value in book.identifiers}
            isbn_value = next(iter(canonical)) if len(canonical) == 1 else ""
            candidates.append({
                "source_name": "google_books", "success": True,
                "title": ": ".join(value for value in (book.title, book.subtitle) if value)[:1000],
                "authors": list(book.authors)[:64],
                "year": book.published_date[:4] if book.published_date and re.match(r"^(?:18|19|20)\d{2}(?:$|-)", book.published_date) else None,
                "metadata": {
                    "identity_confidence": "rejected" if book.match_confidence == "none" else "unconfirmed",
                    "identity_limitation": book.match_reason if book.match_confidence == "unresolved" else None,
                    "isbn": isbn_value, "observed_source_kind": "monograph",
                    "book_edition_metadata": {
                        "volume_id": book.volume_id[:255], "publisher": (book.publisher or "")[:1000],
                        "published_date": (book.published_date or "")[:40],
                        "identifiers": list(book.identifiers)[:16], "record_sha256": book.record_sha256,
                    },
                },
            })
        result = RetrievalResult(
            source_name="google_books", success=bool(candidates),
            error=None if candidates else "No results" if search.outcome == "no_results" else search.error_code or search.outcome,
            metadata={
                "structured_search_attempts": [{"query": search.query or "insufficient-metadata", "outcome": search.outcome,
                    "result_count": len(candidates) if search.outcome in {"results", "no_results"} else None}],
                "book_metadata_candidates": candidates,
            },
        )
        self._record_discovery_attempt(
            category="academic_adapter", provider="google_books", result=result, required=False,
        )
        if fallback:
            return
        trace["limitations"].append(
            "Book catalog dates describe particular editions, not necessarily the copy used. "
            "Only matching observed ISBNs bind an edition comparison; publication and copyright dates may differ."
            if candidates else "The optional book-edition lookup did not establish an edition; this is not evidence that the reference is incorrect."
        )
        if not candidates:
            self._corroborate_book_from_archive(expected)

    def _corroborate_book_from_archive(self, expected) -> None:
        """Ask a second catalogue whether the book exists, never whether it does not.

        Google Books is the only route that can satisfy the book-catalogue
        coverage requirement, so an outage there leaves a monograph with no
        catalogue evidence at all. The Internet Archive is reachable where
        other catalogues are not and its answers are precise: over the
        development corpus every record it returned was accepted by the
        identity rule. Its silence is not: it held nothing for 27 of 38
        books that certainly exist, a 71% false-emptiness rate. So this
        contributes a positive identification and can never contribute
        coverage -- an empty answer reports `no_results_low_coverage`,
        which the review's acceptance test cannot count.
        """
        if expired():
            return
        from app.services.retrieval.internet_archive_catalog import (
            InternetArchiveCatalogRetriever,
        )
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        if trace is None:
            return
        search = InternetArchiveCatalogRetriever().search_metadata_result(
            isbn=expected.isbn or None, title=expected.title or None,
            author=expected.authors[0] if expected.authors else None,
            year=expected.year or None, max_results=5,
        )
        candidates = [{
            "source_name": "internet_archive", "success": True,
            "title": (book.title or "")[:1000],
            "authors": list(book.authors)[:64],
            "year": (book.published_date or "")[:4] or None,
            "metadata": {
                "identity_confidence": "unconfirmed",
                "isbn": next(iter(book.identifiers), ""),
                "observed_source_kind": "monograph",
                "publisher": (book.publisher or "")[:1000],
                # The archive identifier is the candidate key. It cannot
                # reach the Google-Books edition and year rules, which
                # select on `provider == "google_books"` and rest on two
                # independent edition records this catalogue cannot supply.
                "book_edition_metadata": {
                    "volume_id": book.volume_id[:255],
                    "publisher": (book.publisher or "")[:1000],
                    "published_date": (book.published_date or "")[:40],
                    "identifiers": list(book.identifiers)[:16],
                    "record_sha256": book.record_sha256,
                },
            },
        } for book in search.candidates]
        result = RetrievalResult(
            source_name="internet_archive", success=bool(candidates),
            error=None if candidates else search.error_code or search.outcome,
            metadata={
                "structured_search_attempts": [{
                    "query": search.query or "insufficient-metadata",
                    "outcome": search.outcome,
                    "result_count": len(candidates) if search.outcome == "results" else None,
                }],
                "book_metadata_candidates": candidates,
            },
        )
        self._record_discovery_attempt(
            category="academic_adapter", provider="internet_archive",
            result=result, required=False,
        )
        trace["limitations"].append(
            "A second catalogue holds a record for this work; catalogue presence "
            "identifies the work but does not establish the edition cited."
            if candidates else
            "The second catalogue holds no record for this work. Its coverage is "
            "partial, so this is not evidence that the reference is incorrect."
        )

    def resolve(
        self,
        doi: str | None = None,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
        year: str | None = None,
        student_url: str | None = None,
        raw_ref: str | None = None,
        source_kind: str | None = None,
        source_kind_confidence: str = "unknown",
        source_kind_evidence: tuple[str, ...] = (),
        container_title: str = "",
        publisher: str = "",
    ) -> RetrievalResult:
        """Resolve a reference to a source document.

        Returns a RetrievalResult with full_text (or abstract) populated if found.
        Raises SourceResolutionError if no source is found.

        Multi-field verification: when a source is found via title search (not
        via direct DOI/URL), it's verified against multiple reference fields
        (title + author + year) to confirm it's the RIGHT source, not just A
        source with matching keywords. A student may mess up one field (wrong
        URL) but won't mess up title + author + year simultaneously.

        URL failure flagging: when a student URL is provided but fails (403,
        404, blocked), retrieval continues via title search. The result is
        flagged with the URL failure reason so the instructor knows:
        (a) the original URL didn't work, and (b) the found source may differ
        from what the student cited.

        Traditional-media references (films, TV, albums) and physical archives
        skip the academic-database search — they're not indexed there in a
        useful form and title search produces keyword-coincidence false positives.
        """
        url_failure_reason: str | None = None  # track why the URL failed (if it did)
        expected_kind = _expected_source_kind_assessment(
            source_kind=source_kind,
            source_kind_confidence=source_kind_confidence,
            source_kind_evidence=source_kind_evidence,
            raw_ref=raw_ref,
            title=title,
            url=student_url,
        )

        # 0. Route by source type — skip retrieval for traditional media / archives.
        skip_academic_dbs = False
        if raw_ref:
            if is_archive_source(raw_ref):
                link_not_visited(None, 'unsupported_source_type')
                raise SourceResolutionError(
                    f"Physical archive source — cannot be verified automatically: {raw_ref[:60]}"
                )
            skip_academic_dbs = is_traditional_media(raw_ref)

        if student_url:
            student_url = _normalize_cited_url(student_url)

        # 1. Check local S3 cache
        result = self._check_local_cache(
            doi,
            isbn,
            title,
            author=author,
            year=year,
            expected_source_kind=expected_kind,
        )
        self._record_discovery_attempt(
            category="durable_repository",
            provider="local_cache",
            result=result,
            required=False,
        )
        if result.success and result.full_text:
            link_not_visited(None, 'authorized_reuse')
            return result

        # 1.5. Reuse bounded abstract/miss lookup state. Full text always wins
        # from the durable repository. A changed cited URL, acquisition-policy
        # change, access revision, or elapsed refresh interval bypasses this
        # result and runs discovery again.
        lookup_cache = getattr(self, "_lookup_cache", None)
        policy_signature = self._lookup_policy_signature()
        lookup_record = (
            lookup_cache.get(
                doi=doi,
                title=title,
                author=author,
                year=year,
                source_kind=expected_kind.kind,
            )
            if lookup_cache is not None
            else None
        )
        lookup_record_is_fresh = bool(
            lookup_record
            and lookup_record.is_fresh(
            policy_signature=policy_signature,
            student_url=student_url,
            )
        )
        lookup_record_bypassed_for_trace = bool(
            lookup_record_is_fresh and _ACTIVE_DISCOVERY_TRACE.get() is not None
        )
        if lookup_record_bypassed_for_trace:
            trace = _ACTIVE_DISCOVERY_TRACE.get()
            if trace is not None:
                trace["limitations"].append(
                    "An unbound lookup-cache result was bypassed; live routes were rerun for discovery provenance."
                )
        elif lookup_record_is_fresh:
            if lookup_record.result is not None:
                self._record_discovery_attempt(
                    category="durable_repository",
                    provider="retrieval_lookup_cache",
                    result=lookup_record.result,
                    required=False,
                )
                return lookup_record.result
            self._record_discovery_attempt(
                category="durable_repository",
                provider="retrieval_lookup_cache",
                result=RetrievalResult(
                    source_name="retrieval_lookup_cache",
                    success=False,
                    error="No match in fresh cached lookup",
                ),
                required=False,
            )
            raise SourceResolutionError(
                f"Source not found (fresh cached lookup): doi={doi}, "
                f"isbn={isbn}, title={title}"
            )

        # 2. Try student URL — PDF first, then web-page text for HTML sources.
        capabilities = getattr(self, "_acquisition_capabilities", None)
        if student_url and capabilities is not None and 'student_url' not in capabilities:
            link_not_visited(student_url, 'capability_disabled')
        if student_url and (capabilities is None or "student_url" in capabilities):
            if is_library_locator_url(student_url):
                link_not_visited(student_url, 'library_locator_only')
                url_failure_reason = (
                    "Library locator identifies a catalog/discovery record; "
                    "page content is not source evidence"
                )
                self._record_discovery_attempt(
                    category="student_url",
                    provider="student_url",
                    result=RetrievalResult(
                        source_name="student_url",
                        success=False,
                        error="Library locator is not source content",
                    ),
                    required=False,
                )
            elif _url_prefers_pdf(student_url):
                result = self._retry_after_network_error(lambda: self._try_student_url(
                    student_url, doi, title, author, year,
                    expected_source_kind=expected_kind,
                ))
                self._record_discovery_attempt(
                    category="student_url",
                    provider="student_url_pdf",
                    result=result,
                    required=False,
                )
                if result.success and result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return result  # Authoritative — the student's actual source
                # An explicit PDF route may still return an HTML landing page.
                if not result.success and "not a pdf" in (result.error or "").lower():
                    web_result = self._retry_after_network_error(lambda: self._try_web_fetch(
                        student_url, title, expected_source_kind=expected_kind,
                        expected_author=author, expected_year=year, expected_doi=doi,
                    ))
                    self._record_discovery_attempt(
                        category="student_url",
                        provider="student_url_html",
                        result=web_result,
                        required=False,
                    )
                    if web_result.success:
                        self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                        return web_result
                    url_failure_reason = web_result.error or "web fetch failed"
                else:
                    url_failure_reason = result.error or "URL download failed"
            else:
                # Most cited URLs are web pages. Fetch HTML first so news,
                # government, blog and reference content does not pay for a
                # failed PDF-only request before reaching its real route.
                web_result = self._retry_after_network_error(lambda: self._try_web_fetch(
                    student_url, title, expected_source_kind=expected_kind,
                    expected_author=author, expected_year=year, expected_doi=doi,
                ))
                self._record_discovery_attempt(
                    category="student_url",
                    provider="student_url_html",
                    result=web_result,
                    required=False,
                )
                if web_result.success:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return web_result
                if "returned a pdf" in (web_result.error or "").lower():
                    result = self._retry_after_network_error(lambda: self._try_student_url(
                        student_url, doi, title, author, year,
                        expected_source_kind=expected_kind,
                    ))
                    self._record_discovery_attempt(
                        category="student_url",
                        provider="student_url_pdf",
                        result=result,
                        required=False,
                    )
                    if result.success and result.full_text:
                        self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                        return result
                    url_failure_reason = result.error or "PDF validation failed"
                else:
                    url_failure_reason = web_result.error or "web fetch failed"

        # 2.5. Try DOI resolver / campus proxy (institutional deployment).
        #      When DOI_RESOLVER_URL is configured, construct {url}{doi} to
        #      access the paper through the university's library proxy. This
        #      is the key to high full-text retrieval rates for paywalled content
        #      — the proxy handles subscription authentication.
        if (
            doi
            and settings.DOI_RESOLVER_URL
            and (capabilities is None or "doi_resolver" in capabilities)
        ):
            result = self._try_doi_resolver(
                doi, title, expected_source_kind=expected_kind
            )
            self._record_discovery_attempt(
                category="library_metadata",
                provider="doi_resolver",
                result=result,
                required=True,
            )
            if result.success:
                if result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                return result

        # 3. Merge structured-provider evidence into one canonical work graph,
        #    then acquire from its combined location set. Public-domain sources
        #    retain their edition-specific route; public-web discovery remains
        #    the final candidate generator.
        best_abstract_result: RetrievalResult | None = None
        web_retrieval_trace: list[dict] = []
        if skip_academic_dbs:
            logger.info("Skipping academic DBs for a traditional-media reference")
        else:
            public_domain_sources = [
                source for source in self._retrieval_sources
                if source.name in ("gutenberg", "wikisource")
            ]
            web_sources = [
                source for source in self._retrieval_sources
                if "web_discovery" in getattr(source, "capabilities", frozenset())
                or source.name == "web_search"
            ]
            structured_sources = [
                source for source in self._retrieval_sources
                if source not in public_domain_sources and source not in web_sources
            ]
            # An index that cannot hold this kind of work is not asked about
            # it (reference-verification-v1). The adapter declares what it
            # holds; an unclassified reference still consults every index.
            unsuited_sources = [
                source for source in structured_sources
                if not getattr(source, "handles_source_kind", _any_source_kind)(expected_kind.kind)
            ]
            structured_sources = [s for s in structured_sources if s not in unsuited_sources]
            for source, provider_result in self._declined_sources(
                    unsuited_sources, "not suited to this kind of work"):
                # A declined lookup is not applicable, so it never blocks.
                self._record_discovery_attempt(
                    category="academic_adapter", provider=source.name,
                    result=provider_result, required=_route_blocks_completion(source))

            tried_public_domain = False
            if self._is_public_domain_front_route(doi, year):
                tried_public_domain = True
                result = self._try_source_sequence(
                    public_domain_sources, doi, title, author, year,
                    expected_source_kind=expected_kind,
                )
                if result and result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return self._finalize_resolution_result(
                        result, doi, title, author, year, url_failure_reason
                    )
                if result and result.abstract:
                    best_abstract_result = result

            graph = self._last_graph = CanonicalWorkGraph(
                expected_doi=doi,
                expected_title=title,
                expected_author=author,
                expected_year=year,
                expected_container_title=container_title,
                expected_publisher=publisher,
                expected_url=student_url or "",
                expected_source_kind=expected_kind.kind,
                expected_source_kind_confidence=expected_kind.confidence,
                expected_source_kind_evidence=expected_kind.evidence,
            )
            # Accuracy first: every adapter still runs whenever the identity
            # question is open. Efficiency second: once the fast metadata
            # routes have confirmed the work *and* a route to its text exists,
            # the corroboration-only adapters are answering a settled question
            # at real cost, so they are not called. An unconfirmed reference —
            # which is every reference a fabrication finding could concern —
            # is unaffected.
            leading_sources = [
                source for source in structured_sources
                if source.name not in _CORROBORATION_ONLY_SOURCES
            ]
            trailing_sources = [
                source for source in structured_sources
                if source.name in _CORROBORATION_ONLY_SOURCES
            ]
            for source, provider_result in self._lookup_structured_sources(
                leading_sources, doi, title, author, year
            ):
                self._record_discovery_attempt(
                    category="academic_adapter",
                    provider=source.name,
                    result=provider_result,
                    # Deferred adapters are explicitly optional supplements,
                    # not mandatory synchronous search routes.
                    required=_route_blocks_completion(source),
                )
                if provider_result.success:
                    assessment = graph.add(provider_result)
                    if not assessment.accepted:
                        logger.info(
                            "Rejected %s canonical-work candidate: %s",
                            source.name,
                            assessment.reason,
                        )

            if trailing_sources:
                for source, provider_result in self._lookup_structured_sources(
                    trailing_sources, doi, title, author, year
                ) if not self._identity_settled(graph) else self._skipped_sources(
                    trailing_sources
                ):
                    self._record_discovery_attempt(
                        category="academic_adapter",
                        provider=source.name,
                        result=provider_result,
                        required=_route_blocks_completion(source),
                    )
                    if provider_result.success:
                        assessment = graph.add(provider_result)
                        if not assessment.accepted:
                            logger.info(
                                "Rejected %s canonical-work candidate: %s",
                                source.name,
                                assessment.reason,
                            )

            # Title search often discovers a DOI that was absent from the
            # student's reference.  Promote only a high-confidence or
            # independently corroborated DOI, then give DOI-capable adapters a
            # chance to expose their full-text locations.  Previously the DOI
            # stayed trapped in metadata and the acquisition routes continued
            # searching with ``doi=None``.
            enriched_doi = doi or _trusted_graph_doi(graph)
            if not doi and enriched_doi:
                self.prefetch_deferred_dois([enriched_doi])
                doi_sources = [
                    source
                    for source in structured_sources
                    if "doi" in getattr(source, "capabilities", frozenset())
                ]
                for source, provider_result in self._lookup_structured_sources(
                    doi_sources, enriched_doi, title, author, year
                ):
                    self._record_discovery_attempt(
                        category="academic_adapter",
                        provider=source.name,
                        result=provider_result,
                        # Was hardcoded True, so an adapter declared optional
                        # still blocked completion whenever it ran by DOI --
                        # 6 Semantic Scholar attempts in the 2026-09-23
                        # corpus run. The adapter's declaration decides.
                        required=_route_blocks_completion(source),
                    )
                    if provider_result.success:
                        graph.add(provider_result)

            self._title_search_when_doi_unconfirmed(
                structured_sources, doi, title, author, year, expected_kind.kind, graph)

            merged_result = graph.to_result()
            if merged_result.success:
                merged_result = self._download_and_cache(
                    _CANONICAL_GRAPH_SOURCE,
                    merged_result,
                    enriched_doi,
                    title,
                    author,
                    year,
                    expected_source_kind=expected_kind,
                )
                self._record_canonical_location_outcomes(merged_result)
                if merged_result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return self._finalize_resolution_result(
                        merged_result,
                        doi,
                        title,
                        author,
                        year,
                        url_failure_reason,
                    )
                if merged_result.abstract:
                    best_abstract_result = merged_result

            # Unknown-year/no-DOI literary works still get the edition route,
            # but only after structured metadata avoids an unnecessary catalog
            # query for ordinary modern references.
            if (
                not tried_public_domain
                and not enriched_doi
                and self._public_domain_fallback_allowed(expected_kind, year)
            ):
                result = self._try_source_sequence(
                    public_domain_sources, doi, title, author, year,
                    expected_source_kind=expected_kind,
                )
                if result and result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return self._finalize_resolution_result(
                        result, doi, title, author, year, url_failure_reason
                    )
                if result and result.abstract and best_abstract_result is None:
                    best_abstract_result = result

            # Structured metadata above and direct DOI resolution precede paid
            # DOI search. A supplied DOI URL already visited need not be fetched
            # again. Identity-only bibliography resolution remains metadata-only.
            from urllib.parse import unquote
            parsed_student_url = urlparse(student_url or "")
            doi_url_visited = bool(doi and
                parsed_student_url.hostname in {"doi.org", "dx.doi.org"} and
                unquote(parsed_student_url.path).lstrip('/').casefold() == doi.casefold() and
                (capabilities is None or "student_url" in capabilities))
            direct = None
            if (doi and not settings.DOI_RESOLVER_URL and not doi_url_visited
                    and (capabilities is None or "doi_resolver" in capabilities)):
                direct = self._try_doi_resolver(doi, title,
                    expected_source_kind=expected_kind, public=True,
                    expected_author=author, expected_year=year)
                self._record_discovery_attempt(category="student_url",
                    provider="public_doi", result=direct, required=False)
                if direct.success and direct.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return self._finalize_resolution_result(
                        direct, doi, title, author, year, url_failure_reason)
            # MDPI blocks automated requests to its site, including the page a
            # DOI resolves to; its own file host serves the open-access PDF.
            mdpi_blocked = bool(direct is not None and (direct.metadata or {}).get(
                "http_status") in MDPI_BLOCKED_STATUSES) or bool(
                doi_url_visited and re.search(r"(?:http_|status=)(?:401|403|451)\b", url_failure_reason or ""))
            if (doi and is_mdpi_doi(doi) and mdpi_blocked
                    and (capabilities is None or "doi_resolver" in capabilities)):
                mdpi = self._try_mdpi_file_server(doi, title, author, year, expected_kind)
                if mdpi is not None:
                    self._record_discovery_attempt(category="student_url",
                        provider=MDPI_FILE_SERVER_ROUTE, result=mdpi, required=False)
                    if mdpi.success and mdpi.full_text:
                        self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                        return self._finalize_resolution_result(
                            mdpi, doi, title, author, year, url_failure_reason)

            # A completed paid web search that found no source is reused in
            # this scope for SEARCH_REUSE_PAUSE_DAYS (search-reuse-memo-v1).
            # Everything above, including every free academic adapter, has
            # already run; only the paid web tier is skipped.
            if web_sources and self._reuse_search_memo():
                web_sources = []
            for source in web_sources:
                result = self._try_source(
                    source, enriched_doi, title, author, year,
                    expected_source_kind=expected_kind,
                )
                self._record_discovery_attempt(
                    category="bounded_web",
                    provider=source.name,
                    result=result,
                    required=True,
                )
                if result.metadata and result.metadata.get("retrieval_trace"):
                    web_retrieval_trace.extend(result.metadata["retrieval_trace"])
                if result.success and result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return self._finalize_resolution_result(
                        result, doi, title, author, year, url_failure_reason
                    )
                if result.success and result.abstract and best_abstract_result is None:
                    best_abstract_result = result

        # 4. No full text found — abstract fallback.
        if best_abstract_result is not None:
            logger.info(
                "No acquired representation for doi=%s title=%s, but abstract available from %s",
                doi, title, best_abstract_result.source_name,
            )
            if url_failure_reason:
                best_abstract_result.metadata = best_abstract_result.metadata or {}
                best_abstract_result.metadata["url_failure_reason"] = url_failure_reason
            if web_retrieval_trace:
                best_abstract_result.metadata = best_abstract_result.metadata or {}
                best_abstract_result.metadata["supplemental_retrieval_trace"] = (
                    web_retrieval_trace
                )
            if lookup_cache is not None:
                lookup_cache.put_abstract(
                    best_abstract_result,
                    doi=doi,
                    title=title,
                    author=author,
                    year=year,
                    student_url=student_url,
                    policy_signature=policy_signature,
                    source_kind=expected_kind.kind,
                )
            return best_abstract_result

        # If a refresh attempt failed, retain the prior abstract as explicitly
        # stale evidence rather than replacing it with a negative result.
        if (
            lookup_record
            and lookup_record.result is not None
            and not lookup_record_bypassed_for_trace
        ):
            lookup_record.result.metadata = lookup_record.result.metadata or {}
            lookup_record.result.metadata["lookup_cache_stale"] = True
            if url_failure_reason:
                lookup_record.result.metadata["url_failure_reason"] = url_failure_reason
            return lookup_record.result

        if lookup_cache is not None:
            lookup_cache.put_miss(
                doi=doi,
                title=title,
                author=author,
                year=year,
                student_url=student_url,
                policy_signature=policy_signature,
                source_kind=expected_kind.kind,
            )

        raise SourceResolutionError(
            f"Source not found: doi={doi}, isbn={isbn}, title={title}"
            + (f" (URL failed: {url_failure_reason})" if url_failure_reason else ""),
            retrieval_trace=web_retrieval_trace,
        )

    def _lookup_policy_signature(self) -> str:
        """Hash the acquisition context that determines lookup freshness."""
        from app.services.search.brave import BRAVE_RESPONSE_CONTRACT
        from app.services.retrieval.internet_archive import VERSION as ARCHIVE_FILE_CONTRACT
        capabilities = getattr(self, "_acquisition_capabilities", None)
        capability_part = "all" if capabilities is None else ",".join(sorted(capabilities))
        provider_parts = []
        for source in getattr(self, "_retrieval_sources", []):
            source_capabilities = getattr(source, "capabilities", frozenset())
            if not isinstance(source_capabilities, (set, frozenset, list, tuple)):
                source_capabilities = ()
            provider_parts.append(
                f"{source.name}:{','.join(sorted(source_capabilities))}"
            )
        providers = ";".join(provider_parts)
        context = "|".join(
            (
                settings.RETRIEVAL_ACCESS_REVISION,
                settings.SEARCH_POLICY_VERSION,
                BRAVE_RESPONSE_CONTRACT,
                ARCHIVE_FILE_CONTRACT,
                "landing-metadata-identity-v1",
                "landing-title-evidence-v1",
                "landing-undated-identity-v1",
                "identity-only-landing-metadata-v1",
                "identity-only-pdf-metadata-v1",
                "candidate-purpose-ranking-v1",
                "author-etal-comparison-v1",
                "supplied-doi-direct-before-search-v1",
                "search-inspection-capacity-v4",
                ELAPSED_POLICY,
                "journal-registration-title-identity-v1",
                "reference-credibility-v4",
                "metadata-search-observations-v1",
                "screened-metadata-work-binding-v1",
                "corroborated-metadata-identity-v1",
                "bounded-reference-review-v9",
                "publisher-preview-limited-v1",
                "journal-visible-metadata-body-v1",
                "journal-title-author-near-year-v1",
                "provisional-short-title-exclusion-v1",
                str(settings.SEARCH_SEARXNG_FALLBACK_ENABLED),
                str(settings.BRAVE_SEARCH_RETENTION_PERMITTED),
                str(settings.BRAVE_SEARCH_TRANSIENT_ENABLED),
                settings.SEARCH_ESCALATION_MAX_CALLS,
                capability_part,
                providers,
                settings.RETRIEVAL_PROVIDER_CONFIG,
                "doi_resolver" if settings.DOI_RESOLVER_URL else "no_doi_resolver",
            )
        )
        return hashlib.sha256(context.encode("utf-8")).hexdigest()

    def _delete_lookup_cache(
        self,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        source_kind: str | None = None,
    ) -> None:
        lookup_cache = getattr(self, "_lookup_cache", None)
        if lookup_cache is not None:
            lookup_cache.delete(
                doi=doi,
                title=title,
                author=author,
                year=year,
                source_kind=source_kind,
            )

    def prefetch_title_candidates(
        self,
        queries: list[tuple[str, str | None]],
    ) -> dict[str, int]:
        """Run optional provider batch candidate generation for title-only work.

        Only adapters declaring ``batch_title_candidates`` participate. Their
        ordinary per-reference title method remains the correctness fallback
        for every query the grouped candidate set does not resolve.
        """
        outcomes: dict[str, int] = {}
        for source in self._retrieval_sources:
            capabilities = getattr(source, "capabilities", frozenset())
            prefetch = getattr(source, "prefetch_titles", None)
            if "batch_title_candidates" in capabilities and prefetch:
                outcomes[source.name] = prefetch(queries)
        return outcomes

    def lookup_refresh_due(
        self,
        *,
        doi: str | None = None,
        title: str | None = None,
        author: str | None = None,
        year: str | None = None,
        student_url: str | None = None,
        source_kind: str | None = None,
    ) -> bool:
        """Return whether external metadata/full-text discovery should run."""
        lookup_cache = getattr(self, "_lookup_cache", None)
        if lookup_cache is None:
            return True
        normalized_url = _normalize_cited_url(student_url) if student_url else None
        record = lookup_cache.get(
            doi=doi,
            title=title,
            author=author,
            year=year,
            source_kind=source_kind,
        )
        return not (
            record
            and record.is_fresh(
                policy_signature=self._lookup_policy_signature(),
                student_url=normalized_url,
            )
        )

    def prefetch_deferred_dois(self, dois: list[str]) -> dict[str, int]:
        """Batch only unresolved DOI work through installed deferred adapters.

        Assessment orchestration calls this after its normal adapter pass, then
        retries those unresolved references. Single-reference resolution never
        expands a deferred provider into individual network requests.
        """
        outcomes: dict[str, int] = {}
        due_dois = [doi for doi in dois if self.lookup_refresh_due(doi=doi)]
        for source in self._retrieval_sources:
            if not getattr(source, "deferred", False):
                continue
            prefetch = getattr(source, "prefetch_dois", None)
            if prefetch and due_dois:
                outcomes[source.name] = prefetch(due_dois)
        return outcomes

    def resolve_deferred_doi(
        self,
        doi: str,
        title: str | None = None,
        author: str | None = None,
        year: str | None = None,
    ) -> RetrievalResult:
        """Resolve a prefetched DOI only through deferred adapters."""
        best_abstract: RetrievalResult | None = None
        for source in self._retrieval_sources:
            if not getattr(source, "deferred", False):
                continue
            result = self._try_source(source, doi, None, author, year)
            if result.success and result.full_text:
                return result
            if result.success and result.abstract and best_abstract is None:
                best_abstract = result
        if best_abstract is not None:
            return best_abstract
        raise SourceResolutionError(f"Deferred source not found: doi={doi}, title={title}")

    # ── Internal helpers ───────────────────────────────────

    @_timed_retrieval
    def _check_local_cache(
        self,
        doi: str | None,
        isbn: str | None,
        title: str | None,
        *,
        author: str | None = None,
        year: str | None = None,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
    ) -> RetrievalResult:
        """Check if the document is already in S3.

        Documents held for review (review_status != "accepted") are skipped
        so they're not used for citation verification until approved.
        """
        if not self._backend:
            return RetrievalResult(
                source_name="local_cache", success=False,
                error="Source storage unavailable; reuse lookup not attempted",
            )
        session_factory = getattr(self, "_repository_session_factory", None)
        if settings.SOURCE_REPOSITORY_ENABLED and session_factory is not None:
            with session_factory() as session:
                record = find_accepted_representation(
                    session,
                    scope_type="personal_owner",
                    scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
                    doi=doi,
                    isbn=isbn,
                    title=title,
                    work_type=expected_source_kind.kind,
                )
                edition_title_fallback = False
                # An edition qualifier is not part of the work's title. This
                # only finds a candidate; recheck the unchanged citation below.
                base_title = re.sub(r'\s*\(\d+(?:st|nd|rd|th)\s+ed\.?\)\s*$', '', title or '', flags=re.I)
                if record is None and not doi and not isbn and base_title and base_title != title and author and year:
                    record = find_accepted_representation(session,
                        scope_type="personal_owner", scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
                        title=base_title, work_type=expected_source_kind.kind)
                    if record is not None:
                        edition_title_fallback = True
                        if (record.canonical_work.year != year or
                                _normalize_cache_author(record.canonical_work.author) != _normalize_cache_author(author)):
                            record = None
                if record is None:
                    return RetrievalResult(source_name="local_cache", success=False)
                try:
                    data = self._backend.download(record.content_object.storage_key)
                except FileNotFoundError:
                    logger.error(
                        "Accepted durable representation %s points to missing object %s",
                        record.id,
                        record.content_object.storage_key,
                    )
                    return RetrievalResult(
                        source_name="local_cache", success=False,
                        error="Accepted source object unavailable",
                    )
                kind = RepresentationKind(record.representation_kind)
                if record.completeness_verdict == 'incomplete':
                    from app.services.publisher_preview import valid_preview_receipt
                    if (not author or not year or record.canonical_work.year != year
                            or _normalize_cache_author(record.canonical_work.author) != _normalize_cache_author(author)
                            or not valid_preview_receipt((record.validation_evidence or {}).get('publisher_preview'),
                                hashlib.sha256(data).hexdigest(), record.source_url)):
                        return RetrievalResult(source_name='local_cache', success=False,
                            error='Limited preview binding could not be revalidated')
                if edition_title_fallback:
                    if kind != RepresentationKind.PDF:
                        return RetrievalResult(source_name="local_cache", success=False)
                    validation = validate_retrieved_pdf(data, expected_title=title,
                        expected_author=author, expected_year=year, document_kind="book",
                        expected_source_kind=expected_source_kind.kind,
                        expected_source_kind_confidence=expected_source_kind.confidence,
                        expected_source_kind_evidence=expected_source_kind.evidence)
                    if (validation.identity_confidence == "rejected" or validation.completeness == "incomplete"
                            or not _accepted_copy_has_edition_label(data, title)):
                        return RetrievalResult(source_name="local_cache", success=False,
                            error="Edition-qualified library candidate could not be revalidated")
                original_kind = (
                    RepresentationKind(record.original_kind)
                    if record.original_kind else None
                )
                return RetrievalResult(
                    source_name="local_cache",
                    success=True,
                    representation=SourceRepresentation(
                        kind=kind,
                        media_type=record.content_object.media_type,
                        content=data,
                        source_url=record.source_url,
                        original_kind=original_kind,
                        completeness=record.completeness_verdict,
                    ),
                    doi=_registration_bound_doi(record, doi),
                    title=record.canonical_work.display_title,
                    year=record.canonical_work.year,
                    authors=(
                        [record.canonical_work.author]
                        if record.canonical_work.author else []
                    ),
                    metadata={
                        "repository_representation_id": str(record.id),
                        "identity_confidence": record.identity_confidence,
                        "admission_state": record.admission_state,
                        "provenance": record.provenance,
                        "source_kind": record.canonical_work.work_type,
                        "edition_or_version": record.edition_or_version,
                        "publisher_preview": (record.validation_evidence or {}).get('publisher_preview'),
                    },
                )
        for key in self._build_cache_keys(doi, isbn, title):
            # Status gate: exclude documents not yet accepted for use.
            if self._review_status_for_key(key) != "accepted":
                continue
            try:
                data = self._backend.download(key)
                if not data.startswith(PDF_MAGIC):
                    logger.warning(
                        "Ignoring legacy cache object %s: content is not a PDF",
                        key,
                    )
                    continue
                validation = validate_retrieved_pdf(
                    data,
                    expected_doi=doi,
                    expected_title=title,
                    expected_author=author,
                    expected_year=year,
                    expected_isbn=isbn,
                    document_kind=(
                        document_kind_for_source_kind(expected_source_kind.kind)
                        if expected_source_kind.is_known
                        else "article" if doi else "unknown"
                    ),
                    expected_source_kind=expected_source_kind.kind,
                    expected_source_kind_confidence=expected_source_kind.confidence,
                    expected_source_kind_evidence=expected_source_kind.evidence,
                )
                if (
                    validation.identity_confidence != "high"
                    or validation.completeness == "incomplete"
                ):
                    logger.warning(
                        "Ignoring unverified legacy cache object %s: identity=%s "
                        "completeness=%s",
                        key,
                        validation.identity_confidence,
                        validation.completeness,
                    )
                    continue
                return RetrievalResult(
                    source_name="local_cache",
                    success=True,
                    full_text=data,
                    doi=doi,
                    title=title,
                    year=year,
                    authors=[author] if author else [],
                    metadata={
                        "identity_confidence": validation.identity_confidence,
                        "identity_reason": validation.reason,
                        "completeness": validation.completeness,
                        "legacy_cache_revalidated": True,
                    },
                )
            except FileNotFoundError:
                continue
        return RetrievalResult(source_name="local_cache", success=False)

    def _review_status_for_key(self, key: str) -> str:
        """Look up the review_status of a stored document by its S3 key.

        Returns "accepted" when no registry entry exists (e.g. documents
        cached by earlier versions, or retrieval-source downloads that
        bypass the instructor registry). This keeps the gate permissive
        for content not tracked in the in-memory store.
        """
        # Legacy identifier-keyed objects predate durable admission metadata.
        # They remain readable only while the durable repository is disabled.
        return "accepted"

    def _build_cache_keys(
        self,
        doi: str | None,
        isbn: str | None,
        title: str | None,
    ) -> list[str]:
        """Build potential S3 keys for the document."""
        keys: list[str] = []
        if doi:
            keys.append(f"by-doi/{doi}.pdf")
        if isbn:
            keys.append(f"by-isbn/{isbn}.pdf")
        if title:
            h = hashlib.sha256(title.strip().lower().encode()).hexdigest()[:12]
            keys.append(f"by-title-hash/{h}.pdf")
        return keys

    @_timed_retrieval
    def _try_student_url(
        self,
        url: str,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
    ) -> RetrievalResult:
        """Download and verify a student-linked URL.

        Identity-gated like the academic-DB path (REVIEW §2.3/§3.3): the
        downloaded PDF is validated with the consolidated triangulation check
        (DOI + title + author + year) before it is accepted or cached. This
        replaces the older, weaker _verify_pdf_metadata gate (DOI substring or
        title 50-char prefix) so the student-URL path cannot poison the cache
        with a wrong PDF any more than the academic-DB path can.
        """
        try:
            data = self._safe_download(url)
        except Exception as e:
            logger.warning(
                "Student URL download failed for %s: %s",
                private_value_id("url", url),
                type(e).__name__,
            )
            return RetrievalResult(
                source_name="student_url",
                success=False,
                error=f"Student URL download failed ({type(e).__name__})",
            )

        result = RetrievalResult(
            source_name="student_url",
            success=True,
            representation=SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=data,
                source_url=url,
            ),
            doi=doi,
            title=title,
            year=year,
            authors=[author] if author else [],
        )
        accepted, outcome, reason = self._preflight_acquired_representation(
            result,
            expected_doi=doi,
            expected_title=title,
            expected_author=author,
            expected_year=year,
            expected_source_kind=expected_source_kind,
        )
        if not accepted:
            return RetrievalResult(
                source_name="student_url",
                success=False,
                error=f"Student-linked PDF preflight failed: {outcome}",
                doi=doi,
                title=title,
                metadata={**(result.metadata or {}), "preflight_reason": reason},
            )
        if result.parent_representation is not None:
            self._persist_retrieved_representation(
                result,
                ref_doi=doi,
                ref_title=title,
                ref_author=author,
                ref_year=year,
                identity_confidence="high",
                identity_reason=result.metadata.get(
                    "identity_reason", "accepted OCR derivative"
                ),
                downloaded_via_publisher=False,
                safety_report=None,
                expected_source_kind=expected_source_kind,
            )
            if result.metadata.get("durable_admission", {}).get("state") != "accepted":
                self._clear_retrieved_representation(result)
                return RetrievalResult(
                    source_name="student_url",
                    success=False,
                    error="OCR derivative could not enter durable admission",
                    doi=doi,
                    title=title,
                    metadata=result.metadata,
                )
        elif result.metadata.get('publisher_preview') and self._durable_repository_active():
            self._persist_retrieved_representation(result, ref_doi=doi, ref_title=title,
                ref_author=author, ref_year=year, identity_confidence='high',
                identity_reason=result.metadata.get('identity_reason', ''), downloaded_via_publisher=False,
                safety_report=inspect_uploaded_pdf(data), expected_source_kind=expected_source_kind)
        elif self._backend and result.metadata.get("completeness") == "complete":
            key = self._cache_key_for_verified(doi, title)
            self._backend.upload(data, key)
        return result

    def _safe_download(self, url: str, *, timeout: float | None = None,
                       headers: dict[str, str] | None = None) -> bytes:
        """Download a URL with SSRF + size-cap protection and verify it is a PDF.

        Delegates to safe_fetch (REVIEW §2.1/§2.2): rejects non-public hosts
        (loopback/private/link-local, including cloud-metadata endpoints),
        caps the body at STUDENT_URL_MAX_SIZE_MB, and follows redirects one hop
        at a time with per-hop re-validation. Then verifies the body is a PDF.

        Error-string convention: a genuine non-PDF (HTML paywall/login page)
        raises a ValueError whose message contains "not a PDF" — resolve()
        matches on that to route the URL to the web_fetch branch. SSRF blocks
        and size-limit failures deliberately do NOT contain "not a PDF", so a
        blocked internal URL is never silently re-fetched as a web page.
        """
        require_source_candidate(url)
        if timeout is None:
            timeout = settings.STUDENT_URL_TIMEOUT_SECONDS
        max_bytes = settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024
        for transport_attempt in range(2):
            try:
                data = safe_fetch_bytes(
                    url,
                    usage_label="file download",
                    max_bytes=max_bytes,
                    accept_content_types=("application/pdf",),
                    timeout=timeout,
                    max_meta_refreshes=1,
                    **({"headers": headers} if headers else {}),
                )
                break
            except UnsafeUrlError as e:
                raise ValueError(f"URL blocked (SSRF guard): {e}") from e
            except ResponseTooLargeError as e:
                raise ValueError(f"File too large: {e}") from e
            except ValueError as e:
                # Content-type mismatch (HTML page served where a PDF was expected).
                raise ValueError(f"Not a PDF: {e}") from e
            except httpx.TransportError as e:
                if transport_attempt == 0:
                    logger.info(
                        "Retrying PDF once after transient transport failure: %s",
                        private_value_id("url", url),
                    )
                    continue
                raise ValueError(f"Download failed ({type(e).__name__})") from e
            except httpx.HTTPError as e:
                # Status errors are deterministic for this attempt and do not
                # consume the one retry reserved for connection failures.
                # Preserve the observed numeric status: a later diagnosis must
                # be able to distinguish access restriction from an ordinary
                # unavailable response without guessing at the host.
                status = getattr(getattr(e, "response", None), "status_code", None)
                detail = f"{type(e).__name__}" + (f" status={status}" if status else "")
                raise ValueError(f"Download failed ({detail})") from e

        if not data.startswith(PDF_MAGIC):
            raise ValueError("Downloaded content is not a PDF (may be paywall or HTML page)")
        return data

    def _try_oa_landing_page(self, url: str) -> bytes | None:
        """Handle OA URLs that return HTML landing pages instead of direct PDFs.

        Many OA repositories serve a metadata/landing page with a "Download PDF"
        link rather than serving the PDF directly. This method fetches the page,
        finds the actual PDF link, and downloads it.

        Detection methods (in priority order):
        1. <meta name="citation_pdf_url" content="..."> — academic standard
           (used by most repositories, publishers, and preprint servers)
        2. <a> tags with href ending in .pdf
        3. <a> tags with text containing "PDF", "Download", "Full Text", "View/Open"

        Returns PDF bytes if found, None if no PDF link on the page.
        """
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        try:
            require_source_candidate(url)
            resp = safe_request(
                url,
                usage_label="landing page",
                headers={"Accept": "text/html,application/xhtml+xml,*/*"},
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug(
                "OA landing page fetch failed for %s: %s",
                private_value_id("url", url),
                type(e).__name__,
            )
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_url = None

        # Method 1: citation_pdf_url meta tag (highest confidence — academic standard)
        meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta and meta.get("content"):
            pdf_url = meta["content"]
            logger.info(
                "Found PDF via citation_pdf_url meta tag: %s",
                private_value_id("url", pdf_url),
            )

        # Method 2: <a> tags with href ending in .pdf
        if not pdf_url:
            for a in soup.find_all("a", href=True):
                href = a["href"].lower()
                if href.endswith(".pdf") or ".pdf?" in href:
                    pdf_url = urljoin(str(resp.url), a["href"])
                    logger.info(
                        "Found PDF via .pdf link: %s",
                        private_value_id("url", pdf_url),
                    )
                    break

        # Method 3: <a> tags with PDF-related text
        if not pdf_url:
            pdf_keywords = ("full text", "download pdf", "view/open", "download article",
                            "open access", "get pdf", "pdf download")
            for a in soup.find_all("a"):
                text = a.get_text(strip=True).lower()
                href = a.get("href", "")
                if any(kw in text for kw in pdf_keywords) and href:
                    pdf_url = urljoin(str(resp.url), href)
                    logger.info(
                        "Found PDF via bounded link text; %s",
                        private_value_id("url", pdf_url),
                    )
                    break

        if not pdf_url:
            return None

        # Download the found PDF URL (SSRF-guarded + size-capped via safe_fetch;
        # pdf_url comes from the parsed landing-page HTML, so it must be
        # re-validated like any other attacker-influenced URL).
        try:
            require_source_candidate(pdf_url)
            pdf_data = safe_fetch_bytes(
                pdf_url,
                usage_label="file download",
                accept_content_types=("application/pdf",),
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                max_meta_refreshes=1,
            )
            if pdf_data[:5] == PDF_MAGIC:
                logger.info("OA landing page PDF download successful: %d bytes", len(pdf_data))
                return pdf_data
        except Exception as e:
            logger.debug(
                "OA landing page PDF download failed: %s", type(e).__name__
            )

        return None

    def _inspect_candidate(self, content: bytes, *, title=None, author=None, year=None,
                           doi=None, source_kind='unknown', pdf=False) -> dict | None:
        """Permission-bounded observations only; no alternate admission path."""
        if expired():
            return {'status': 'not_run', 'reason_code': 'reference_elapsed_budget_timeout', 'decision_applied': False}
        if not settings.SOURCE_INSPECTION_ENABLED:
            return None
        from app.services.source_identity_confirmer import (
            ExpectedSourceIdentity, SourceIdentityEvidenceBundle, SourceIdentityEvidenceItem,
            build_source_identity_evidence, inspect_source_observations,
        )
        from app.services.llm_service import chat_completion_json
        endpoint = urlparse(settings.LLM_BASE_URL or '')
        local = endpoint.hostname in {'localhost', '127.0.0.1', '::1'}
        if (getattr(self, '_acquisition_capabilities', None) is not None
                or (not local and (not settings.SOURCE_INSPECTION_REMOTE_ALLOWED
                    or settings.REPORT_AUTH_MODE not in {'personal_local', 'personal_bearer'}))):
            return {'status': 'not_run', 'reason_code': 'processing_not_authorized', 'decision_applied': False}
        from app.services.llm_input_boundary import redact_direct_identifiers
        submitted = (_ACTIVE_DISCOVERY_TRACE.get() or {}).get('submitted_reference', '')
        submitted = re.sub(r'https?://\S+', '[submitted link omitted]', submitted)
        expected = ExpectedSourceIdentity(source_id=hashlib.sha256(content).hexdigest()[:32],
            title=title or 'Unknown', author_or_contributors=author, year=year, doi=doi,
            source_kind=source_kind,
            submitted_reference=redact_direct_identifiers(submitted[:4000]).text or None)
        import json
        from app.services.source_identity_confirmer import INSPECTION_VERSION
        cache_context = json.dumps({
            'model': settings.LLM_MODEL, 'endpoint': settings.LLM_BASE_URL,
            'contract': INSPECTION_VERSION, 'pdf': pdf,
            'ocr': settings.PURE_SCAN_OCR_ENABLED,
        }, sort_keys=True).encode()
        key = hashlib.sha256(content+expected.model_dump_json().encode()+cache_context).hexdigest()
        cache = getattr(self, '_source_inspection_cache', None)
        if cache is None:
            cache = self._source_inspection_cache = {}
        if key in cache:
            return cache[key]
        if len(cache) >= max(0, min(settings.SOURCE_INSPECTION_MAX_CALLS, 12)):
            return {'status': 'not_run', 'reason_code': 'inspection_budget_exhausted', 'decision_applied': False}
        cache[key] = {'status': 'incomplete', 'reason_code': 'inspection_unavailable', 'decision_applied': False}
        try:
            limitations = []
            if pdf:
                # Reuse the established book front-matter window when the
                # native first-three-page title check is inconclusive.
                from app.services.source_validator import _has_prominent_front_title_support
                front_limit = (6 if source_kind in {'monograph', 'edited_collection'}
                    and not _has_prominent_front_title_support(content, title) else 3)
                from app.services.ocr_derivative import OcrDerivativeError
                try:
                    bundle = build_source_identity_evidence(content, expected,
                        include_ocr=settings.PURE_SCAN_OCR_ENABLED, inspection_v3=True,
                        front_page_limit=front_limit)
                except OcrDerivativeError as exc:
                    if str(exc) != 'Identity observation source exceeds its bound.':
                        raise
                    limitations.append('supplementary_ocr_source_size_limit')
                    bundle = build_source_identity_evidence(content, expected,
                        include_ocr=False, inspection_v3=True, front_page_limit=front_limit)
            else:
                from app.services.llm_input_boundary import redact_direct_identifiers
                text = redact_direct_identifiers(content.decode('utf-8', errors='strict')[:2500]).text
                bundle = SourceIdentityEvidenceBundle(content_sha256=hashlib.sha256(content).hexdigest(),
                    page_count=1, expected=expected,
                    evidence=[SourceIdentityEvidenceItem(evidence_id='e000', role='web_header', text=text)])
            result = inspect_source_observations(bundle,
                response_provider=lambda system, user: chat_completion_json(system, user,
                    max_tokens=1600, max_retries=0, disable_thinking=True),
                processing_boundary='local' if local else 'authorized_remote')
            result['model'] = settings.LLM_MODEL
            result['endpoint_host'] = endpoint.hostname
            result['extraction_limitations'] = limitations
            cache[key] = result
        except Exception:
            pass
        return cache[key]

    def _preflight_acquired_representation(
        self,
        result: RetrievalResult,
        *,
        expected_doi: str | None,
        expected_title: str | None,
        expected_author: str | None,
        expected_year: str | None,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
        advertised_by_confirmed_landing: bool = False,
    ) -> tuple[bool, str, str]:
        """Validate one acquired representation before accepting its location.

        A transport success is not a source success.  This method runs the
        hostile-file, identity and completeness boundaries while the graph
        still has alternate locations available.  Only high-confidence source
        identity can become verification evidence; bounded uncertainty remains
        an inspectable failed attempt rather than an accepted representation.
        """
        representation = result.representation
        if representation is None:
            return False, "unavailable", "location produced no representation"

        metadata = result.metadata or {}
        result.metadata = metadata
        kind = representation.kind
        content = representation.content

        provider_kind = classify_provider_source_kind(metadata)
        provider_compatibility = compare_source_kinds(
            expected_source_kind, provider_kind
        )
        if (
            provider_compatibility.verdict == "incompatible"
            and not _doi_identity_matches(expected_doi, result.doi)
        ):
            metadata["expected_source_kind"] = expected_source_kind.kind
            metadata["observed_source_kind"] = provider_kind.kind
            metadata["source_kind_verdict"] = "incompatible"
            return False, "type_rejected", provider_compatibility.reason[:160]

        if kind is RepresentationKind.PDF:
            safety_report = None
            if self._durable_repository_active():
                try:
                    safety_report = inspect_uploaded_pdf(content)
                except FileSafetyUnavailable:
                    source_validation_observed(representation.source_url, hashlib.sha256(content).hexdigest(), 'safety_unavailable')
                    return False, "safety_unavailable", "malware_scanner_unavailable"
                if safety_report.verdict is SafetyVerdict.REJECTED:
                    source_validation_observed(representation.source_url, hashlib.sha256(content).hexdigest(), 'safety_rejected')
                    reason = "; ".join(safety_report.findings) or "hostile-file rejection"
                    return False, "safety_rejected", reason[:160]

            validation = validate_retrieved_pdf(
                content,
                inspection_reconciliation=True,
                inspection_provider=(lambda data, validation: self._inspect_candidate(data,
                    title=expected_title, author=expected_author, year=expected_year,
                    doi=expected_doi, source_kind=expected_source_kind.kind, pdf=True))
                    if safety_report and safety_report.verdict is SafetyVerdict.CLEAN else None,
                expected_doi=expected_doi,
                expected_title=expected_title,
                expected_author=expected_author,
                expected_year=expected_year,
                document_kind=_document_kind_for_result(
                    result,
                    has_doi=bool(expected_doi or result.doi),
                    expected_source_kind=expected_source_kind.kind,
                ),
                expected_page_range=_expected_page_range(result),
                expected_source_kind=expected_source_kind.kind,
                expected_source_kind_confidence=expected_source_kind.confidence,
                expected_source_kind_evidence=expected_source_kind.evidence,
            )
            identity_confidence = validation.identity_confidence
            if getattr(validation, 'reason_code', None) == 'identity_insufficient_observations':
                identity_confidence = 'low'
                metadata['identity_reason_code'] = validation.reason_code
            if getattr(validation, 'source_inspection', None) is not None:
                metadata['source_inspection'] = validation.source_inspection
            identity_reason = validation.reason
            completeness = getattr(validation, "completeness", "not_assessed")
            from app.services.publisher_preview import publisher_preview_url
            if publisher_preview_url(representation.source_url):
                completeness = 'incomplete'
            text_quality = getattr(validation, "text_quality", "not_assessed")
            representation.completeness = completeness
            observed_kind = SourceKindAssessment(
                getattr(validation, "observed_source_kind", "unknown"),
                "high"
                if getattr(validation, "source_kind_verdict", "unknown")
                == "incompatible"
                else "unknown",
            )
            kind_verdict = getattr(validation, "source_kind_verdict", "unknown")
            if text_quality == "pure_scan" and settings.PURE_SCAN_OCR_ENABLED:
                if not self._durable_repository_active():
                    return (
                        False,
                        "ocr_repository_required",
                        "Pure-scan OCR requires the durable source repository.",
                    )
                prepared = prepare_pure_scan_ocr(
                    content,
                    safety_verified=bool(
                        safety_report
                        and safety_report.verdict is SafetyVerdict.CLEAN
                    ),
                    source_url=representation.source_url,
                    expected_doi=expected_doi,
                    expected_title=expected_title,
                    expected_author=expected_author,
                    expected_year=expected_year,
                    document_kind=_document_kind_for_result(
                        result,
                        has_doi=bool(expected_doi or result.doi),
                        expected_source_kind=expected_source_kind.kind,
                    ),
                    expected_page_range=_expected_page_range(result),
                    expected_source_kind=expected_source_kind.kind,
                    expected_source_kind_confidence=expected_source_kind.confidence,
                    expected_source_kind_evidence=expected_source_kind.evidence,
                )
                if (
                    prepared.status != "ready"
                    or prepared.parent is None
                    or prepared.derivative is None
                    or prepared.derivative_record is None
                    or prepared.validation is None
                ):
                    return False, prepared.status, prepared.reason[:160]
                result.parent_representation = prepared.parent
                result.set_representation(prepared.derivative)
                representation = prepared.derivative
                kind = representation.kind
                content = representation.content
                validation = prepared.validation
                identity_confidence = validation.identity_confidence
                identity_reason = validation.reason
                completeness = validation.completeness
                text_quality = validation.text_quality
                observed_kind = SourceKindAssessment(
                    validation.observed_source_kind,
                    "medium",
                    ("validated local OCR derivative",),
                )
                kind_verdict = validation.source_kind_verdict
                metadata["ocr_derivative"] = {
                    "derivation_method": prepared.derivative.metadata[
                        "ocr_derivative_version"
                    ],
                    "parent_content_sha256": (
                        prepared.derivative_record.parent_content_sha256
                    ),
                    "derivative_content_sha256": (
                        prepared.derivative_record.content_sha256
                    ),
                    "derivation_manifest_sha256": (
                        prepared.derivative_record.manifest_sha256
                    ),
                    "manifest": prepared.derivative_record.manifest,
                    "page_labels": list(prepared.page_labels),
                    "page_mapping_method": prepared.page_mapping_method,
                    "page_mapping_sha256": prepared.page_mapping_sha256,
                    "parent_file_safety": {
                        "verdict": safety_report.verdict.value,
                        "structural_verdict": (
                            safety_report.structural_verdict.value
                        ),
                        "malware_verdict": safety_report.malware_verdict.value,
                    },
                }
        else:
            provider_identity = self._verify_source_identity(
                result,
                expected_doi or result.doi,
                expected_title or result.title,
                expected_author,
                expected_year,
            )
            content_identity, content_reason = _verify_text_content_identity(
                content,
                expected_doi or result.doi,
                expected_title or result.title,
                expected_author,
                expected_year,
            )
            identity_confidence = _combine_identity_confidence(
                provider_identity,
                content_identity,
            )
            identity_reason = (
                f"provider metadata identity={provider_identity}; "
                f"content identity={content_identity} ({content_reason})"
            )
            completeness = representation.completeness or "not_assessed"
            text_quality = "digital"
            representation_kind = representation.metadata.get(
                "observed_source_kind"
            )
            representation_confidence = representation.metadata.get(
                "observed_source_kind_confidence"
            )
            observed_kind = (
                SourceKindAssessment(
                    normalize_source_kind(representation_kind),
                    representation_confidence
                    if representation_confidence in {"high", "medium", "low"}
                    else "unknown",
                    ("acquired HTML structural/content evidence",),
                )
                if representation_kind
                else classify_content_source_kind(
                    content.decode("utf-8", errors="replace"),
                    source_url=representation.source_url,
                )
            )
            content_compatibility = compare_source_kinds(
                expected_source_kind, observed_kind
            )
            kind_verdict = content_compatibility.verdict
            structured_text_type_unconfirmed = bool(
                representation.original_kind
                in {RepresentationKind.HTML, RepresentationKind.XML}
                and _requires_confirmed_non_web_source_kind(expected_source_kind)
                and kind_verdict != "compatible"
                and provider_compatibility.verdict != "compatible"
                # A page whose own title, author and year confirmed the cited
                # work is that work republished online (Hess, 2026-09-30).
                and REPUBLISHED_WORK_EVIDENCE not in (representation.metadata or {}).get(
                    "observed_source_kind_evidence", [])
            )
            if (
                kind_verdict == "incompatible"
                or structured_text_type_unconfirmed
            ):
                metadata["expected_source_kind"] = expected_source_kind.kind
                metadata["observed_source_kind"] = observed_kind.kind
                metadata["source_kind_verdict"] = kind_verdict
                outcome = (
                    "type_unconfirmed"
                    if structured_text_type_unconfirmed
                    else "type_rejected"
                )
                return False, outcome, content_compatibility.reason[:160]

        metadata["identity_confidence"] = identity_confidence
        metadata["identity_reason"] = identity_reason
        metadata["representation_kind"] = kind.value
        metadata["completeness"] = completeness
        metadata["text_quality"] = text_quality
        metadata["expected_source_kind"] = expected_source_kind.kind
        metadata["observed_source_kind"] = (
            observed_kind.kind if observed_kind.is_known else provider_kind.kind
        )
        metadata["source_kind_verdict"] = kind_verdict

        if kind_verdict == "incompatible":
            return False, "type_rejected", identity_reason[:160]
        if kind is RepresentationKind.PDF:
            metadata['submitted_response_identity'] = validated_response_identity(
                representation.source_url, hashlib.sha256(content).hexdigest(), identity_confidence)
        # A record page that named the cited work, and then named its own full
        # text, corroborates the file it points at. Doster's honours thesis
        # opens on a scanned handwritten approval form, so its own bytes can
        # never establish a title; without this the repository's bibliographic
        # record won acquisition and the citation showed limited text with
        # nothing to read. Only a clean, readable PDF of a compatible kind
        # qualifies, and only from the page that advertised it.
        corroborated = (
            advertised_by_confirmed_landing
            and identity_confidence == "medium"
            and kind is RepresentationKind.PDF
            and kind_verdict != "incompatible"
            and text_quality in {"digital", "born_digital", "scan_ocr"}
            and bool(safety_report and safety_report.verdict is SafetyVerdict.CLEAN)
        )
        if corroborated:
            metadata["identity_corroborated_by_landing_page"] = True
            identity_reason = (
                f"{identity_reason}; corroborated by the landing page that names "
                "this work and advertises this file as its full text"
            )
        if identity_confidence != "high" and not corroborated:
            self._retain_provisional_candidate(result, confidence=identity_confidence,
                reason=identity_reason, completeness=completeness,
                text_quality=text_quality, kind_verdict=kind_verdict,
                expected_title=expected_title, expected_author=expected_author,
                expected_year=expected_year)
            outcome = (
                "identity_rejected"
                if identity_confidence == "rejected"
                else "identity_unconfirmed"
            )
            return False, outcome, identity_reason[:160]
        if kind is RepresentationKind.PDF and completeness == "incomplete":
            from app.services.publisher_preview import preview_receipt
            receipt = preview_receipt(content, representation.source_url,
                identity=identity_confidence, completeness=completeness,
                source_kind=expected_source_kind.kind, text_quality=text_quality,
                cleanliness=safety_report.verdict.value if safety_report else 'not_assessed')
            if receipt is None:
                source_validation_observed(representation.source_url, hashlib.sha256(content).hexdigest(), 'completeness_rejected')
                return False, "completeness_rejected", validation.reason[:160]
            metadata['publisher_preview'] = receipt
        if kind is RepresentationKind.PDF and completeness not in {'complete', 'not_applicable'}:
            source_validation_observed(representation.source_url, hashlib.sha256(content).hexdigest(), 'completeness_uncertain')
        metadata["accepted_representation_sha256"] = hashlib.sha256(content).hexdigest()
        return True, "acquired", identity_reason[:160]

    def _acquire_from_locations(
        self,
        result: RetrievalResult,
        *,
        expected_doi: str | None = None,
        expected_title: str | None = None,
        expected_author: str | None = None,
        expected_year: str | None = None,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
        advertised_by_confirmed_landing: bool = False,
    ) -> bool:
        """Try ranked locations until one passes transport and source preflight."""
        if not result.locations and result.full_text_url:
            result.locations = [
                AcquisitionLocation(
                    url=result.full_text_url,
                    provider=result.source_name,
                    is_best=True,
                )
            ]
        ranked = sorted(result.locations, key=_location_rank)
        root_location_urls = {location.url for location in ranked}
        required_root_attempts = min(
            _MIN_ROOT_LOCATION_ATTEMPTS, len(root_location_urls)
        )
        attempts: list[dict] = []
        pending = list(ranked)
        attempted_urls: set[str] = set()
        location_ranks = {
            location.url: rank for rank, location in enumerate(ranked, start=1)
        }

        def rank_for(location: AcquisitionLocation) -> int:
            if location.url not in location_ranks:
                location_ranks[location.url] = len(location_ranks) + 1
            return location_ranks[location.url]

        def record_unattempted(reason_code: str) -> None:
            seen = set(attempted_urls)
            seen.update(str(item.get("url") or "") for item in attempts)
            for remaining in pending:
                if remaining.url in seen:
                    continue
                seen.add(remaining.url)
                attempts.append(
                    {
                        "url": remaining.url,
                        "provider": remaining.provider,
                        "kind": (
                            remaining.representation_kind.value
                            if remaining.representation_kind
                            else None
                        ),
                        "candidate_title": remaining.metadata.get("search_title"),
                        "discovery_provider": remaining.metadata.get(
                            "search_provider"
                        ),
                        "discovery_engine_group": remaining.metadata.get(
                            "search_engine_group"
                        ),
                        "origin_providers": list(
                            remaining.metadata.get("providers")
                            or [remaining.provider]
                        ),
                        "rank": rank_for(remaining),
                        "outcome": "not_attempted",
                        "reason_code": reason_code,
                    }
                )
        html_fallback: tuple[
            str, str, str, SourceKindAssessment, bool | None
        ] | None = None
        def record_attempt(attempt: dict) -> None:
            """Stamp how long this fetch-and-validate actually took.

            Without it a retrieval-timing question can only be answered by
            comparing whole runs, and run-to-run variance on one paper reached
            five sources and seven hundred seconds — larger than any effect
            being measured. A per-fetch duration answers "did a successful
            fetch ever exceed this budget" from a single run.
            """
            started = attempt.pop("_started_monotonic", None)
            if started is not None:
                attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            attempts.append(attempt)

        while pending and len(attempts) < _MAX_LOCATION_ATTEMPTS:
            location = pending.pop(0)
            if location.url in attempted_urls:
                continue
            attempted_urls.add(location.url)
            attempt = {
                "url": location.url,
                "provider": location.provider,
                "kind": location.representation_kind.value
                if location.representation_kind else None,
                "candidate_title": location.metadata.get("search_title"),
                "discovery_provider": location.metadata.get("search_provider"),
                "discovery_engine_group": location.metadata.get(
                    "search_engine_group"
                ),
                "origin_providers": list(
                    location.metadata.get("providers") or [location.provider]
                ),
                "rank": rank_for(location),
                "_started_monotonic": time.monotonic(),
            }
            if location.metadata.get("discovered_from_location_sha256"):
                attempt["discovered_from_location_sha256"] = location.metadata["discovered_from_location_sha256"]
            if location.metadata.get("discovered_from_response_sha256"):
                attempt["discovered_from_response_sha256"] = location.metadata["discovered_from_response_sha256"]
            try:
                require_source_candidate(location.url)
                from app.services.retrieval.internet_archive import item_identifier, public_pdf_locations
                if item_identifier(location.url):
                    discovered = public_pdf_locations(
                        location.url, timeout=settings.DISCOVERY_CANDIDATE_TIMEOUT_SECONDS,
                        max_bytes=settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024,
                    )
                    # Queue behind existing roots; existing attempt and candidate
                    # budgets still govern every file and its validation.
                    for candidate in discovered:
                        candidate = replace(candidate, metadata={
                            **candidate.metadata,
                            'search_provider': location.metadata.get('search_provider'),
                            'search_engine_group': location.metadata.get('search_engine_group'),
                            'search_title': location.metadata.get('search_title'),
                            'discovered_from_location_sha256': hashlib.sha256(location.url.encode()).hexdigest(),
                        })
                        if candidate.url not in attempted_urls:
                            pending.append(candidate)
                            rank_for(candidate)
                    attempt['outcome'] = 'unavailable'
                    attempt['reason_code'] = 'archive_files_discovered' if discovered else 'archive_no_public_pdf'
                    attempt['discovered_urls'] = [c.url for c in discovered]
                    record_attempt(attempt)
                    continue
                if location.representation_kind is RepresentationKind.PDF:
                    data = self._safe_download(
                        location.url,
                        timeout=settings.DISCOVERY_CANDIDATE_TIMEOUT_SECONDS,
                        **({"headers": mdpi_request_headers()}
                           if location.metadata.get("constructed_route") == MDPI_FILE_SERVER_ROUTE else {}),
                    )
                    result.set_representation(
                        SourceRepresentation(
                            kind=RepresentationKind.PDF,
                            media_type="application/pdf",
                            content=data,
                            source_url=location.url,
                        )
                    )
                elif location.representation_kind in (
                    RepresentationKind.XML,
                    RepresentationKind.PLAIN_TEXT,
                ):
                    response = safe_request(
                        location.url,
                        usage_label="discovered location",
                        headers={"Accept": location.media_type or "application/xml,text/plain,*/*"},
                        timeout=settings.DISCOVERY_CANDIDATE_TIMEOUT_SECONDS,
                    )
                    text = _extract_text_representation(
                        response.content,
                        location.representation_kind,
                    )
                    if len(text) < 500:
                        raise ValueError("representation too short to be useful full text")
                    result.set_representation(
                        SourceRepresentation(
                            kind=RepresentationKind.PLAIN_TEXT,
                            media_type="text/plain",
                            content=text.encode("utf-8"),
                            source_url=str(response.url),
                            original_kind=location.representation_kind,
                            charset="utf-8",
                            completeness="not_assessed",
                        )
                    )
                else:
                    response = safe_request(
                        location.url,
                        usage_label="discovered location",
                        headers={"Accept": "text/html,application/xhtml+xml,*/*"},
                        timeout=settings.DISCOVERY_CANDIDATE_TIMEOUT_SECONDS,
                    )
                    discovered = discover_scholarly_locations(
                        response.text,
                        str(response.url),
                        provider=location.provider,
                    )
                    if discovered:
                        inherited_search_provider = location.metadata.get(
                            "search_provider"
                        )
                        inherited_search_engine_group = location.metadata.get(
                            "search_engine_group"
                        )
                        inherited_search_title = location.metadata.get("search_title")
                        discovered = [
                            replace(
                                candidate,
                                metadata={
                                    **candidate.metadata,
                                    "search_provider": inherited_search_provider,
                                    "search_engine_group": (
                                        inherited_search_engine_group
                                    ),
                                    "search_title": inherited_search_title,
                                    "discovered_from_location_sha256": hashlib.sha256(
                                        location.url.encode("utf-8")
                                    ).hexdigest(),
                                    "discovered_from_response_sha256": hashlib.sha256(response.content).hexdigest(),
                                },
                            )
                            for candidate in discovered
                        ]
                        for candidate in discovered:
                            rank_for(candidate)
                    page_titles = _extract_html_titles(response.text)
                    observed_page = extract_web_source_metadata(response.text, str(response.url))
                    # Record what this page showed even when it is not the cited
                    # work and no identity observation is preserved. Without it a
                    # fetched page and an unvisited one reach the bounded review
                    # as the same silence. The value is a category name, not
                    # retained search-result content.
                    attempt["title_reason"] = observed_page.get("title_reason")
                    page_title_match = (
                        _html_title_matches(expected_title, page_titles)
                        if expected_title and page_titles
                        else None
                    )
                    if page_title_match is False:
                        # A generic application/login/results title describes
                        # the interface, not necessarily the requested work.
                        # Require independent bibliographic corroboration before
                        # converting a title mismatch into an identity rejection.
                        if not (observed_page.get("title") and (
                            observed_page.get("doi") or observed_page.get("authors")
                        )):
                            raise ValueError("landing page bibliography unavailable")
                        raise ValueError(
                            "landing page title does not match the cited reference"
                        )
                    landing_identity = self._landing_metadata_identity(
                        response, expected_doi=expected_doi, expected_title=expected_title,
                        expected_author=expected_author, expected_year=expected_year,
                    )
                    if landing_identity:
                        attempt["landing_metadata_identity"] = landing_identity
                    else:
                        # The page's own title and byline, as an observation,
                        # never identity: a same-titled page credited to
                        # someone else is shown as an author difference
                        # (Kozlovic, owner decision 2026-09-30).
                        attempt["landing_page_observation"] = self._landing_metadata_observation(response)
                    from app.services.web_completeness import extract_complete_article_body
                    if extract_complete_article_body(response.text, ""):
                        # Reuse the safe response and ordinary observed-field /
                        # article-body checks. A complete HTML article need not
                        # wait behind its PDF link or unrelated root locations.
                        article = self._try_web_fetch(
                            str(response.url), expected_title,
                            expected_author=expected_author, expected_year=expected_year,
                            expected_doi=expected_doi, expected_source_kind=expected_source_kind,
                            _response=response,
                        )
                        if (article.success and article.representation is not None
                                and article.representation.completeness == "complete"
                                and (article.metadata or {}).get("identity_confidence") == "high"):
                            result.set_representation(article.representation)
                            result.title, result.authors, result.year, result.doi = (
                                article.title, article.authors, article.year, article.doi)
                            result.metadata = {**(result.metadata or {}), **(article.metadata or {})}
                            result.full_text_url = str(response.url)
                            attempt.update(outcome="acquired", reason_code="accepted_complete_html",
                                validated_identity_content_sha256=hashlib.sha256(response.content).hexdigest(),
                                independent_source_url=str(response.url))
                            record_attempt(attempt)
                            record_unattempted("accepted_candidate_found")
                            result.metadata["location_attempts"] = attempts
                            return True
                    discovery_signal = location.metadata.get("discovery_signal")
                    is_explicit_full_text = (
                        location.intended_application == "text-mining"
                        or discovery_signal in {
                            "citation_fulltext_html_url",
                            "link_rel",
                            "json_ld",
                        }
                        or (
                            location.representation_kind is RepresentationKind.HTML
                            and location.landing_page_url is not None
                            and location.url != location.landing_page_url
                        )
                    )
                    if is_explicit_full_text:
                        article_text = _extract_scholarly_html(response.text, result.title)
                        if article_text:
                            structured_kind = classify_html_source_kind(
                                response.text, str(response.url)
                            )
                            content_kind = classify_content_source_kind(
                                article_text, source_url=str(response.url)
                            )
                            observed_html_kind = (
                                content_kind
                                if content_kind.confidence == "high"
                                else structured_kind
                            )
                            html_fallback = (
                                location.provider,
                                str(response.url),
                                article_text,
                                observed_html_kind,
                                page_title_match,
                            )
                    elif html_fallback is None and _page_states_cited_work(
                            landing_identity or attempt.get("landing_page_observation"),
                            expected_title, expected_author, expected_year):
                        # A page whose own title, author and year confirm the
                        # cited work may itself be the work, without marking an
                        # article body (Hess's essay in the Jump Cut archive,
                        # 2026-09-30). Its text is kept as the last resort, with
                        # completeness not established, so it reads as limited text.
                        article_text = _extract_scholarly_html(response.text, result.title)
                        if article_text and len(article_text.split()) >= 300:
                            page_kind = classify_html_source_kind(response.text, str(response.url))
                            from app.services.web_completeness import stated_page_completeness
                            # Rule A (owner decision 2026-09-30): complete only with no
                            # cut-off sign, 1,000+ words and a length fitting the cited kind.
                            stated = stated_page_completeness(response.text, article_text,
                                                              expected_source_kind.kind)
                            attempt["stated_page_completeness"] = {k: v for k, v in stated.items()
                                                                   if k != "html_sha256"}
                            evidence = (*page_kind.evidence, REPUBLISHED_WORK_EVIDENCE,
                                        *((STATED_COMPLETE_EVIDENCE,) if stated["verdict"] == "complete" else ()))
                            html_fallback = (
                                location.provider, str(response.url), article_text,
                                SourceKindAssessment(page_kind.kind, page_kind.confidence, evidence),
                                page_title_match,
                            )
                    pending_urls = {candidate.url for candidate in pending}
                    unseen_discovered = [
                        candidate
                        for candidate in sorted(discovered, key=_location_rank)
                        if candidate.url not in attempted_urls
                        and candidate.url not in pending_urls
                    ]
                    # A landing page can expose several weak or inaccessible
                    # derivatives.  Do not let one branch consume the entire
                    # bounded attempt budget before the strongest independent
                    # search results have been tried.  After three root leads,
                    # derived full-text links may again take priority.
                    attempted_roots = len(attempted_urls & root_location_urls)
                    roots_to_reserve = max(
                        0, required_root_attempts - attempted_roots
                    )
                    reserved_prefix: list[AcquisitionLocation] = []
                    remaining_pending = list(pending)
                    while remaining_pending and roots_to_reserve:
                        queued = remaining_pending.pop(0)
                        reserved_prefix.append(queued)
                        if queued.url in root_location_urls:
                            roots_to_reserve -= 1
                    pending = (
                        reserved_prefix + unseen_discovered + remaining_pending
                    )
                    if not discovered and html_fallback:
                        (
                            _,
                            source_url,
                            article_text,
                            observed_html_kind,
                            page_title_match,
                        ) = html_fallback
                        result.set_representation(
                            SourceRepresentation(
                                kind=RepresentationKind.PLAIN_TEXT,
                                media_type="text/plain",
                                content=article_text.encode("utf-8"),
                                source_url=source_url,
                                original_kind=RepresentationKind.HTML,
                                charset="utf-8",
                                completeness=("complete" if STATED_COMPLETE_EVIDENCE in observed_html_kind.evidence
                                              else "not_assessed"),
                                metadata={
                                    "observed_source_kind": observed_html_kind.kind,
                                    "observed_source_kind_confidence": (
                                        observed_html_kind.confidence
                                    ),
                                    "page_title_match": page_title_match,
                                    "observed_source_kind_evidence": list(observed_html_kind.evidence),
                                    **_stated_page_marker(observed_html_kind, article_text, expected_source_kind),
                                },
                            )
                        )
                    else:
                        if discovered:
                            attempt.update(
                                outcome="unavailable",
                                reason_code="landing_page_discovered_full_text_candidates",
                                discovered_location_sha256=[hashlib.sha256(candidate.url.encode()).hexdigest()
                                    for candidate in discovered],
                                discovery_response_sha256=hashlib.sha256(response.content).hexdigest(),
                            )
                            record_attempt(attempt)
                            continue
                        raise ValueError(
                            f"landing page yielded {len(discovered)} acquisition candidate(s)"
                        )
                accepted, outcome, reason = self._preflight_acquired_representation(
                    result,
                    expected_doi=expected_doi or result.doi,
                    expected_title=expected_title or result.title,
                    expected_author=(
                        expected_author
                        or (result.authors[0] if result.authors else None)
                    ),
                    expected_year=expected_year or result.year,
                    expected_source_kind=expected_source_kind,
                    advertised_by_confirmed_landing=advertised_by_confirmed_landing,
                )
                attempt["outcome"] = outcome
                if outcome in {"acquired", "completeness_rejected"} and (result.metadata or {}).get("identity_confidence") == "high":
                    # Preserve source-bound identity independently of admission.
                    # A partial copy can establish identity without full text.
                    attempt["validated_identity_content_sha256"] = hashlib.sha256(result.representation.content).hexdigest()
                    attempt["independent_source_url"] = result.representation.source_url
                attempt["reason_code"] = (
                    "accepted_representation" if accepted else outcome
                )
                attempt["observed_source_kind"] = (result.metadata or {}).get(
                    "observed_source_kind"
                )
                attempt["source_kind_verdict"] = (result.metadata or {}).get(
                    "source_kind_verdict"
                )
                if not accepted:
                    attempt["reason"] = reason
                    record_attempt(attempt)
                    self._clear_retrieved_representation(result)
                    continue
                record_attempt(attempt)
                result.full_text_url = location.url
                result.source_name = location.provider
                result.metadata = result.metadata or {}
                record_unattempted("accepted_candidate_found")
                result.metadata["location_attempts"] = attempts
                return True
            except Exception as exc:
                self._clear_retrieved_representation(result)
                outcome, reason_code = _location_exception_disposition(exc)
                attempt["outcome"] = outcome
                attempt["reason_code"] = reason_code
                attempt["reason"] = safe_exception_code(exc)
                record_attempt(attempt)
                # MDPI blocks automated requests to its site; its own file
                # host serves the same open-access PDF. Try that next, once.
                active_trace = _ACTIVE_DISCOVERY_TRACE.get()
                tried_mdpi = (active_trace.setdefault("mdpi_file_server_urls", set())
                              if active_trace is not None else set())
                for fallback in _mdpi_file_server_locations(
                        location, exc, expected_doi or result.doi):
                    if fallback.url in attempted_urls or fallback.url in tried_mdpi or any(
                            queued.url == fallback.url for queued in pending):
                        continue
                    tried_mdpi.add(fallback.url)
                    pending.insert(0, fallback)
                    rank_for(fallback)
        if html_fallback:
            (
                provider,
                source_url,
                article_text,
                observed_html_kind,
                page_title_match,
            ) = html_fallback
            result.set_representation(
                SourceRepresentation(
                    kind=RepresentationKind.PLAIN_TEXT,
                    media_type="text/plain",
                    content=article_text.encode("utf-8"),
                    source_url=source_url,
                    original_kind=RepresentationKind.HTML,
                    charset="utf-8",
                    completeness=("complete" if STATED_COMPLETE_EVIDENCE in observed_html_kind.evidence
                                  else "not_assessed"),
                    metadata={
                        "observed_source_kind": observed_html_kind.kind,
                        "observed_source_kind_confidence": (
                            observed_html_kind.confidence
                        ),
                        "page_title_match": page_title_match,
                        "observed_source_kind_evidence": list(observed_html_kind.evidence),
                        **_stated_page_marker(observed_html_kind, article_text, expected_source_kind),
                    },
                )
            )
            accepted, outcome, reason = self._preflight_acquired_representation(
                result,
                expected_doi=expected_doi or result.doi,
                expected_title=expected_title or result.title,
                expected_author=(
                    expected_author or (result.authors[0] if result.authors else None)
                ),
                expected_year=expected_year or result.year,
                expected_source_kind=expected_source_kind,
                advertised_by_confirmed_landing=advertised_by_confirmed_landing,
            )
            attempt = {
                "url": source_url,
                "provider": provider,
                "kind": RepresentationKind.HTML.value,
                "outcome": "acquired_fallback" if accepted else outcome,
                "reason_code": (
                    "accepted_html_fallback" if accepted else outcome
                ),
                "rank": location_ranks.get(source_url),
                "origin_providers": [provider],
            }
            if not accepted:
                attempt["reason"] = reason
                self._clear_retrieved_representation(result)
            else:
                attempt["validated_identity_content_sha256"] = hashlib.sha256(result.representation.content).hexdigest()
                attempt["independent_source_url"] = result.representation.source_url
                result.full_text_url = source_url
                result.source_name = provider
            record_attempt(attempt)
            result.metadata = result.metadata or {}
            record_unattempted(
                "accepted_candidate_found" if accepted else "attempt_limit_reached"
            )
            result.metadata["location_attempts"] = attempts
            return accepted
        result.metadata = result.metadata or {}
        record_unattempted("attempt_limit_reached")
        result.metadata["location_attempts"] = attempts
        return False

    @staticmethod
    def _landing_metadata_observation(response) -> dict:
        """The fetched page's bibliographic fields, kept without its address."""
        observed = extract_web_source_metadata(response.text, str(response.url))
        fields = {key: observed[key] for key in ("title", "authors", "year", "doi")}
        fields["metadata"] = {key: observed[key] for key in ("container_title", "pages") if observed.get(key)}
        return {"observed": fields, "content_sha256": hashlib.sha256(response.content).hexdigest()}

    @staticmethod
    def _landing_metadata_identity(response, *, expected_doi, expected_title, expected_author, expected_year):
        """Identity-only evidence from the fetched page, never its search snippet.

        A metadata page is not an acquired full-text representation. Preserve
        this observation only when the supplied bibliographic fields agree.
        """
        observed = extract_web_source_metadata(response.text, str(response.url))
        fields = {key: observed[key] for key in ("title", "authors", "year", "doi")}
        fields["metadata"] = {
            key: observed[key] for key in ("container_title", "pages") if observed.get(key)
        }
        if observed.get("title_reason"):
            fields["metadata"]["title_reason"] = observed["title_reason"]
        active_trace = _ACTIVE_DISCOVERY_TRACE.get()
        # Use the same complete expected fields as the final trace before
        # declaring identity or finalizing transient-provider audit counts.
        expected = active_trace["expected"] if active_trace is not None else ExpectedBibliographicFields(
            title=expected_title or "", doi=expected_doi or "",
            authors=[expected_author] if expected_author else [], year=expected_year or "",
        )
        candidate = build_reference_discovery_candidate(
            attempt_id="landing-identity", provider="independent_landing_metadata",
            expected=expected,
            result=RetrievalResult(source_name="independent_landing_metadata", success=True, **fields),
        )
        confirmed = (
            not expected.reference_parse_review
            and candidate.plausible_identity_match and not candidate.has_material_conflict
            and not candidate.has_unresolved_supplied_identity_fields
            and (candidate.authoritative_identifier_match or sum(
                c.outcome in {'agreement', 'minor_difference'} for c in candidate.comparisons
                if c.field_name in {'title', 'author', 'year', 'doi', 'isbn'}
            ) >= 2)
        )
        if not confirmed:
            return None
        validated_response_identity(str(response.url), hashlib.sha256(response.content).hexdigest(),
            'high', [c.field_name for c in candidate.comparisons])
        return {"observed": fields, "content_sha256": hashlib.sha256(response.content).hexdigest(),
                "source_url": str(response.url)}

    def _verify_source_identity(
        self,
        result: RetrievalResult,
        expected_doi: str | None,
        expected_title: str | None,
        expected_author: str | None,
        expected_year: str | None,
    ) -> str:
        """Verify a found source matches the cited reference via multi-field matching.

        A student may mess up one field (wrong URL) but won't mess up title +
        author + year simultaneously. This scores how many fields match to
        determine confidence that we found the RIGHT source.

        Returns: "high" | "medium" | "low"
          - high: DOI match, OR title + author + year all match
          - medium: title + at least one of (author, year) match
          - low: title only matches (possible but uncertain)
        """
        import re as _re

        score = 0
        max_score = 0

        # DOI (strongest signal — definitive if present)
        if expected_doi:
            max_score += 3
            if result.doi and result.doi.lower() == expected_doi.lower():
                return "high"  # DOI match = definitive, skip other checks

        # Title (strong signal)
        if expected_title and result.title:
            max_score += 2
            # Token overlap between expected and found titles
            exp_tokens = {t.lower() for t in _re.split(r"[^A-Za-z0-9]+", expected_title) if len(t) >= 3}
            got_tokens = {t.lower() for t in _re.split(r"[^A-Za-z0-9]+", result.title) if len(t) >= 3}
            if exp_tokens and got_tokens:
                overlap = len(exp_tokens & got_tokens) / len(exp_tokens)
                if overlap >= 0.6:
                    score += 2
                elif overlap >= 0.3:
                    score += 1

        # Author (supporting signal)
        if expected_author and result.authors:
            max_score += 1
            # Does the expected author's surname appear in the result's authors?
            expected_surname = expected_author.split(",")[0].strip().lower()
            if expected_surname and any(expected_surname in a.lower() for a in result.authors):
                score += 1

        # Year (supporting signal)
        if expected_year and result.year:
            max_score += 1
            if expected_year in result.year or result.year in expected_year:
                score += 1

        if max_score == 0:
            return "low"  # nothing to compare

        ratio = score / max_score
        if ratio >= 0.7:
            return "high"
        elif ratio >= 0.4:
            return "medium"
        return "low"

    def _retry_after_network_error(self, attempt) -> RetrievalResult:
        """A student's own link is tried once more after a network-level failure
        (owner request 2026-09-29): one dropped connection must not leave the
        cited source unretrieved. A wrong page, a missing page or an access
        refusal is an answer, not a network failure, and is not retried."""
        result = attempt()
        if result.success or not _network_failure(result.error):
            return result
        first = _network_failure(result.error)
        time.sleep(STUDENT_LINK_RETRY_DELAY_SECONDS)
        retried = attempt()
        retried.metadata = {**(retried.metadata or {}), "student_link_retry": {"first_failure": first}}
        return retried

    @_timed_retrieval
    def _try_web_fetch(
        self,
        url: str,
        title: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
        expected_author: str | None = None,
        expected_year: str | None = None,
        expected_doi: str | None = None,
        _response: httpx.Response | None = None,
    ) -> RetrievalResult:
        """Fetch a web page (HTML) and extract its readable text.

        Used when a student-provided URL is NOT a PDF (the _safe_download path
        rejects it). Many student citations point to web pages — news articles,
        reference sites (Britannica, Wikipedia), government reports, blogs.
        These are legitimate text-based sources that need verification.

        Identity gate (REVIEW §2b #18): before trusting the page as the cited
        source, its titles (<title>, og:title, citation_title) are compared to
        the cited reference title. A login page, error page, or a different
        article on the same site will mismatch and be rejected — previously the
        app trusted whatever was at the URL unconditionally.

        Uses trafilatura to extract article text (strips navigation, ads,
        sidebars). The extracted text is returned as the source content for
        the verification engine to check the citation against.

        Return observed bibliographic fields and an explicit identity decision
        to the ordinary scoped source-admission path. Fetch success alone is
        not permission to use the text as evidence.
        """
        from app.services.retrieval.internet_archive import item_identifier
        if item_identifier(url):
            result = RetrievalResult(
                source_name='internet_archive', success=False,
                locations=[AcquisitionLocation(url=url, provider='internet_archive')],
                metadata={'requested_url_sha256': hashlib.sha256(url.encode()).hexdigest()},
            )
            result.success = self._acquire_from_locations(
                result, expected_doi=expected_doi, expected_title=title,
                expected_author=expected_author, expected_year=expected_year,
                expected_source_kind=expected_source_kind,
            )
            if not result.success:
                result.error = 'archive_source_not_acquired'
            return result
        try:
            require_source_candidate(url)
            resp = _response if _response is not None else safe_request(
                url,
                usage_label="reference URL",
                headers={
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(
                "Web fetch failed for %s: %s",
                private_value_id("url", url),
                type(e).__name__,
            )
            if isinstance(e, CandidateBudgetExceeded):
                link_not_visited(url, 'candidate_budget_exhausted')
            return RetrievalResult(
                source_name="web_fetch",
                success=False,
                error=f"web_fetch_failed:{safe_exception_code(e)}",
                metadata={'candidate_budget_skipped': isinstance(e, CandidateBudgetExceeded),
                    'web_fetch_diagnostic': WebFetchDiagnostic(reason=(
                    'candidate_budget_exhausted' if isinstance(e, CandidateBudgetExceeded)
                    else 'access_restricted' if _location_exception_disposition(e)[0] == 'access_restricted'
                    else 'transport_failure' if _location_exception_disposition(e)[0] == 'transport_failure'
                    else 'fetch_unavailable')).model_dump()},
            )

        def diagnostic(reason: str) -> dict:
            page_observed(url, hashlib.sha256(resp.content).hexdigest(), reason)
            return {'web_fetch_diagnostic': WebFetchDiagnostic(
                reason=reason, observed_content_sha256=hashlib.sha256(resp.content).hexdigest(),
            ).model_dump()}

        content_type = resp.headers.get("content-type", "").lower()
        if resp.content.startswith(PDF_MAGIC) or "application/pdf" in content_type:
            return RetrievalResult(
                source_name="web_fetch",
                success=False,
                error="URL returned a PDF; use PDF route",
                metadata=diagnostic('pdf_route_required'),
            )

        final_url = str(resp.url)
        observed = extract_web_source_metadata(resp.text, final_url)
        if urlparse(final_url).hostname == 'lantern.mediahist.org' and urlparse(final_url).path.startswith('/catalog/'):
            # A catalog record may establish identity, but it is not the scan.
            # Follow only explicit source links through existing acquisition.
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            locations = discover_scholarly_locations(resp.text, final_url)
            for anchor in BeautifulSoup(resp.text, 'html.parser').find_all('a', href=True):
                target = urljoin(final_url, anchor['href'])
                if item_identifier(target) and not any(x.url == target for x in locations):
                    locations.append(AcquisitionLocation(url=target, provider='internet_archive', landing_page_url=final_url))
            catalog = RetrievalResult(source_name='web_fetch', success=False,
                title=observed['title'], authors=observed['authors'], year=observed['year'],
                doi=observed['doi'], locations=locations[:2], metadata={'web_identity': observed})
            if locations:
                catalog.success = self._acquire_from_locations(catalog,
                    expected_doi=expected_doi, expected_title=title, expected_author=expected_author,
                    expected_year=expected_year, expected_source_kind=expected_source_kind)
            landing = self._landing_metadata_identity(resp, expected_doi=expected_doi,
                expected_title=title, expected_author=expected_author, expected_year=expected_year)
            if landing:
                catalog.metadata.setdefault('location_attempts', []).append({
                    'url': final_url, 'outcome': 'unavailable', 'landing_metadata_identity': landing,
                    'reason_code': 'catalog_record_not_source_text'})
            if not catalog.success:
                catalog.error = 'Catalog record found; source representation not acquired'
            return catalog
        from app.services.retrieval.dspace import is_dspace_shell, repository_item
        if is_dspace_shell(resp.text):
            # A DSpace 7 repository page is an empty JavaScript shell; the item
            # and its files come from the repository's own API on the same host.
            def fetch_json(address):
                return safe_request(address, usage_label="repository record",
                                    headers={"Accept": "application/json"},
                                    timeout=settings.STUDENT_URL_TIMEOUT_SECONDS).json()
            try:
                record = repository_item(final_url, fetch_json)
            except Exception as exc:
                logger.info("Repository record unavailable (%s)", safe_exception_code(exc))
                record = None
            if record and record["files"]:
                identity = {**observed, "title": record["title"], "authors": record["authors"],
                            "year": record["year"]}
                repository = RetrievalResult(
                    source_name='web_fetch', success=False, title=record["title"], authors=record["authors"],
                    year=record["year"], doi=observed.get('doi'),
                    locations=[AcquisitionLocation(url=f["url"], provider='dspace_repository', media_type='application/pdf',
                                                   representation_kind=RepresentationKind.PDF, landing_page_url=final_url)
                               for f in record["files"]],
                    metadata={'web_identity': identity, 'repository_record': 'dspace7'})
                repository.success = self._acquire_from_locations(
                    repository, expected_doi=expected_doi, expected_title=title, expected_author=expected_author,
                    expected_year=expected_year, expected_source_kind=expected_source_kind)
                if not repository.success:
                    repository.error = 'Repository record found; file not acquired'
                return repository
        # Identity check: confirm the page IS the cited source. If we can
        # extract a page title and it doesn't match the reference, reject — this
        # is a wrong page (login, error, or a different article). If no title is
        # extractable (some JS-rendered pages), fall through to content extraction.
        landing_title_confirmed = False
        if title:
            page_titles = _extract_html_titles(resp.text)
            landing_title_confirmed = bool(
                page_titles and _landing_title_confirms(title, page_titles)
            )
            if page_titles and not _html_title_matches(title, page_titles):
                from app.services.submitted_links import is_site_homepage
                homepage = is_site_homepage(resp.text, final_url)
                logger.info(
                    "Web fetch identity mismatch for %s",
                    private_value_id("url", url),
                )
                return RetrievalResult(
                    source_name="web_fetch",
                    success=False,
                    error="Page title does not match the cited reference — likely wrong page",
                    metadata=diagnostic('site_homepage' if homepage else 'cross_script_title_unresolved' if all(
                        cross_script_comparison_unresolved(title, value) for value in page_titles
                    ) else 'page_title_mismatch_unconfirmed'),
                )

        # A repository or publisher landing page advertises its own full text
        # through the scholarly `citation_pdf_url` standard. Prefer that over
        # treating the landing page as the source: the page is a record about
        # the work, not the work. Ordinary identity, safety and validation
        # gates still apply to the acquired bytes, and a failed acquisition
        # falls through to the existing HTML handling below.
        advertised = [
            location
            for location in discover_scholarly_locations(
                resp.text, final_url, provider="landing_page"
            )
            if location.representation_kind is RepresentationKind.PDF
        ]
        if advertised:
            landing = RetrievalResult(
                source_name="landing_page", success=False,
                title=observed["title"], authors=observed["authors"],
                year=observed["year"], doi=observed["doi"],
                locations=advertised[:2],
                metadata={
                    "web_identity": observed,
                    "requested_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                    "landing_page_url": final_url,
                },
            )
            if self._acquire_from_locations(
                landing,
                expected_doi=expected_doi, expected_title=title,
                expected_author=expected_author, expected_year=expected_year,
                expected_source_kind=expected_source_kind,
                # This page named the cited work and then named its own full
                # text. A thesis whose first pages are a scanned approval form
                # cannot prove its title from its own bytes; the record that
                # points at it can, and did.
                advertised_by_confirmed_landing=landing_title_confirmed,
            ):
                landing.success = True
                return landing
            self._clear_retrieved_representation(landing)

        # Extract readable text
        import trafilatura
        page_text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_links=False,
            include_tables=False,
            favor_recall=True,
        )
        if not page_text or len(page_text) < 100:
            result = RetrievalResult(
                source_name="web_fetch",
                success=False,
                title=observed['title'], authors=observed['authors'],
                year=observed['year'], doi=observed['doi'],
                error="Page loaded but no readable article text extracted",
                metadata=diagnostic('readable_text_unavailable'),
            )
            identity, comparisons, confirmed = _web_identity_comparison(result,
                title=title, author=expected_author, year=expected_year, doi=expected_doi)
            reason = ('bibliographic_fields_conflict' if identity.has_material_conflict
                else 'bibliographic_identity_confirmed' if confirmed else 'bibliographic_identity_unconfirmed')
            identity_observed(url, hashlib.sha256(resp.content).hexdigest(), reason,
                [c.field_name for c in comparisons if c.outcome == 'material_conflict'],
                differences=_web_link_differences(comparisons, result, title=title,
                    author=expected_author, year=expected_year, doi=expected_doi))
            result.metadata['identity_comparisons'] = [c.model_dump(mode='json') for c in comparisons]
            # No representation, accepted hash, success, or admission grant.
            return result

        from app.services.web_completeness import extract_complete_article_body
        page_text = extract_complete_article_body(resp.text, page_text)
        structured_kind = classify_html_source_kind(resp.text, final_url)
        content_kind = classify_content_source_kind(page_text, source_url=final_url)
        observed_kind = (
            content_kind if content_kind.confidence == "high" else structured_kind
        )
        kind_compatibility = compare_source_kinds(
            expected_source_kind, observed_kind
        )
        requires_confirmed_non_web_type = (
            _requires_confirmed_non_web_source_kind(expected_source_kind)
        )
        if (
            kind_compatibility.verdict == "incompatible"
            or (
                requires_confirmed_non_web_type
                and kind_compatibility.verdict != "compatible"
            )
        ):
            return RetrievalResult(
                source_name="web_fetch",
                success=False,
                error=(
                    "Page work type does not confirm the cited reference — "
                    + kind_compatibility.reason
                ),
                metadata={
                    "expected_source_kind": expected_source_kind.kind,
                    "observed_source_kind": observed_kind.kind,
                    "source_kind_verdict": kind_compatibility.verdict,
                    **diagnostic('source_kind_unconfirmed'),
                },
            )

        logger.info(
            "Web fetch extracted %d chars from %s",
            len(page_text),
            private_value_id("url", url),
        )
        from app.services.web_completeness import assess_web_completeness, length_fits_kind
        web_coverage = assess_web_completeness(resp.text, page_text)
        if (web_coverage.get("verdict") == "complete"
                and length_fits_kind(len(re.findall(r"\w+", page_text)), expected_source_kind.kind) is False):
            # Its length cannot be a whole work of the cited kind (2026-09-30).
            web_coverage = {**web_coverage, "verdict": "not_assessed", "reason": "length_does_not_fit_kind"}
        result = RetrievalResult(
            source_name="web_fetch",
            success=True,
            title=observed["title"],
            authors=observed["authors"],
            year=observed["year"],
            doi=observed["doi"],
            representation=SourceRepresentation(
                kind=RepresentationKind.PLAIN_TEXT,
                media_type="text/plain",
                content=page_text.encode("utf-8"),
                source_url=final_url,
                original_kind=RepresentationKind.HTML,
                charset="utf-8",
                completeness=web_coverage['verdict'],
                metadata={'web_completeness':web_coverage},
            ),
            # Transitional compatibility for the verification path that still
            # consumes web article text through the limited-evidence field.
            abstract=page_text,
            full_text_url=final_url,
            metadata={
                "web_identity": observed,
                "web_completeness": web_coverage,
                "requested_url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                "representation_kind": RepresentationKind.PLAIN_TEXT.value,
                "expected_source_kind": expected_source_kind.kind,
                "observed_source_kind": observed_kind.kind,
                "observed_source_kind_confidence": observed_kind.confidence,
                "source_kind_verdict": kind_compatibility.verdict,
            },
        )
        identity, comparisons, confirmed = _web_identity_comparison(result,
            title=title, author=expected_author, year=expected_year, doi=expected_doi)
        # A one-year bibliographic error must not make a clearly identified
        # journal article unreadable. Keep the year conflict in discovery/link
        # evidence; this is work-level acquisition, not reference correctness.
        fields = {c.field_name: c.outcome for c in comparisons}
        nearby_publication_year = (
            expected_source_kind.kind == 'journal_article'
            and observed_kind.kind == 'journal_article' and observed_kind.confidence == 'high'
            and fields.get('title') == fields.get('author') == 'agreement'
            and {c.field_name for c in comparisons if c.outcome == 'material_conflict'} == {'year'}
            and re.fullmatch(r'\d{4}', str(expected_year or ''))
            and re.fullmatch(r'\d{4}', str(result.year or ''))
            and abs(int(expected_year) - int(result.year)) == 1
            and all(c.outcome in {'agreement', 'minor_difference'} for c in comparisons if c.field_name != 'year')
        )
        if not confirmed and not identity.has_material_conflict:
            # Use independently retrieved page content, never search snippets.
            inspected = self._inspect_candidate(result.full_text, title=title,
                author=expected_author, year=expected_year, doi=expected_doi,
                source_kind=expected_source_kind.kind)
            if inspected is not None:
                result.metadata['source_inspection'] = inspected
        result.metadata["identity_confidence"] = "high" if nearby_publication_year else "rejected" if identity.has_material_conflict else "high" if confirmed else "medium"
        if nearby_publication_year:
            result.metadata['work_identity_basis'] = 'journal-title-author-near-year-v1'
        result.metadata["identity_reason"] = "Observed webpage title/author/date/identifier comparisons"
        result.metadata["identity_comparisons"] = [c.model_dump(mode="json") for c in comparisons]
        result.metadata["text_quality"] = "digital"
        result.metadata.update(diagnostic(
            'bibliographic_fields_conflict' if identity.has_material_conflict
            else 'bibliographic_identity_confirmed' if confirmed else 'bibliographic_identity_unconfirmed'))
        identity_observed(url, hashlib.sha256(resp.content).hexdigest(),
            result.metadata['web_fetch_diagnostic']['reason'],
            [c.field_name for c in comparisons if c.outcome == 'material_conflict'],
            differences=_web_link_differences(comparisons, result, title=title,
                author=expected_author, year=expected_year, doi=expected_doi))
        if confirmed or nearby_publication_year:
            result.metadata["accepted_representation_sha256"] = hashlib.sha256(result.full_text).hexdigest()
        if identity.has_material_conflict and not nearby_publication_year:
            result.success = False
            result.error = "Observed webpage bibliographic fields conflict with citation"
        elif not confirmed and not nearby_publication_year:
            # Readable bytes are not an accepted source. Preserve this candidate
            # in the trace and continue the existing permitted discovery chain.
            result.success = False
            result.error = "Observed webpage bibliographic identity remains unconfirmed"
        return result

    def _validated_doi_resolver_pdf(
        self,
        payload: bytes,
        source_url: str,
        doi: str,
        title: str | None,
        *,
        landing_page_url: str | None = None,
        location_metadata: dict | None = None,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
        expected_author: str | None = None,
        expected_year: str | None = None,
        access_type: str | None = "institutional",
    ) -> RetrievalResult:
        """Apply the common PDF identity gate to resolver-acquired bytes."""
        result = RetrievalResult(
            source_name="doi_resolver",
            success=True,
            representation=SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=payload,
                source_url=source_url,
                completeness="not_assessed",
            ),
            locations=[
                AcquisitionLocation(
                    url=source_url,
                    provider="doi_resolver",
                    media_type="application/pdf",
                    representation_kind=RepresentationKind.PDF,
                    landing_page_url=landing_page_url,
                    access_type=access_type,
                    is_best=True,
                    metadata=location_metadata or {},
                )
            ],
            doi=doi,
            title=title,
        )
        accepted, outcome, reason = self._preflight_acquired_representation(
            result,
            expected_doi=doi,
            expected_title=title,
            expected_author=expected_author,
            expected_year=expected_year,
            expected_source_kind=expected_source_kind,
        )
        if not accepted:
            logger.info(
                "DOI resolver PDF rejected for %s (outcome=%s)",
                private_value_id("doi", doi),
                outcome,
            )
            result.success = False
            if outcome in {"identity_rejected", "identity_unconfirmed"}:
                result.metadata[outcome] = True
                result.error = f"DOI resolver PDF failed source identity: {outcome}"
            elif outcome == "completeness_rejected":
                result.metadata["completeness_rejected"] = True
                result.error = "DOI resolver PDF failed completeness validation"
            else:
                result.error = f"DOI resolver PDF preflight failed: {outcome}"
            self._clear_retrieved_representation(result)
            return result
        if result.parent_representation is not None:
            self._persist_retrieved_representation(
                result,
                ref_doi=doi,
                ref_title=title,
                ref_author=expected_author,
                ref_year=expected_year,
                identity_confidence="high",
                identity_reason=result.metadata.get(
                    "identity_reason", "accepted OCR derivative"
                ),
                downloaded_via_publisher=True,
                safety_report=None,
                expected_source_kind=expected_source_kind,
            )
            if result.metadata.get("durable_admission", {}).get("state") != "accepted":
                result.success = False
                result.error = "OCR derivative could not enter durable admission"
                self._clear_retrieved_representation(result)
        return result

    @_timed_retrieval
    def _try_mdpi_file_server(self, doi, title, author, year,
                              expected_kind: SourceKindAssessment) -> RetrievalResult | None:
        """Acquire an MDPI article's PDF from MDPI's own file host.

        Built from the DOI only; the downloaded file passes the ordinary
        acquisition, safety and identity gates like any other location. The
        result carries no bibliographic fields of its own, so identity rests
        on the file's content alone. Returns None when nothing new is to try.
        """
        trace = _ACTIVE_DISCOVERY_TRACE.get()
        tried = trace.setdefault("mdpi_file_server_urls", set()) if trace is not None else set()
        urls = [url for url in mdpi_file_server_urls(doi) if url not in tried]
        if not urls:
            return None
        tried.update(urls)
        candidate = RetrievalResult(
            source_name=MDPI_FILE_SERVER_ROUTE,
            success=True,
            doi=doi,
            # No fallback URL: the later publisher-PDF step would otherwise
            # fetch the same file again with a browser user agent.
            full_text_url=None,
            locations=[
                AcquisitionLocation(
                    url=url, provider=MDPI_FILE_SERVER_ROUTE, media_type="application/pdf",
                    representation_kind=RepresentationKind.PDF, is_best=index == 0,
                    metadata={"constructed_route": MDPI_FILE_SERVER_ROUTE},
                )
                for index, url in enumerate(urls)
            ],
        )
        try:
            result = self._download_and_cache(
                _MDPI_FILE_SERVER_SOURCE, candidate, doi, title, author, year,
                expected_source_kind=expected_kind,
            )
        except Exception as exc:
            return RetrievalResult(source_name=MDPI_FILE_SERVER_ROUTE, success=False,
                                   error=safe_exception_code(exc), doi=doi)
        if not result.full_text:
            result.success = False
            result.error = result.error or "MDPI file server copy unavailable"
        return result

    @_timed_retrieval
    def _try_doi_resolver(
        self,
        doi: str,
        title: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
        public: bool = False,
        expected_author: str | None = None,
        expected_year: str | None = None,
    ) -> RetrievalResult:
        """Access a paper through the institution's DOI resolver / campus proxy.

        When DOI_RESOLVER_URL is configured (e.g., an EZproxy or OpenURL
        resolver), the app constructs {DOI_RESOLVER_URL}{doi} to access the
        paper through the university's library subscription. The response is
        untrusted: a resolver menu, login page, wrong article, or malformed PDF
        must pass the same representation-specific identity checks used by the
        rest of acquisition before it can become verification evidence.

        This is the highest-yield retrieval path for paywalled academic content
        at institutions with library subscriptions — it can return full-text
        PDFs that no OA source has.
        """
        base = "https://doi.org/" if public else settings.DOI_RESOLVER_URL or ""
        trusted_base = None if public else base
        # Validate + URL-encode the DOI before placing it in the trusted
        # resolver URL (REVIEW §2.4). The DOI is student-controlled; raw
        # concatenation let a DOI like "10.1/x?url=http://internal/..." inject
        # a query string into the proxy URL (chained SSRF through the proxy).
        from app.services.ref_field_extractor import decode_doi
        doi = decode_doi(doi)
        if not re.fullmatch(r"10\.\d{4,9}/[A-Za-z0-9._:()\-/]+", doi):
            logger.warning("DOI resolver rejected a malformed DOI")
            return RetrievalResult(
                source_name="doi_resolver",
                success=False,
                error="Malformed DOI",
                doi=doi,
            )
        from urllib.parse import quote
        resolver_url = f"{base}{quote(doi, safe='/')}"

        try:
            # trust_prefix: the configured resolver base is an operator-trusted
            # host (may be an on-campus EZproxy on a private network). The DOI
            # is sanitized above, and every redirect hop is re-validated by
            # safe_request, so this trust does not extend to publisher targets.
            require_source_candidate(resolver_url)
            resp = safe_request(
                resolver_url,
                usage_label="DOI resolver",
                headers={"Accept": "text/html,application/pdf,*/*"},
                max_bytes=settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024,
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                trust_prefix=trusted_base,
            )
        except Exception as e:
            logger.warning(
                "DOI resolver failed for %s: %s",
                private_value_id("doi", doi),
                type(e).__name__,
            )
            return RetrievalResult(
                source_name="doi_resolver",
                success=False,
                error=f"DOI resolver failed ({type(e).__name__})",
                doi=doi,
                metadata={"candidate_budget_skipped": isinstance(e, CandidateBudgetExceeded),
                          **({"http_status": _blocked_status(e)} if _blocked_status(e) else {})},
            )

        final_url = str(getattr(resp, "url", None) or resolver_url)

        # A resolver PDF enters the common PDF identity validator. It remains
        # in-memory here; retention is selected by the configured source-store
        # policy rather than implicitly caching subscription content.
        if resp.content[:5] == PDF_MAGIC:
            return self._validated_doi_resolver_pdf(
                resp.content,
                final_url,
                doi,
                title,
                expected_source_kind=expected_source_kind,
                expected_author=expected_author, expected_year=expected_year,
                access_type=None if public else "institutional",
            )

        # HTML is accepted only when it is an identity-matching scholarly text.
        # Length alone is never evidence: long login menus and resolver pages
        # were the original false-acceptance path.
        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            public_identity = (self._landing_metadata_identity(resp,
                expected_doi=doi, expected_title=title, expected_author=expected_author,
                expected_year=expected_year) if public else None)
            discovered = discover_scholarly_locations(
                resp.text,
                final_url,
                provider="doi_resolver",
            )
            for candidate in sorted(discovered, key=_location_rank)[:_MAX_LOCATION_ATTEMPTS]:
                if candidate.representation_kind is not RepresentationKind.PDF:
                    continue
                try:
                    require_source_candidate(candidate.url)
                    candidate_response = safe_request(
                        candidate.url,
                        usage_label="file download",
                        headers={"Accept": "application/pdf,*/*"},
                        max_bytes=settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024,
                        timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                    trust_prefix=(
                        trusted_base
                        if trusted_base and trusted_url_matches_prefix(candidate.url, trusted_base)
                        else None
                    ),
                    )
                except CandidateBudgetExceeded:
                    return RetrievalResult(source_name="doi_resolver", success=False,
                        metadata={"candidate_budget_skipped": True})
                except Exception as exc:
                    logger.info(
                        "DOI resolver landing-page candidate unavailable for %s: %s",
                        private_value_id("doi", doi),
                        type(exc).__name__,
                    )
                    continue
                if not candidate_response.content.startswith(PDF_MAGIC):
                    continue
                validated = self._validated_doi_resolver_pdf(
                    candidate_response.content,
                    str(candidate_response.url),
                    doi,
                    title,
                    landing_page_url=final_url,
                    location_metadata=candidate.metadata,
                    expected_source_kind=expected_source_kind,
                    expected_author=expected_author, expected_year=expected_year,
                    access_type=None if public else "institutional",
                )
                if validated.success:
                    return validated

            page_titles = _extract_html_titles(resp.text)
            if title and page_titles and not _html_title_matches(title, page_titles):
                logger.info(
                    "DOI resolver HTML title mismatch for %s",
                    private_value_id("doi", doi),
                )
                return RetrievalResult(
                    source_name="doi_resolver",
                    success=False,
                    error="DOI resolver HTML title does not match cited source",
                    doi=doi,
                    metadata={"identity_rejected": True},
                )

            # Public DOI destinations do not inherit institutional assumptions.
            # Reuse observed-field identity, not the requested DOI appearing in
            # arbitrary body text, before exposing HTML as source evidence.
            if public and public_identity is None:
                return RetrievalResult(source_name="doi_resolver", success=False,
                    error="Public DOI landing identity remains unconfirmed", doi=doi)
            page_text = _extract_scholarly_html(resp.text, title)
            if page_text:
                content = page_text.encode("utf-8")
                structured_kind = classify_html_source_kind(resp.text, final_url)
                content_kind = classify_content_source_kind(
                    page_text, source_url=final_url
                )
                observed_kind = (
                    content_kind
                    if content_kind.confidence == "high"
                    else structured_kind
                )
                kind_compatibility = compare_source_kinds(
                    expected_source_kind, observed_kind
                )
                if (
                    kind_compatibility.verdict == "incompatible"
                    or (
                        _requires_confirmed_non_web_source_kind(
                            expected_source_kind
                        )
                        and kind_compatibility.verdict != "compatible"
                    )
                ):
                    return RetrievalResult(
                        source_name="doi_resolver",
                        success=False,
                        error="DOI resolver HTML work type conflicts with citation",
                        doi=doi,
                        metadata={
                            "identity_rejected": True,
                            "expected_source_kind": expected_source_kind.kind,
                            "observed_source_kind": observed_kind.kind,
                            "source_kind_verdict": kind_compatibility.verdict,
                        },
                    )
                identity_confidence, identity_reason = _verify_text_content_identity(
                    content,
                    doi,
                    title,
                    expected_author,
                    expected_year,
                )
                if identity_confidence == "high":
                    logger.info(
                        "DOI resolver returned identity-gated HTML for %s (%d chars)",
                        doi,
                        len(page_text),
                    )
                    return RetrievalResult(
                        source_name="doi_resolver",
                        success=True,
                        representation=SourceRepresentation(
                            kind=RepresentationKind.PLAIN_TEXT,
                            media_type="text/plain",
                            content=content,
                            source_url=final_url,
                            original_kind=RepresentationKind.HTML,
                            charset="utf-8",
                            completeness="not_assessed",
                        ),
                        locations=[
                            AcquisitionLocation(
                                url=final_url,
                                provider="doi_resolver",
                                media_type="text/html",
                                representation_kind=RepresentationKind.HTML,
                                access_type=None if public else "institutional",
                                is_best=True,
                            )
                        ],
                        # Transitional compatibility for consumers that still
                        # expose acquired web text through this field.
                        abstract=page_text,
                        doi=doi,
                        title=page_titles[0] if page_titles else title,
                        authors=public_identity['observed'].get('authors', []) if public_identity else [],
                        year=public_identity['observed'].get('year') if public_identity else None,
                        metadata={
                            "identity_confidence": identity_confidence,
                            "identity_reason": identity_reason,
                            "representation_kind": RepresentationKind.PLAIN_TEXT.value,
                            "original_representation_kind": RepresentationKind.HTML.value,
                            "expected_source_kind": expected_source_kind.kind,
                            "observed_source_kind": observed_kind.kind,
                            "source_kind_verdict": kind_compatibility.verdict,
                        },
                    )
                logger.info(
                    "DOI resolver HTML rejected for %s: %s",
                    doi,
                    identity_reason,
                )
            else:
                logger.info(
                    "DOI resolver returned no complete identity-matching HTML article for %s",
                    doi,
                )

        return RetrievalResult(
            source_name="doi_resolver",
            success=False,
            error="DOI resolver returned content that failed representation or identity validation",
            doi=doi,
        )

    @_timed_retrieval
    def _try_source(
        self, source, doi, title, author, year=None, *,
        expected_source_kind=SourceKindAssessment(),
        identity_only=False,
    ):
        """Keep cleanup active even when acquisition raises/cancels."""
        try:
            return self._try_source_impl(source, doi, title, author, year,
                                         expected_source_kind=expected_source_kind,
                                         identity_only=identity_only)
        finally:
            # Legacy/custom retrievers remain untouched. API-first never
            # caches Brave responses; this also clears stale in-memory entries.
            cache = getattr(source, "_policy_query_cache", None)
            if isinstance(cache, dict):
                for key in list(cache):
                    if isinstance(key, tuple) and key[0] == "brave":
                        del cache[key]

    def _try_source_impl(
        self,
        source: RetrievalSource,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
        identity_only: bool = False,
    ) -> RetrievalResult:
        """Try a single retrieval source."""
        is_web_discovery = (
            "web_discovery" in getattr(source, "capabilities", frozenset())
            or source.name == "web_search"
        )
        retrieval_trace: list[dict] = []
        attempted_urls: set[str] = set()
        tried_search_providers: set[str] = set()
        last_result = RetrievalResult(source_name=source.name, success=False)
        landing_inspections = 0
        pdf_inspections = 0
        inspections_by_url = {}

        def identity_satisfied():
            if not identity_only:
                return False
            trace = _ACTIVE_DISCOVERY_TRACE.get()
            if not trace:
                return False
            for attempt in inspections_by_url.values():
                observation = attempt.get("landing_metadata_identity") or attempt.get("landing_metadata_observation")
                if not observation or attempt.get("outcome") not in {"unavailable", "identity_unconfirmed"}:
                    continue
                candidate = build_reference_discovery_candidate(attempt_id="identity-stop",
                    provider="independent_content", expected=trace["expected"],
                    result=RetrievalResult(source_name="independent_content", success=True, **observation["observed"]))
                # Work identity does not require classifying/acquiring its text.
                # All supplied bibliographic fields still need agreement; this
                # stop does not promote the observation to admitted evidence or
                # change its transient confirmed-identity count.
                unresolved = any(c.expected_sha256 and c.field_name != "source_kind"
                    and c.outcome not in {"agreement", "minor_difference"}
                    and not (c.field_name == "year" and trace["expected"].year == "n.d.")
                    for c in candidate.comparisons)
                if (not trace["expected"].reference_parse_review and candidate.plausible_identity_match
                        and not candidate.has_material_conflict and not unresolved):
                    return True
            return False

        def acquire(result: RetrievalResult, phase: str) -> RetrievalResult:
            nonlocal last_result, landing_inspections, pdf_inspections
            result.metadata = result.metadata or {}
            search_attempts = result.metadata.get("search_attempts", [])
            from app.services.search.candidate_audit import location_audit_fields
            candidate_locations = [
                {
                    "url": location.url,
                    "provider": location.provider,
                    "kind": location.representation_kind.value
                    if location.representation_kind
                    else None,
                    "candidate_title": location.metadata.get("search_title"),
                    "discovery_provider": location.metadata.get("search_provider"),
                    "discovery_engine_group": location.metadata.get(
                        "search_engine_group"
                    ),
                    "outcome": "not_attempted" if identity_only else "metadata_only",
                    **({"reason_code": "identity_only_no_source_acquisition"} if identity_only else {}),
                    # Empty unless the development candidate audit is on.
                    **location_audit_fields(location),
                }
                for location in result.locations
            ]
            candidate_locations.extend(result.metadata.get("candidate_dispositions", []))
            from app.services.search.transient import is_transient_brave, finalize_transient_brave, transient_acquisition_scope
            from app.log_safety import transient_discovery_logging
            from contextlib import nullcontext
            transient = is_transient_brave(result)
            reused_attempts = []
            for location in result.locations:
                prior = inspections_by_url.get(location.url)
                if prior:
                    reused_attempts.append({**prior, "url": location.url,
                        "discovery_provider": location.metadata.get("search_provider"),
                        "inspection_reused": True})
            result.metadata["_reused_location_attempts"] = reused_attempts
            def preserve_inspections():
                attempts = result.metadata.setdefault("location_attempts", [])
                attempts[:0] = result.metadata.pop("_reused_location_attempts", [])
                for item in attempts:
                    if item.get("url") and item.get("outcome") != "not_attempted":
                        inspections_by_url[item["url"]] = {key: item[key] for key in (
                            "outcome", "reason_code", "landing_metadata_identity", "landing_metadata_observation", "landing_page_observation",
                            "validated_identity_content_sha256", "independent_source_url") if key in item}
            for attempt in search_attempts:
                provider = str(attempt.get("provider", "")).lower()
                if provider:
                    tried_search_providers.add(provider)
            if result.locations:
                result.locations = [
                    location
                    for location in result.locations
                    if location.url not in attempted_urls and location.url not in inspections_by_url
                ]
                if result.locations:
                    result.full_text_url = result.locations[0].url
                else:
                    result.full_text_url = None
            if identity_only and result.success and result.locations:
                from app.services.identity_landing import inspect, MAX_PAGES
                from app.services.identity_pdf import inspect as inspect_pdf, MAX_FILES
                from app.services.reference_review_scope import scope
                expected = _ACTIVE_DISCOVERY_TRACE.get()
                expected = expected['expected'] if expected else ExpectedBibliographicFields(
                    title=title or '', authors=[author] if author else [], year=year or '', doi=doi or '')
                try:
                    with (transient_discovery_logging() if transient else nullcontext()), (transient_acquisition_scope(attempted_urls, inspections_by_url) if transient else nullcontext()):
                        for location in result.locations:
                            if scope(title, location.metadata.get('search_title')) == 'outside_bound':
                                continue
                            if location.representation_kind == RepresentationKind.PDF:
                                if pdf_inspections >= MAX_FILES:
                                    continue
                                pdf_inspections += 1
                                attempt = inspect_pdf(location, expected)
                            else:
                                if landing_inspections >= MAX_PAGES:
                                    continue
                                landing_inspections += 1
                                attempt = inspect(location, expected)
                            result.metadata.setdefault('location_attempts', []).append(attempt)
                except BaseException as exc:
                    if transient:
                        exc.args = ('Transient metadata inspection interrupted',)
                        exc.__cause__ = exc.__context__ = None
                        exc.__suppress_context__ = True
                    raise
                finally:
                    preserve_inspections()
                    if transient:
                        finalize_transient_brave(result)
                        candidate_locations.clear()
            if not identity_only and result.success and (result.locations or result.full_text_url or result.full_text):
                try:
                    with (transient_discovery_logging() if transient else nullcontext()), (transient_acquisition_scope(attempted_urls, inspections_by_url) if transient else nullcontext()):
                        result = self._download_and_cache(
                            source, result, doi, title, author, year,
                            expected_source_kind=expected_source_kind,
                        )
                        if transient:
                            finalize_transient_brave(result)
                except Exception:
                    if not transient:
                        raise
                    # Unexpected sink failures must not export a discovery URL
                    # through an exception message/chain in a task checkpoint.
                    raise RuntimeError("Transient source processing failed") from None
                except BaseException as exc:
                    if transient:
                        exc.args = ("Transient source processing interrupted",)
                        exc.__cause__ = exc.__context__ = None
                        exc.__suppress_context__ = True
                    raise
                finally:
                    preserve_inspections()
                    if transient:
                        finalize_transient_brave(result)
                        candidate_locations.clear()
            preserve_inspections()
            if transient:
                finalize_transient_brave(result)
                candidate_locations.clear()
            location_attempts = (result.metadata or {}).get("location_attempts", [])
            attempted_urls.update(
                str(attempt.get("url"))
                for attempt in location_attempts
                if attempt.get("url")
            )
            retrieval_trace.append(
                {
                    "phase": phase,
                    "source": source.name,
                    "search_attempts": search_attempts,
                    "candidate_locations": candidate_locations,
                    "location_attempts": location_attempts,
                    "transient_search_audits": result.metadata.get("transient_search_audits", []),
                    "outcome": "acquired" if result.full_text else (
                        "candidates_rejected" if result.success else "no_candidates"
                    ),
                }
            )
            result.metadata = result.metadata or {}
            result.metadata["retrieval_trace"] = list(retrieval_trace)
            last_result = result
            return result

        policy_providers = getattr(source, "_policy_providers", None)
        api_first = is_web_discovery and isinstance(policy_providers, dict) and bool(policy_providers)
        if api_first:
            result = acquire(source.search_reference(doi=doi, title=title, author=author, year=year), "api_first")
            if result.full_text or identity_satisfied():
                return result

        # Legacy DOI/title sequence is retained for the historical policy.
        if doi and not api_first:
            result = source.search_by_doi(doi)
            if result.success:
                result = acquire(result, "doi")
                if result.full_text or not is_web_discovery:
                    return result
            elif is_web_discovery:
                acquire(result, "doi")

        # Fall back to title search
        if title and not api_first:
            if is_web_discovery:
                result = source.search_by_title_author(title, author, year)  # type: ignore[call-arg]
            else:
                result = source.search_by_title_author(title, author)
            if result.success:
                result = acquire(result, "title_author")
                if result.full_text or not is_web_discovery:
                    return result
            elif is_web_discovery:
                acquire(result, "title_author")

        # A search provider returning URLs is not a retrieval success. After
        # every URL from that tier fails transport, identity, type or
        # completeness validation, ask the adapter for the next configured
        # provider and run the same shared acquisition gates.
        next_tier = getattr(source, "search_after_failed_candidates", None)
        while is_web_discovery and callable(next_tier):
            result = next_tier(
                doi=doi,
                title=title,
                author=author,
                year=year,
                tried_providers=tried_search_providers,
            )
            if not result.success:
                acquire(result, "post_validation_escalation")
                break
            result = acquire(result, "post_validation_escalation")
            if result.full_text or identity_satisfied():
                return result

        last_result.metadata = last_result.metadata or {}
        last_result.metadata["retrieval_trace"] = retrieval_trace
        return last_result

    @_timed_retrieval
    def _lookup_source(
        self,
        source: RetrievalSource,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None = None,
    ) -> RetrievalResult:
        """Collect provider evidence without acquiring or caching content."""
        attempts = []
        result = RetrievalResult(source_name=source.name, success=False, error="No applicable lookup", metadata={"lookup_applicable": False})
        lookups = []
        capabilities = getattr(source, "capabilities", frozenset())
        # Empty declarations retain the legacy interface; explicit capability
        # declarations prevent calling a DOI-only adapter as a title search.
        if doi and (not capabilities or "doi" in capabilities):
            lookups.append((f"doi:{doi}", lambda: source.search_by_doi(doi)))
        if title and (not capabilities or {"title_author", "title_search"} & capabilities):
            lookups.append((f"title:{title}" + (f" author:{author}" if author else ""),
                            lambda: source.search_by_title_author(title, author)))
        for query, lookup in lookups:
            try:
                result = (RetrievalResult(source_name=source.name, success=False,
                          error="reference_elapsed_budget_timeout") if expired() else lookup())
            except Exception as exc:
                result = RetrievalResult(source_name=source.name, success=False, error=safe_exception_code(exc))
            observation = {"query": query, "outcome": _search_execution_outcome(result)}
            metadata = result.metadata or {}
            if metadata.get('bounded_review_screen'):
                observation['bounded_review_screen'] = metadata['bounded_review_screen']
            if type(metadata.get("identity_search_result_count")) is int:
                observation["result_count"] = metadata["identity_search_result_count"]
            if metadata.get("identity_search_reason_code") in INCOMPLETE_METADATA_SEARCH_REASONS:
                observation["reason_code"] = metadata["identity_search_reason_code"]
            if source.name == "crossref" and query.startswith("doi:") and metadata.get("identifier_check") == "not_registered":
                observation["reason_code"] = "identifier_not_registered"
            attempts.append(observation)
            result.metadata = result.metadata or {}
            result.metadata["structured_search_attempts"] = list(attempts)
            if result.success:
                break
        return result

    def _lookup_structured_sources(
        self,
        sources: list[RetrievalSource],
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
    ) -> list[tuple[RetrievalSource, RetrievalResult]]:
        """Query independent metadata adapters concurrently, preserving order.

        Each adapter remains internally rate-limited and source acquisition is
        still serialized through the canonical graph. Only independent
        metadata lookups overlap, so one slow provider no longer adds its full
        timeout to every other configured provider's latency.
        """
        if len(sources) <= 1:
            results = [
                (source, self._lookup_source(source, doi, title, author, year))
                for source in sources
            ]
            return [(source, reserve_metadata_candidate(source.name, result)) for source, result in results]
        with ThreadPoolExecutor(
            max_workers=min(_MAX_STRUCTURED_PROVIDER_WORKERS, len(sources)),
            thread_name_prefix="source-metadata",
        ) as executor:
            futures = {
                source.name: executor.submit(
                    copy_context().run,
                    self._lookup_source,
                    source,
                    doi,
                    title,
                    author,
                    year,
                )
                for source in sources
            }
            results = [
                (source, futures[source.name].result())
                for source in sources
            ]
            # Reserve in configured order, not network-completion order.
            return [(source, reserve_metadata_candidate(source.name, result)) for source, result in results]

    def _try_source_sequence(
        self,
        sources: list[RetrievalSource],
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
    ) -> RetrievalResult | None:
        """Try an ordered source-specific route while retaining an abstract."""
        best_abstract: RetrievalResult | None = None
        for source in sources:
            result = self._try_source(
                source, doi, title, author, year,
                expected_source_kind=expected_source_kind,
            )
            self._record_discovery_attempt(
                category="academic_adapter",
                provider=source.name,
                result=result,
                required=_route_blocks_completion(source),
            )
            if result.success and result.full_text:
                return result
            if result.success and result.abstract and best_abstract is None:
                best_abstract = result
        return best_abstract

    @staticmethod
    def _is_public_domain_front_route(
        doi: str | None,
        year: str | None,
    ) -> bool:
        """Prioritize edition discovery only for clearly older DOI-less works."""
        if doi or not year:
            return False
        match = re.search(r"\d{4}", year)
        # This is only a conservative routing optimization, not a legal
        # public-domain determination. The adapters make the availability
        # decision and later admission retains licence evidence.
        return bool(match and int(match.group()) <= 1900)

    # Works published from this year onward are not in these catalogs. Like
    # `_is_public_domain_front_route` above this is a routing bound, not a
    # legal determination; the adapters and later admission still decide.
    _PUBLIC_DOMAIN_FALLBACK_LATEST_YEAR = 1930

    @classmethod
    def _public_domain_fallback_allowed(
        cls,
        expected_source_kind: SourceKindAssessment,
        year: str | None = None,
    ) -> bool:
        """Avoid literary-edition catalogs for modern works.

        The kind test alone sent every DOI-less book to Gutenberg and
        Wikisource. Measured over the 11-paper corpus run on 2026-09-23, those
        two were queried 20 times each for books published **1978 to 2023**,
        found nothing, and spent 615 seconds -- 9.5% of all attempt time -- on
        public-domain archives that cannot hold a modern textbook. An unknown
        year still passes: absence of a date is not evidence of recency.
        """
        if expected_source_kind.kind not in {
            "unknown",
            "monograph",
            "edited_collection",
            "book_section",
        }:
            return False
        match = re.search(r"\d{4}", year or "")
        if not match:
            return True
        return int(match.group()) <= cls._PUBLIC_DOMAIN_FALLBACK_LATEST_YEAR

    def _finalize_resolution_result(
        self,
        result: RetrievalResult,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        url_failure_reason: str | None,
    ) -> RetrievalResult:
        """Attach record-identity and failed cited-URL evidence to a result."""
        metadata = result.metadata or {}
        graph = metadata.get("canonical_work") or {}
        evidence = graph.get("identity_evidence") or []
        if evidence:
            rank = {"low": 0, "medium": 1, "high": 2}
            confidence = max(
                (item.get("confidence", "low") for item in evidence),
                key=lambda value: rank.get(value, 0),
            )
        else:
            confidence = self._verify_source_identity(
                result, doi, title, author, year
            )
        metadata["source_match_confidence"] = confidence
        if url_failure_reason:
            metadata["url_failure_reason"] = url_failure_reason
            metadata["source_note"] = (
                f"The cited URL was unavailable ({url_failure_reason}). "
                f"A representation was acquired via {result.source_name}; "
                f"canonical-work match confidence is {confidence}."
            )
        result.metadata = metadata
        return result

    @_timed_retrieval
    def _download_and_cache(
        self,
        source: RetrievalSource,
        result: RetrievalResult,
        ref_doi: str | None = None,
        ref_title: str | None = None,
        ref_author: str | None = None,
        ref_year: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
    ) -> RetrievalResult:
        """Download full text from a retrieval result and cache it.

        Tries, in order:
        1. Custom download_full_text (Gutenberg text fetch) — always cacheable
        2. OA PDF URL (from OpenAlex/S2) — always cacheable (open access)
        3. Publisher PDF URL (campus-network access) — cacheability depends on
           whether the article is actually OA (from metadata) or paywalled

        Identity gate (cache-poisoning defense, REVIEW §2.3): every candidate
        location is checked against the cited reference before it can end graph
        traversal. Wrong, incomplete and identity-uncertain representations are
        recorded as failed attempts and the next location is tried. Only a
        high-confidence representation can be returned, cached or admitted.

        Caching policy (applied only to identity-verified content):
        - OA content + public-domain texts: ALWAYS cached (free to store)
        - Paywalled content (accessed via campus IP): cached only if
          CACHE_PAYWALLED_PDFS=True (institution's copyright-policy decision)
        """
        downloaded_via_publisher = False
        safety_report = None
        source_candidate_key = None

        # If the source has already populated full_text (some do), or has a
        # custom download method, use it.
        try:
            if result.full_text is None:
                # Check if the source overrides download_full_text
                if type(source).download_full_text is not RetrievalSource.download_full_text:
                    source_candidate_key = result.full_text_url or f"adapter:{source.name}:{result.doi or result.title}"
                    require_source_candidate(source_candidate_key)
                    result = source.download_full_text(result)
                elif result.locations or result.full_text_url:
                    self._acquire_from_locations(
                        result,
                        expected_doi=ref_doi or result.doi,
                        expected_title=ref_title or result.title,
                        expected_author=ref_author,
                        expected_year=ref_year,
                        expected_source_kind=expected_source_kind,
                    )

            # If OA download failed or wasn't available, try publisher PDF
            # (works on campus networks with IP-based access)
            capabilities = getattr(self, "_acquisition_capabilities", None)
            if (
                result.full_text is None
                and result.doi
                and (capabilities is None or "publisher_url" in capabilities)
            ):
                from app.services.publisher_urls import try_download_publisher_pdf
                publisher = _extract_publisher(result)
                pdf_bytes = try_download_publisher_pdf(
                    doi=result.doi,
                    publisher=publisher,
                    oa_url=result.full_text_url,
                )
                if pdf_bytes:
                    result.set_representation(
                        SourceRepresentation(
                            kind=RepresentationKind.PDF,
                            media_type="application/pdf",
                            content=pdf_bytes,
                            source_url=result.full_text_url,
                        )
                    )
                    downloaded_via_publisher = True
                    logger.info(
                        "Downloaded PDF for %s via publisher URL (publisher=%s)",
                        result.doi, publisher or "unknown",
                    )

            # ── Identity gate (cache-poisoning defense, REVIEW §2.3) ─────────
            # Before any cache write, confirm the downloaded PDF IS the cited
            # source. Previously this path cached on a magic-byte check alone,
            # so a wrong PDF found via title search was stored under the DOI
            # key and served to every later paper citing that DOI — the exact
            # wrong-source failure documented in STATE.md §11.
            #
            # Completeness now receives an explicit source kind. Unknown inputs
            # abstain from book-only rules; confirmed incomplete PDFs never
            # become full-text evidence or cache entries.
            if result.full_text:
                require_source_candidate(source_candidate_key or
                    (result.representation.source_url if result.representation else None) or
                    result.full_text_url or "bytes:" + hashlib.sha256(result.full_text).hexdigest())
                representation_kind = (
                    result.representation.kind
                    if result.representation
                    else RepresentationKind.PDF
                    if result.full_text.startswith(PDF_MAGIC)
                    else RepresentationKind.PLAIN_TEXT
                )
                completeness = (
                    result.representation.completeness
                    if result.representation is not None
                    else "not_assessed"
                )
                if (
                    representation_kind is RepresentationKind.PDF
                    and self._durable_repository_active()
                ):
                    try:
                        safety_report = inspect_uploaded_pdf(result.full_text)
                    except FileSafetyUnavailable as exc:
                        result.metadata = result.metadata or {}
                        result.metadata["durable_admission"] = {
                            "state": "not_stored",
                            "reason": "required hostile-file inspection unavailable",
                            "retryable": True,
                        }
                        self._clear_retrieved_representation(result)
                        logger.warning(
                            "Required safety inspection unavailable for retrieved %s: %s",
                            ref_doi or ref_title or result.doi,
                            exc,
                        )
                        return result
                    if safety_report.verdict is SafetyVerdict.REJECTED:
                        result.metadata = result.metadata or {}
                        result.metadata["durable_admission"] = {
                            "state": "rejected",
                            "reason": "hostile-file inspection rejected representation",
                            "findings": list(safety_report.findings),
                        }
                        self._clear_retrieved_representation(result)
                        logger.warning(
                            "Rejected retrieved PDF before durable admission: %s",
                            "; ".join(safety_report.findings),
                        )
                        return result
                result.metadata = result.metadata or {}
                accepted_preflight_hash = result.metadata.get(
                    "accepted_representation_sha256"
                )
                preflight_matches = bool(
                    accepted_preflight_hash
                    and accepted_preflight_hash
                    == hashlib.sha256(result.full_text).hexdigest()
                    and result.metadata.get("identity_confidence") == "high"
                )
                if preflight_matches:
                    identity_confidence = "high"
                    identity_reason = result.metadata.get(
                        "identity_reason",
                        "accepted location preflight",
                    )
                    completeness = result.metadata.get(
                        "completeness",
                        "not_assessed",
                    )
                    text_quality = result.metadata.get(
                        "text_quality",
                        "not_assessed",
                    )
                elif representation_kind is RepresentationKind.PDF:
                    validation = validate_retrieved_pdf(
                        result.full_text,
                        inspection_reconciliation=True,
                        inspection_provider=(lambda data, validation: self._inspect_candidate(data,
                            title=ref_title or result.title, author=ref_author, year=ref_year,
                            doi=ref_doi or result.doi, source_kind=expected_source_kind.kind, pdf=True))
                            if safety_report and safety_report.verdict is SafetyVerdict.CLEAN else None,
                        expected_doi=ref_doi or result.doi,
                        expected_title=ref_title or result.title,
                        expected_author=ref_author,
                        expected_year=ref_year,
                        document_kind=_document_kind_for_result(
                            result,
                            has_doi=bool(ref_doi or result.doi),
                            expected_source_kind=expected_source_kind.kind,
                        ),
                        expected_page_range=_expected_page_range(result),
                        expected_source_kind=expected_source_kind.kind,
                        expected_source_kind_confidence=expected_source_kind.confidence,
                        expected_source_kind_evidence=expected_source_kind.evidence,
                    )
                    if getattr(validation, 'source_inspection', None) is not None:
                        result.metadata['source_inspection'] = validation.source_inspection
                    identity_confidence = validation.identity_confidence
                    if getattr(validation, 'reason_code', None) == 'identity_insufficient_observations':
                        identity_confidence = 'low'
                        result.metadata['identity_reason_code'] = validation.reason_code
                    identity_reason = validation.reason
                    completeness = getattr(
                        validation, "completeness", "not_assessed"
                    )
                    from app.services.publisher_preview import publisher_preview_url
                    if result.representation and publisher_preview_url(result.representation.source_url):
                        completeness = 'incomplete'
                    text_quality = getattr(
                        validation, "text_quality", "not_assessed"
                    )
                    if result.representation is not None:
                        result.representation.completeness = completeness
                else:
                    provider_identity = self._verify_source_identity(
                        result,
                        ref_doi or result.doi,
                        ref_title or result.title,
                        ref_author,
                        ref_year,
                    )
                    content_identity, content_reason = _verify_text_content_identity(
                        result.full_text,
                        ref_doi or result.doi,
                        ref_title or result.title,
                        ref_author,
                        ref_year,
                    )
                    identity_confidence = _combine_identity_confidence(
                        provider_identity, content_identity
                    )
                    identity_reason = (
                        f"Typed {representation_kind.value} representation; "
                        f"provider metadata identity={provider_identity}; "
                        f"content identity={content_identity} ({content_reason}); "
                        "content completeness remains independently assessed"
                    )
                    result.metadata = result.metadata or {}
                    result.metadata["provider_identity_confidence"] = provider_identity
                    result.metadata["content_identity_confidence"] = content_identity
                    text_quality = "digital"
                result.metadata = result.metadata or {}
                result.metadata["identity_confidence"] = identity_confidence
                result.metadata["identity_reason"] = identity_reason
                result.metadata["representation_kind"] = representation_kind.value
                result.metadata["completeness"] = completeness
                result.metadata["text_quality"] = text_quality

                if identity_confidence != "high":
                    self._retain_provisional_candidate(result, confidence=identity_confidence,
                        reason=identity_reason, completeness=completeness,
                        text_quality=text_quality,
                        expected_title=ref_title or result.title,
                        expected_author=ref_author, expected_year=ref_year,
                        kind_verdict=getattr(validation, 'source_kind_verdict', 'unknown')
                            if representation_kind is RepresentationKind.PDF else 'unknown')
                    # Wrong or insufficiently identified source — do not return
                    # it as a hit or cache it.  Resolve can continue through the
                    # remaining permitted retrieval routes.
                    logger.info(
                        "Rejecting %s from %s: identity was not high-confidence (%s)",
                        ref_doi or ref_title or result.doi, source.name,
                        identity_reason,
                    )
                    self._clear_retrieved_representation(result)
                    if identity_confidence == "rejected":
                        result.metadata["identity_rejected"] = True
                    else:
                        result.metadata["identity_unconfirmed"] = True
                    return result

                if (
                    representation_kind is RepresentationKind.PDF
                    and completeness == "incomplete"
                ):
                    from app.services.publisher_preview import preview_receipt
                    receipt = preview_receipt(result.full_text, result.representation.source_url,
                        identity=identity_confidence, completeness=completeness,
                        source_kind=expected_source_kind.kind, text_quality=text_quality,
                        cleanliness=safety_report.verdict.value if safety_report else 'not_assessed')
                    if receipt is None:
                        self._clear_retrieved_representation(result)
                        result.metadata["completeness_rejected"] = True
                        return result
                    result.metadata['publisher_preview'] = receipt

                # Determine cacheability:
                # - Content from custom/OA sources (Gutenberg, OA URL) → cacheable
                #   (only now that identity is confirmed)
                # - Content from publisher URL → check if it's actually OA.
                #   If OA, cache. If not OA (paywalled), cache only if
                #   CACHE_PAYWALLED_PDFS=True.
                # - Identity "low"/"skipped" → never cached (uncertain source).
                is_oa = _check_is_oa(result)
                should_cache = (
                    representation_kind is RepresentationKind.PDF
                    and identity_confidence == "high"
                    and completeness == "complete"
                )

                if should_cache and downloaded_via_publisher and not is_oa:
                    # Paywalled content accessed via campus IP
                    if not settings.CACHE_PAYWALLED_PDFS:
                        should_cache = False
                        logger.info(
                            "Paywalled PDF for %s verified in-memory, NOT cached "
                            "(OA=False, CACHE_PAYWALLED_PDFS=False)",
                            result.doi,
                        )

                if (
                    identity_confidence == "high"
                    and self._durable_repository_active()
                ):
                    self._persist_retrieved_representation(
                        result,
                        ref_doi=ref_doi,
                        ref_title=ref_title,
                        ref_author=ref_author,
                        ref_year=ref_year,
                        identity_confidence=identity_confidence,
                        identity_reason=identity_reason,
                        downloaded_via_publisher=downloaded_via_publisher,
                        safety_report=safety_report,
                        expected_source_kind=expected_source_kind,
                    )
                elif should_cache and self._backend:
                    key = self._cache_key_for_verified(result.doi, result.title)
                    self._backend.upload(result.full_text, key)
                    # Audit log: record the identity verdict + the reference fields
                    # the gate checked against, so cached objects can be audited
                    # without reconstructing the mapping from keys alone.
                    logger.info(
                        "Cached a validated source representation "
                        "(source=%s, identity=%s, object=%s)",
                        source.name,
                        identity_confidence,
                        private_value_id("storage_key", key),
                    )
                    if downloaded_via_publisher:
                        logger.info(
                            "Cached %s PDF (oa=%s)",
                            "OA" if is_oa else "paywalled", is_oa,
                        )
                elif not should_cache:
                    # Not cached because identity confidence is low/skipped, or
                    # because paywall policy forbids it. The PDF is still
                    # returned (flagged) so a human reviewer can see it.
                    logger.info(
                        "Not caching candidate from %s: identity_confidence=%s",
                        source.name,
                        identity_confidence,
                    )
        except CandidateBudgetExceeded:
            result.metadata = result.metadata or {}
            result.metadata["candidate_budget_skipped"] = True
            result.success = False
            self._clear_retrieved_representation(result)
        except Exception as e:
            logger.warning(
                "Failed to download full text from %s (type=%s)",
                source.name,
                type(e).__name__,
            )
        finally:
            from app.services.search.transient import finalize_transient_brave
            finalize_transient_brave(result)

        return result

    def _durable_repository_active(self) -> bool:
        return bool(
            settings.SOURCE_REPOSITORY_ENABLED
            and self._backend is not None
            and getattr(self, "_repository_session_factory", None) is not None
        )

    @staticmethod
    def _clear_retrieved_representation(result: RetrievalResult) -> None:
        result.full_text = None
        result.representation = None
        result.parent_representation = None

    @staticmethod
    def _retrieval_retention_decision(
        result: RetrievalResult,
        *,
        downloaded_via_publisher: bool,
    ) -> tuple[str | None, str]:
        metadata = result.metadata or {}
        explicit = metadata.get("license_class")
        if explicit in LICENSE_CLASSES:
            return explicit, "explicit_representation_metadata"
        if result.source_name in {"gutenberg", "wikisource"}:
            return "public_domain", "public_domain_adapter_policy"
        if _check_is_oa(result) or any(
            location.access_type in {"open_access", "public_domain"}
            for location in result.locations
        ):
            return "open_access", "provider_oa_or_licence_evidence"
        if (
            settings.CACHE_PAYWALLED_PDFS
            and (downloaded_via_publisher or result.source_name == "elsevier")
        ):
            return "paywalled_db_retrieved", "deployment_subscription_retention_policy"

        representation = result.representation
        source_url = (
            representation.source_url if representation is not None else None
        ) or result.full_text_url
        publicly_downloaded = bool(
            source_url
            and source_url.casefold().startswith(("http://", "https://"))
            and not downloaded_via_publisher
            and result.source_name != "elsevier"
            and not any(
                location.access_type == "institutional"
                for location in result.locations
            )
        )
        if (
            publicly_downloaded
            and settings.PUBLIC_RETRIEVAL_RETENTION_POLICY == "store_scoped"
        ):
            # Public reachability is not a licence conclusion. The deployment
            # has chosen scoped retention, so keep the representation under the
            # neutral restricted class and preserve the policy basis.
            return "rights_unclassified", "deployment_public_download_policy"
        return None, "deployment_policy_did_not_authorize_retention"

    def _persist_retrieved_representation(
        self,
        result: RetrievalResult,
        *,
        ref_doi: str | None,
        ref_title: str | None,
        ref_author: str | None,
        ref_year: str | None,
        identity_confidence: str,
        identity_reason: str,
        downloaded_via_publisher: bool,
        safety_report,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
    ) -> None:
        """Admit a validated provider representation through the durable core."""
        from app.services.search.transient import finalize_transient_brave
        finalize_transient_brave(result)
        representation = result.representation
        if representation is None or not representation.content:
            return
        license_class, retention_basis = self._retrieval_retention_decision(
            result,
            downloaded_via_publisher=downloaded_via_publisher,
        )
        result.metadata = result.metadata or {}
        if license_class is None:
            result.metadata["durable_admission"] = {
                "state": "not_stored",
                "reason": "retention policy does not authorize this representation",
                "retention_basis": retention_basis,
            }
            return

        completeness = (
            representation.completeness
            or result.metadata.get("completeness")
            or "not_assessed"
        ).casefold()
        if completeness == "skipped":
            completeness = "not_assessed"
        if representation.kind is RepresentationKind.PDF:
            cleanliness = (
                safety_report.verdict.value if safety_report else "not_assessed"
            )
            safety_evidence = {
                "verdict": cleanliness,
                "structural_verdict": (
                    safety_report.structural_verdict.value if safety_report else None
                ),
                "malware_verdict": (
                    safety_report.malware_verdict.value if safety_report else None
                ),
            }
        else:
            # Only normalized UTF-8 text is retained; executable HTML/XML/EPUB
            # bytes are not the stored representation.
            cleanliness = "clean"
            safety_evidence = {
                "verdict": "clean",
                "basis": "normalized non-executable text representation",
                "original_kind": (
                    representation.original_kind.value
                    if representation.original_kind else None
                ),
            }

        license_expires_at = None
        if (
            license_class == "paywalled_db_retrieved"
            and settings.CAMPUS_ACCESS_TTL_DAYS > 0
        ):
            license_expires_at = datetime.now(timezone.utc) + timedelta(
                days=settings.CAMPUS_ACCESS_TTL_DAYS
            )

        factory = self._repository_session_factory
        with factory() as session:
            try:
                work = WorkIdentity(
                    title=ref_title or result.title or ref_doi or result.doi or "",
                    work_type=(
                        expected_source_kind.kind
                        if expected_source_kind.is_known
                        else classify_provider_source_kind(result.metadata).kind
                        if classify_provider_source_kind(result.metadata).is_known
                        else "academic_work"
                    ),
                    doi=ref_doi or result.doi,
                    author=ref_author
                    or (result.authors[0] if result.authors else None),
                    year=ref_year or result.year,
                )
                validation_evidence = {
                    "publisher_preview": result.metadata.get("publisher_preview"),
                    "identity_reason": identity_reason,
                    "identity_confidence": identity_confidence,
                    "expected_source_kind": expected_source_kind.kind,
                    "expected_source_kind_confidence": (
                        expected_source_kind.confidence
                    ),
                    "expected_source_kind_evidence": list(
                        expected_source_kind.evidence
                    ),
                    "observed_source_kind": result.metadata.get(
                        "observed_source_kind", "unknown"
                    ),
                    "source_kind_verdict": result.metadata.get(
                        "source_kind_verdict", "unknown"
                    ),
                    "file_safety": safety_evidence,
                    "location_attempts": _withhold_search_result_urls(
                        result.metadata.get("location_attempts", [])
                    ),
                    "transient_search_audits": result.metadata.get("transient_search_audits", []),
                    "search_retention_policy": result.metadata.get("search_retention_policy"),
                    "canonical_work": result.metadata.get("canonical_work", {}),
                    "retention_basis": retention_basis,
                    "public_retrieval_retention_policy": (
                        settings.PUBLIC_RETRIEVAL_RETENTION_POLICY
                    ),
                }
                ocr_evidence = result.metadata.get("ocr_derivative")
                if ocr_evidence and result.parent_representation is not None:
                    validation_evidence["ocr_derivative"] = ocr_evidence
                    parent_record, record = admit_derived_representation_pair(
                        session,
                        self._backend,
                        parent_request=AdmissionRequest(
                            work=work,
                            representation=result.parent_representation,
                            provenance="local_ocr_parent",
                            license_class=license_class,
                            scope_type="personal_owner",
                            scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
                            identity_verdict="verified",
                            identity_confidence=0.8,
                            completeness_verdict=completeness,
                            cleanliness_verdict="clean",
                            text_quality="pure_scan",
                            expires_at=license_expires_at,
                            admitted_by="retrieval_pipeline",
                            validation_evidence={
                                "ocr_parent": {
                                    "derivative_content_sha256": ocr_evidence[
                                        "derivative_content_sha256"
                                    ],
                                    "derivation_manifest_sha256": ocr_evidence[
                                        "derivation_manifest_sha256"
                                    ],
                                    "file_safety": ocr_evidence[
                                        "parent_file_safety"
                                    ],
                                }
                            },
                            request_acceptance=False,
                        ),
                        derivative_request=AdmissionRequest(
                            work=work,
                            representation=representation,
                            provenance="local_ocr_derivative",
                            license_class=license_class,
                            scope_type="personal_owner",
                            scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
                            identity_verdict="verified",
                            identity_confidence=0.8,
                            completeness_verdict=completeness,
                            cleanliness_verdict="clean",
                            text_quality="scan_ocr",
                            expires_at=license_expires_at,
                            admitted_by="retrieval_pipeline",
                            validation_evidence=validation_evidence,
                        ),
                    )
                else:
                    parent_record = None
                    record = admit_representation(
                        session,
                        self._backend,
                        AdmissionRequest(
                            work=work,
                            representation=representation,
                            provenance=result.source_name,
                            license_class=license_class,
                            scope_type="personal_owner",
                            scope_id=settings.SOURCE_REPOSITORY_SCOPE_ID,
                            identity_verdict="verified",
                            identity_confidence=(
                                1.0 if identity_confidence == "high" else 0.7
                            ),
                            completeness_verdict=completeness,
                            cleanliness_verdict=cleanliness,
                            text_quality=result.metadata.get(
                                "text_quality", "not_assessed"
                            ),
                            edition_or_version=next(
                                (
                                    location.version
                                    for location in result.locations
                                    if location.version
                                ),
                                result.metadata.get("edition_or_version"),
                            ),
                            expires_at=license_expires_at,
                            admitted_by="retrieval_pipeline",
                            validation_evidence=validation_evidence,
                        ),
                    )
                commit_source_admissions(session)
                result.metadata["durable_admission"] = {
                    "state": record.admission_state,
                    "representation_id": str(record.id),
                    "parent_representation_id": (
                        str(parent_record.id) if parent_record is not None else None
                    ),
                    "license_class": license_class,
                    "retention_basis": retention_basis,
                }
            except AdmissionError as exc:
                rollback_source_admissions(session, self._backend)
                result.metadata["durable_admission"] = {
                    "state": "not_stored",
                    "reason": "admission_rejected",
                }
                logger.warning(
                    "Durable retrieval admission rejected (type=%s)",
                    type(exc).__name__,
                )
            except Exception:
                rollback_source_admissions(session, self._backend)
                raise

    def _cache_key_for_verified(
        self,
        doi: str | None,
        title: str | None,
    ) -> str:
        """Generate an S3 key for a verified document."""
        if doi:
            return f"by-doi/{doi}.pdf"
        if title:
            h = hashlib.sha256(title.strip().lower().encode()).hexdigest()[:12]
            return f"by-title-hash/{h}.pdf"
        raise ValueError("Need DOI or title for cache key")


def _extract_publisher(result: RetrievalResult) -> str | None:
    """Extract the publisher name from a RetrievalResult's metadata.

    Used to construct publisher-specific PDF URLs for campus-network access.
    """
    if not result.metadata:
        return None
    # Crossref stores publisher in metadata["publisher"] or metadata["message"]["publisher"]
    if "publisher" in result.metadata:
        return result.metadata["publisher"]
    msg = result.metadata.get("message", {})
    if isinstance(msg, dict) and "publisher" in msg:
        return msg["publisher"]
    return None


def _metadata_records(result: RetrievalResult) -> list[dict]:
    """Return bounded provider metadata records relevant to validation."""
    root = result.metadata if isinstance(result.metadata, dict) else {}
    records = [root]
    message = root.get("message")
    if isinstance(message, dict):
        records.append(message)
    providers = root.get("provider_metadata")
    if isinstance(providers, dict):
        for value in providers.values():
            if not isinstance(value, dict):
                continue
            records.append(value)
            nested = value.get("message")
            if isinstance(nested, dict):
                records.append(nested)
    return records


def _document_kind_for_result(
    result: RetrievalResult,
    *,
    has_doi: bool,
    expected_source_kind: str | None = None,
) -> str:
    expected_document_kind = document_kind_for_source_kind(expected_source_kind)
    if expected_document_kind != "unknown":
        return expected_document_kind
    type_map = {
        "journal-article": "article",
        "proceedings-article": "article",
        "posted-content": "article",
        "book-chapter": "chapter",
        "book-section": "chapter",
        "book": "book",
        "monograph": "book",
        "edited-book": "book",
        "reference-book": "book",
    }
    for metadata in _metadata_records(result):
        raw = metadata.get("work_type") or metadata.get("type")
        if isinstance(raw, str) and raw.casefold() in type_map:
            return type_map[raw.casefold()]
    return "article" if has_doi else "unknown"


# Adapters kept for bibliographic corroboration rather than for reaching the
# work. Their cost is real — CORE alone serializes every call behind a ten
# second interval — and across all stored jobs they supplied the acquired
# representation once in seventy-five. When identity is already confirmed and a
# route to the text already exists, they have no remaining question to answer.
_CORROBORATION_ONLY_SOURCES = frozenset(
    {"core", "semantic_scholar", "elsevier", "datacite", "open_library", "eric"}
)
_CONFIRMED_DISCOVERY_OUTCOMES = frozenset(
    {"confirmed", "confirmed_with_minor_differences"}
)


def _expected_page_range(result: RetrievalResult) -> tuple[int, int] | None:
    """Extract an inclusive provider page range without guessing editions."""
    range_re = re.compile(r"(?<!\d)(\d{1,6})\s*[-–—]\s*(\d{1,6})(?!\d)")
    for metadata in _metadata_records(result):
        raw = metadata.get("page")
        if not isinstance(raw, str):
            continue
        match = range_re.search(raw)
        if not match:
            continue
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end and 1 < end - start + 1 <= 500:
            return start, end
    return None


def _check_is_oa(result: RetrievalResult) -> bool:
    """Determine whether a retrieved work is open access.

    Checks the OA flags from the source metadata:
    - OpenAlex: best_oa_location.is_oa, open_access.is_oa, open_access.oa_status
    - Semantic Scholar: presence of openAccessPdf
    - If the full_text_url came from an OA source, assume OA

    When in doubt (no metadata signal), assumes NOT OA (conservative — better
    to over-protect copyrighted content than to cache it accidentally).
    """
    if any(
        location.access_type == "open_access" or bool(location.license)
        for location in result.locations
    ):
        return True

    if not result.metadata:
        return False

    meta = result.metadata

    # OpenAlex: check multiple OA indicators
    best_oa = meta.get("best_oa_location") or {}
    if best_oa.get("is_oa"):
        return True
    oa_info = meta.get("open_access") or {}
    if oa_info.get("is_oa") or oa_info.get("oa_status") in ("gold", "green", "hybrid", "bronze"):
        return True

    # Semantic Scholar: openAccessPdf present indicates OA
    if meta.get("openAccessPdf"):
        return True

    # Canonical work graphs retain each adapter's unmodified metadata under
    # provider_metadata instead of flattening conflicting provider fields.
    provider_metadata = meta.get("provider_metadata") or {}
    if isinstance(provider_metadata, dict):
        for provider_meta in provider_metadata.values():
            if not isinstance(provider_meta, dict):
                continue
            best_oa = provider_meta.get("best_oa_location") or {}
            oa_info = provider_meta.get("open_access") or {}
            if best_oa.get("is_oa") or oa_info.get("is_oa"):
                return True
            if oa_info.get("oa_status") in (
                "gold", "green", "hybrid", "bronze"
            ):
                return True
            if provider_meta.get("openAccessPdf"):
                return True

    # If the result came from an OA URL (not a publisher paywall URL),
    # and the URL looks like a known OA host
    if result.full_text_url:
        oa_hosts = ("doi.org", "ncbi.nlm.nih.gov", "arxiv.org", "biorxiv.org",
                     "plos.org", "frontiersin.org", "mdpi.com", "doaj.org")
        if any(host in result.full_text_url for host in oa_hosts):
            return True

    return False
