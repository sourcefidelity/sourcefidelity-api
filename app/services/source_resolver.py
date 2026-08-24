"""Source resolution orchestrator.

Implements the resolution priority chain:
    Local S3 cache -> Student URL -> Retrieval Sources -> Fail
"""

import hashlib
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx

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
from app.services.source_validator import validate_retrieved_pdf
from app.services.source_repository import (
    LICENSE_CLASSES,
    AdmissionError,
    AdmissionRequest,
    WorkIdentity,
    admit_representation,
    find_accepted_representation,
)
from app.services.safe_fetch import (
    UnsafeUrlError,
    ResponseTooLargeError,
    safe_request,
    safe_fetch_bytes,
)

logger = logging.getLogger(__name__)

# Magic-byte check for PDF
PDF_MAGIC = b"%PDF-"
_MAX_LOCATION_ATTEMPTS = 3


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


def _doi_identity_matches(expected: str | None, observed: str | None) -> bool:
    if not expected or not observed:
        return False
    def clean(value: str) -> str:
        return re.sub(
            r"^https?://(?:dx\.)?doi\.org/", "", value.strip().casefold()
        )

    return clean(expected) == clean(observed)


def _location_rank(location: AcquisitionLocation) -> tuple[int, int, int]:
    """Prefer direct, useful representations while retaining provider ranking."""
    kind_rank = {
        RepresentationKind.PDF: 0,
        RepresentationKind.XML: 1,
        RepresentationKind.PLAIN_TEXT: 2,
        RepresentationKind.EPUB: 3,
        RepresentationKind.HTML: 4,
        None: 5,
    }
    version_rank = {
        "publishedVersion": 0,
        "acceptedVersion": 1,
        "submittedVersion": 2,
    }.get(location.version, 3)
    return (
        kind_rank.get(location.representation_kind, 5),
        0 if location.is_best else 1,
        version_rank,
    )


def _extract_text_representation(payload: bytes, kind: RepresentationKind) -> str:
    """Normalize XML or plain bytes without claiming that metadata is complete."""
    text = payload.decode("utf-8", errors="replace")
    if kind is RepresentationKind.XML:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(text, "xml").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


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
    if expected_doi and expected_doi.lower() in compact_text:
        return "high", "exact DOI appears in acquired text"

    title_overlap = 0.0
    if expected_title:
        title_tokens = {
            token for token in re.findall(r"[a-z0-9]+", expected_title.lower())
            if len(token) >= 3
        }
        content_tokens = set(re.findall(r"[a-z0-9]+", compact_text))
        if title_tokens:
            title_overlap = len(title_tokens & content_tokens) / len(title_tokens)

    support = 0
    if expected_author:
        surname = expected_author.split(",", 1)[0].strip().lower()
        if len(surname) >= 3 and surname in compact_text:
            support += 1
    if expected_year and expected_year in compact_text:
        support += 1

    if title_overlap >= 0.8 and support:
        return "high", f"title token overlap={title_overlap:.2f} with author/year support"
    if title_overlap >= 0.6:
        return "medium", f"title token overlap={title_overlap:.2f} in acquired text"
    return "low", f"insufficient content-level identity evidence (title overlap={title_overlap:.2f})"


def _combine_identity_confidence(provider: str, content: str) -> str:
    """Require both record identity and representation identity for high confidence."""
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
            logger.warning("Storage backend unavailable — caching disabled: %s", str(e)[:80])
            self._backend = None
        self._repository_session_factory = (
            SessionLocal if settings.SOURCE_REPOSITORY_ENABLED else None
        )
        self._retrieval_sources = get_retrieval_sources()
        self._lookup_cache = RetrievalLookupCache()

    # ── Public API ────────────────────────────────────────

    def resolve_reference(
        self,
        reference,
        *,
        isbn: str | None = None,
    ) -> RetrievalResult:
        """Resolve a ParsedReference without dropping its typed identity data."""
        return self.resolve(
            doi=getattr(reference, "doi", None) or None,
            isbn=isbn,
            title=getattr(reference, "title", None) or None,
            author=getattr(reference, "author", None) or None,
            year=getattr(reference, "year", None) or None,
            student_url=getattr(reference, "url", None) or None,
            raw_ref=getattr(reference, "raw_ref", None) or None,
            source_kind=getattr(reference, "source_kind", None) or None,
            source_kind_confidence=(
                getattr(reference, "source_kind_confidence", "unknown")
            ),
            source_kind_evidence=tuple(
                getattr(reference, "source_kind_evidence", ()) or ()
            ),
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
        if result.success and result.full_text:
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
        if lookup_record and lookup_record.is_fresh(
            policy_signature=policy_signature,
            student_url=student_url,
        ):
            if lookup_record.result is not None:
                return lookup_record.result
            raise SourceResolutionError(
                f"Source not found (fresh cached lookup): doi={doi}, "
                f"isbn={isbn}, title={title}"
            )

        # 2. Try student URL — PDF first, then web-page text for HTML sources.
        capabilities = getattr(self, "_acquisition_capabilities", None)
        if student_url and (capabilities is None or "student_url" in capabilities):
            if is_library_locator_url(student_url):
                url_failure_reason = (
                    "Library locator identifies a catalog/discovery record; "
                    "page content is not source evidence"
                )
            elif _url_prefers_pdf(student_url):
                result = self._try_student_url(
                    student_url, doi, title, author, year,
                    expected_source_kind=expected_kind,
                )
                if result.success and result.full_text:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return result  # Authoritative — the student's actual source
                # An explicit PDF route may still return an HTML landing page.
                if not result.success and "not a pdf" in (result.error or "").lower():
                    web_result = self._try_web_fetch(
                        student_url, title, expected_source_kind=expected_kind
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
                web_result = self._try_web_fetch(
                    student_url, title, expected_source_kind=expected_kind
                )
                if web_result.success:
                    self._delete_lookup_cache(doi, title, author, year, expected_kind.kind)
                    return web_result
                if "returned a pdf" in (web_result.error or "").lower():
                    result = self._try_student_url(
                        student_url, doi, title, author, year,
                        expected_source_kind=expected_kind,
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
            logger.info(
                "Skipping academic DBs for traditional-media reference: %s",
                (raw_ref or title or "")[:60],
            )
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

            graph = CanonicalWorkGraph(
                expected_doi=doi,
                expected_title=title,
                expected_author=author,
                expected_year=year,
                expected_source_kind=expected_kind.kind,
                expected_source_kind_confidence=expected_kind.confidence,
                expected_source_kind_evidence=expected_kind.evidence,
            )
            for source in structured_sources:
                provider_result = self._lookup_source(
                    source, doi, title, author, year
                )
                if provider_result.success:
                    assessment = graph.add(provider_result)
                    if not assessment.accepted:
                        logger.info(
                            "Rejected %s canonical-work candidate: %s",
                            source.name,
                            assessment.reason,
                        )

            merged_result = graph.to_result()
            if merged_result.success:
                merged_result = self._download_and_cache(
                    _CANONICAL_GRAPH_SOURCE,
                    merged_result,
                    doi,
                    title,
                    author,
                    year,
                    expected_source_kind=expected_kind,
                )
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
            if not tried_public_domain and not doi:
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

            for source in web_sources:
                result = self._try_source(
                    source, doi, title, author, year,
                    expected_source_kind=expected_kind,
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
        if lookup_record and lookup_record.result is not None:
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
            return RetrievalResult(source_name="local_cache", success=False)
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
                    return RetrievalResult(source_name="local_cache", success=False)
                kind = RepresentationKind(record.representation_kind)
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
                    doi=record.canonical_work.doi,
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
            logger.warning("Student URL download failed: %s", e)
            return RetrievalResult(source_name="student_url", success=False, error=str(e))

        # Identity check via the same validator the academic-DB path uses.
        validation = validate_retrieved_pdf(
            data,
            expected_doi=doi,
            expected_title=title,
            expected_author=author,
            expected_year=year,
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
            validation.identity_confidence == "high"
            and validation.completeness != "incomplete"
        ):
            # Verified — cache (if available) and return.
            if self._backend and validation.completeness == "complete":
                key = self._cache_key_for_verified(doi, title)
                self._backend.upload(data, key)
            return RetrievalResult(
                source_name="student_url",
                success=True,
                representation=SourceRepresentation(
                    kind=RepresentationKind.PDF,
                    media_type="application/pdf",
                    content=data,
                    source_url=url,
                    completeness=validation.completeness,
                ),
                doi=doi,
                title=title,
                metadata={
                    "identity_confidence": validation.identity_confidence,
                    "identity_reason": validation.reason,
                    "completeness": validation.completeness,
                    "expected_source_kind": expected_source_kind.kind,
                    "observed_source_kind": validation.observed_source_kind,
                    "source_kind_verdict": validation.source_kind_verdict,
                },
            )

        if validation.completeness == "incomplete":
            return RetrievalResult(
                source_name="student_url",
                success=False,
                error="Student-linked PDF failed completeness validation",
                doi=doi,
                title=title,
                metadata={
                    "identity_confidence": validation.identity_confidence,
                    "completeness": validation.completeness,
                    "completeness_rejected": True,
                },
            )

        # Anything short of high-confidence identity is not the accepted cited
        # representation. Keep looking through the remaining retrieval routes.
        logger.warning(
            "Student URL identity mismatch (%s): %s — discarding.",
            validation.identity_confidence, validation.reason[:80],
        )
        # Keep the "not a PDF"-style routing cue intact: a genuine non-PDF is
        # handled earlier by _safe_download's error; here the URL was a PDF but
        # the wrong one.
        return RetrievalResult(
            source_name="student_url",
            success=False,
            error=f"Identity mismatch ({validation.identity_confidence}): "
                  f"content does not match cited reference",
        )

    def _safe_download(self, url: str) -> bytes:
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
        timeout = settings.STUDENT_URL_TIMEOUT_SECONDS
        max_bytes = settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024
        for transport_attempt in range(2):
            try:
                data = safe_fetch_bytes(
                    url,
                    max_bytes=max_bytes,
                    accept_content_types=("application/pdf",),
                    timeout=timeout,
                    max_meta_refreshes=1,
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
                        url[:80],
                    )
                    continue
                raise ValueError(f"Download failed: {str(e)[:120]}") from e
            except httpx.HTTPError as e:
                # Status errors are deterministic for this attempt and do not
                # consume the one retry reserved for connection failures.
                raise ValueError(f"Download failed: {str(e)[:120]}") from e

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
            resp = safe_request(
                url,
                headers={"Accept": "text/html,application/xhtml+xml,*/*"},
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.debug("OA landing page fetch failed for %s: %s", url[:60], e)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_url = None

        # Method 1: citation_pdf_url meta tag (highest confidence — academic standard)
        meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta and meta.get("content"):
            pdf_url = meta["content"]
            logger.info("Found PDF via citation_pdf_url meta tag: %s", pdf_url[:80])

        # Method 2: <a> tags with href ending in .pdf
        if not pdf_url:
            for a in soup.find_all("a", href=True):
                href = a["href"].lower()
                if href.endswith(".pdf") or ".pdf?" in href:
                    pdf_url = urljoin(str(resp.url), a["href"])
                    logger.info("Found PDF via .pdf link: %s", pdf_url[:80])
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
                    logger.info("Found PDF via link text '%s': %s", text[:30], pdf_url[:80])
                    break

        if not pdf_url:
            return None

        # Download the found PDF URL (SSRF-guarded + size-capped via safe_fetch;
        # pdf_url comes from the parsed landing-page HTML, so it must be
        # re-validated like any other attacker-influenced URL).
        try:
            pdf_data = safe_fetch_bytes(
                pdf_url,
                accept_content_types=("application/pdf",),
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                max_meta_refreshes=1,
            )
            if pdf_data[:5] == PDF_MAGIC:
                logger.info("OA landing page PDF download successful: %d bytes", len(pdf_data))
                return pdf_data
        except Exception as e:
            logger.debug("OA landing page PDF download failed: %s", str(e)[:60])

        return None

    def _preflight_acquired_representation(
        self,
        result: RetrievalResult,
        *,
        expected_doi: str | None,
        expected_title: str | None,
        expected_author: str | None,
        expected_year: str | None,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
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
            if self._durable_repository_active():
                try:
                    safety_report = inspect_uploaded_pdf(content)
                except FileSafetyUnavailable as exc:
                    return False, "safety_unavailable", str(exc)[:160]
                if safety_report.verdict is SafetyVerdict.REJECTED:
                    reason = "; ".join(safety_report.findings) or "hostile-file rejection"
                    return False, "safety_rejected", reason[:160]

            validation = validate_retrieved_pdf(
                content,
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
            identity_reason = validation.reason
            completeness = getattr(validation, "completeness", "not_assessed")
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
        if identity_confidence != "high":
            outcome = (
                "identity_rejected"
                if identity_confidence == "rejected"
                else "identity_unconfirmed"
            )
            return False, outcome, identity_reason[:160]
        if kind is RepresentationKind.PDF and completeness == "incomplete":
            return False, "completeness_rejected", validation.reason[:160]
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
        attempts: list[dict] = []
        pending = list(ranked)
        attempted_urls: set[str] = set()
        html_fallback: tuple[
            str, str, str, SourceKindAssessment, bool | None
        ] | None = None
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
            }
            try:
                if location.representation_kind is RepresentationKind.PDF:
                    data = self._safe_download(location.url)
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
                        headers={"Accept": location.media_type or "application/xml,text/plain,*/*"},
                        timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
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
                        headers={"Accept": "text/html,application/xhtml+xml,*/*"},
                        timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                    )
                    discovered = discover_scholarly_locations(
                        response.text,
                        str(response.url),
                        provider=location.provider,
                    )
                    page_titles = _extract_html_titles(response.text)
                    page_title_match = (
                        _html_title_matches(expected_title, page_titles)
                        if expected_title and page_titles
                        else None
                    )
                    if page_title_match is False:
                        raise ValueError(
                            "landing page title does not match the cited reference"
                        )
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
                    pending = [
                        candidate for candidate in sorted(discovered, key=_location_rank)
                        if candidate.url not in attempted_urls
                    ] + pending
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
                                completeness="not_assessed",
                                metadata={
                                    "observed_source_kind": observed_html_kind.kind,
                                    "observed_source_kind_confidence": (
                                        observed_html_kind.confidence
                                    ),
                                    "page_title_match": page_title_match,
                                },
                            )
                        )
                    else:
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
                )
                attempt["outcome"] = outcome
                if not accepted:
                    attempt["reason"] = reason
                    attempts.append(attempt)
                    self._clear_retrieved_representation(result)
                    continue
                attempts.append(attempt)
                result.full_text_url = location.url
                result.source_name = location.provider
                result.metadata = result.metadata or {}
                result.metadata["location_attempts"] = attempts
                return True
            except Exception as exc:
                self._clear_retrieved_representation(result)
                attempt["outcome"] = "unavailable"
                attempt["reason"] = str(exc)[:160]
                attempts.append(attempt)
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
                    completeness="not_assessed",
                    metadata={
                        "observed_source_kind": observed_html_kind.kind,
                        "observed_source_kind_confidence": (
                            observed_html_kind.confidence
                        ),
                        "page_title_match": page_title_match,
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
            )
            attempt = {
                "url": source_url,
                "provider": provider,
                "kind": RepresentationKind.HTML.value,
                "outcome": "acquired_fallback" if accepted else outcome,
            }
            if not accepted:
                attempt["reason"] = reason
                self._clear_retrieved_representation(result)
            else:
                result.full_text_url = source_url
                result.source_name = provider
            attempts.append(attempt)
            result.metadata = result.metadata or {}
            result.metadata["location_attempts"] = attempts
            return accepted
        result.metadata = result.metadata or {}
        result.metadata["location_attempts"] = attempts
        return False

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

    def _try_web_fetch(
        self,
        url: str,
        title: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
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

        The page content is NOT persisted to S3 — it's processed in-memory
        and returned. Only the verification result is stored in the report.
        """
        try:
            resp = safe_request(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning("Web fetch failed for %s: %s", url[:60], e)
            return RetrievalResult(
                source_name="web_fetch",
                success=False,
                error=f"Web fetch failed: {str(e)[:80]}",
            )

        content_type = resp.headers.get("content-type", "").lower()
        if resp.content.startswith(PDF_MAGIC) or "application/pdf" in content_type:
            return RetrievalResult(
                source_name="web_fetch",
                success=False,
                error="URL returned a PDF; use PDF route",
            )

        # Identity check: confirm the page IS the cited source. If we can
        # extract a page title and it doesn't match the reference, reject — this
        # is a wrong page (login, error, or a different article). If no title is
        # extractable (some JS-rendered pages), fall through to content extraction.
        if title:
            page_titles = _extract_html_titles(resp.text)
            if page_titles and not _html_title_matches(title, page_titles):
                logger.info(
                    "Web fetch identity mismatch for %s: cited %r vs page %r",
                    url[:60], title[:40], page_titles[0][:40],
                )
                return RetrievalResult(
                    source_name="web_fetch",
                    success=False,
                    error="Page title does not match the cited reference — likely wrong page",
                )

        # Extract readable text
        import trafilatura
        page_text = trafilatura.extract(
            resp.text,
            include_links=False,
            include_tables=False,
            favor_recall=True,
        )
        if not page_text or len(page_text) < 100:
            return RetrievalResult(
                source_name="web_fetch",
                success=False,
                error="Page loaded but no readable article text extracted",
            )

        structured_kind = classify_html_source_kind(resp.text, url)
        content_kind = classify_content_source_kind(page_text, source_url=url)
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
                },
            )

        logger.info("Web fetch extracted %d chars from %s", len(page_text), url[:60])
        return RetrievalResult(
            source_name="web_fetch",
            success=True,
            representation=SourceRepresentation(
                kind=RepresentationKind.PLAIN_TEXT,
                media_type="text/plain",
                content=page_text.encode("utf-8"),
                source_url=url,
                original_kind=RepresentationKind.HTML,
                charset="utf-8",
                completeness="not_assessed",
            ),
            # Transitional compatibility for the verification path that still
            # consumes web article text through the limited-evidence field.
            abstract=page_text,
            full_text_url=url,
            metadata={
                "representation_kind": RepresentationKind.PLAIN_TEXT.value,
                "expected_source_kind": expected_source_kind.kind,
                "observed_source_kind": observed_kind.kind,
                "observed_source_kind_confidence": observed_kind.confidence,
                "source_kind_verdict": kind_compatibility.verdict,
            },
        )

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
    ) -> RetrievalResult:
        """Apply the common PDF identity gate to resolver-acquired bytes."""
        validation = validate_retrieved_pdf(
            payload,
            expected_doi=doi,
            expected_title=title,
            document_kind=(
                document_kind_for_source_kind(expected_source_kind.kind)
                if expected_source_kind.is_known else "article"
            ),
            expected_source_kind=expected_source_kind.kind,
            expected_source_kind_confidence=expected_source_kind.confidence,
            expected_source_kind_evidence=expected_source_kind.evidence,
        )
        if validation.identity_confidence != "high":
            logger.info(
                "DOI resolver PDF rejected for %s: %s",
                doi,
                validation.reason,
            )
            return RetrievalResult(
                source_name="doi_resolver",
                success=False,
                error="DOI resolver PDF failed source identity validation",
                doi=doi,
                metadata={
                    "identity_confidence": validation.identity_confidence,
                    "identity_reason": validation.reason,
                    "identity_rejected": True,
                    "expected_source_kind": expected_source_kind.kind,
                    "observed_source_kind": validation.observed_source_kind,
                    "source_kind_verdict": validation.source_kind_verdict,
                },
            )
        if validation.completeness == "incomplete":
            return RetrievalResult(
                source_name="doi_resolver",
                success=False,
                error="DOI resolver PDF failed completeness validation",
                doi=doi,
                title=title,
                metadata={
                    "identity_confidence": validation.identity_confidence,
                    "identity_reason": validation.reason,
                    "completeness": validation.completeness,
                    "completeness_rejected": True,
                },
            )
        logger.info(
            "DOI resolver returned identity-gated PDF for %s (%d bytes)",
            doi,
            len(payload),
        )
        return RetrievalResult(
            source_name="doi_resolver",
            success=True,
            representation=SourceRepresentation(
                kind=RepresentationKind.PDF,
                media_type="application/pdf",
                content=payload,
                source_url=source_url,
                completeness=validation.completeness,
            ),
            locations=[
                AcquisitionLocation(
                    url=source_url,
                    provider="doi_resolver",
                    media_type="application/pdf",
                    representation_kind=RepresentationKind.PDF,
                    landing_page_url=landing_page_url,
                    access_type="institutional",
                    is_best=True,
                    metadata=location_metadata or {},
                )
            ],
            doi=doi,
            title=title,
            metadata={
                "identity_confidence": validation.identity_confidence,
                "identity_reason": validation.reason,
                "completeness": validation.completeness,
                "representation_kind": RepresentationKind.PDF.value,
                "expected_source_kind": expected_source_kind.kind,
                "observed_source_kind": validation.observed_source_kind,
                "source_kind_verdict": validation.source_kind_verdict,
            },
        )

    def _try_doi_resolver(
        self,
        doi: str,
        title: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
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
        base = settings.DOI_RESOLVER_URL or ""
        # Validate + URL-encode the DOI before placing it in the trusted
        # resolver URL (REVIEW §2.4). The DOI is student-controlled; raw
        # concatenation let a DOI like "10.1/x?url=http://internal/..." inject
        # a query string into the proxy URL (chained SSRF through the proxy).
        if not re.fullmatch(r"10\.\d{4,9}/[A-Za-z0-9._:()\-/]+", doi):
            logger.warning("DOI resolver: rejecting malformed DOI %r", doi[:40])
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
            resp = safe_request(
                resolver_url,
                headers={"Accept": "text/html,application/pdf,*/*"},
                timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                trust_prefix=base,
            )
        except Exception as e:
            logger.warning("DOI resolver failed for %s: %s", doi, str(e)[:80])
            return RetrievalResult(
                source_name="doi_resolver",
                success=False,
                error=f"DOI resolver failed: {str(e)[:80]}",
                doi=doi,
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
            )

        # HTML is accepted only when it is an identity-matching scholarly text.
        # Length alone is never evidence: long login menus and resolver pages
        # were the original false-acceptance path.
        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            discovered = discover_scholarly_locations(
                resp.text,
                final_url,
                provider="doi_resolver",
            )
            for candidate in sorted(discovered, key=_location_rank)[:_MAX_LOCATION_ATTEMPTS]:
                if candidate.representation_kind is not RepresentationKind.PDF:
                    continue
                try:
                    candidate_response = safe_request(
                        candidate.url,
                        headers={"Accept": "application/pdf,*/*"},
                        max_bytes=settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024,
                        timeout=settings.STUDENT_URL_TIMEOUT_SECONDS,
                        trust_prefix=(base if candidate.url.startswith(base) else None),
                    )
                except Exception as exc:
                    logger.info(
                        "DOI resolver landing-page candidate unavailable for %s: %s",
                        doi,
                        str(exc)[:100],
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
                )
                if validated.success:
                    return validated

            page_titles = _extract_html_titles(resp.text)
            if title and page_titles and not _html_title_matches(title, page_titles):
                logger.info(
                    "DOI resolver HTML title mismatch for %s: cited %r vs page %r",
                    doi,
                    title[:60],
                    page_titles[0][:60],
                )
                return RetrievalResult(
                    source_name="doi_resolver",
                    success=False,
                    error="DOI resolver HTML title does not match cited source",
                    doi=doi,
                    metadata={"identity_rejected": True},
                )

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
                    None,
                    None,
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
                                access_type="institutional",
                                is_best=True,
                            )
                        ],
                        # Transitional compatibility for consumers that still
                        # expose acquired web text through this field.
                        abstract=page_text,
                        doi=doi,
                        title=page_titles[0] if page_titles else title,
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

    def _try_source(
        self,
        source: RetrievalSource,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None = None,
        *,
        expected_source_kind: SourceKindAssessment = SourceKindAssessment(),
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

        def acquire(result: RetrievalResult, phase: str) -> RetrievalResult:
            nonlocal last_result
            result.metadata = result.metadata or {}
            search_attempts = result.metadata.get("search_attempts", [])
            for attempt in search_attempts:
                provider = str(attempt.get("provider", "")).lower()
                if provider:
                    tried_search_providers.add(provider)
            if result.locations:
                result.locations = [
                    location
                    for location in result.locations
                    if location.url not in attempted_urls
                ]
                if result.locations:
                    result.full_text_url = result.locations[0].url
                else:
                    result.full_text_url = None
            if result.success and (result.locations or result.full_text_url or result.full_text):
                result = self._download_and_cache(
                    source,
                    result,
                    doi,
                    title,
                    author,
                    year,
                    expected_source_kind=expected_source_kind,
                )
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
                    "location_attempts": location_attempts,
                    "outcome": "acquired" if result.full_text else (
                        "candidates_rejected" if result.success else "no_candidates"
                    ),
                }
            )
            result.metadata = result.metadata or {}
            result.metadata["retrieval_trace"] = list(retrieval_trace)
            last_result = result
            return result

        # Try DOI first (most precise)
        if doi:
            result = source.search_by_doi(doi)
            if result.success:
                result = acquire(result, "doi")
                if result.full_text or not is_web_discovery:
                    return result
            elif is_web_discovery:
                acquire(result, "doi")

        # Fall back to title search
        if title:
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
            if result.full_text:
                return result

        last_result.metadata = last_result.metadata or {}
        last_result.metadata["retrieval_trace"] = retrieval_trace
        return last_result

    def _lookup_source(
        self,
        source: RetrievalSource,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None = None,
    ) -> RetrievalResult:
        """Collect provider evidence without acquiring or caching content."""
        if doi:
            result = source.search_by_doi(doi)
            if result.success:
                return result
        if title:
            result = source.search_by_title_author(title, author)
            if result.success:
                return result
        return RetrievalResult(source_name=source.name, success=False)

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

        # If the source has already populated full_text (some do), or has a
        # custom download method, use it.
        try:
            if result.full_text is None:
                # Check if the source overrides download_full_text
                if type(source).download_full_text is not RetrievalSource.download_full_text:
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
                representation_kind = (
                    result.representation.kind
                    if result.representation
                    else RepresentationKind.PDF
                    if result.full_text.startswith(PDF_MAGIC)
                    else RepresentationKind.PLAIN_TEXT
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
                if (
                    representation_kind is RepresentationKind.PDF
                    and preflight_matches
                ):
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
                    identity_confidence = validation.identity_confidence
                    identity_reason = validation.reason
                    completeness = getattr(
                        validation, "completeness", "not_assessed"
                    )
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
                if representation_kind is RepresentationKind.PDF:
                    result.metadata["completeness"] = completeness
                result.metadata["text_quality"] = text_quality

                if identity_confidence != "high":
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
                    logger.info(
                        "Rejecting incomplete PDF representation for %s from %s",
                        ref_doi or ref_title or result.doi,
                        source.name,
                    )
                    self._clear_retrieved_representation(result)
                    result.metadata["completeness_rejected"] = True
                    return result

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
                        "CACHED %s | source=%s | identity=%s | ref: doi=%s title=%r author=%s year=%s | pdf: doi=%s title=%r",
                        key, source.name, identity_confidence,
                        ref_doi, (ref_title or "")[:60], ref_author, ref_year,
                        result.doi, (result.title or "")[:60],
                    )
                    if downloaded_via_publisher:
                        logger.info(
                            "Cached %s PDF for %s (oa=%s)",
                            "OA" if is_oa else "paywalled", result.doi, is_oa,
                        )
                elif not should_cache:
                    # Not cached because identity confidence is low/skipped, or
                    # because paywall policy forbids it. The PDF is still
                    # returned (flagged) so a human reviewer can see it.
                    logger.info(
                        "Not caching %s from %s: identity_confidence=%s",
                        result.doi or result.title, source.name,
                        identity_confidence,
                    )
        except Exception as e:
            logger.warning("Failed to download full text from %s: %s", source.name, e)

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
                record = admit_representation(
                    session,
                    self._backend,
                    AdmissionRequest(
                        work=WorkIdentity(
                            title=ref_title or result.title or ref_doi or result.doi or "",
                            work_type=(
                                expected_source_kind.kind
                                if expected_source_kind.is_known
                                else classify_provider_source_kind(
                                    result.metadata
                                ).kind
                                if classify_provider_source_kind(result.metadata).is_known
                                else "academic_work"
                            ),
                            doi=ref_doi or result.doi,
                            author=ref_author or (result.authors[0] if result.authors else None),
                            year=ref_year or result.year,
                        ),
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
                        text_quality="not_assessed",
                        edition_or_version=next(
                            (location.version for location in result.locations if location.version),
                            None,
                        ),
                        expires_at=license_expires_at,
                        admitted_by="retrieval_pipeline",
                        validation_evidence={
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
                            "location_attempts": result.metadata.get("location_attempts", []),
                            "canonical_work": result.metadata.get("canonical_work", {}),
                            "retention_basis": retention_basis,
                            "public_retrieval_retention_policy": (
                                settings.PUBLIC_RETRIEVAL_RETENTION_POLICY
                            ),
                        },
                    ),
                )
                session.commit()
                result.metadata["durable_admission"] = {
                    "state": record.admission_state,
                    "representation_id": str(record.id),
                    "license_class": license_class,
                    "retention_basis": retention_basis,
                }
            except AdmissionError as exc:
                session.rollback()
                result.metadata["durable_admission"] = {
                    "state": "not_stored",
                    "reason": str(exc),
                }
                logger.warning("Durable retrieval admission rejected: %s", exc)

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
