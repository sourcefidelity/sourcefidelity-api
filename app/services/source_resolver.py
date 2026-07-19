"""Source resolution orchestrator.

Implements the resolution priority chain:
    Local S3 cache -> Student URL -> Retrieval Sources -> Fail
"""

import hashlib
import logging
import re

import httpx

from app.config import settings
from app.services.storage import get_storage_backend
from app.services.retrieval import get_retrieval_sources, RetrievalSource, RetrievalResult
from app.services.source_type import is_traditional_media, is_archive_source

logger = logging.getLogger(__name__)

# Magic-byte check for PDF
PDF_MAGIC = b"%PDF-"


class SourceResolutionError(Exception):
    """Raised when a source cannot be found."""


class SourceResolver:
    """Resolves a reference to a source document."""

    def __init__(self) -> None:
        self._backend = get_storage_backend()
        self._retrieval_sources = get_retrieval_sources()

    # ── Public API ────────────────────────────────────────

    def resolve(
        self,
        doi: str | None = None,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
        student_url: str | None = None,
        raw_ref: str | None = None,
    ) -> RetrievalResult:
        """Resolve a reference to a document.

        Returns a RetrievalResult with full_text (or abstract) populated if found.
        Raises SourceResolutionError if no source is found.

        Traditional-media references (films, TV, albums) and physical archives
        skip the academic-database search — they're not indexed there in a
        useful form and title search produces keyword-coincidence false positives.
        """
        # 0. Route by source type — skip retrieval for traditional media / archives.
        #     Traditional media (films etc.) can still be resolved via student_url
        #     or S3 if an instructor uploaded the script/screenplay, so we only
        #     skip the academic-DB step, not the earlier steps.
        skip_academic_dbs = False
        if raw_ref:
            if is_archive_source(raw_ref):
                raise SourceResolutionError(
                    f"Physical archive source — cannot be verified automatically: {raw_ref[:60]}"
                )
            skip_academic_dbs = is_traditional_media(raw_ref)

        # 1. Check local S3 cache
        result = self._check_local_cache(doi, isbn, title)
        if result.success and result.full_text:
            return result

        # 2. Try student URL
        if student_url:
            result = self._try_student_url(student_url, doi, title, author)
            if result.success and result.full_text:
                return result

        # 3. Try retrieval sources in priority order (academic databases)
        best_abstract_result: RetrievalResult | None = None
        if skip_academic_dbs:
            logger.info(
                "Skipping academic DBs for traditional-media reference: %s",
                (raw_ref or title or "")[:60],
            )
        else:
            for source in self._retrieval_sources:
                result = self._try_source(source, doi, title, author)
                if result.success and result.full_text:
                    return result
                # Track the best abstract-only result as a fallback (for paywalled
                # sources where no OA PDF is available).
                if result.success and result.abstract and best_abstract_result is None:
                    best_abstract_result = result

        # 4. No full text found — but if we have an abstract, return it for
        #    lower-confidence (abstract-only) verification.
        if best_abstract_result is not None:
            logger.info(
                "No full-text PDF found for doi=%s title=%s, but abstract available from %s",
                doi, title, best_abstract_result.source_name,
            )
            return best_abstract_result

        raise SourceResolutionError(
            f"Source not found: doi={doi}, isbn={isbn}, title={title}"
        )

    # ── Internal helpers ───────────────────────────────────

    def _check_local_cache(
        self,
        doi: str | None,
        isbn: str | None,
        title: str | None,
    ) -> RetrievalResult:
        """Check if the document is already in S3.

        Documents held for review (review_status != "accepted") are skipped
        so they're not used for citation verification until approved.
        """
        for key in self._build_cache_keys(doi, isbn, title):
            # Status gate: exclude documents not yet accepted for use.
            if self._review_status_for_key(key) != "accepted":
                continue
            try:
                data = self._backend.download(key)
                return RetrievalResult(
                    source_name="local_cache",
                    success=True,
                    full_text=data,
                    doi=doi,
                    title=title,
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
        # Lazy import avoids a circular dependency at module load time.
        try:
            from app.routers.sources import _stored_documents
        except Exception:
            return "accepted"

        for doc in _stored_documents:
            if doc.get("s3_key") == key:
                return doc.get("review_status", "accepted")
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
    ) -> RetrievalResult:
        """Download and verify a student-linked URL."""
        try:
            data = self._safe_download(url)
        except Exception as e:
            logger.warning("Student URL download failed: %s", e)
            return RetrievalResult(source_name="student_url", success=False, error=str(e))

        # Verify the PDF content matches what was cited
        if self._verify_pdf_metadata(data, doi, title, author):
            # Cache to S3
            key = self._cache_key_for_verified(doi, title)
            self._backend.upload(data, key)
            return RetrievalResult(
                source_name="student_url",
                success=True,
                full_text=data,
                doi=doi,
                title=title,
            )
        logger.warning("Student URL metadata mismatch. Discarding.")
        return RetrievalResult(
            source_name="student_url",
            success=False,
            error="Metadata mismatch — content does not match cited reference",
        )

    def _safe_download(self, url: str) -> bytes:
        """Download a URL with safety checks (R9).

        Validates scheme, enforces a size limit, and verifies the body is
        actually a PDF (not an HTML paywall/login page).
        """
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme: {url}")

        timeout = settings.STUDENT_URL_TIMEOUT_SECONDS

        # HEAD request to check content type and size before downloading.
        head = httpx.head(url, timeout=timeout, follow_redirects=True)
        head.raise_for_status()

        content_type = head.headers.get("content-type", "").lower()
        content_length = int(head.headers.get("content-length", 0))
        max_bytes = settings.STUDENT_URL_MAX_SIZE_MB * 1024 * 1024
        if content_length and content_length > max_bytes:
            raise ValueError(f"File too large: {content_length} bytes")
        # Hint if it's clearly not a PDF by content-type.
        if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
            raise ValueError(f"Not a PDF (content-type: {content_type})")

        # Download
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()

        # Verify magic bytes — guards against HTML pages served as 200.
        if not resp.content.startswith(PDF_MAGIC):
            raise ValueError("Downloaded content is not a PDF (may be paywall or HTML page)")

        return resp.content

    def _verify_pdf_metadata(
        self,
        data: bytes,
        doi: str | None,
        title: str | None,
        author: str | None,  # noqa: ARG002 (reserved for future author checks)
    ) -> bool:
        """Verify that a downloaded PDF matches the cited metadata.

        Extracts text from the first few pages and checks for a DOI match
        or a title match.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=data, filetype="pdf")
            text = ""
            for page in doc[:3]:  # first 3 pages
                text += page.get_text()
            doc.close()
        except Exception:
            return False  # if text extraction fails entirely, reject

        # Check DOI
        if doi:
            doi_clean = doi.removeprefix("https://doi.org/").strip()
            if doi_clean and doi_clean in text:
                return True

        # Check title (fuzzy prefix match)
        if title:
            t_clean = re.sub(r"\s+", " ", title.strip().lower())
            text_clean = re.sub(r"\s+", " ", text.lower())
            if t_clean[:50] in text_clean:
                return True

        return False

    def _try_source(
        self,
        source: RetrievalSource,
        doi: str | None,
        title: str | None,
        author: str | None,
    ) -> RetrievalResult:
        """Try a single retrieval source."""
        # Try DOI first (most precise)
        if doi:
            result = source.search_by_doi(doi)
            if result.success:
                return self._download_and_cache(source, result)

        # Fall back to title search
        if title:
            result = source.search_by_title_author(title, author)
            if result.success:
                return self._download_and_cache(source, result)

        return RetrievalResult(source_name=source.name, success=False)

    def _download_and_cache(
        self, source: RetrievalSource, result: RetrievalResult
    ) -> RetrievalResult:
        """Download full text from a retrieval result and cache it.

        Tries, in order:
        1. Custom download_full_text (Gutenberg text fetch) — always cacheable
        2. OA PDF URL (from OpenAlex/S2) — always cacheable (open access)
        3. Publisher PDF URL (campus-network access) — cacheability depends on
           whether the article is actually OA (from metadata) or paywalled

        Caching policy:
        - OA content + public-domain texts: ALWAYS cached (free to store)
        - Paywalled content (accessed via campus IP): cached only if
          CACHE_PAYWALLED_PDFS=True (institution's copyright-policy decision)
        """
        downloaded_via_publisher = False

        # If the source has already populated full_text (some do), or has a
        # custom download method, use it.
        try:
            if result.full_text is None:
                # Check if the source overrides download_full_text
                if type(source).download_full_text is not RetrievalSource.download_full_text:
                    result = source.download_full_text(result)
                elif result.full_text_url:
                    data = self._safe_download(result.full_text_url)
                    result.full_text = data

            # If OA download failed or wasn't available, try publisher PDF
            # (works on campus networks with IP-based access)
            if result.full_text is None and result.doi:
                from app.services.publisher_urls import try_download_publisher_pdf
                publisher = _extract_publisher(result)
                pdf_bytes = try_download_publisher_pdf(
                    doi=result.doi,
                    publisher=publisher,
                    oa_url=result.full_text_url,
                )
                if pdf_bytes:
                    result.full_text = pdf_bytes
                    downloaded_via_publisher = True
                    logger.info(
                        "Downloaded PDF for %s via publisher URL (publisher=%s)",
                        result.doi, publisher or "unknown",
                    )

            # Determine cacheability:
            # - Content from custom/OA sources (Gutenberg, OA URL) → always cache
            # - Content from publisher URL → check if it's actually OA
            #   using the metadata's open-access flag. If OA, always cache.
            #   If not OA (paywalled), cache only if CACHE_PAYWALLED_PDFS=True.
            if result.full_text:
                is_oa = _check_is_oa(result)
                should_cache = True

                if downloaded_via_publisher and not is_oa:
                    # Paywalled content accessed via campus IP
                    if not settings.CACHE_PAYWALLED_PDFS:
                        should_cache = False
                        logger.info(
                            "Paywalled PDF for %s verified in-memory, NOT cached "
                            "(OA=False, CACHE_PAYWALLED_PDFS=False)",
                            result.doi,
                        )

                if should_cache:
                    key = self._cache_key_for_verified(result.doi, result.title)
                    self._backend.upload(result.full_text, key)
                    if downloaded_via_publisher:
                        logger.info(
                            "Cached %s PDF for %s (oa=%s)",
                            "OA" if is_oa else "paywalled", result.doi, is_oa,
                        )
        except Exception as e:
            logger.warning("Failed to download full text from %s: %s", source.name, e)

        return result

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


def _check_is_oa(result: RetrievalResult) -> bool:
    """Determine whether a retrieved work is open access.

    Checks the OA flags from the source metadata:
    - OpenAlex: best_oa_location.is_oa, open_access.is_oa, open_access.oa_status
    - Semantic Scholar: presence of openAccessPdf
    - If the full_text_url came from an OA source, assume OA

    When in doubt (no metadata signal), assumes NOT OA (conservative — better
    to over-protect copyrighted content than to cache it accidentally).
    """
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

    # If the result came from an OA URL (not a publisher paywall URL),
    # and the URL looks like a known OA host
    if result.full_text_url:
        oa_hosts = ("doi.org", "ncbi.nlm.nih.gov", "arxiv.org", "biorxiv.org",
                     "plos.org", "frontiersin.org", "mdpi.com", "doaj.org")
        if any(host in result.full_text_url for host in oa_hosts):
            return True

    return False


