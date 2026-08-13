"""Source resolution orchestrator.

Implements the resolution priority chain:
    Local S3 cache -> Student URL -> Retrieval Sources -> Fail
"""

import hashlib
import html
import logging
import re

import httpx

from app.config import settings
from app.services.storage import get_storage_backend
from app.services.retrieval import get_retrieval_sources, RetrievalSource, RetrievalResult
from app.services.source_type import is_traditional_media, is_archive_source
from app.services.source_validator import validate_retrieved_pdf
from app.services.safe_fetch import (
    UnsafeUrlError,
    ResponseTooLargeError,
    safe_request,
    safe_fetch_bytes,
)

logger = logging.getLogger(__name__)

# Magic-byte check for PDF
PDF_MAGIC = b"%PDF-"


class SourceResolutionError(Exception):
    """Raised when a source cannot be found."""


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
        # Storage backend is optional — if S3/MinIO is unavailable, the
        # resolver still works (just without caching/persistence). This makes
        # the resolver robust to S3 outages and simplifies testing without Docker.
        try:
            self._backend = get_storage_backend()
        except Exception as e:
            logger.warning("Storage backend unavailable — caching disabled: %s", str(e)[:80])
            self._backend = None
        self._retrieval_sources = get_retrieval_sources()

    # ── Public API ────────────────────────────────────────

    def resolve(
        self,
        doi: str | None = None,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
        year: str | None = None,
        student_url: str | None = None,
        raw_ref: str | None = None,
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

        # 0. Route by source type — skip retrieval for traditional media / archives.
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

        # 2. Try student URL — PDF first, then web-page text for HTML sources.
        if student_url:
            result = self._try_student_url(student_url, doi, title, author, year)
            if result.success and result.full_text:
                return result  # Authoritative — the student's actual source
            # If the URL wasn't a PDF (it was HTML), try fetching the web page
            if not result.success and "not a pdf" in (result.error or "").lower():
                web_result = self._try_web_fetch(student_url, title)
                if web_result.success:
                    return web_result  # Authoritative — the student's actual web source
                url_failure_reason = web_result.error or "web fetch failed"
            else:
                url_failure_reason = result.error or "URL download failed"

        # 2.5. Try DOI resolver / campus proxy (institutional deployment).
        #      When DOI_RESOLVER_URL is configured, construct {url}{doi} to
        #      access the paper through the university's library proxy. This
        #      is the key to high full-text retrieval rates for paywalled content
        #      — the proxy handles subscription authentication.
        if doi and settings.DOI_RESOLVER_URL:
            result = self._try_doi_resolver(doi, title)
            if result.success:
                return result

        # 3. Try retrieval sources in priority order (academic databases).
        #    If the student URL failed, we still try to find the source — but
        #    we verify it's the RIGHT source via multi-field matching, and flag
        #    that it was found via title search (may differ from the cited URL).
        best_abstract_result: RetrievalResult | None = None
        if skip_academic_dbs:
            logger.info(
                "Skipping academic DBs for traditional-media reference: %s",
                (raw_ref or title or "")[:60],
            )
        else:
            # Iteration order. Front-end PD route: for likely public-domain
            # literary works (no DOI + year ≤ 1928), try Wikisource → Gutenberg
            # BEFORE the slower academic chain. They are the canonical sources
            # for PD literary texts, so running them first saves ~5 academic-DB
            # calls per such ref (the English/History-department use case). For
            # everything else, keep the configured order.
            sources_ordered = self._retrieval_sources
            if (not doi) and year:
                try:
                    if int(re.search(r"\d{4}", year).group()) <= 1928:
                        pd = [s for s in self._retrieval_sources
                              if s.name in ("wikisource", "gutenberg")]
                        rest = [s for s in self._retrieval_sources
                                if s.name not in ("wikisource", "gutenberg")]
                        sources_ordered = pd + rest
                        logger.info(
                            "PD-first route (pre-1929, no DOI): trying "
                            "Wikisource/Gutenberg before academic chain: %s",
                            (title or "")[:60],
                        )
                except (ValueError, AttributeError):
                    pass

            for source in sources_ordered:
                # Back-end PD gate: Gutenberg/Wikisource only index old works;
                # skip them for references clearly too recent to be PD. 1928 is
                # the US public-domain threshold for published works (as of
                # 2024). When no year is known we still try.
                if source.name in ("gutenberg", "wikisource") and year:
                    try:
                        if int(re.search(r"\d{4}", year).group()) > 1928:
                            continue
                    except (ValueError, AttributeError):
                        pass
                result = self._try_source(source, doi, title, author, year)
                if result.success and result.full_text:
                    # Multi-field verification: confirm this is the right source
                    match_confidence = self._verify_source_identity(
                        result, doi, title, author, year
                    )
                    result.metadata = result.metadata or {}
                    result.metadata["source_match_confidence"] = match_confidence
                    if url_failure_reason:
                        result.metadata["url_failure_reason"] = url_failure_reason
                        result.metadata["source_note"] = (
                            f"Original URL failed ({url_failure_reason}). "
                            f"Source found via {result.source_name} title search "
                            f"(match confidence: {match_confidence}). "
                            f"May differ from what the student cited."
                        )
                        logger.info("URL failed (%s) — source found via %s (%s confidence)",
                                     url_failure_reason, result.source_name, match_confidence)
                    return result
                if result.success and result.abstract and best_abstract_result is None:
                    best_abstract_result = result

        # 4. No full text found — abstract fallback.
        if best_abstract_result is not None:
            logger.info(
                "No full-text PDF found for doi=%s title=%s, but abstract available from %s",
                doi, title, best_abstract_result.source_name,
            )
            if url_failure_reason:
                best_abstract_result.metadata = best_abstract_result.metadata or {}
                best_abstract_result.metadata["url_failure_reason"] = url_failure_reason
            return best_abstract_result

        raise SourceResolutionError(
            f"Source not found: doi={doi}, isbn={isbn}, title={title}"
            + (f" (URL failed: {url_failure_reason})" if url_failure_reason else "")
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
        if not self._backend:
            return RetrievalResult(source_name="local_cache", success=False)
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
        year: str | None = None,
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
            skip_completeness=True,
        )

        if validation.identity_confidence in ("high", "medium"):
            # Verified — cache (if available) and return.
            if self._backend:
                key = self._cache_key_for_verified(doi, title)
                self._backend.upload(data, key)
            return RetrievalResult(
                source_name="student_url",
                success=True,
                full_text=data,
                doi=doi,
                title=title,
                metadata={"identity_confidence": validation.identity_confidence,
                          "identity_reason": validation.reason},
            )

        # low / rejected / skipped — discard. Do not cache an unverified PDF.
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
        try:
            data = safe_fetch_bytes(
                url,
                max_bytes=max_bytes,
                accept_content_types=("application/pdf",),
                timeout=timeout,
            )
        except UnsafeUrlError as e:
            raise ValueError(f"URL blocked (SSRF guard): {e}") from e
        except ResponseTooLargeError as e:
            raise ValueError(f"File too large: {e}") from e
        except ValueError as e:
            # Content-type mismatch (HTML page served where a PDF was expected).
            raise ValueError(f"Not a PDF: {e}") from e
        except httpx.HTTPError as e:
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
            )
            if pdf_data[:5] == PDF_MAGIC:
                logger.info("OA landing page PDF download successful: %d bytes", len(pdf_data))
                return pdf_data
        except Exception as e:
            logger.debug("OA landing page PDF download failed: %s", str(e)[:60])

        return None

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

    def _try_web_fetch(self, url: str, title: str | None = None) -> RetrievalResult:
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

        logger.info("Web fetch extracted %d chars from %s", len(page_text), url[:60])
        return RetrievalResult(
            source_name="web_fetch",
            success=True,
            # Store web-page text in `abstract` field (it's the source text for
            # verification — analogous to an abstract but from a web page).
            # full_text stays None (not a PDF). The verification engine handles
            # abstract-source verification at medium confidence.
            abstract=page_text,
            full_text_url=url,
        )

    def _try_doi_resolver(self, doi: str, title: str | None = None) -> RetrievalResult:
        """Access a paper through the institution's DOI resolver / campus proxy.

        When DOI_RESOLVER_URL is configured (e.g., an EZproxy or OpenURL
        resolver), the app constructs {DOI_RESOLVER_URL}{doi} to access the
        paper through the university's library subscription. The proxy handles
        authentication; the app gets the PDF.

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

        # Check if we got a PDF
        if resp.content[:5] == PDF_MAGIC:
            logger.info("DOI resolver returned PDF for %s (%d bytes)", doi, len(resp.content))
            return RetrievalResult(
                source_name="doi_resolver",
                success=True,
                full_text=resp.content,
                full_text_url=resolver_url,
                doi=doi,
                title=title,
            )

        # Check if we got HTML (might be a landing page with a download link,
        # or a login page if the proxy isn't authenticated)
        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            # Try to extract readable text — might be the article page
            import trafilatura
            page_text = trafilatura.extract(resp.text, favor_recall=True)
            if page_text and len(page_text) > 500:
                logger.info("DOI resolver returned HTML text for %s (%d chars)", doi, len(page_text))
                return RetrievalResult(
                    source_name="doi_resolver",
                    success=True,
                    abstract=page_text,  # HTML text as source for verification
                    full_text_url=resolver_url,
                    doi=doi,
                    title=title,
                )
            # Probably a login/abstract page — not useful
            logger.warning("DOI resolver returned HTML but no readable article for %s", doi)

        return RetrievalResult(
            source_name="doi_resolver",
            success=False,
            error="DOI resolver returned non-PDF, non-article content",
            doi=doi,
        )

    def _try_source(
        self,
        source: RetrievalSource,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None = None,
    ) -> RetrievalResult:
        """Try a single retrieval source."""
        # Try DOI first (most precise)
        if doi:
            result = source.search_by_doi(doi)
            if result.success:
                return self._download_and_cache(
                    source, result, doi, title, author, year
                )

        # Fall back to title search
        if title:
            result = source.search_by_title_author(title, author)
            if result.success:
                return self._download_and_cache(
                    source, result, doi, title, author, year
                )

        return RetrievalResult(source_name=source.name, success=False)

    def _download_and_cache(
        self,
        source: RetrievalSource,
        result: RetrievalResult,
        ref_doi: str | None = None,
        ref_title: str | None = None,
        ref_author: str | None = None,
        ref_year: str | None = None,
    ) -> RetrievalResult:
        """Download full text from a retrieval result and cache it.

        Tries, in order:
        1. Custom download_full_text (Gutenberg text fetch) — always cacheable
        2. OA PDF URL (from OpenAlex/S2) — always cacheable (open access)
        3. Publisher PDF URL (campus-network access) — cacheability depends on
           whether the article is actually OA (from metadata) or paywalled

        Identity gate (cache-poisoning defense, REVIEW §2.3): before anything is
        written to the cache, the downloaded bytes are checked against the cited
        reference (DOI + title + author + year triangulation). A PDF that fails
        identity ("rejected") is dropped — it is NOT the cited source, so neither
        returning it nor caching it is safe. Low-confidence PDFs are returned
        (flagged, for manual review) but not cached, so a wrong or uncertain
        source can never poison the cache and be served to later papers.

        Caching policy (applied only to identity-verified content):
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
                    try:
                        data = self._safe_download(result.full_text_url)
                        result.full_text = data
                    except Exception:
                        # OA URL returned HTML landing page instead of direct PDF.
                        # Try to extract the actual PDF link from the page.
                        logger.info("Direct OA download failed, trying landing-page handler: %s",
                                     result.full_text_url[:60])
                        pdf_data = self._try_oa_landing_page(result.full_text_url)
                        if pdf_data:
                            result.full_text = pdf_data

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

            # ── Identity gate (cache-poisoning defense, REVIEW §2.3) ─────────
            # Before any cache write, confirm the downloaded PDF IS the cited
            # source. Previously this path cached on a magic-byte check alone,
            # so a wrong PDF found via title search was stored under the DOI
            # key and served to every later paper citing that DOI — the exact
            # wrong-source failure documented in STATE.md §11.
            #
            # skip_completeness=True: completeness heuristics are book-oriented
            # and would falsely reject valid short articles; identity is the
            # only signal that should gate caching here.
            if result.full_text:
                validation = validate_retrieved_pdf(
                    result.full_text,
                    expected_doi=ref_doi or result.doi,
                    expected_title=ref_title or result.title,
                    expected_author=ref_author,
                    expected_year=ref_year,
                    skip_completeness=True,
                )
                result.metadata = result.metadata or {}
                result.metadata["identity_confidence"] = validation.identity_confidence
                result.metadata["identity_reason"] = validation.reason

                if validation.identity_confidence == "rejected":
                    # Wrong source — do not return it as a hit, do not cache.
                    # Clearing full_text makes resolve() skip this result and
                    # continue to the next source.
                    logger.info(
                        "Rejecting %s from %s: identity check failed (%s)",
                        ref_doi or ref_title or result.doi, source.name,
                        validation.reason,
                    )
                    result.full_text = None
                    result.metadata["identity_rejected"] = True
                    return result

                # Determine cacheability:
                # - Content from custom/OA sources (Gutenberg, OA URL) → cacheable
                #   (only now that identity is confirmed)
                # - Content from publisher URL → check if it's actually OA.
                #   If OA, cache. If not OA (paywalled), cache only if
                #   CACHE_PAYWALLED_PDFS=True.
                # - Identity "low"/"skipped" → never cached (uncertain source).
                is_oa = _check_is_oa(result)
                should_cache = validation.identity_confidence in ("high", "medium")

                if should_cache and downloaded_via_publisher and not is_oa:
                    # Paywalled content accessed via campus IP
                    if not settings.CACHE_PAYWALLED_PDFS:
                        should_cache = False
                        logger.info(
                            "Paywalled PDF for %s verified in-memory, NOT cached "
                            "(OA=False, CACHE_PAYWALLED_PDFS=False)",
                            result.doi,
                        )

                if should_cache and self._backend:
                    key = self._cache_key_for_verified(result.doi, result.title)
                    self._backend.upload(result.full_text, key)
                    # Audit log: record the identity verdict + the reference fields
                    # the gate checked against, so cached objects can be audited
                    # without reconstructing the mapping from keys alone.
                    logger.info(
                        "CACHED %s | source=%s | identity=%s | ref: doi=%s title=%r author=%s year=%s | pdf: doi=%s title=%r",
                        key, source.name, validation.identity_confidence,
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
                        validation.identity_confidence,
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


