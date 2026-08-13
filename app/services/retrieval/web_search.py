"""Web-search PDF retrieval adapter.

After the academic-DB chain (OpenAlex, Crossref, etc.) fails to find a source,
this adapter searches the web for open-access PDF copies. Uses the pluggable
search provider (Google Custom Search / Bing) to find PDFs on author homepages,
institutional repositories, preprint servers, and other legitimate OA locations.

Source-access neutrality (§3.5): the app verifies against whatever it finds.
Does NOT access Sci-Hub or pirated copies — only results from the search
provider, which indexes public web pages. If a search result points to an
author homepage or institutional repository, that's legitimate OA access.

Validation: any found PDF is validated by title-match (same logic as the
student-URL path) before being accepted — prevents accepting a wrong paper
that happens to share keywords with the search query.
"""

import logging
from typing import Optional

import httpx

from app.services.retrieval.base import RetrievalResult, RetrievalSource
from app.services.search import get_search_provider
from app.services.safe_fetch import safe_fetch_bytes
from app.services.search.base import SearchProvider

logger = logging.getLogger(__name__)

# Max PDFs to try downloading per source (each download is a network call;
# try a few candidates in case the first is a wrong paper or a dead link)
_MAX_DOWNLOAD_ATTEMPTS = 3

# Max PDF size to accept (50MB — same as student-URL limit)
_MAX_PDF_SIZE = 50 * 1024 * 1024


class WebSearchRetriever(RetrievalSource):
    """Find PDFs via web search when academic-DB chain fails.

    Uses the configured search provider to search for source titles +
    'filetype:pdf'. Downloads and validates candidate PDFs.
    Returns the first PDF whose title matches the search query.

    DuckDuckGo automatic fallback: if the primary provider fails (rate limit,
    exhausted credits, API down, or not configured), DuckDuckGo is used as
    the fallback. DuckDuckGo is free, needs no API key, and is always
    available — so web_search works out of the box with zero configuration
    beyond adding 'web_search' to RETRIEVAL_SOURCES.

    If no primary provider is configured (SEARCH_PROVIDER not set), DuckDuckGo
    is used as the primary. If a primary IS configured, DuckDuckGo is the
    automatic fallback when the primary returns nothing or errors.
    """

    name = "web_search"

    def __init__(self, search_provider: Optional[SearchProvider] = None):
        from app.services.search.duckduckgo import DuckDuckGoSearch
        self._search = search_provider or get_search_provider()
        self._fallback = DuckDuckGoSearch()  # always available — free, no key
        if not self._search:
            logger.info("WebSearchRetriever: no primary search provider configured — "
                        "using DuckDuckGo as primary (free, no API key needed)")

    def search_by_doi(self, doi: str) -> RetrievalResult:
        """Search for a PDF by DOI.

        DOI search is deterministic — the DOI is a unique identifier. No LLM
        selection needed; just search for {doi} filetype:pdf and download.
        """
        query = f'{doi} filetype:pdf'
        return self._search_and_download(query, doi=doi, title=None,
                                          use_llm_selection=False)

    def search_by_title_author(
        self,
        title: str,
        author: str | None = None,
        year: str | None = None,
    ) -> RetrievalResult:
        """Search for a PDF by title (and optional author/year).

        Title search uses LLM-mediated result selection — the LLM picks the
        right result from the search results before downloading. This prevents
        downloading wrong sources (syllabi, reviews, different papers with
        similar titles) that deterministic token-matching can't distinguish.
        """
        parts = [f'"{title}"']
        if author:
            parts.append(author)
        if year:
            parts.append(year)
        parts.append("filetype:pdf")
        query = " ".join(parts)
        return self._search_and_download(query, doi=None, title=title,
                                          author=author, year=year,
                                          use_llm_selection=True)

    def _search_and_download(
        self,
        query: str,
        doi: Optional[str],
        title: Optional[str],
        author: Optional[str] = None,
        year: Optional[str] = None,
        use_llm_selection: bool = True,
    ) -> RetrievalResult:
        """Run the search, select the right result (LLM), download + validate.

        Two modes:
        - DOI search (use_llm_selection=False): DOI is unique, no LLM needed.
          Download first PDF candidate and validate by DOI presence.
        - Title search (use_llm_selection=True): LLM selects the right result
          from the search results list before downloading. Prevents wrong-source
          PDFs (syllabi, reviews, different papers with similar titles).
        """
        results = self._run_search(query)

        # If primary returned nothing, try DuckDuckGo fallback
        if not results and self._search and self._search.name != "DuckDuckGo":
            logger.info("Primary search returned nothing for '%s' — trying DuckDuckGo fallback", query[:60])
            results = self._fallback.search(query, num_results=10)

        if not results:
            return RetrievalResult(source_name=self.name, success=False, error="no search results")

        if use_llm_selection:
            # LLM-mediated selection: pick the right result before downloading
            from app.services.source_llm_selector import select_matching_result
            selection = select_matching_result(
                results=results,
                title=title or "",
                author=author or "",
                year=year or "",
            )
            if selection.selected_index < 0:
                logger.info("LLM selected no matching result for '%s' (%s)",
                             query[:60], selection.reason[:60])
                return RetrievalResult(source_name=self.name, success=False,
                                       error=f"LLM: no matching result ({selection.reason[:60]})")

            # Honor confidence (REVIEW §3.1): only auto-download at medium/high.
            # A "low" selection means the LLM is unsure this is the cited
            # source — skip rather than risk a wrong-source download.
            if selection.confidence == "low":
                logger.info("LLM match for '%s' was low-confidence — not downloading",
                             query[:60])
                return RetrievalResult(
                    source_name=self.name, success=False,
                    error=f"LLM: low-confidence match, skipped ({selection.reason[:60]})",
                )

            # Download only the LLM-selected result
            selected = results[selection.selected_index]
            logger.info("LLM selected result %d: %s (confidence=%s)",
                         selection.selected_index + 1, selected.url[:60], selection.confidence)
            candidates = [selected]
        else:
            # DOI search: filter for PDFs, try all candidates
            pdf_results = [r for r in results if r.is_pdf or r.url.lower().endswith(".pdf")]
            candidates = pdf_results if pdf_results else results[:_MAX_DOWNLOAD_ATTEMPTS]

        logger.info("Web search '%s': %d results, %d PDF candidates",
                     query[:60], len(results), len(candidates))

        for candidate in candidates[:_MAX_DOWNLOAD_ATTEMPTS]:
            pdf_bytes = self._try_download_pdf(candidate.url)
            if pdf_bytes is None:
                continue

            # Full validation: text quality → identity (triangulation) → completeness
            from app.services.source_validator import validate_retrieved_pdf
            validation = validate_retrieved_pdf(
                pdf_bytes,
                expected_doi=doi,
                expected_title=title,
                expected_author=author,
                expected_year=year,
            )
            if validation.accept:
                logger.info("Web search found valid PDF: %s (%s)",
                            candidate.url[:80], validation.reason[:60])
                return RetrievalResult(
                    source_name=self.name,
                    success=True,
                    full_text=pdf_bytes,
                    full_text_url=candidate.url,
                    doi=doi,
                    title=title,
                    metadata={
                        "identity_confidence": validation.identity_confidence,
                        "completeness": validation.completeness,
                        "text_quality": validation.text_quality,
                        "page_count": validation.page_count,
                    },
                )
            else:
                logger.info("Web search PDF rejected: %s — %s",
                            candidate.url[:80], validation.reason[:60])

        return RetrievalResult(source_name=self.name, success=False, error="no valid PDF found")

    def _run_search(self, query: str) -> list:
        """Run search with cascading engines for SearXNG, fallback to DuckDuckGo.

        For SearXNG: tries Google Scholar first (academic precision), then
        Google (broad coverage), then Bing (last resort). Stops at the first
        engine that returns PDF candidates. This is faster than querying all
        engines simultaneously and prioritizes academic-quality results.

        For other providers: single search, then DuckDuckGo fallback.
        """
        from app.services.search.searxng import SearXNGSearch

        if isinstance(self._search, SearXNGSearch):
            # Cascading: Google Scholar → Google → Bing (user preference: Bing last)
            results: list = []  # bound before the loop so the tail return is safe
            for engine in ("google scholar", "google", "bing"):
                try:
                    results = self._search.search(query, num_results=10, engines=engine)
                    if results:
                        # Check if any results are PDFs or look promising
                        pdf_results = [r for r in results if r.is_pdf or r.url.lower().endswith(".pdf")]
                        if pdf_results or len(results) >= 3:
                            logger.info("SearXNG engine '%s' returned %d results (%d PDFs)",
                                         engine, len(results), len(pdf_results))
                            return results
                        logger.info("SearXNG engine '%s' returned %d results, no PDFs — trying next engine",
                                     engine, len(results))
                except Exception as e:
                    logger.warning("SearXNG engine '%s' failed: %s", engine, str(e)[:60])
            # All engines exhausted — return whatever we got from the last attempt
            return results

        # Non-SearXNG provider: single search
        if not self._search:
            return self._fallback.search(query, num_results=10)
        try:
            return self._search.search(query, num_results=10)
        except Exception as e:
            logger.warning("Primary search failed for '%s': %s — trying DuckDuckGo fallback", query[:60], e)
            return self._fallback.search(query, num_results=10)

    def _try_download_pdf(self, url: str) -> Optional[bytes]:
        """Download a PDF from a URL with SSRF + size-cap + magic-byte checks.

        The URL comes from search-provider results (arbitrary web pages), so it
        is treated as untrusted: safe_fetch rejects non-public hosts and aborts
        downloads past _MAX_PDF_SIZE before they can exhaust memory.
        """
        try:
            data = safe_fetch_bytes(
                url,
                max_bytes=_MAX_PDF_SIZE,
                accept_content_types=("application/pdf",),
                timeout=30,
            )
            if not data[:5] == b"%PDF-":
                logger.debug("URL returned non-PDF content: %s", url[:60])
                return None
            return data
        except Exception as e:
            logger.debug("PDF download failed for %s: %s", url[:60], e)
            return None
