"""DuckDuckGo search provider — free, no API key required.

Uses DuckDuckGo's HTML endpoint (unofficial, not an API). Parses results from
the HTML page. No rate limits, no API key, no cost — but less reliable than a
real API (DDG may change their HTML format, rate-limit scraping, or serve
CAPTCHAs under heavy use).

This is a LAST-RESORT fallback when no other search provider is available.
For reliable search, use SearXNG (self-hosted) or a commercial API.

The HTML endpoint: https://html.duckduckgo.com/html/?q=<query>
Returns results in <a class="result__a" href="...">title</a> format,
with snippets in <a class="result__snippet">...</a>.
"""

import logging
import re

import httpx
from bs4 import BeautifulSoup

from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

_DDG_URL = "https://html.duckduckgo.com/html/"


class DuckDuckGoSearch(SearchProvider):
    """DuckDuckGo HTML-scraping search provider.

    Free, no API key. Unofficial — may break if DDG changes their HTML format.
    Use as a last-resort fallback when SearXNG/commercial APIs are unavailable.
    """

    @property
    def name(self) -> str:
        return "DuckDuckGo"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via DuckDuckGo HTML endpoint."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
        }
        data = {"q": query}

        try:
            resp = httpx.post(_DDG_URL, headers=headers, data=data, timeout=15,
                              follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("DuckDuckGo search failed for '%s': %s", query[:60], e)
            return []

        results = []
        soup = BeautifulSoup(resp.text, "html.parser")

        # DDG HTML results: each result is in a div with class "result"
        # Title: <a class="result__a" href="...">
        # Snippet: <a class="result__snippet">...</a>
        for result_div in soup.find_all("div", class_="result")[:num_results]:
            link = result_div.find("a", class_="result__a")
            if not link:
                continue
            url = link.get("href", "")
            title = link.get_text(strip=True)

            snippet_tag = result_div.find("a", class_="result__snippet")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            # DDG wraps URLs in a redirect: //duckduckgo.com/l/?uddg=<actual_url>
            # Extract the actual URL
            if "uddg=" in url:
                from urllib.parse import parse_qs, urlparse
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                url = params.get("uddg", [url])[0]

            is_pdf = url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))

        return results
