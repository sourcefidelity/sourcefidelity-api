"""Wikisource retrieval adapter (multilingual, 83 languages).

Wikisource (wikisource.org) is the Wikimedia Foundation's free-content library
of primary texts — novels, plays, poems, historical documents, philosophical
works. It spans 83 language editions, making it the single best source for
global public-domain text coverage: languages with no dedicated repository
(Korean, Hindi, Arabic, Persian, Turkish, many African languages) are covered here.

API: MediaWiki Action API (keyless). Search by title, fetch page wikitext,
strip markup to get clean text. Multilingual: each language edition is a
separate subdomain (en.wikisource.org, fr.wikisource.org, zh.wikisource.org).

Note: Wikimedia infrastructure may be regionally unavailable in some
Chinese-web-optimized deployments. The source works globally in regions
where Wikimedia is reachable.
"""

import logging
import re

import httpx

from app.services.relevance import score_relevance
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

# Language editions to try, in order of likelihood for the cited work.
# English first (most common in academic writing globally), then major
# non-English editions. The search tries each until a match is found.
_WIKISOURCE_LANGS = ("en", "fr", "de", "zh", "es", "it", "ru", "ja", "pt", "ar")

_HEADERS = {"User-Agent": "SourceFidelity/0.1.0 (open-source source-verification tool)"}


class WikisourceRetriever(RetrievalSource):
    """Retrieves public-domain texts from Wikisource (multilingual)."""

    name = "wikisource"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        # Literary/historical works don't have DOIs.
        return RetrievalResult(source_name=self.name, success=False, error="Wikisource does not use DOIs")

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        search_q = title
        if author:
            from app.services.relevance import extract_surnames
            surname = (extract_surnames(author) or [""])[0]
            if surname:
                search_q += f" {surname}"

        # Try each language edition until we find a relevant match
        for lang in _WIKISOURCE_LANGS:
            try:
                result = self._search_edition(lang, title, author, search_q)
                if result.success:
                    return result
            except Exception as e:
                logger.debug("Wikisource %s search failed: %s", lang, e)
                continue

        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="No relevant match across language editions",
        )

    def _search_edition(
        self, lang: str, title: str, author: str | None, search_q: str
    ) -> RetrievalResult:
        """Search a single Wikisource language edition."""
        base = f"https://{lang}.wikisource.org/w/api.php"

        # Step 1: Search for the work
        resp = httpx.get(
            base,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_q,
                "format": "json",
                "srlimit": 5,
            },
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return RetrievalResult(source_name=self.name, success=False, error=f"No results in {lang}")

        # Step 2: Filter by relevance (title + author)
        for r in results:
            page_title = r["title"]
            # The search result title may include subpage paths
            # (e.g. "Hard Times/First Book/Chapter I"). For matching,
            # extract the root work name.
            display_title = page_title.split("/")[-1] if "/" in page_title else page_title

            # Fetch the page to get author info from the {{header}} template
            page_data = self._fetch_page(lang, page_title)
            if not page_data:
                continue

            page_text, page_author = page_data
            # Clean the wikitext
            clean_text = _strip_wikitext(page_text)

            rel = score_relevance(title, display_title, author, page_author or None)
            if rel.is_relevant:
                result = RetrievalResult(
                    source_name=self.name,
                    success=True,
                    title=page_title,
                    authors=[page_author] if page_author else [],
                    full_text=clean_text.encode("utf-8") if clean_text else None,
                    full_text_url=f"https://{lang}.wikisource.org/wiki/{page_title.replace(' ', '_')}",
                    metadata={"lang": lang, "wikisource_title": page_title},
                )
                # If the page is very short (likely an index), note it
                if clean_text and len(clean_text) < 200:
                    logger.info(
                        "Wikisource page %s is short (%d chars) — may be an index page",
                        page_title, len(clean_text),
                    )
                return result
            logger.debug("Wikisource match rejected: %s", rel.detail[:80])

        return RetrievalResult(source_name=self.name, success=False, error=f"No relevant match in {lang}")

    def _fetch_page(self, lang: str, page_title: str) -> tuple[str, str] | None:
        """Fetch a Wikisource page's wikitext and extract author.

        Returns (wikitext, author_name) or None on failure.
        """
        base = f"https://{lang}.wikisource.org/w/api.php"
        try:
            resp = httpx.get(
                base,
                params={
                    "action": "parse",
                    "page": page_title,
                    "prop": "wikitext",
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
            if not wikitext:
                return None

            # Extract author from the {{header}} template
            author = _extract_header_author(wikitext)
            return wikitext, author
        except Exception as e:
            logger.debug("Wikisource page fetch failed for %s: %s", page_title, e)
            return None


def _extract_header_author(wikitext: str) -> str:
    """Extract the author from a Wikisource {{header}} template.

    The header looks like:
        {{header
         | title    = ...
         | author   = [[Author:Charles Dickens|Charles Dickens]]
         | override_author = ...
        }}
    """
    # Match author or override_author field
    m = re.search(r"\|\s*(?:override_)?author\s*=\s*(.+?)(?:\n\s*\||\n\}\})", wikitext)
    if not m:
        return ""
    raw = m.group(1).strip()
    # Strip wikilinks: [[Author:Charles Dickens|Charles Dickens]] -> Charles Dickens
    link_m = re.search(r"\[\[(?:[^\]]*\|)?([^\]]+)\]\]", raw)
    if link_m:
        return link_m.group(1).strip()
    return raw.strip()


def _strip_wikitext(wikitext: str) -> str:
    """Strip wikitext markup to get clean readable text.

    Removes: templates ({{...}}), wikilinks ([[...]]), HTML tags, wiki
    formatting ('''bold''', ''italic''), and section headers (===).
    """
    # Remove the {{header}} template block entirely
    text = re.sub(r"\{\{header[^}]*\}\}", "", wikitext, flags=re.DOTALL | re.IGNORECASE)
    # Remove other templates ({{...}}) — simple non-nested removal
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Convert wikilinks: [[Link|Display text]] -> Display text
    text = re.sub(r"\[\[[^\]]*\|([^\]]+)\]\]", r"\1", text)
    # Convert simple wikilinks: [[Article]] -> Article
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove remaining brackets
    text = text.replace("[[", "").replace("]]", "")
    # Remove wiki formatting
    text = re.sub(r"'{2,}", "", text)  # bold/italic
    text = re.sub(r"^=+([^=]+)=+$", r"\1", text, flags=re.MULTILINE)  # headers
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
