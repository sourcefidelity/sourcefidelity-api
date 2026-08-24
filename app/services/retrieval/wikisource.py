"""Multilingual primary-text retrieval through the MediaWiki Action API."""

import logging
import re
from urllib.parse import quote

from bs4 import BeautifulSoup
import httpx

from app.services.relevance import extract_surnames, score_relevance
from app.services.retrieval.base import (
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
    SourceRepresentation,
)
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy

logger = logging.getLogger(__name__)
_DEFAULT_LANGS = ("en", "fr", "de", "zh", "es", "it", "ru", "ja", "pt", "ar")
_HEADERS = {
    "User-Agent": "SourceFidelity/0.1 (https://github.com/sourcefidelity/sourcefidelity-api)"
}
_NON_CONTENT_PREFIXES = ("author:", "index:", "page:", "category:", "template:", "portal:")


class WikisourceRetriever(RetrievalSource):
    name = "wikisource"
    capabilities = frozenset({"title_search", "full_text", "public_domain", "multilingual"})
    documentation_url = "https://www.mediawiki.org/wiki/API:Action_API"

    def __init__(self) -> None:
        self.policy = provider_policy(
            self.name,
            ProviderPolicy(timeout_seconds=15, max_batches=len(_DEFAULT_LANGS)),
        )
        self.languages = _DEFAULT_LANGS[: self.policy.max_batches or len(_DEFAULT_LANGS)]

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name, success=False, error="Wikisource does not use DOIs"
        )

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        if not self.policy.enabled:
            return RetrievalResult(source_name=self.name, success=False, error="Provider disabled")
        surname = (extract_surnames(author) or [""])[0] if author else ""
        query = f"{title} {surname}".strip()
        for lang in self.languages:
            try:
                result = self._search_edition(lang, title, author, query)
                if result.success:
                    return result
            except Exception as exc:
                logger.debug("Wikisource %s search failed: %s", lang, exc)
        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="No complete relevant page across configured language editions",
        )

    def _search_edition(
        self, lang: str, title: str, author: str | None, query: str
    ) -> RetrievalResult:
        endpoint = f"https://{lang}.wikisource.org/w/api.php"
        response = httpx.get(
            endpoint,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 0,
                "srlimit": 5,
                "format": "json",
                "formatversion": 2,
            },
            headers=_HEADERS,
            timeout=self.policy.timeout_seconds,
        )
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])
        for candidate in results:
            page_title = candidate.get("title", "")
            if not page_title or page_title.lower().startswith(_NON_CONTENT_PREFIXES):
                continue
            root_title = page_title.split("/", 1)[0]
            page = self._fetch_rendered_page(lang, root_title)
            if not page:
                continue
            clean_text, page_author = page
            relevance = score_relevance(title, root_title, author, page_author or None)
            # Fail closed on short root pages, which are commonly tables of
            # contents pointing to chapter subpages rather than the cited work.
            if not relevance.is_relevant or len(clean_text) < 2_000:
                continue
            return RetrievalResult(
                source_name=self.name,
                success=True,
                title=root_title,
                authors=[page_author] if page_author else [],
                representation=SourceRepresentation(
                    kind=RepresentationKind.PLAIN_TEXT,
                    media_type="text/plain",
                    content=clean_text.encode("utf-8"),
                    original_kind=RepresentationKind.HTML,
                    charset="utf-8",
                    completeness="not_assessed",
                ),
                full_text_url=f"https://{lang}.wikisource.org/wiki/{quote(root_title.replace(' ', '_'))}",
                metadata={
                    "lang": lang,
                    "wikisource_title": root_title,
                    "license_class": "public_domain",
                    "representation_scope": "rendered_root_page",
                    "completeness": "not_assessed",
                },
            )
        return RetrievalResult(source_name=self.name, success=False, error=f"No relevant match in {lang}")

    def _fetch_rendered_page(self, lang: str, page_title: str) -> tuple[str, str] | None:
        endpoint = f"https://{lang}.wikisource.org/w/api.php"
        response = httpx.get(
            endpoint,
            params={
                "action": "parse",
                "page": page_title,
                "prop": "text|wikitext",
                "format": "json",
                "formatversion": 2,
            },
            headers=_HEADERS,
            timeout=self.policy.timeout_seconds,
        )
        response.raise_for_status()
        parsed = response.json().get("parse", {})
        html = parsed.get("text", "")
        wikitext = parsed.get("wikitext", "")
        if isinstance(html, dict):
            html = html.get("*", "")
        if isinstance(wikitext, dict):
            wikitext = wikitext.get("*", "")
        if not html:
            return None
        return _extract_rendered_text(html), _extract_header_author(wikitext)


def _extract_header_author(wikitext: str) -> str:
    match = re.search(
        r"\|\s*(?:override_)?author\s*=\s*(.+?)(?:\n\s*\||\n\}\})", wikitext
    )
    if not match:
        return ""
    raw = match.group(1).strip()
    link = re.search(r"\[\[(?:[^\]]*\|)?([^\]]+)\]\]", raw)
    return (link.group(1) if link else raw).strip()


def _extract_rendered_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for selector in (
        "script", "style", "nav", ".mw-editsection", ".ws-noexport", ".noprint",
        ".mw-collapsible-toggle", "table.metadata", "table.licenseContainer",
    ):
        for node in soup.select(selector):
            node.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
