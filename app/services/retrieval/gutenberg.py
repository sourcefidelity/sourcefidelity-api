"""Project Gutenberg retrieval adapter (via Gutendex API).

Retrieves the full text of public-domain literary works — novels, plays,
poems — from Project Gutenberg. This is the primary-text retrieval tier,
used AFTER academic databases (which return secondary criticism, not the
works themselves).

Search: Gutendex (gutendex.com), a keyless wrapper around the Gutenberg
catalog. Text download: gutenberg.org plain-text format.

Note: gutenberg.org may be unreachable from some networks (geoIP blocking
in certain countries). The search API (gutendex.com) is on a separate host.
If the text-download step fails, the result carries the URL for a later retry.
"""

import logging
import re

import httpx

from app.services.relevance import score_relevance
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

GUTENDEX_BASE = "https://gutendex.com/books/"

# Gutenberg plain-text boilerplate markers
_START_MARKER = re.compile(r"\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)
_END_MARKER = re.compile(r"\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE)


class GutenbergRetriever(RetrievalSource):
    """Retrieves public-domain literary texts from Project Gutenberg."""

    name = "gutenberg"

    def search_by_doi(self, doi: str) -> RetrievalResult:
        # Literary works don't have DOIs — this source is title-search only.
        return RetrievalResult(source_name=self.name, success=False, error="Gutenberg does not use DOIs")

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        try:
            # Search Gutendex by title + author surname. Gutendex's search
            # breaks on commas/initials, so extract just the surname.
            search_q = title
            if author:
                surnames = _extract_search_surname(author)
                if surnames:
                    search_q += f" {surnames}"
            resp = httpx.get(
                GUTENDEX_BASE,
                params={"search": search_q},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return RetrievalResult(source_name=self.name, success=False, error="No results")

            # Apply relevance filtering (title + author verification) to avoid
            # wrong-book matches on generic titles.
            for book in results[:5]:
                book_title = book.get("title", "")
                book_authors = [a.get("name", "") for a in book.get("authors", [])]
                rel = score_relevance(title, book_title, author, book_authors)
                if rel.is_relevant:
                    return self._parse_book(book)
                logger.debug("Gutenberg match rejected: %s", rel.detail[:100])

            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"No relevant match (top {min(5, len(results))} results rejected)",
            )
        except Exception as e:
            logger.warning("Gutenberg search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def download_full_text(self, result: RetrievalResult) -> RetrievalResult:
        """Fetch the actual plain text from gutenberg.org and strip boilerplate."""
        if not result.full_text_url:
            return result
        try:
            resp = httpx.get(result.full_text_url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            raw = resp.text
            # Strip the Project Gutenberg boilerplate headers/footers
            clean = _strip_boilerplate(raw)
            result.full_text = clean.encode("utf-8")
            logger.info(
                "Gutenberg text fetched: %s (%d chars after boilerplate strip)",
                result.full_text_url, len(clean),
            )
        except Exception as e:
            logger.warning("Gutenberg text download failed: %s", e)
        return result

    def _parse_book(self, book: dict) -> RetrievalResult:
        """Parse a Gutendex book result into a RetrievalResult."""
        formats = book.get("formats", {}) or {}
        # Prefer UTF-8 plain text
        txt_url = (
            formats.get("text/plain; charset=utf-8")
            or formats.get("text/plain")
        )
        authors = [a.get("name", "") for a in book.get("authors", []) if a.get("name")]

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=None,
            title=book.get("title", ""),
            year=str(book.get("copyright_year") or "") or None,
            authors=authors,
            full_text_url=txt_url,
            metadata=book,
        )


def _strip_boilerplate(raw_text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate from plain text.

    Gutenberg texts are wrapped with:
        *** START OF THE PROJECT GUTENBERG EBOOK [TITLE] ***
        ... [license text] ...
        [actual book text]
        ... [license text] ...
        *** END OF THE PROJECT GUTENBERG EBOOK [TITLE] ***

    This extracts just the book text between the markers.
    """
    start_match = _START_MARKER.search(raw_text)
    end_match = _END_MARKER.search(raw_text)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)

    return raw_text[start_idx:end_idx].strip()


def _extract_search_surname(author: str) -> str:
    """Extract just the surname from an author string for Gutendex search.

    Gutendex search breaks on commas and initials ("Dickens, C." → 0 results),
    so we extract just the surname ("Dickens") for the search query.
    """
    # Reuse the surname extraction from relevance.py
    from app.services.relevance import extract_surnames
    surnames = extract_surnames(author)
    return surnames[0] if surnames else ""

