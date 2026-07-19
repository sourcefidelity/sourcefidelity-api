"""OpenAlex retrieval adapter."""

import logging
import re

import httpx

from app.config import settings
from app.services.retrieval.base import RetrievalSource, RetrievalResult

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexRetriever(RetrievalSource):
    name = "openalex"

    def _headers(self) -> dict:
        # OpenAlex identifies polite-pool users by mailto in User-Agent.
        email = settings.OPENALEX_EMAIL or "support@sourcefidelity.org"
        return {"User-Agent": f"SourceFidelity/{settings.APP_VERSION} (mailto:{email})"}

    def _auth_params(self) -> dict:
        # Since Feb 13 2025, OpenAlex requires an API key passed as the
        # "api_key" query parameter (NOT an Authorization header).
        # See https://developers.openalex.org/api-reference/authentication
        if settings.OPENALEX_API_KEY:
            return {"api_key": settings.OPENALEX_API_KEY}
        return {}

    def search_by_doi(self, doi: str) -> RetrievalResult:
        url = f"{OPENALEX_BASE}/works/doi:{doi}"
        try:
            resp = httpx.get(url, headers=self._headers(), params=self._auth_params(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_work(data)
        except Exception as e:
            logger.warning("OpenAlex DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        try:
            # Use the title relevance search only. OpenAlex's filter syntax
            # (authorships.author.display_name.search:) uses commas/colons as
            # delimiters, which corrupts on real author names like "York, A.E."
            # The title search alone ranks well enough for our purposes.
            #
            # Fetch a few candidates (not just 1) and apply a relevance filter
            # so keyword-coincidence matches are rejected (e.g. "Rain Man" the
            # film should not match a diabetology paper whose title has "Man").
            #
            # Strip wildcard characters (* and ?) from the query — OpenAlex
            # treats them as wildcards, not literals, so "How costly is
            # protectionism?" triggers a 400 error.
            clean_title = re.sub(r"[*?]", "", title).strip()
            params = {"search": clean_title, "per_page": 5}
            params.update(self._auth_params())
            url = f"{OPENALEX_BASE}/works"
            resp = httpx.get(url, headers=self._headers(), params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return RetrievalResult(source_name=self.name, success=False, error="No results")

            from app.services.relevance import score_relevance

            for work in results:
                result = self._parse_work(work)
                matched_title = result.title or ""
                matched_authors = result.authors or []
                rel = score_relevance(title, matched_title, author, matched_authors)
                if rel.is_relevant:
                    return result
                logger.debug(
                    "OpenAlex match rejected: %s", rel.detail[:100],
                )

            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"No relevant match (top {len(results)} results were keyword coincidences)",
            )
        except Exception as e:
            logger.warning("OpenAlex title search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def _parse_work(self, data: dict) -> RetrievalResult:
        """Parse an OpenAlex work object into a RetrievalResult."""
        # OpenAlex returns doi as a URL ("https://doi.org/10.xxx/yyy").
        # Use removeprefix, NOT lstrip — lstrip strips a character set and
        # would corrupt DOIs whose leading chars happen to be in the prefix.
        raw_doi = data.get("doi") or ""
        doi = raw_doi.removeprefix("https://doi.org/").strip() or None

        authors = []
        for authorship in data.get("authorships", []):
            name = authorship.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)

        pub_year = data.get("publication_year")
        year = str(pub_year) if pub_year else "n.d."

        title = data.get("title") or data.get("display_name", "")

        # Check for OA PDF
        best_oa = data.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url")

        # Extract abstract from OpenAlex's inverted-index format
        abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=pdf_url,
            abstract=abstract,
            metadata=data,
        )


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as {word: [position1, position2, ...]}.
    This rebuilds the original word order.
    """
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions) if positions else None
