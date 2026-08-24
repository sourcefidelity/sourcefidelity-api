"""OpenAlex retrieval adapter."""

import copy
import logging
import re

import httpx

from app.config import settings
from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
)
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexRequestError(RuntimeError):
    """A credential-safe OpenAlex failure that never includes the request URL."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class OpenAlexRetriever(RetrievalSource):
    name = "openalex"
    capabilities = frozenset(
        {
            "doi",
            "batch_doi",
            "title_author",
            "batch_title_candidates",
            "metadata",
            "abstract",
            "locations",
        }
    )
    documentation_url = "https://developers.openalex.org/api-reference/introduction"
    default_policy = ProviderPolicy(timeout_seconds=15.0, batch_size=25)

    def __init__(self) -> None:
        if not settings.OPENALEX_GROUPED_TITLE_PREFETCH_ENABLED:
            self.capabilities = self.capabilities - {"batch_title_candidates"}
        self.policy = provider_policy(self.name, self.default_policy)
        self._authentication_failed = False
        self._doi_cache: dict[str, RetrievalResult] = {}
        self._title_cache: dict[tuple[str, str], RetrievalResult] = {}
        self._grouped_title_attempted: set[tuple[str, str]] = set()
        self.provider_metrics = {
            "calls": 0,
            "grouped_doi_calls": 0,
            "grouped_doi_items": 0,
            "grouped_doi_cache_hits": 0,
            "grouped_title_calls": 0,
            "grouped_title_items": 0,
            "grouped_title_matches": 0,
            "grouped_title_cache_hits": 0,
            "individual_title_fallbacks": 0,
            "network_errors": 0,
            "authentication_failures": 0,
        }

    def _headers(self) -> dict:
        # OpenAlex identifies polite-pool users by mailto in User-Agent.
        email = settings.OPENALEX_EMAIL or "support@sourcefidelity.org"
        return {"User-Agent": f"SourceFidelity/{settings.APP_VERSION} (mailto:{email})"}

    def _auth_params(self) -> dict:
        # Since Feb 13 2025, OpenAlex requires an API key passed as the
        # "api_key" query parameter (NOT an Authorization header).
        # See https://developers.openalex.org/api-reference/authentication
        if settings.OPENALEX_API_KEY:
            return {"api_key": settings.OPENALEX_API_KEY.strip()}
        return {}

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        """Request without allowing a query-string credential into logs/errors."""
        if self._authentication_failed:
            raise OpenAlexRequestError(
                401, "OpenAlex authentication circuit is open after an earlier 401"
            )
        try:
            self.provider_metrics["calls"] += 1
            response = httpx.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self.policy.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            self.provider_metrics["network_errors"] += 1
            raise RuntimeError(
                f"OpenAlex network error ({type(exc).__name__})"
            ) from exc
        if response.status_code == 401:
            self._authentication_failed = True
            self.provider_metrics["authentication_failures"] += 1
            raise OpenAlexRequestError(
                401,
                "OpenAlex authentication failed (401); provider circuit opened",
            )
        if response.status_code >= 400:
            raise OpenAlexRequestError(
                response.status_code,
                f"OpenAlex request failed (HTTP {response.status_code})",
            )
        return response

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        normalized = doi.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return normalized

    def prefetch_dois(self, dois: list[str]) -> int:
        """Group DOI filters into bounded OpenAlex requests and cache misses."""
        normalized = list(
            dict.fromkeys(self._normalize_doi(doi) for doi in dois if doi.strip())
        )
        missing = [doi for doi in normalized if doi not in self._doi_cache]
        batch_size = min(self.policy.batch_size, 100)
        item_limit = (
            len(missing)
            if self.policy.max_batches == 0
            else batch_size * self.policy.max_batches
        )
        prefetched = 0
        for start in range(0, min(len(missing), item_limit), batch_size):
            chunk = missing[start:start + batch_size]
            try:
                params = {
                    "filter": "doi:" + "|".join(
                        f"https://doi.org/{doi}" for doi in chunk
                    ),
                    "per_page": len(chunk),
                }
                params.update(self._auth_params())
                response = self._get(f"{OPENALEX_BASE}/works", params=params)
                payload = response.json()
                results = payload.get("results") or []
                found: dict[str, RetrievalResult] = {}
                for work in results:
                    if not isinstance(work, dict):
                        continue
                    result = self._parse_work(work)
                    if result.doi:
                        found[self._normalize_doi(result.doi)] = result
                for doi in chunk:
                    if doi in found:
                        self._doi_cache[doi] = found[doi]
                        prefetched += 1
                    else:
                        self._doi_cache[doi] = RetrievalResult(
                            source_name=self.name,
                            success=False,
                            error="Not found",
                        )
                self.provider_metrics["grouped_doi_calls"] += 1
                self.provider_metrics["grouped_doi_items"] += len(chunk)
            except Exception as exc:
                logger.warning("OpenAlex grouped DOI prefetch failed: %s", exc)
                break
        return prefetched

    @staticmethod
    def _title_key(title: str, author: str | None) -> tuple[str, str]:
        normalized_title = re.sub(r"\s+", " ", title).strip().casefold()
        normalized_author = re.sub(r"\s+", " ", author or "").strip().casefold()
        return normalized_title, normalized_author

    @staticmethod
    def _quoted_title(title: str) -> str:
        """Return a literal phrase safe for an OpenAlex Boolean search."""
        clean = re.sub(r"[*?]", "", title).strip()
        clean = clean.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{clean}"'

    @staticmethod
    def _title_chunks(
        queries: list[tuple[str, str | None]],
        batch_size: int,
        max_query_chars: int = 3000,
    ) -> list[list[tuple[str, str | None]]]:
        """Bound grouped searches by both item count and request-URL length."""
        chunks: list[list[tuple[str, str | None]]] = []
        current: list[tuple[str, str | None]] = []
        current_chars = 2  # surrounding parentheses
        for query in queries:
            phrase_chars = len(OpenAlexRetriever._quoted_title(query[0]))
            separator_chars = 4 if current else 0  # `` OR ``
            if current and (
                len(current) >= batch_size
                or current_chars + separator_chars + phrase_chars > max_query_chars
            ):
                chunks.append(current)
                current = []
                current_chars = 2
                separator_chars = 0
            current.append(query)
            current_chars += separator_chars + phrase_chars
        if current:
            chunks.append(current)
        return chunks

    def prefetch_titles(
        self,
        queries: list[tuple[str, str | None]],
    ) -> int:
        """Generate candidates for several titles, caching only strong matches.

        OpenAlex title search is not a keyed batch API: one Boolean request
        returns a globally ranked union without per-title attribution. Returned
        works are therefore matched locally against every query. An unresolved
        query is deliberately *not* cached as a miss, so its later ordinary
        lookup performs the required individual fallback.
        """
        if not settings.OPENALEX_GROUPED_TITLE_PREFETCH_ENABLED:
            return 0

        unique: dict[tuple[str, str], tuple[str, str | None]] = {}
        for title, author in queries:
            if title and title.strip():
                unique.setdefault(self._title_key(title, author), (title, author))
        pending = [
            query
            for key, query in unique.items()
            if key not in self._title_cache and key not in self._grouped_title_attempted
        ]
        if not pending:
            return 0

        # Ten long bibliographic titles comfortably fit below OpenAlex's
        # approximate 4 KB request-URL limit; the character bound handles
        # unusually long titles independently of the configured DOI batch size.
        batch_size = min(self.policy.batch_size, 10)
        chunks = self._title_chunks(pending, batch_size)
        if self.policy.max_batches:
            chunks = chunks[:self.policy.max_batches]

        from app.services.relevance import score_relevance

        matched_count = 0
        for chunk in chunks:
            keys = [self._title_key(title, author) for title, author in chunk]
            self._grouped_title_attempted.update(keys)
            search = "(" + " OR ".join(
                self._quoted_title(title) for title, _author in chunk
            ) + ")"
            try:
                params = {"search.exact": search, "per_page": 100}
                params.update(self._auth_params())
                response = self._get(f"{OPENALEX_BASE}/works", params=params)
                payload = response.json()
                works = [work for work in payload.get("results") or [] if isinstance(work, dict)]
                parsed = [(work, self._parse_work(work)) for work in works]
                for (title, author), key in zip(chunk, keys):
                    best: tuple[float, RetrievalResult] | None = None
                    for _work, result in parsed:
                        relevance = score_relevance(
                            title,
                            result.title or "",
                            author,
                            result.authors or [],
                        )
                        if not relevance.is_relevant:
                            continue
                        if best is None or relevance.score > best[0]:
                            best = (relevance.score, result)
                    if best is not None:
                        self._title_cache[key] = copy.deepcopy(best[1])
                        matched_count += 1
                self.provider_metrics["grouped_title_calls"] += 1
                self.provider_metrics["grouped_title_items"] += len(chunk)
            except Exception as exc:
                logger.warning("OpenAlex grouped title prefetch failed: %s", exc)
                break
        self.provider_metrics["grouped_title_matches"] += matched_count
        return matched_count

    def preflight(self, require_api_key: bool = False) -> tuple[bool, str]:
        """Make one small credential/reachability check before a long baseline."""
        if require_api_key and not settings.OPENALEX_API_KEY:
            return False, "OpenAlex API key is not configured"
        try:
            self._get(
                f"{OPENALEX_BASE}/works/W2741809807",
                params=self._auth_params(),
            )
            mode = "configured key" if settings.OPENALEX_API_KEY else "unkeyed access"
            return True, f"OpenAlex preflight succeeded using {mode}"
        except Exception as exc:
            return False, str(exc)

    def search_by_doi(self, doi: str) -> RetrievalResult:
        normalized = self._normalize_doi(doi)
        if normalized in self._doi_cache:
            self.provider_metrics["grouped_doi_cache_hits"] += 1
            return copy.deepcopy(self._doi_cache[normalized])
        url = f"{OPENALEX_BASE}/works/doi:{doi}"
        try:
            resp = self._get(url, params=self._auth_params())
            data = resp.json()
            return self._parse_work(data)
        except Exception as e:
            logger.warning("OpenAlex DOI search failed: %s", e)
            return RetrievalResult(source_name=self.name, success=False, error=str(e))

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        key = self._title_key(title, author)
        if key in self._title_cache:
            self.provider_metrics["grouped_title_cache_hits"] += 1
            return copy.deepcopy(self._title_cache[key])
        if key in self._grouped_title_attempted:
            self.provider_metrics["individual_title_fallbacks"] += 1
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
            resp = self._get(url, params=params)
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
                    self._title_cache[key] = copy.deepcopy(result)
                    return result
                logger.debug(
                    "OpenAlex match rejected: %s", rel.detail[:100],
                )

            result = RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"No relevant match (top {len(results)} results were keyword coincidences)",
            )
            self._title_cache[key] = copy.deepcopy(result)
            return result
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
        for authorship in data.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author_record = authorship.get("author") or {}
            if not isinstance(author_record, dict):
                continue
            name = author_record.get("display_name", "")
            if name:
                authors.append(name)

        pub_year = data.get("publication_year")
        year = str(pub_year) if pub_year else "n.d."

        title = data.get("title") or data.get("display_name", "")

        # Preserve the complete OpenAlex/Unpaywall location graph. Choosing one
        # URL here caused false misses when the preferred location was blocked,
        # dead or HTML-only while another repository held a usable copy.
        locations = _parse_locations(data)
        best_location = next((loc for loc in locations if loc.is_best), None)
        pdf_location = next(
            (loc for loc in locations if loc.representation_kind is RepresentationKind.PDF),
            None,
        )
        preferred = best_location or pdf_location or (locations[0] if locations else None)

        # Extract abstract from OpenAlex's inverted-index format
        abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))

        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=doi,
            title=title,
            year=year,
            authors=authors,
            full_text_url=preferred.url if preferred else None,
            locations=locations,
            abstract=abstract,
            metadata=data,
        )


def _parse_locations(data: dict) -> list[AcquisitionLocation]:
    """Normalize and deduplicate every OpenAlex acquisition location."""
    best = data.get("best_oa_location") or {}
    raw_locations = [best, data.get("primary_location") or {}]
    raw_locations.extend(data.get("locations") or [])
    parsed: list[AcquisitionLocation] = []
    seen: set[str] = set()
    for raw in raw_locations:
        if not isinstance(raw, dict) or not raw:
            continue
        source = raw.get("source") or {}
        if not isinstance(source, dict):
            source = {}
        landing_url = raw.get("landing_page_url")
        pdf_url = raw.get("pdf_url")
        for url, kind, media_type in (
            (pdf_url, RepresentationKind.PDF, "application/pdf"),
            (landing_url, RepresentationKind.HTML, "text/html"),
        ):
            if not url or url in seen:
                continue
            seen.add(url)
            parsed.append(
                AcquisitionLocation(
                    url=url,
                    provider="openalex",
                    media_type=media_type,
                    representation_kind=kind,
                    landing_page_url=landing_url,
                    host_type=source.get("host_organization_name")
                    or source.get("type"),
                    version=raw.get("version"),
                    license=raw.get("license"),
                    access_type="open_access" if raw.get("is_oa") else None,
                    is_best=(raw == best),
                    metadata={
                        "source_id": source.get("id"),
                        "source_name": source.get("display_name"),
                        "is_accepted": raw.get("is_accepted"),
                        "is_published": raw.get("is_published"),
                    },
                )
            )
    return parsed


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as {word: [position1, position2, ...]}.
    This rebuilds the original word order.
    """
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs or []:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions) if positions else None
