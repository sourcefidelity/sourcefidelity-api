"""DataCite DOI metadata — theses, reports and repository deposits.

Crossref registers journal and book DOIs; DataCite registers the rest, which is
where a large share of student-cited grey literature actually lives: honours and
doctoral theses, agency reports, working papers and institutional-repository
deposits. Those entries previously had no identifier route at all, so a genuine
thesis and an invented one looked equally unfindable.

Public API, no authentication. This adapter observes metadata only.
"""

import logging
import re

import httpx

from app.log_safety import safe_exception_code
from app.services.doi_cache import doi_request_segment
from app.services.retrieval.base import SCHOLARLY_PAPER_KINDS, OBSERVED_AUTHOR_LIMIT, RetrievalResult, RetrievalSource
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy
from app.services.processing_metrics import record_provider_request

logger = logging.getLogger(__name__)

DATACITE_DOIS = "https://api.datacite.org/dois"
_HEADERS = {
    "User-Agent": "SourceFidelity/0.1 (academic source verification; contact via repository)"
}
_MAX_ROWS = 5


# Characters that steer the query language rather than match text. A cited
# title routinely contains colons, question marks and quotation marks.
_QUERY_SYNTAX = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _sanitize_query(value: str) -> str:
    return " ".join(_QUERY_SYNTAX.sub(" ", value).split())


def _attribute_title(attributes: dict) -> str | None:
    for entry in attributes.get("titles") or []:
        value = str((entry or {}).get("title") or "").strip()
        if value:
            return value
    return None


def _attribute_authors(attributes: dict) -> list[str]:
    names = []
    for entry in attributes.get("creators") or []:
        value = str((entry or {}).get("name") or "").strip()
        if value:
            names.append(value)
    return names[:OBSERVED_AUTHOR_LIMIT]


class DataCiteRetriever(RetrievalSource):
    """Look up DataCite-registered works by DOI or by title and author."""
    # Registers DOIs for theses, reports and deposits Crossref does not.
    required_for_search_completion = True

    name = "datacite"
    # Repository DOIs: preprints, reports, theses, data; not book catalogues.
    supported_source_kinds = SCHOLARLY_PAPER_KINDS | {"dataset", "software"}
    capabilities = frozenset(
        {"doi", "title_author", "metadata", "metadata_only_search"}
    )
    documentation_url = "https://support.datacite.org/docs/api"

    def __init__(self) -> None:
        self.policy = provider_policy(
            self.name,
            ProviderPolicy(timeout_seconds=20.0, min_interval_seconds=1.0),
        )
        self.provider_metrics: dict[str, int] = {"calls": 0, "timeouts": 0, "errors": 0}

    def _request(self, url: str, params: dict) -> RetrievalResult | dict:
        self.provider_metrics["calls"] += 1
        record_provider_request("datacite")
        try:
            response = httpx.get(
                url, params=params, headers=_HEADERS,
                timeout=self.policy.timeout_seconds,
            )
        except httpx.TimeoutException:
            self.provider_metrics["timeouts"] += 1
            return RetrievalResult(
                source_name=self.name, success=False, error="read_timeout"
            )
        except httpx.HTTPError as exc:
            self.provider_metrics["errors"] += 1
            return RetrievalResult(
                source_name=self.name, success=False,
                error=f"datacite:{safe_exception_code(exc)}",
            )
        if response.status_code == 404:
            return RetrievalResult(
                source_name=self.name, success=False, error="No results",
                metadata={"identity_search_result_count": 0},
            )
        if response.status_code != 200:
            return RetrievalResult(
                source_name=self.name, success=False,
                error=f"DataCite HTTP {response.status_code}",
            )
        try:
            return response.json()
        except ValueError:
            return RetrievalResult(
                source_name=self.name, success=False, error="response_invalid"
            )

    def _result_from(self, attributes: dict, row_count: int) -> RetrievalResult:
        published = attributes.get("publicationYear")
        return RetrievalResult(
            source_name=self.name,
            success=True,
            doi=str(attributes.get("doi") or "") or None,
            title=_attribute_title(attributes),
            authors=_attribute_authors(attributes),
            year=str(published) if published else None,
            metadata={
                "identity_search_result_count": row_count,
                "publisher": attributes.get("publisher"),
                "resource_type": ((attributes.get("types") or {}).get("resourceTypeGeneral")),
            },
        )

    def search_by_doi(self, doi: str) -> RetrievalResult:
        segment = doi_request_segment(str(doi or ""))
        if segment is None:
            return RetrievalResult(
                source_name=self.name, success=False, error="No DOI supplied"
            )
        payload = self._request(f"{DATACITE_DOIS}/{segment}", {})
        if isinstance(payload, RetrievalResult):
            return payload
        attributes = (payload.get("data") or {}).get("attributes") or {}
        if not attributes:
            return RetrievalResult(
                source_name=self.name, success=False, error="No results",
                metadata={"identity_search_result_count": 0},
            )
        return self._result_from(attributes, 1)

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        cleaned = " ".join(str(title or "").split())
        if not cleaned:
            return RetrievalResult(
                source_name=self.name, success=False, error="No title supplied"
            )
        # Title alone. Appending the author narrows a genuine deposit out of its
        # own index, as measured on Semantic Scholar.
        #
        # Two passes. The field-scoped phrase is precise but demands the whole
        # cited title match the registered one; a deposit whose title carries a
        # different subtitle or punctuation is missed. When it returns nothing,
        # retry the same words unscoped before concluding the work is absent.
        # Characters that steer the query language are removed from both.
        sanitized = _sanitize_query(cleaned)[:250]
        if not sanitized:
            return RetrievalResult(
                source_name=self.name, success=False, error="No usable title terms"
            )
        for query in (f'titles.title:"{sanitized}"', sanitized):
            payload = self._request(
                DATACITE_DOIS, {"query": query, "page[size]": _MAX_ROWS}
            )
            if isinstance(payload, RetrievalResult):
                return payload
            rows = payload.get("data") or []
            if rows:
                return self._result_from(
                    (rows[0] or {}).get("attributes") or {}, len(rows)
                )
        return RetrievalResult(
            source_name=self.name, success=False, error="No results",
            metadata={"identity_search_result_count": 0},
        )
