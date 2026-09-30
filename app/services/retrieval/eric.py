"""ERIC — education literature, and only that.

This adapter is **positive-only** (`positive-only-corroboration-v1`). ERIC
indexes education research, so its silence about a law article or a translation
journal says nothing at all, and must never be read as evidence that a work does
not exist. Measured 2026-09-22 against this corpus: of four genuine
education-adjacent titles it holds one, and of three labelled-fabricated titles
it holds none. Low recall, and that is fine — under the positive-only contract a
hit protects a genuine reference from a false accusation and a miss costs
nothing, because the bounded review excludes this provider from coverage quorum
and resolves its unresolved leads out of bounds.

Free, no key, no registration.
"""

import logging
import re

import httpx

from app.log_safety import safe_exception_code
from app.services.retrieval.base import SCHOLARLY_PAPER_KINDS, OBSERVED_AUTHOR_LIMIT, RetrievalResult, RetrievalSource
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy
from app.services.processing_metrics import record_provider_request

logger = logging.getLogger(__name__)

ERIC_SEARCH = "https://api.ies.ed.gov/eric/"
_HEADERS = {
    "User-Agent": "SourceFidelity/0.1 (academic source verification; contact via repository)"
}
_FIELDS = "id,title,author,publicationdateyear,source"
_MAX_ROWS = 5
# The query is wrapped in title:"…". A quotation mark inside the title would
# close that wrapper early and turn the remainder into stray query syntax;
# measured separately, an unquoted search returns the entire database.
_QUERY_SYNTAX = re.compile(r'["\\:()\[\]{}^~*?]')


def _sanitize(value: str) -> str:
    return " ".join(_QUERY_SYNTAX.sub(" ", str(value or "")).split())


class EricRetriever(RetrievalSource):
    """Confirm an education work exists; never testify that one does not."""
    # Unchanged for now. ARCHITECTURE records ERIC as positive-only
    # education literature whose silence about another field is not
    # evidence, which argues for False -- but that would REMOVE a block on
    # fabrication findings, so it needs its own measured decision rather
    # than being folded into this refactor.
    required_for_search_completion = True

    name = "eric"
    # Education articles, reports and theses; not a book catalogue.
    supported_source_kinds = SCHOLARLY_PAPER_KINDS
    capabilities = frozenset({"title_author", "metadata", "metadata_only_search"})
    documentation_url = "https://eric.ed.gov/?api"
    # Read by the bounded review through POSITIVE_ONLY_PROVIDERS; declared here
    # so the constraint is visible where the adapter is.
    positive_only = True

    def __init__(self) -> None:
        self.policy = provider_policy(
            self.name,
            ProviderPolicy(timeout_seconds=20.0, min_interval_seconds=1.0),
        )
        self.provider_metrics: dict[str, int] = {"calls": 0, "timeouts": 0, "errors": 0}

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="ERIC is searched by title, not by DOI",
        )

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        """Search by title alone; the author is compared by the caller.

        Adding the author changed nothing measurable here and narrows recall on
        every other adapter tested in this project, so it is left out.
        """
        cleaned = _sanitize(title)[:250]
        if not cleaned:
            return RetrievalResult(
                source_name=self.name, success=False, error="No usable title terms"
            )
        self.provider_metrics["calls"] += 1
        record_provider_request("eric")
        try:
            response = httpx.get(
                ERIC_SEARCH,
                params={
                    "search": f'title:"{cleaned}"',
                    "format": "json",
                    "rows": _MAX_ROWS,
                    "fields": _FIELDS,
                },
                headers=_HEADERS,
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
                error=f"eric:{safe_exception_code(exc)}",
            )
        if response.status_code != 200:
            return RetrievalResult(
                source_name=self.name, success=False,
                error=f"ERIC HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError:
            return RetrievalResult(
                source_name=self.name, success=False, error="response_invalid"
            )
        rows = ((payload.get("response") or {}).get("docs")) or []
        if not rows:
            # Not evidence of absence: this index holds education research only.
            return RetrievalResult(
                source_name=self.name, success=False, error="No results",
                metadata={"identity_search_result_count": 0},
            )
        top = rows[0] or {}
        authors = top.get("author") or []
        if isinstance(authors, str):
            authors = [authors]
        year = top.get("publicationdateyear")
        return RetrievalResult(
            source_name=self.name,
            success=True,
            title=str(top.get("title") or "") or None,
            authors=[str(value) for value in authors][:OBSERVED_AUTHOR_LIMIT],
            year=str(year) if year else None,
            metadata={
                "identity_search_result_count": len(rows),
                "eric_id": top.get("id"),
                "container_title": top.get("source"),
                "positive_only_policy_version": "positive-only-corroboration-v1",
            },
        )
