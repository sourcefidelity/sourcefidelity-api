"""Europe PMC — open-access full text held in a repository (owner decision 2026-10-02).

Publishers increasingly refuse automated requests (Heliyon on cell.com and MDPI
return HTTP 403 to the application), while the same open-access articles are
deposited in PubMed Central and served by Europe PMC as JATS XML. This adapter
finds a work by DOI or title and offers that XML as an open-access location;
the resolver's existing location acquisition, identity and completeness checks
decide whether it is admitted.

Positive-only: Europe PMC holds the life sciences and a minority of other
fields, so its silence never counts towards a search being complete.
Free, no key. REST API: https://europepmc.org/RestfulWebService
"""

import logging
import re

import httpx

from app.log_safety import safe_exception_code
from app.services.processing_metrics import record_provider_request
from app.services.retrieval.base import (
    OBSERVED_AUTHOR_LIMIT,
    SCHOLARLY_PAPER_KINDS,
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    RetrievalSource,
)
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy

logger = logging.getLogger(__name__)

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULL_TEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
ARTICLE_URL = "https://europepmc.org/articles/{pmcid}"
_HEADERS = {"User-Agent": "SourceFidelity/0.1 (academic source verification; contact via repository)"}
# Query syntax characters that would end the quoted field early.
_QUERY_SYNTAX = re.compile(r'["\\:()\[\]{}^~*?]')
_PMCID = re.compile(r"^PMC\d{1,10}$")


def _sanitize(value: str) -> str:
    return " ".join(_QUERY_SYNTAX.sub(" ", str(value or "")).split())


class EuropePmcRetriever(RetrievalSource):
    """Find a work's open-access repository copy; never testify that one is absent."""

    required_for_search_completion = False
    positive_only = True
    name = "europepmc"
    supported_source_kinds = SCHOLARLY_PAPER_KINDS
    capabilities = frozenset({"doi", "title_author", "metadata", "full_text", "open_access"})
    documentation_url = "https://europepmc.org/RestfulWebService"

    def __init__(self) -> None:
        self.policy = provider_policy(self.name, ProviderPolicy(timeout_seconds=20.0, min_interval_seconds=0.2))
        self.provider_metrics: dict[str, int] = {"calls": 0, "timeouts": 0, "errors": 0}

    def search_by_doi(self, doi: str) -> RetrievalResult:
        cleaned = _sanitize(re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(doi or "").strip(), flags=re.I))
        if not cleaned:
            return RetrievalResult(source_name=self.name, success=False, error="No usable DOI")
        return self._search(f'DOI:"{cleaned}"')

    def search_by_title_author(self, title: str, author: str | None = None) -> RetrievalResult:
        cleaned = _sanitize(title)[:250]
        if not cleaned:
            return RetrievalResult(source_name=self.name, success=False, error="No usable title terms")
        return self._search(f'TITLE:"{cleaned}"')

    def _search(self, query: str) -> RetrievalResult:
        if not self.policy.enabled:
            return RetrievalResult(source_name=self.name, success=False, error="Provider disabled")
        self.provider_metrics["calls"] += 1
        record_provider_request(self.name)
        try:
            response = httpx.get(SEARCH_URL, params={"query": query, "format": "json", "resultType": "core",
                                                     "pageSize": 3}, headers=_HEADERS,
                                 timeout=self.policy.timeout_seconds)
        except httpx.TimeoutException:
            self.provider_metrics["timeouts"] += 1
            return RetrievalResult(source_name=self.name, success=False, error="read_timeout")
        except httpx.HTTPError as exc:
            self.provider_metrics["errors"] += 1
            return RetrievalResult(source_name=self.name, success=False,
                                   error=f"europepmc:{safe_exception_code(exc)}")
        if response.status_code != 200:
            return RetrievalResult(source_name=self.name, success=False,
                                   error=f"Europe PMC HTTP {response.status_code}")
        try:
            rows = ((response.json().get("resultList") or {}).get("result")) or []
        except ValueError:
            return RetrievalResult(source_name=self.name, success=False, error="response_invalid")
        if not rows:
            # Not evidence of absence: this index holds mainly the life sciences.
            return RetrievalResult(source_name=self.name, success=False, error="No results",
                                   metadata={"identity_search_result_count": 0})
        return self._result(rows[0] or {}, len(rows))

    def _result(self, row: dict, count: int) -> RetrievalResult:
        authors = [str(a.get("fullName") or a.get("lastName") or "")
                   for a in ((row.get("authorList") or {}).get("author") or []) if isinstance(a, dict)]
        pmcid = str(row.get("pmcid") or "")
        open_text = (row.get("isOpenAccess") == "Y" and row.get("inEPMC") == "Y" and bool(_PMCID.match(pmcid)))
        locations = []
        if open_text:
            locations.append(AcquisitionLocation(
                url=FULL_TEXT_URL.format(pmcid=pmcid), provider=self.name, media_type="application/xml",
                representation_kind=RepresentationKind.XML, landing_page_url=ARTICLE_URL.format(pmcid=pmcid),
                host_type="repository", access_type="open_access", is_best=True,
                metadata={"pmcid": pmcid, "route": "europepmc_full_text_xml"}))
        journal = ((row.get("journalInfo") or {}).get("journal") or {}).get("title")
        return RetrievalResult(
            source_name=self.name,
            success=True,
            title=str(row.get("title") or "").rstrip(".") or None,
            authors=[a for a in authors if a][:OBSERVED_AUTHOR_LIMIT],
            year=str(row.get("pubYear") or "") or None,
            doi=str(row.get("doi") or "") or None,
            locations=locations,
            full_text_url=ARTICLE_URL.format(pmcid=pmcid) if open_text else None,
            metadata={
                "identity_search_result_count": count,
                "container_title": journal,
                "pmcid": pmcid or None,
                "is_open_access": open_text,
                "license_class": "open_access" if open_text else None,
                "positive_only_policy_version": "positive-only-corroboration-v1",
            },
        )
