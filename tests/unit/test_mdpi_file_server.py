"""MDPI's own file server as a fallback when its site blocks automated requests.

Owner approved 2026-09-29. A blocked MDPI location is followed by one plain
request to mdpi-res.com built from the DOI; the file passes the ordinary gates.
Network doubles only.
"""
from unittest.mock import Mock

import httpx
import pytest

from app.config import settings
from app.services.publisher_urls import is_mdpi_location, mdpi_file_server_urls
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalResult
from app.services.source_resolver import SourceResolutionError, SourceResolver

DOI = "10.3390/literature5020007"
FILE = ("https://mdpi-res.com/d_attachment/literature/literature-05-00007/"
        "article_deploy/literature-05-00007.pdf")


@pytest.mark.parametrize("doi,expected", [
    (DOI, [FILE]),
    ("https://doi.org/10.3390/LITERATURE5020007", [FILE]),
    ("10.3390/ijerph17010001",
     ["https://mdpi-res.com/d_attachment/ijerph/ijerph-17-00001/article_deploy/ijerph-17-00001.pdf"]),
    # Five-digit article numbers, once a volume passes article 9999.
    ("10.3390/ijms241512345",
     ["https://mdpi-res.com/d_attachment/ijms/ijms-24-12345/article_deploy/ijms-24-12345.pdf"]),
    # The DOI code is used as the slug even where MDPI's slug differs
    # (Sustainability); that address answers 404 and is unavailable.
    ("10.3390/su12010001",
     ["https://mdpi-res.com/d_attachment/su/su-12-00001/article_deploy/su-12-00001.pdf"]),
    ("10.3390/books978-3-03897", []),
    ("10.3390/proceedings", []),
    ("10.1080/10509208.2019.1660132", []),
    (None, []),
])
def test_file_server_address_is_built_from_the_doi(doi, expected):
    assert mdpi_file_server_urls(doi) == expected


def test_mdpi_locations_are_recognized():
    assert is_mdpi_location("https://www.mdpi.com/2410-9789/5/2/7/pdf")
    assert is_mdpi_location("https://doi.org/10.3390/literature5020007")
    assert not is_mdpi_location("https://doi.org/10.1080/x")
    assert not is_mdpi_location("https://mdpi.example.org/x")


def _status_error(status):
    request = httpx.Request("GET", "https://www.mdpi.com/x")
    return httpx.HTTPStatusError("blocked", request=request, response=httpx.Response(status, request=request))


def _resolver(downloads, headers_seen):
    resolver = SourceResolver.__new__(SourceResolver)

    def download(url, *, timeout=None, headers=None):
        headers_seen[url] = headers
        outcome = downloads[url]
        if isinstance(outcome, Exception):
            # _safe_download wraps HTTP status errors like this.
            raise ValueError(f"Download failed (HTTPStatusError status={outcome.response.status_code})") from outcome
        return outcome

    resolver._safe_download = download
    resolver._preflight_acquired_representation = Mock(return_value=(True, "acquired", "identity ok"))
    return resolver


def _acquire(resolver, url, doi=DOI):
    result = RetrievalResult(source_name="openalex", success=True, doi=doi, locations=[
        AcquisitionLocation(url=url, provider="openalex", representation_kind=RepresentationKind.PDF)])
    accepted = resolver._acquire_from_locations(result, expected_doi=doi, expected_title="A title")
    return accepted, result


def test_a_blocked_mdpi_location_is_followed_by_the_file_server(monkeypatch):
    headers = {}
    site = "https://www.mdpi.com/2410-9789/5/2/7/pdf"
    resolver = _resolver({site: _status_error(403), FILE: b"%PDF-mdpi"}, headers)
    accepted, result = _acquire(resolver, site)
    assert accepted and result.full_text_url == FILE
    outcomes = [(a["url"], a["outcome"]) for a in result.metadata["location_attempts"]]
    assert outcomes[0][0] == site and outcomes[1] == (FILE, "acquired")
    # The ordinary gates ran on the downloaded file.
    resolver._preflight_acquired_representation.assert_called_once()
    # A plain request that names the application: no browser user agent.
    assert headers[site] is None
    assert "Mozilla" not in headers[FILE]["User-Agent"]
    assert headers[FILE]["User-Agent"].startswith("SourceFidelity/")


def test_a_missing_file_server_copy_is_simply_unavailable():
    headers = {}
    site = "https://www.mdpi.com/2410-9789/5/2/7/pdf"
    resolver = _resolver({site: _status_error(403), FILE: _status_error(404)}, headers)
    accepted, result = _acquire(resolver, site)
    assert not accepted
    assert [a["url"] for a in result.metadata["location_attempts"]] == [site, FILE]
    resolver._preflight_acquired_representation.assert_not_called()


@pytest.mark.parametrize("url,status,doi", [
    ("https://www.mdpi.com/2410-9789/5/2/7/pdf", 500, DOI),           # not a block
    ("https://publisher.example/article.pdf", 403, DOI),               # not an MDPI location
    ("https://www.mdpi.com/2410-9789/5/2/7/pdf", 403, "10.1000/other"),  # no MDPI DOI
])
def test_no_fallback_outside_a_blocked_mdpi_location(url, status, doi):
    headers = {}
    resolver = _resolver({url: _status_error(status)}, headers)
    accepted, result = _acquire(resolver, url, doi=doi)
    assert not accepted
    assert [a["url"] for a in result.metadata["location_attempts"]] == [url]


def test_a_doi_link_to_an_mdpi_work_supplies_the_doi():
    headers = {}
    link = "https://doi.org/10.3390/literature5020007"
    resolver = _resolver({link: _status_error(403), FILE: b"%PDF-mdpi"}, headers)
    accepted, _ = _acquire(resolver, link, doi=None)
    assert accepted


def test_the_public_doi_route_falls_back_when_mdpi_blocks_it(monkeypatch):
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", None)
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = []
    resolver._acquisition_capabilities = None
    resolver._lookup_cache = None
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name="local_cache", success=False))
    resolver._try_doi_resolver = Mock(return_value=RetrievalResult(
        source_name="doi_resolver", success=False, error="DOI resolver failed (HTTPStatusError)",
        metadata={"http_status": 403}))
    acquired = RetrievalResult(source_name="mdpi_file_server", success=True, full_text=b"%PDF-mdpi")
    resolver._try_mdpi_file_server = Mock(return_value=acquired)
    resolver._finalize_resolution_result = lambda result, *args: result
    assert resolver.resolve(doi=DOI, title="A title") is acquired
    resolver._try_mdpi_file_server.assert_called_once()

    resolver._try_doi_resolver.return_value = RetrievalResult(
        source_name="doi_resolver", success=False, error="DOI resolver failed (ConnectError)", metadata={})
    resolver._try_mdpi_file_server.reset_mock()
    with pytest.raises(SourceResolutionError):
        resolver.resolve(doi=DOI, title="A title")
    resolver._try_mdpi_file_server.assert_not_called()


def test_the_file_server_route_uses_the_ordinary_acquisition_path():
    resolver = SourceResolver.__new__(SourceResolver)
    seen = {}

    def download_and_cache(source, result, doi, title, author, year, **kwargs):
        seen["urls"] = [location.url for location in result.locations]
        seen["fields"] = (result.title, result.authors, result.full_text_url)
        return result

    resolver._download_and_cache = download_and_cache
    outcome = resolver._try_mdpi_file_server(DOI, "A title", "Writer", "2020", Mock())
    assert seen["urls"] == [FILE]
    # No bibliographic fields of its own: identity rests on the file content,
    # and no fallback URL invites a second, browser-headed fetch.
    assert seen["fields"] == (None, [], None)
    assert outcome.success is False
