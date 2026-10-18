"""Six library switches for every service (ARCHITECTURE §4 *Library permissions*).

Every registered adapter, the resolver's own routes and the licensed services
have search, availability, abstract, acquisition, retention and remote-model
switches. Public services start as they worked before; licensed services
start off (the publisher-PDF route only on a campus network), and licensed
text is used only with switches the deployment turned on itself. On a campus
network, text without open-access evidence counts as licensed text.
"""
import json
from unittest.mock import Mock

import httpx
import pytest
from pydantic import SecretStr

from app.config import settings
from app.services import library_permissions as lp
from app.services.retrieval import _RETRIEVER_CLASSES, get_retrieval_sources
from app.services.retrieval.base import (
    AcquisitionLocation, RepresentationKind, RetrievalResult, RetrievalSource, SourceRepresentation,
)
from app.services.retrieval.elsevier import ElsevierRetriever
from app.services.source_resolver import (
    SourceResolver, _campus_resolver_active, _search_execution_outcome,
)

LICENSED_ON = {"acquisition": True, "remote_model": True}


def _set(monkeypatch, switches=None, *, location="off_campus"):
    monkeypatch.setattr(settings, "LIBRARY_PERMISSIONS", json.dumps(switches or {}))
    monkeypatch.setattr(settings, "NETWORK_LOCATION", location)


def _pdf_result(source_name="openalex", *, oa=False):
    return RetrievalResult(
        source_name=source_name, success=True, doi="10.1234/x", title="Example",
        abstract="An abstract.", full_text=b"%PDF-1.7 text", full_text_url="https://host.example/x.pdf",
        representation=SourceRepresentation(kind=RepresentationKind.PDF, media_type="application/pdf",
                                            content=b"%PDF-1.7 text"),
        metadata={"open_access": {"is_oa": True}} if oa else {})


class _Adapter(RetrievalSource):
    name = "openalex"

    def search_by_doi(self, doi):
        return RetrievalResult(
            source_name=self.name, success=True, doi=doi, title="Example", abstract="An abstract.",
            full_text_url="https://host.example/x.pdf",
            locations=[AcquisitionLocation(url="https://host.example/x.pdf", provider="openalex", access_type="open_access")])

    def search_by_title_author(self, title, author=None):
        return self.search_by_doi("10.1234/x")


class TestRegistry:
    def test_every_registered_adapter_has_all_six_switches(self):
        table = lp.table()
        for name in _RETRIEVER_CLASSES:
            assert set(table[name]) == set(lp.PERMISSIONS)
        for name in lp.ROUTE_SERVICES + lp.LICENSED_SERVICES:
            assert set(table[name]) == set(lp.PERMISSIONS)

    @pytest.mark.parametrize("location", ["off_campus", "on_campus", "both"])
    def test_public_services_start_on_and_licensed_services_off(self, monkeypatch, location):
        _set(monkeypatch, location=location)
        table = lp.table()
        assert all(all(table[name].values()) for name in _RETRIEVER_CLASSES)
        assert all(all(table[name].values()) for name in lp.ROUTE_SERVICES)
        for name in ("doi_resolver", "elsevier_institutional"):
            assert not any(table[name].values())

    @pytest.mark.parametrize("location,on", [("off_campus", True), ("on_campus", False), ("both", False)])
    def test_the_publisher_pdf_route_starts_on_only_off_campus(self, monkeypatch, location, on):
        _set(monkeypatch, location=location)
        assert all(lp.table()["publisher_pdf"].values()) is on

    def test_nothing_is_set_by_default(self):
        assert lp.explicit("{}") == {}
        assert settings.NETWORK_LOCATION == "off_campus"

    @pytest.mark.parametrize("raw", ["not json", "[]", '{"unknown": {"search": false}}',
                                     '{"openalex": {"everything": false}}', '{"openalex": {"search": "no"}}'])
    def test_an_invalid_setting_changes_nothing(self, raw):
        assert lp.explicit(raw) == {}

    def test_one_switch_leaves_the_others(self, monkeypatch):
        _set(monkeypatch, {"openalex": {"remote_model": False}})
        assert lp.table()["openalex"] == {**{p: True for p in lp.PERMISSIONS}, "remote_model": False}
        assert all(lp.table()["crossref"].values())

    def test_the_factory_applies_the_switches(self, monkeypatch):
        _set(monkeypatch, {"crossref": {"search": False}})
        crossref = next(s for s in get_retrieval_sources() if s.name == "crossref")
        assert _search_execution_outcome(crossref.search_by_doi("10.1234/x")) == "access_restricted"


class TestAdapterSwitches:
    def test_an_adapter_with_every_switch_on_is_untouched(self, monkeypatch):
        _set(monkeypatch)
        adapter = _Adapter()
        assert "search_by_doi" not in vars(lp.restrict(adapter))

    def test_search_off_is_an_unattempted_route_never_a_miss(self, monkeypatch):
        _set(monkeypatch, {"openalex": {"search": False}})
        result = lp.restrict(_Adapter()).search_by_doi("10.1234/x")
        assert result.success is False
        assert _search_execution_outcome(result) == "access_restricted"

    def test_abstract_off_withholds_only_the_abstract(self, monkeypatch):
        _set(monkeypatch, {"openalex": {"abstract": False}})
        result = lp.restrict(_Adapter()).search_by_doi("10.1234/x")
        assert result.abstract is None and result.title == "Example" and result.locations

    @pytest.mark.parametrize("switch", ["acquisition", "availability"])
    def test_acquisition_or_availability_off_withholds_its_links(self, monkeypatch, switch):
        _set(monkeypatch, {"openalex": {switch: False}})
        result = lp.restrict(_Adapter()).search_by_doi("10.1234/x")
        assert not result.locations and result.full_text_url is None
        assert result.abstract == "An abstract."

    def test_remote_model_off_withholds_all_text(self, monkeypatch):
        """No local-only path exists, so its text would reach the model arms."""
        _set(monkeypatch, {"openalex": {"remote_model": False}})
        result = lp.apply(_pdf_result())
        assert result.full_text is None and result.abstract is None and not result.full_text_url
        assert result.title == "Example"
        assert {w["reason"] for w in result.metadata["library_permission_withheld"]} == {"remote_model_off"}

    def test_retention_off_keeps_nothing(self, monkeypatch):
        _set(monkeypatch, {"openalex": {"retention": False}})
        decided, basis = SourceResolver._retrieval_retention_decision(
            _pdf_result(oa=True), downloaded_via_publisher=False)
        assert (decided, basis) == (None, "library_retention_off")

    def test_route_services_switch_on_the_final_result(self, monkeypatch):
        _set(monkeypatch, {"student_url": {"remote_model": False}})
        resolver = SourceResolver.__new__(SourceResolver)
        resolver._resolve_routes = Mock(return_value=_pdf_result("student_url"))
        assert resolver.resolve(title="Example").full_text is None

    def test_a_copy_already_held_is_left_alone(self, monkeypatch):
        _set(monkeypatch, {"openalex": {"remote_model": False}})
        assert lp.apply(_pdf_result("local_cache")).full_text is not None


class TestOnCampus:
    @pytest.mark.parametrize("location", ["on_campus", "both"])
    def test_text_without_open_access_evidence_counts_as_licensed(self, monkeypatch, location):
        _set(monkeypatch, location=location)
        result = lp.apply(_pdf_result())
        assert result.full_text is None and result.abstract == "An abstract."
        assert result.metadata["library_permission_withheld"][0]["reason"] == "licensed_acquisition_not_permitted"

    def test_open_access_text_is_used(self, monkeypatch):
        _set(monkeypatch, location="on_campus")
        assert lp.apply(_pdf_result(oa=True)).full_text is not None

    def test_licensed_text_is_used_with_switches_the_deployment_turned_on(self, monkeypatch):
        _set(monkeypatch, {"openalex": LICENSED_ON}, location="on_campus")
        assert lp.apply(_pdf_result()).full_text is not None

    def test_publisher_route_text_follows_the_publisher_route_switches(self, monkeypatch):
        _set(monkeypatch, {"publisher_pdf": LICENSED_ON}, location="on_campus")
        result = _pdf_result()
        result.metadata["library_service"] = "publisher_pdf"
        assert lp.apply(result).full_text is not None
        _set(monkeypatch, location="on_campus")
        assert lp.apply(_pdf_result()).full_text is None

    def test_a_licensed_copy_from_the_public_doi_route_is_not_kept_by_default(self, monkeypatch):
        _set(monkeypatch, location="on_campus")
        from app.services.storage_policy import StoragePolicy
        policy = StoragePolicy.model_validate({"sources": {"keep_licensed_copies": True}})
        result = _pdf_result("doi_resolver")
        decided, _ = SourceResolver._retrieval_retention_decision(
            result, downloaded_via_publisher=True, policy=policy)
        assert decided is None
        _set(monkeypatch, {"public_doi": {"retention": True}}, location="on_campus")
        decided, _ = SourceResolver._retrieval_retention_decision(
            result, downloaded_via_publisher=True, policy=policy)
        assert decided == "paywalled_db_retrieved"

    def test_off_campus_nothing_changes(self, monkeypatch):
        _set(monkeypatch)
        assert lp.apply(_pdf_result()).full_text is not None


class TestLicensedServices:
    def _publisher_attempt(self, monkeypatch, switches, location="on_campus"):
        _set(monkeypatch, switches, location=location)
        download = Mock(return_value=None)
        monkeypatch.setattr("app.services.publisher_urls.try_download_publisher_pdf", download)
        resolver = SourceResolver.__new__(SourceResolver)
        resolver._backend = None
        resolver._retrieval_sources = []
        resolver._acquisition_capabilities = None
        result = resolver._download_and_cache(_Adapter(), RetrievalResult(
            source_name="metadata", success=True, doi="10.1234/example", title="Example"))
        return download, result

    def test_publisher_pdf_works_as_before_off_campus(self, monkeypatch):
        download, result = self._publisher_attempt(monkeypatch, {}, location="off_campus")
        download.assert_called_once()

    def test_publisher_pdf_switched_off_off_campus_is_honoured(self, monkeypatch):
        download, _ = self._publisher_attempt(
            monkeypatch, {"publisher_pdf": {"acquisition": False}}, location="off_campus")
        download.assert_not_called()

    def test_publisher_pdf_not_attempted_on_campus_by_default(self, monkeypatch):
        download, result = self._publisher_attempt(monkeypatch, {})
        download.assert_not_called()
        assert result.metadata["publisher_pdf_not_attempted"] == "licensed_acquisition_not_permitted"

    def test_publisher_pdf_attempted_when_turned_on(self, monkeypatch):
        download, _ = self._publisher_attempt(monkeypatch, {"publisher_pdf": LICENSED_ON})
        download.assert_called_once()

    def test_acquisition_alone_is_refused_until_local_processing_exists(self, monkeypatch):
        download, result = self._publisher_attempt(monkeypatch, {"publisher_pdf": {"acquisition": True}})
        download.assert_not_called()
        assert result.metadata["publisher_pdf_not_attempted"] == "licensed_text_local_processing_not_built"

    def test_another_services_switch_does_not_enable_it(self, monkeypatch):
        download, _ = self._publisher_attempt(monkeypatch, {"doi_resolver": LICENSED_ON})
        download.assert_not_called()

    @pytest.mark.parametrize("location", ["off_campus", "on_campus", "both"])
    def test_a_configured_campus_resolver_stays_off_until_turned_on(self, monkeypatch, location):
        monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
        _set(monkeypatch, location=location)
        assert not _campus_resolver_active()
        _set(monkeypatch, {"doi_resolver": LICENSED_ON}, location=location)
        assert _campus_resolver_active()

    def test_an_unpermitted_campus_resolver_leaves_the_public_doi_route(self, monkeypatch):
        """A required route that never ran would make every DOI search incomplete."""
        monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
        _set(monkeypatch)
        resolver = SourceResolver.__new__(SourceResolver)
        resolver._backend = None
        resolver._retrieval_sources = []
        resolver._acquisition_capabilities = None
        resolver._lookup_cache = None
        resolver._check_local_cache = Mock(
            return_value=RetrievalResult(source_name="local_cache", success=False))
        acquired = RetrievalResult(source_name="doi_resolver", success=True, full_text=b"%PDF-x")
        resolver._try_doi_resolver = Mock(return_value=acquired)
        resolver._finalize_resolution_result = lambda result, *args: result
        assert resolver.resolve(doi="10.1234/example", title="Example") is acquired
        assert resolver._try_doi_resolver.call_args.kwargs.get("public") is True

    def test_public_doi_search_off_skips_the_route(self, monkeypatch):
        monkeypatch.setattr(settings, "DOI_RESOLVER_URL", None)
        _set(monkeypatch, {"public_doi": {"search": False}})
        resolver = SourceResolver.__new__(SourceResolver)
        resolver._backend = None
        resolver._retrieval_sources = []
        resolver._acquisition_capabilities = None
        resolver._lookup_cache = None
        resolver._check_local_cache = Mock(
            return_value=RetrievalResult(source_name="local_cache", success=False))
        resolver._try_doi_resolver = Mock()
        with pytest.raises(Exception):
            resolver.resolve(doi="10.1234/example", title="Example")
        resolver._try_doi_resolver.assert_not_called()


def _elsevier(monkeypatch, *, is_oa, switches):
    monkeypatch.setattr(settings, "ELSEVIER_API_KEY", SecretStr("fixture-key"))
    _set(monkeypatch, switches)
    request = httpx.Request("GET", "https://api.elsevier.com/content/article/doi/10.1016/j.x")
    body = {"full-text-retrieval-response": {"coredata": {
        "dc:title": "Example", "openaccessArticle": is_oa}}}
    monkeypatch.setattr("app.services.retrieval.elsevier.httpx.get",
                        Mock(return_value=httpx.Response(200, request=request, json=body)))
    adapter = ElsevierRetriever(health_store=Mock())
    adapter._fetch_full_text = Mock(return_value="Article text")
    return adapter, adapter.search_by_doi("10.1016/j.x")


class TestElsevier:
    def test_paywalled_text_is_not_requested_by_default(self, monkeypatch):
        adapter, result = _elsevier(monkeypatch, is_oa=False, switches={})
        adapter._fetch_full_text.assert_not_called()
        assert result.metadata["full_text_not_requested"] == "licensed_acquisition_not_permitted"

    def test_open_access_text_needs_no_licensed_switch(self, monkeypatch):
        adapter, result = _elsevier(monkeypatch, is_oa=True, switches={})
        adapter._fetch_full_text.assert_called_once()
        assert result.representation is not None

    def test_paywalled_text_is_requested_when_turned_on(self, monkeypatch):
        adapter, _ = _elsevier(monkeypatch, is_oa=False, switches={"elsevier_institutional": LICENSED_ON})
        adapter._fetch_full_text.assert_called_once()
