"""Europe PMC open-access repository copies (owner decision 2026-10-02)."""
from types import SimpleNamespace

from app.services.retrieval import europe_pmc
from app.services.retrieval.base import RepresentationKind
from app.services.source_resolver import _extract_text_representation

ROW = {"title": "A study of narrative forms.", "pubYear": "2020", "doi": "10.1/x", "pmcid": "PMC123",
       "isOpenAccess": "Y", "inEPMC": "Y", "authorList": {"author": [{"fullName": "Lopez I"}]},
       "journalInfo": {"journal": {"title": "Example Journal"}}}


def fake_get(rows):
    return lambda url, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"resultList": {"result": rows}})


def test_an_open_access_record_offers_its_full_text_xml(monkeypatch):
    monkeypatch.setattr(europe_pmc.httpx, "get", fake_get([ROW]))
    result = europe_pmc.EuropePmcRetriever().search_by_doi("https://doi.org/10.1/x")
    assert result.success and result.title == "A study of narrative forms" and result.year == "2020"
    [location] = result.locations
    assert location.url.endswith("/PMC123/fullTextXML") and location.representation_kind is RepresentationKind.XML
    assert location.access_type == "open_access"


def test_a_closed_record_is_identity_only_and_silence_is_not_absence(monkeypatch):
    monkeypatch.setattr(europe_pmc.httpx, "get", fake_get([dict(ROW, isOpenAccess="N")]))
    assert europe_pmc.EuropePmcRetriever().search_by_title_author("A study").locations == []
    monkeypatch.setattr(europe_pmc.httpx, "get", fake_get([]))
    assert not europe_pmc.EuropePmcRetriever().search_by_title_author("A study").success
    assert europe_pmc.EuropePmcRetriever.blocks_search_completion() is False


def test_jats_text_keeps_paragraphs_and_drops_the_reference_list():
    xml = ("<article><front><article-meta><title-group><article-title>A study</article-title></title-group>"
           "<abstract><p>The abstract says one thing about narrative forms in digital culture today.</p></abstract>"
           "</article-meta></front><body><sec><title>Introduction</title>"
           + "".join(f"<p>Paragraph {n} discusses narrative forms and creative processes at some length here.</p>"
                     for n in range(12))
           + "</sec></body><back><ref-list><ref>Cited Work, 1999.</ref></ref-list></back></article>").encode()
    text = _extract_text_representation(xml, RepresentationKind.XML)
    assert text.startswith("A study\n\nThe abstract says") and "Paragraph 11" in text
    assert "Cited Work" not in text
