"""A page that is only an abstract, or too short to be the work, yields an abstract
(landing-abstract-back-matter-v1 and doi-abstract-over-short-page-v1, owner
decisions 2026-10-13). Invented fixtures only."""
from app.services.discovery_recorder import Finished
import hashlib
from unittest.mock import Mock, patch

import httpx
import pytest

from app.config import settings
from app.services.abstract_only_pages import (
    DOI_ABSTRACT_POLICY, LANDING_POLICY, landing_abstract_and_back_matter, short_page_words)
from app.services.retrieval.base import (
    AcquisitionLocation, RepresentationKind, RetrievalResult, RetrievalSource, SourceRepresentation)
from app.services.source_resolver import SourceResolver, _CONFIRMED_DISCOVERY_OUTCOMES
from app.services.source_type import SourceKindAssessment

TITLE = "Tariff Shadows over Coastal Trade Networks"
DOI = "10.5555/synthetic-tariff-shadows"
URL = "https://journals.example/article/10.5555/synthetic-tariff-shadows"
ARTICLE = SourceKindAssessment("journal_article", "high", ("reference structure",))

ABSTRACT = (
    "This article examines how temporary tariff measures change the routes that coastal "
    "shipping networks take between small ports. Using an invented panel of harbour records, "
    "we find that measures announced as temporary shift cargo towards larger hubs and that the "
    "shift persists after the measures lapse. The effect is strongest for perishable goods and "
    "for ports without rail links, which suggests that network position conditions exposure."
)
NOTES = "\n".join(
    f"{n}. The harbour ledger for this period records the cargo by weight and by declared value, "
    f"and the figures used here follow the second of these conventions throughout the sample."
    for n in range(1, 31))
REFERENCES = "\n".join(
    f"Author{n} A, Writer{n} B (20{n % 20:02d}) Ports and policies in a changing sea. Journal of Invented Trade {n}:1-20"
    for n in range(1, 40))

BACK_MATTER_ONLY = f"""Tariff Shadows over Coastal Trade Networks
Original Paper
Published: 03 May 2019
Abstract
{ABSTRACT}
Access this article
Log in via an institution
Buy article PDF 39,95 EUR
Price includes VAT
1 Unit = 1 Article or 1 Chapter
Instant access to the full article PDF.
Similar content being viewed by others
Harbours and hinterlands in the long run
Shipping costs and regional growth
Notes
{NOTES}
References
{REFERENCES}
Acknowledgements
We thank the invented harbour archive for access to its ledgers.
Author information
Authors and Affiliations
Institute of Invented Studies, Exampletown
About this article
Issue date: November 2019
DOI: https://doi.org/{DOI}
"""

BODY = "\n".join(
    "Coastal trade networks connect small ports through a few large hubs, and the routes that "
    "cargo takes depend on relative costs, on the regularity of sailings and on the policies "
    "that govern each crossing, so a change in any one of them alters the whole pattern."
    for _ in range(12))


def _landing_html(text: str) -> str:
    paragraphs = "".join(
        f"<h2>{line}</h2>" if len(line.split()) <= 6 and not line.endswith(".") else f"<p>{line}</p>"
        for line in text.splitlines() if line.strip())
    return (f'<html><head><title>{TITLE} | Review of Invented Economics</title>'
            f'<meta name="citation_title" content="{TITLE}">'
            f'<meta name="citation_doi" content="{DOI}">'
            '<meta name="citation_author" content="Harbour, Ada">'
            '<meta name="citation_publication_date" content="2019/05/03">'
            f'</head><body><article>{paragraphs}</article></body></html>')


class TestLandingDetector:
    def test_a_back_matter_only_landing_page_yields_its_abstract(self):
        found = landing_abstract_and_back_matter(BACK_MATTER_ONLY)
        assert found is not None and found["policy"] == LANDING_POLICY
        assert found["abstract"] == ABSTRACT
        assert {"notes", "references", "acknowledgements", "access this article"} <= set(
            found["back_matter_sections"])
        assert found["abstract_sha256"] == hashlib.sha256(ABSTRACT.encode()).hexdigest()

    @pytest.mark.parametrize("heading", ["Introduction", "1 Introduction", "2. Data and sources", "Methods"])
    def test_a_landing_page_with_a_body_section_is_unchanged(self, heading):
        text = BACK_MATTER_ONLY.replace("Notes\n", f"{heading}\n{BODY}\nNotes\n")
        assert landing_abstract_and_back_matter(text) is None

    def test_a_substantial_paragraph_outside_the_back_matter_keeps_the_page(self):
        text = BACK_MATTER_ONLY.replace("Access this article\n", f"Shipping after the measures\n{BODY}\nAccess this article\n")
        assert landing_abstract_and_back_matter(text) is None

    def test_an_open_page_without_an_access_prompt_is_unchanged(self):
        text = "\n".join(line for line in BACK_MATTER_ONLY.splitlines()
                         if not any(cue in line for cue in ("Access this", "Log in", "Buy article", "Price", "Instant")))
        assert landing_abstract_and_back_matter(text) is None

    def test_no_abstract_heading_is_unchanged(self):
        assert landing_abstract_and_back_matter(BACK_MATTER_ONLY.replace("Abstract\n", "")) is None

    def test_an_abstract_with_only_an_access_box_is_an_abstract(self):
        head = BACK_MATTER_ONLY.split("Similar content")[0]
        assert landing_abstract_and_back_matter(head)["abstract"] == ABSTRACT

    @pytest.mark.parametrize("after", ["Similar content being viewed by others\n", "Acknowledgements\n"])
    def test_a_body_under_an_unlisted_heading_after_short_back_matter_is_unchanged(self, after):
        text = BACK_MATTER_ONLY.replace(after, f"{after}Shipping after the measures\n{BODY}\n", 1)
        assert landing_abstract_and_back_matter(text) is None


def _acquire(text, *, doi=DOI, abstract=None):
    html = _landing_html(text)
    response = httpx.Response(200, text=html, headers={"content-type": "text/html"},
                              request=httpx.Request("GET", URL))
    resolver = SourceResolver.__new__(SourceResolver)
    result = RetrievalResult(
        source_name="canonical_work_graph", success=True, title=TITLE, doi=doi, abstract=abstract,
        locations=[AcquisitionLocation(url=URL, provider="openalex", representation_kind=RepresentationKind.HTML,
                                       metadata={"discovery_signal": "citation_fulltext_html_url"})],
        metadata={})
    with patch("app.services.source_resolver.safe_request", return_value=response), \
            patch.object(SourceResolver, "_preflight_acquired_representation",
                         return_value=(True, "acquired", "")):
        accepted = resolver._acquire_from_locations(
            result, expected_title=TITLE, expected_doi=doi, expected_author="Harbour, A",
            expected_year="2019", expected_source_kind=ARTICLE)
    return accepted, result


class TestLandingPageAcquisition:
    def test_the_landing_page_is_not_limited_text_and_its_abstract_is_kept(self):
        accepted, result = _acquire(BACK_MATTER_ONLY)
        assert not accepted and result.representation is None and not result.full_text
        assert result.abstract == ABSTRACT
        provenance = result.metadata["abstract_provenance"]
        assert provenance["policy"] == LANDING_POLICY and provenance["basis"] == "publisher_landing_page"
        assert provenance["provider"] == "openalex" and provenance["doi"] == DOI
        assert "source_url" not in provenance and "source_url_sha256" not in provenance
        attempt = result.metadata["location_attempts"][-1]
        assert attempt["reason_code"] == "landing_abstract_and_back_matter_only"
        assert attempt["landing_abstract_only"]["page_identity_confirmed"] is True
        assert "abstract" not in attempt["landing_abstract_only"]

    def test_an_adapter_abstract_already_held_is_kept(self):
        accepted, result = _acquire(BACK_MATTER_ONLY, abstract="An adapter abstract of the work.")
        assert not accepted and result.representation is None
        assert result.abstract == "An adapter abstract of the work."
        assert "abstract_provenance" not in result.metadata

    def test_a_landing_page_with_an_introduction_and_body_is_unchanged(self):
        text = BACK_MATTER_ONLY.replace("Notes\n", f"Introduction\n{BODY}\nNotes\n")
        accepted, result = _acquire(text)
        assert accepted and result.representation is not None
        assert "Coastal trade networks connect" in result.representation.content.decode()
        assert "abstract_provenance" not in result.metadata


def _page(words, *, completeness="not_assessed", kind=RepresentationKind.PLAIN_TEXT,
          original=RepresentationKind.HTML):
    text = " ".join(["harbour"] * words)
    return RetrievalResult(source_name="web_search", success=True, representation=SourceRepresentation(
        kind=kind, media_type="text/plain", content=text.encode(), original_kind=original,
        completeness=completeness))


def _holder(*, confidence="high", doi=DOI, provider="openalex"):
    return RetrievalResult(
        source_name="canonical_work_graph", success=True, abstract=ABSTRACT, doi=DOI, title=TITLE,
        metadata={"canonical_work": {
            "abstract_provider": provider, "abstract_providers": [provider],
            "identity_evidence": [
                {"provider": "crossref", "confidence": confidence, "doi": doi.upper()},
                {"provider": "openalex", "confidence": confidence, "doi": f"https://doi.org/{doi}"}]}})


class TestShortPage:
    def test_only_an_incomplete_web_page_below_a_whole_work_is_short(self):
        assert short_page_words(_page(595), "journal_article") == 595
        assert short_page_words(_page(595), None) == 595
        assert short_page_words(_page(2500), "journal_article") is None       # substantial limited text
        assert short_page_words(_page(595, completeness="complete"), "journal_article") is None
        # A web page of the cited kind can be whole at 595 words.
        assert short_page_words(_page(595), "webpage") is None
        # A PDF (a several-page preview) keeps its own rule.
        pdf = _page(595, kind=RepresentationKind.PDF, original=None)
        assert short_page_words(pdf, "journal_article") is None


class TestDoiAbstractPreference:
    def _resolver(self):
        resolver = SourceResolver.__new__(SourceResolver)
        resolver._discovery_finished = lambda preview=False: Finished(None, {"outcome": next(iter(_CONFIRMED_DISCOVERY_OUTCOMES))}, None)
        return resolver

    def test_a_short_page_yields_to_the_confirmed_doi_abstract(self):
        page = _page(595)
        page.metadata = {"durable_admission": {"state": "accepted", "representation_id": "r1"}}
        preferred = self._resolver()._abstract_instead_of_short_page(page, _holder(), DOI, ARTICLE)
        assert preferred is not None and preferred.representation is None and not preferred.full_text
        assert preferred.abstract == ABSTRACT and preferred.success
        provenance = preferred.metadata["abstract_provenance"]
        assert provenance["policy"] == DOI_ABSTRACT_POLICY and provenance["provider"] == "openalex"
        assert provenance["doi"] == DOI and provenance["doi_confirmed_by"] == ["crossref", "openalex"]
        assert provenance["displaced_page"]["words"] == 595
        assert provenance["displaced_page"]["durable_admission_state"] == "accepted"
        assert "durable_admission" not in preferred.metadata

    def test_the_acquired_graph_page_itself_yields_and_loses_its_admission(self):
        holder = _holder()
        holder.set_representation(_page(595).representation)
        holder.metadata["durable_admission"] = {"state": "accepted", "representation_id": "r1"}
        preferred = self._resolver()._abstract_instead_of_short_page(holder, holder, DOI, ARTICLE)
        assert preferred.representation is None and "durable_admission" not in preferred.metadata
        assert holder.representation is not None                       # the original is untouched

    @pytest.mark.parametrize("holder,doi", [
        (_holder(confidence="medium"), DOI),                             # DOI not confirmed
        (_holder(provider="semantic_scholar"), DOI),                     # abstract from an unconfirmed record
        (_holder(), "10.5555/another-work"),                             # a different DOI
        (_holder(), None),
        (None, DOI),
    ])
    def test_without_a_confirmed_doi_abstract_the_page_stands(self, holder, doi):
        assert self._resolver()._abstract_instead_of_short_page(_page(595), holder, doi, ARTICLE) is None

    def test_full_or_substantial_text_still_wins(self):
        resolver = self._resolver()
        assert resolver._abstract_instead_of_short_page(_page(2500), _holder(), DOI, ARTICLE) is None
        assert resolver._abstract_instead_of_short_page(
            _page(595, completeness="complete"), _holder(), DOI, ARTICLE) is None

    def test_an_unconfirmed_reference_keeps_its_page(self):
        resolver = SourceResolver.__new__(SourceResolver)
        resolver._discovery_finished = lambda preview=False: Finished(None, {"outcome": "bibliographic_conflict"}, None)
        assert resolver._abstract_instead_of_short_page(_page(595), _holder(), DOI, ARTICLE) is None


class _Index(RetrievalSource):
    capabilities = frozenset({"doi", "metadata", "abstract", "locations"})

    def __init__(self, name, abstract=None):
        self.name, self._abstract, self.calls = name, abstract, 0

    def search_by_doi(self, doi):
        self.calls += 1
        return RetrievalResult(source_name=self.name, success=True, doi=DOI, title=TITLE,
                               authors=["Harbour, Ada"], year="2019", abstract=self._abstract,
                               metadata={"type": "journal-article"})

    def search_by_title_author(self, title, author=None):
        return RetrievalResult(source_name=self.name, success=False)


class _Web(RetrievalSource):
    name = "web_search"
    capabilities = frozenset({"web_discovery"})

    def search_by_doi(self, doi):
        return RetrievalResult(source_name=self.name, success=False)

    def search_by_title_author(self, title, author=None, year=None):
        return RetrievalResult(source_name=self.name, success=False)


def _resolve(monkeypatch, sources, short_page, *, settled=True):
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", None)
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = sources
    resolver._acquisition_capabilities = frozenset({"student_url"})   # no public DOI route
    resolver._lookup_cache = None
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name="local_cache", success=False))
    resolver._download_and_cache = lambda source, result, *a, **k: result
    resolver._record_canonical_location_outcomes = lambda result: None
    resolver._reuse_search_memo = lambda: False
    resolver._identity_settled = lambda graph: settled
    resolver._discovery_finished = lambda preview=False: Finished(None, {"outcome": "confirmed"}, None)
    resolver._try_source = Mock(return_value=short_page)
    resolver._finalize_resolution_result = lambda result, *args: result
    return resolver.resolve(doi=DOI, title=TITLE, author="Harbour, A", year="2019",
                            source_kind="journal_article", source_kind_confidence="high")


def test_resolution_prefers_the_doi_abstract_over_a_short_search_page(monkeypatch):
    # The indexes confirm the DOI and hold its abstract; search
    # finds a short third-party page that states the work.
    resolved = _resolve(monkeypatch, [_Index("crossref", ABSTRACT), _Index("openalex"), _Web()], _page(595))
    assert resolved.representation is None and resolved.abstract == ABSTRACT
    provenance = resolved.metadata["abstract_provenance"]
    assert provenance["provider"] == "crossref" and provenance["doi"] == DOI


def test_resolution_keeps_a_substantial_search_page(monkeypatch):
    page = _page(2500)
    assert _resolve(monkeypatch, [_Index("crossref", ABSTRACT), _Web()], page) is page


def test_settled_identity_still_asks_for_an_abstract_when_none_is_held(monkeypatch):
    scholar = _Index("semantic_scholar", ABSTRACT)
    core = _Index("core")
    resolved = _resolve(monkeypatch, [_Index("crossref"), scholar, core, _Web()], _page(595))
    assert scholar.calls == 1 and core.calls == 0
    assert resolved.abstract == ABSTRACT
    assert resolved.metadata["abstract_provenance"]["provider"] == "semantic_scholar"


def test_settled_identity_with_an_abstract_skips_the_corroborating_index(monkeypatch):
    scholar = _Index("semantic_scholar", "Another abstract.")
    _resolve(monkeypatch, [_Index("crossref", ABSTRACT), scholar, _Web()], _page(595))
    assert scholar.calls == 0


@pytest.mark.parametrize("body,abstract_only", [(False, True), (True, False)])
def test_the_public_doi_page_with_only_an_abstract_keeps_only_its_abstract(body, abstract_only):
    text = BACK_MATTER_ONLY.replace("Notes\n", f"Introduction\n{BODY}\nNotes\n") if body else BACK_MATTER_ONLY
    response = httpx.Response(200, text=_landing_html(text), headers={"content-type": "text/html"},
                              request=httpx.Request("GET", URL))
    resolver = SourceResolver.__new__(SourceResolver)
    with patch("app.services.source_resolver.safe_request", return_value=response), \
            patch("app.services.source_resolver._verify_text_content_identity", return_value=("high", "fixture")), \
            patch("app.services.source_resolver.compare_source_kinds",
                  return_value=Mock(verdict="compatible")):
        result = resolver._try_doi_resolver(DOI, TITLE, public=True, expected_author="Harbour, A",
                                            expected_year="2019", expected_source_kind=ARTICLE)
    if abstract_only:
        assert not result.success and result.representation is None and result.abstract == ABSTRACT
        assert result.metadata["abstract_provenance"]["basis"] == "publisher_landing_page"
        assert result.metadata["landing_abstract_only"]["policy"] == LANDING_POLICY
        from app.services.source_resolver import _landing_abstract_holder
        holder = _landing_abstract_holder(result)
        assert holder.success and holder.abstract == ABSTRACT and holder.representation is None
    else:
        assert result.success and result.representation is not None
        assert "abstract_provenance" not in (result.metadata or {})


def test_resolution_falls_back_to_a_kept_landing_abstract(monkeypatch):
    landing = RetrievalResult(
        source_name="doi_resolver", success=False, abstract=ABSTRACT, doi=DOI,
        metadata={"abstract_provenance": {"policy": LANDING_POLICY, "basis": "publisher_landing_page"}})
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", None)
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = [_Index("crossref"), _Web()]
    resolver._acquisition_capabilities = None
    resolver._lookup_cache = None
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name="local_cache", success=False))
    resolver._download_and_cache = lambda source, result, *a, **k: result
    resolver._record_canonical_location_outcomes = lambda result: None
    resolver._reuse_search_memo = lambda: False
    resolver._try_doi_resolver = Mock(return_value=landing)
    resolver._try_source = Mock(return_value=RetrievalResult(source_name="web_search", success=False))
    resolved = resolver.resolve(doi=DOI, title=TITLE, author="Harbour, A", year="2019",
                                source_kind="journal_article", source_kind_confidence="high")
    assert resolved.success and resolved.representation is None and resolved.abstract == ABSTRACT
    assert resolved.metadata["abstract_provenance"]["basis"] == "publisher_landing_page"


def test_the_abstract_evidence_keeps_its_provenance():
    from app.services.paper_workflow import _bounded_abstract_evidence
    provenance = {"policy": DOI_ABSTRACT_POLICY, "provider": "openalex", "doi": DOI}
    evidence = _bounded_abstract_evidence(ABSTRACT, source_name="canonical_work_graph", provenance=provenance)
    assert evidence["provenance"] == provenance and evidence["text"] == ABSTRACT
    assert "provenance" not in _bounded_abstract_evidence(ABSTRACT, source_name="openalex")


# ---- notes-and-listings pages are never whole works (2026-10-13) -----------

_NOTE_LINES = "\n".join(
    f"{n}. Quill, A., “Tariff walls and coastal ports,” Harbour Review 12 (19{60 + n % 30}), pp. {n}-{n + 9}Google Scholar."
    for n in range(1, 120))
_ABSTRACT = ("This article examines how tariff walls reshape coastal trade, drawing on port records and "
             "merchant correspondence to show how protection redirected shipping. ") * 2
_ARTICLE = "\n".join(
    f"{n} Section heading run into its paragraph: ports adapted their warehousing as tariffs rose, "
    "and merchants moved cargo inland along canals that had carried little before (Quill, 1972). " * 3
    for n in range(1, 16))


def test_a_page_of_notes_with_a_short_abstract_is_not_a_whole_work():
    from app.services.abstract_only_pages import mostly_notes_or_listings
    from app.services.web_completeness import stated_page_completeness
    page = "Abstract\n" + _ABSTRACT + "\n" + _NOTE_LINES + "\nCited by\nNo data available."
    assert mostly_notes_or_listings(page)
    verdict = stated_page_completeness(f"<html><body><p>{page}</p></body></html>", page, "journal_article")
    assert verdict["verdict"] != "complete" and verdict["reason"] == "mostly_notes_or_listings"


def test_an_article_whose_paragraphs_open_with_section_numbers_is_prose():
    from app.services.abstract_only_pages import body_prose_words, mostly_notes_or_listings
    assert not mostly_notes_or_listings(_ARTICLE) and body_prose_words(_ARTICLE) > 1000


def test_a_bibliographic_index_record_is_not_the_work():
    from app.services.source_validator import _detect_nonwork_listing
    record = ("Tariff walls and coastal ports Download full text from publisher As the access to this document "
              "is restricted, you may want to search for a different version of it. References listed on Indexa "
              "Quill, A., 1972. Ports. Most related items These are the items that most often cite the same works. "
              "More about this item Keywords tariffs; ports")
    assert _detect_nonwork_listing(record) == "a bibliographic index record"
    assert _detect_nonwork_listing("An article that mentions most related items once.") is None


def test_continuous_dense_script_prose_is_not_read_as_notes():
    from app.services.abstract_only_pages import mostly_notes_or_listings
    regulation = "\n".join(f"第{n}条 为了规范网络服务活动，保护公民合法权益，制定本规定，依法管理。" * 4 for n in range(1, 40))
    assert not mostly_notes_or_listings("1. Note\n" + regulation)
