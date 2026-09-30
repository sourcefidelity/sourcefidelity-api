"""Journal, volume and issue are read from landing pages that publish them.

Found 2026-09-23 while checking why a reference naming the wrong journal was
never flagged. The reference gives Khan's "The separation of platforms and
commerce" as Harvard Law Review 131(5); the article is Columbia Law Review 119
(2019), open access, and the first web result is the publisher's own page.
Retrieval found it. Identity could not use it, for three separate reasons that
all had to be fixed before any comparison existed:

  - Digital Commons (bepress) publishes the Highwire tags under its own
    prefix, and the extractor read only the unprefixed names, so those pages
    yielded no title, author, journal or volume at all.
  - `citation_journal_title` was never read -- only `citation_inbook_title`,
    which a journal article does not carry.
  - volume and issue were not extracted, and not forwarded by the landing
    inspection even when present.
"""
from app.services.web_source_metadata import extract_web_source_metadata

# Reduced from the live response of the Columbia Law School repository.
_BEPRESS = """<html><head>
<meta name="bepress_citation_title" content="The Separation of Platforms and Commerce">
<meta name="bepress_citation_author" content="Khan, Lina M.">
<meta name="bepress_citation_journal_title" content="Colum. L. Rev.">
<meta name="bepress_citation_volume" content="119">
<meta name="bepress_citation_firstpage" content="973">
<meta name="bepress_citation_date" content="2019">
</head><body><h1>The Separation of Platforms and Commerce</h1></body></html>"""

_HIGHWIRE = """<html><head>
<meta name="citation_title" content="A study">
<meta name="citation_author" content="Doe, J.">
<meta name="citation_journal_title" content="Journal of Examples">
<meta name="citation_volume" content="12">
<meta name="citation_issue" content="3">
</head><body></body></html>"""


def test_a_bepress_repository_page_yields_its_full_record() -> None:
    observed = extract_web_source_metadata(_BEPRESS, "https://scholarship.law.columbia.edu/x/1/")

    assert observed["title"] == "The Separation of Platforms and Commerce"
    assert observed["authors"] == ["Khan, Lina M."]
    assert observed["container_title"] == "Colum. L. Rev."
    assert observed["volume"] == "119"
    assert observed["year"] == "2019"


def test_a_journal_article_names_its_journal_as_the_container() -> None:
    observed = extract_web_source_metadata(_HIGHWIRE, "https://example.org/a")

    assert observed["container_title"] == "Journal of Examples"
    assert observed["volume"] == "12"
    assert observed["issue"] == "3"


def test_a_page_that_publishes_nothing_observes_nothing() -> None:
    """Absence stays absence; it is never filled in from the citation."""
    observed = extract_web_source_metadata(
        "<html><head><title>Home</title></head><body></body></html>",
        "https://example.org/")

    assert observed["container_title"] is None
    assert observed["volume"] is None
    assert observed["issue"] is None


def test_conflicting_repeated_values_stay_unresolved() -> None:
    page = ("<html><head>"
            '<meta name="citation_volume" content="119">'
            '<meta name="citation_volume" content="120">'
            "</head><body></body></html>")

    assert extract_web_source_metadata(page, "https://example.org/a")["volume"] is None


def test_the_landing_inspection_reports_the_volume_the_page_publishes(monkeypatch) -> None:
    """The whole point of the repair: the comparison now has a value to make.

    The reference says volume 131; the page says 119. Before this, the
    inspection aborted at `landing_page_bibliography_unavailable` because it
    saw neither title nor author on a bepress page.
    """
    import httpx

    from app.services import identity_landing
    from app.services.reference_discovery import ExpectedBibliographicFields
    from app.services.retrieval.base import AcquisitionLocation, RepresentationKind

    url = "https://scholarship.law.columbia.edu/faculty_scholarship/2789/"
    request = httpx.Request("GET", url)
    response = httpx.Response(200, request=request, text=_BEPRESS,
                              headers={"content-type": "text/html"})
    monkeypatch.setattr(identity_landing, "safe_request", lambda *a, **k: response)
    monkeypatch.setattr(identity_landing, "require_source_candidate", lambda _u: None)

    expected = ExpectedBibliographicFields(
        title="The separation of platforms and commerce",
        authors=["Khan, L"], year="2018",
        container_title="Harvard Law Review", volume="131", issue="5")
    location = AcquisitionLocation(url=url, provider="web_search",
                                   representation_kind=RepresentationKind.HTML)

    attempt = identity_landing.inspect(location, expected)

    observed = (attempt.get("landing_metadata_observation") or {}).get("observed") or {}
    assert observed["metadata"]["volume"] == "119"
    assert observed["metadata"]["container_title"] == "Colum. L. Rev."
    assert observed["year"] == "2019"
    assert attempt["reason_code"] == "identity_landing_metadata_observed"
