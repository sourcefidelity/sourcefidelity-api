"""One page-identity rule for DOI/landing and web pages (owner direction 2026-10-14)."""
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate
from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import identity_confirms_work

EXPECTED = ExpectedBibliographicFields(
    title="Trade protection and the role of non-tariff barriers",
    authors=["Kinzius, L.", "Sandkamp, A.", "Yalcin, E."], year="2019", doi="10.1007/s10290-019-00341-6",
    container_title="Review of World Economics", volume="155", issue="4", pages="603-643")


def _page(**metadata):
    result = RetrievalResult(source_name="independent_landing_metadata", success=True,
                             title="Trade protection and the role of non-tariff barriers",
                             authors=["Kinzius, Luisa", "Sandkamp, Alexander", "Yalcin, Erdal"], year="2019",
                             doi="10.1007/s10290-019-00341-6", metadata=metadata)
    return build_reference_discovery_candidate(attempt_id="x", provider="independent_landing_metadata",
                                               expected=EXPECTED, result=result)


def test_an_unshown_volume_and_issue_do_not_block_a_matching_publisher_page():
    # Springer's page shows the journal and pages but not volume or issue.
    candidate = _page(container_title="Review of World Economics", pages="603-643")
    assert identity_confirms_work(candidate)
    # A finding that accuses the reference still uses the strict form.
    assert candidate.has_unresolved_supplied_identity_fields


def test_a_shown_conflicting_journal_still_blocks():
    assert not identity_confirms_work(_page(container_title="Journal of Unrelated Studies", pages="603-643"))


def test_an_unshown_supplied_year_or_author_still_blocks():
    result = RetrievalResult(source_name="x", success=True, title=EXPECTED.title, authors=[], year=None,
                             doi=EXPECTED.doi)
    candidate = build_reference_discovery_candidate(attempt_id="x", provider="x", expected=EXPECTED, result=result)
    assert not identity_confirms_work(candidate)


def test_a_third_party_page_restating_the_publishers_abstract_yields_to_it():
    from app.services.abstract_only_pages import publisher_abstract_covers_page
    from app.services.retrieval.base import RepresentationKind, SourceRepresentation
    abstract = " ".join(["Non-tariff barriers restrict trade between countries in measurable ways."] * 12)
    landing = RetrievalResult(source_name="doi_resolver", success=True, abstract=abstract,
                              metadata={"abstract_provenance": {"basis": "publisher_landing_page"}})

    def page(text, completeness="not_assessed"):
        return RetrievalResult(source_name="web_fetch", success=True, representation=SourceRepresentation(
            kind=RepresentationKind.PLAIN_TEXT, media_type="text/plain", content=text.encode(),
            original_kind=RepresentationKind.HTML, completeness=completeness))
    restated = page("Trade protection and the role of non-tariff barriers. " + abstract + " JEL: F13, F14.")
    assert publisher_abstract_covers_page(restated, landing) is not None
    # A page with much more prose, a complete page or no publisher abstract stands.
    longer = page(abstract + " " + " ".join(["The authors estimate gravity models across sectors and years."] * 60))
    assert publisher_abstract_covers_page(longer, landing) is None
    assert publisher_abstract_covers_page(page(abstract, "complete"), landing) is None
    adapter = RetrievalResult(source_name="openalex", success=True, abstract=abstract, metadata={})
    assert publisher_abstract_covers_page(restated, adapter) is None
