"""Owner-reported spend 2026-10-04 (paper 10): finished searches are reused."""
from app.services import source_reuse as rerun_reuse
from app.services.source_reuse import earlier_result



def _reusable(item, reference=None):
    """The earlier-result decision, through source_reuse's public interface."""
    return earlier_result(item, reference).reusable

def test_an_optional_retry_does_not_force_a_new_search_when_required_searches_finished(monkeypatch):
    item = {"status": "metadata_only", "reason_code": "bibliography_identity_only",
            "retryable_provider_dependencies": ["openalex"],
            "reference_discovery": {"outcome": "search_incomplete", "attempts": []}}
    monkeypatch.setattr(rerun_reuse, "_required_searches_completed", lambda item, reference: True)
    assert _reusable(item, {"reference_id": "r"})
    monkeypatch.setattr(rerun_reuse, "_required_searches_completed", lambda item, reference: False)
    assert not _reusable(item, {"reference_id": "r"})
    assert not _reusable({**item, "full_text_search_incomplete_providers": ["core"]}, {})


def test_an_uncited_web_pages_answered_search_is_reused():
    answered = {"attempts": [{"route_category": "bounded_web", "outcome": "candidate_found"},
                             {"route_category": "bounded_web", "outcome": "no_match"}]}
    item = {"status": "link_check_only", "web_page_check": {"link": "telling"}, "reference_discovery": answered}
    assert _reusable(item, {})
    failed = {"attempts": [{"route_category": "bounded_web", "outcome": "operational_failure"}]}
    assert not _reusable({**item, "reference_discovery": failed}, {})
    assert not _reusable({**item, "web_page_check": None}, {})


def test_an_unconfirmed_chapter_without_its_book_lookup_is_searched_again():
    chapter = {"source_kind": "book_section", "container_title": "Global media"}
    item = {"status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": {"outcome": "unlocated_after_search"}}
    assert not _reusable(item, chapter)
    found = {**item, "reference_discovery": {"outcome": "unlocated_after_search", "container_identity": {"status": "identified"}}}
    assert _reusable(found, chapter)


def test_an_incomplete_search_with_no_provider_at_fault_is_reused(monkeypatch):
    # Owner decision 2026-10-07: a repeat would end the same way.
    from app.services import paper_workflow
    item = {"status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": {"outcome": "search_incomplete", "expected": {},
                                    "attempts": [{"route_category": "bounded_web", "outcome": "candidate_found"}]}}
    monkeypatch.setattr(rerun_reuse, "_required_searches_completed", lambda item, reference: False)
    monkeypatch.setattr(paper_workflow, "_blocking_providers_from_record", lambda record: [])
    assert _reusable(item, {"reference_id": "r"})
    # A provider whose failure holds it incomplete, or an incomplete parse, forces a new search.
    monkeypatch.setattr(paper_workflow, "_blocking_providers_from_record", lambda record: ["exa"])
    assert not _reusable(item, {"reference_id": "r"})
    monkeypatch.setattr(paper_workflow, "_blocking_providers_from_record", lambda record: [])
    parse = {**item, "reference_discovery": {**item["reference_discovery"], "expected": {"reference_parse_review": True}}}
    assert not _reusable(parse, {"reference_id": "r"})
