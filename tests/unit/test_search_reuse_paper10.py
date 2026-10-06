"""Owner-reported spend 2026-10-04 (paper 10): finished searches are reused."""
from app.services.search import rerun_reuse


def test_an_optional_retry_does_not_force_a_new_search_when_required_searches_finished(monkeypatch):
    item = {"status": "metadata_only", "reason_code": "bibliography_identity_only",
            "retryable_provider_dependencies": ["openalex"],
            "reference_discovery": {"outcome": "search_incomplete", "attempts": []}}
    monkeypatch.setattr(rerun_reuse, "_required_searches_completed", lambda item, reference: True)
    assert rerun_reuse._reusable(item, {"reference_id": "r"})
    monkeypatch.setattr(rerun_reuse, "_required_searches_completed", lambda item, reference: False)
    assert not rerun_reuse._reusable(item, {"reference_id": "r"})
    assert not rerun_reuse._reusable({**item, "full_text_search_incomplete_providers": ["core"]}, {})


def test_an_uncited_web_pages_answered_search_is_reused():
    answered = {"attempts": [{"route_category": "bounded_web", "outcome": "candidate_found"},
                             {"route_category": "bounded_web", "outcome": "no_match"}]}
    item = {"status": "link_check_only", "web_page_check": {"link": "telling"}, "reference_discovery": answered}
    assert rerun_reuse._reusable(item, {})
    failed = {"attempts": [{"route_category": "bounded_web", "outcome": "operational_failure"}]}
    assert not rerun_reuse._reusable({**item, "reference_discovery": failed}, {})
    assert not rerun_reuse._reusable({**item, "web_page_check": None}, {})


def test_an_unconfirmed_chapter_without_its_book_lookup_is_searched_again():
    chapter = {"source_kind": "book_section", "container_title": "Global media"}
    item = {"status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": {"outcome": "unlocated_after_search"}}
    assert not rerun_reuse._reusable(item, chapter)
    found = {**item, "reference_discovery": {"outcome": "unlocated_after_search", "container_identity": {"status": "identified"}}}
    assert rerun_reuse._reusable(found, chapter)
