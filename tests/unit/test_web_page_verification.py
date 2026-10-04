"""Cited web pages are verified by their link, an archived copy and web search
(`webpage-verification-v1`, owner decision 2026-10-03)."""
from types import SimpleNamespace

from app.services.reference_verification import assess_reference_verification
from app.services.web_page_verification import link_state, near_title, web_page_check


def request(**fields):
    return [{"kind": "url", "requests": [{"completed_at": "2026-10-03T00:00:00Z", **fields}]}]


def test_the_link_decides_whether_the_page_can_be_assessed():
    assert link_state(None, "https://www.example.org")[0] == "telling"               # home page only
    assert link_state(request(outcome="not_found", http_status=404), "https://e.org/a")[0] == "telling"
    assert link_state(request(outcome="access_refused", http_status=403), "https://e.org/a")[0] == "not_telling"
    assert link_state(request(outcome="response", http_status=200, destination_identity="confirmed"),
                      "https://e.org/a")[0] == "confirmed"
    other = request(outcome="response", http_status=200, destination_identity="bibliographic_conflict",
                    identity_fields=["title"], identity_differences=[{"field": "title", "destination": "Annual Statistics 2022"}])
    assert link_state(other, "https://e.org/a", "Postclassical Hollywood narrative experiments")[0] == "telling"
    shell = request(outcome="response", http_status=202, destination_identity="bibliographic_conflict",
                    identity_fields=["title"], identity_differences=[{"field": "title", "destination": "JavaScript is disabled"}])
    assert link_state(shell, "https://e.org/a", "A Film (1951)")[0] == "not_telling"


def test_a_near_title_on_the_linked_page_is_a_possible_match_not_a_flag():
    assert near_title("The Day the Earth Stood Still (1951 Film) Analysis. [Web log post]",
                      "The Day the Earth Stood Still (1951 Film) Study Guide: Analysis")
    ref = SimpleNamespace(reference_id="r", source_kind="webpage", title="A Cited Page Title Here", raw_ref="x")
    result = assess_reference_verification(ref, None, {"link": "near", "archive": "not_checked"})
    assert result["status"] == "possible_match"


def test_a_web_page_is_unverifiable_only_after_a_telling_link_no_archive_and_completed_web_search():
    ref = SimpleNamespace(reference_id="r", source_kind="webpage", title="Postclassical Hollywood and narrative experimentation",
                          raw_ref="Site. (2011). Postclassical Hollywood and narrative experimentation. https://www.example.org")
    queries = [{"query_id": f"q-{e}", "execution_provider": e, "execution_outcome": "no_results",
                "normalized_query": "postclassical hollywood and narrative experimentation"} for e in ("brave", "exa")]
    discovery = {"outcome": "unlocated_after_search", "expected": {"title": ref.title, "source_kind": "webpage"},
                 "queries": queries, "candidates": [],
                 "attempts": [{"route_category": "bounded_web", "provider": "web_search", "permitted": True,
                               "completed_at": "t", "outcome": "no_match", "query_ids": ["q-brave", "q-exa"]}]}
    flagged = assess_reference_verification(ref, discovery, {"link": "telling", "archive": "none"})
    assert flagged["status"] == "cannot_be_verified"
    assert assess_reference_verification(ref, discovery, {"link": "telling", "archive": "matched"})["status"] == "verified"
    assert assess_reference_verification(ref, discovery, {"link": "not_telling"})["status"] == "not_assessed"
    assert assess_reference_verification(ref, discovery)["reason_code"] == "not_held_by_academic_indexes"


def test_a_web_page_without_a_usable_title_or_link_is_not_checked():
    garbled = SimpleNamespace(source_kind="webpage", url="https://e.org/a", title="Retrieved May 5, 2023, from")
    assert web_page_check(garbled, None) is None
    no_link = SimpleNamespace(source_kind="webpage", url="", title="A Real Page Title")
    assert web_page_check(no_link, None) is None
