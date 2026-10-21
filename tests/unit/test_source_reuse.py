"""Source reuse (2026-10-15): one rule set for reusing an earlier check's result
or a library text, and every refusal says why."""
from types import SimpleNamespace

from app.services.retrieval.base import RepresentationKind
from app.services.search.rerun_reuse import reference_signature, reuse_refusal
from app.services.source_reuse import (
    LANTERN_PAGES_VERSION, ROUTE_VERSIONS, STUDENT_LINK_ROUTES_VERSION, earlier_result, stored_copy,
    stored_text_today,
)

REF = {"reference_id": "r1", "raw_ref": "Ames, A. (2020). Harbour cranes. Port Studies.", "title": "Harbour cranes",
       "author": "Ames, A.", "year": "2020", "source_kind": "journal_article"}
CONFIRMED = {"outcome": "confirmed", "attempts": []}


def test_every_route_version_says_what_bumping_it_reads_again():
    assert STUDENT_LINK_ROUTES_VERSION == ROUTE_VERSIONS["student_link_routes"].version
    assert LANTERN_PAGES_VERSION == ROUTE_VERSIONS["lantern_pages"].version
    assert all(entry.rereads for entry in ROUTE_VERSIONS.values())


def test_a_refused_earlier_result_names_its_reason():
    assert earlier_result({"status": "abstract_only", "reference_discovery": CONFIRMED,
                           "abstract_evidence": {"provenance": {"abstract_provider": "crossref"}}}, REF).reusable
    # An abstract stored without its source record is searched again once (v7).
    assert not earlier_result({"status": "abstract_only", "reference_discovery": CONFIRMED}, REF).reusable
    stale = earlier_result({"status": "durable_authorized", "representation_id": "x", "source_name": "lantern"}, REF)
    assert (stale.reusable, stale.reason) == (False, "lantern_pages_changed")
    incomplete = earlier_result({"status": "unavailable", "reason_code": "full_text_search_incomplete",
                                 "reference_discovery": CONFIRMED}, REF)
    assert (incomplete.reusable, incomplete.reason) == (False, "result_incomplete")


def test_the_rerun_index_remembers_why_a_work_was_not_reused():
    signature = reference_signature(REF, identity_only=False)
    index = {("refused",) + signature: "network_failed"}
    assert reuse_refusal(index, REF, identity_only=False) == "network_failed"
    assert reuse_refusal({}, REF, identity_only=False) is None


def test_a_stored_text_is_tested_against_todays_admission_rules():
    notes = "Notes\n" + "\n".join(f"{i}. Ibid., p. {i}." for i in range(1, 40))
    decision, completeness = stored_text_today(notes, title=None, completeness="complete")
    assert decision.reusable and completeness == "not_assessed"
    decision, completeness = stored_text_today("Harbour cranes in river ports. " * 40, title=None,
                                               completeness="complete")
    assert decision.reusable and completeness == "complete"


def test_a_library_copy_is_checked_in_the_original_order():
    record = SimpleNamespace(completeness_verdict="complete", validation_evidence={}, source_url="https://x",
                             canonical_work=SimpleNamespace(year="2020", author="Ames, A."))
    common = dict(title="Harbour cranes", author="Ames, A.", year="2020", normalize_author=str)
    text = b"Harbour cranes in river ports. " * 40
    decision, _ = stored_copy(record, text, kind=RepresentationKind.PLAIN_TEXT, edition_check=lambda: True, **common)
    assert decision.reason == "edition_candidate_not_pdf"
    decision, _ = stored_copy(record, b"%PDF", kind=RepresentationKind.PDF, edition_check=lambda: False, **common)
    assert decision.reason == "edition_candidate_not_revalidated"
    preview = SimpleNamespace(**{**vars(record), "completeness_verdict": "incomplete"})
    decision, _ = stored_copy(preview, b"%PDF", kind=RepresentationKind.PDF, edition_check=lambda: False, **common)
    assert decision.reason == "limited_preview_not_revalidated"     # checked before the edition
    assert stored_copy(record, text, kind=RepresentationKind.PLAIN_TEXT, **common)[0].reusable
