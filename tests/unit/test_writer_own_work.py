"""Parts about the writer's own work are not judged (writer-own-work-v1, owner direction 2026-10-18).

Fixtures are invented: no student text or real paper.
"""
from types import SimpleNamespace

from app.services import judgment_reasons
from app.services.writer_own_work import OWN_WORK, own_work_parts

ROUTE = SimpleNamespace(arm_id="deepseek", model="fixture-model")
PARTS = [("a", "it draws on the harbour-film tradition"),
         ("b", "Lanterns, which sets its story in a flooded city")]


def answer(*pairs):
    return lambda *args, **kwargs: {"parts": [{"id": i, "about": about} for i, about in pairs]}


def test_only_a_part_about_the_writers_own_work_is_skipped():
    found = own_work_parts("Statement.", "My film Tidewater is a hybrid.", "Quill (2020). Lanterns.", PARTS,
                           route=ROUTE, call=answer(("a", OWN_WORK), ("b", "cited_source")))
    assert found["own_work"] == {"a"} and found["status"] == "model"
    unsure = own_work_parts("Statement.", "", "ref", PARTS, route=ROUTE, call=answer(("a", "unsure"), ("b", "unsure")))
    assert unsure["own_work"] == set()


def test_a_failed_or_malformed_answer_skips_nothing():
    def fail(*args, **kwargs):
        raise RuntimeError("down")
    assert own_work_parts("S.", "", "ref", PARTS, route=ROUTE, call=fail)["own_work"] == set()
    assert own_work_parts("S.", "", "ref", PARTS, route=ROUTE, call=lambda *a, **k: {"parts": "x"})["own_work"] == set()
    bad = own_work_parts("S.", "", "ref", PARTS, route=ROUTE, call=answer(("a", OWN_WORK), ("b", "maybe")))
    assert bad["own_work"] == set() and bad["status"] == "invalid"
    assert own_work_parts("S.", "", "ref", PARTS, route=None)["status"] == "not_run"


def test_an_answer_is_reused_for_the_same_input():
    stored = {}
    calls = []

    def call(*args, **kwargs):
        calls.append(1)
        return {"parts": [{"id": "a", "about": OWN_WORK}, {"id": "b", "about": "cited_source"}]}
    for _ in range(2):
        found = own_work_parts("S.", "before", "ref", PARTS, route=ROUTE, call=call,
                               cache_lookup=stored.get, cache_store=stored.__setitem__)
        assert found["own_work"] == {"a"}
    assert len(calls) == 1


def test_the_window_note_is_the_owner_wording():
    assert judgment_reasons.fixed_note("writer_own_work") == (
        "Not judged: this part describes the student's own work, not a claim made by the source.")


def test_a_part_carrying_only_other_references_markers_is_not_this_sources():
    from app.services.judgment_report import names_only_other_sources as _names_only_other_sources
    payload = {"source_binding": {"reference_id": "r1"},
               "claim": {"text": "x", "citation_markers": [
                   {"text": "Quill (1984)", "reference_ids": ["r1"]},
                   {"text": "(Marsh, 2017)", "reference_ids": ["r2"]}]},
               "verification_candidates": {"candidates": [
                   {"candidate_id": "a", "text": "This study uses Quill (1984)'s approach"},
                   {"candidate_id": "b", "text": "while building on Tidewater (Marsh, 2017)"},
                   {"candidate_id": "c", "text": "a clause with no marker"}]}}
    assert _names_only_other_sources(payload, "b")
    assert not _names_only_other_sources(payload, "a") and not _names_only_other_sources(payload, "c")


def test_a_chapter_upload_can_be_its_whole_book():
    from app.routers.report import _book_identities
    from app.services.book_of_part import book_of_cited_part, cited_book_title
    assert cited_book_title("Lanterns at Sea: Essays (2nd ed.)") == "Lanterns at Sea: Essays"
    assert _book_identities({"source_kind": "journal_article", "container_title": "Fanmag"}, None) == []
    # One rule names the book for the report, the search and upload matching.
    container = {"is_monograph": True, "located_record": {"title": "Lanterns at Sea", "authors": ["Moor, L."]}}
    source = {"source_kind": "book_section", "container_title": "Lanterns at Sea (2nd ed.)", "author": "Moor, L."}
    assert book_of_cited_part(container)["title"] == _book_identities(source, container)[0][0] == "Lanterns at Sea"


def test_a_whole_book_is_matched_first_as_the_catalogue_records_it():
    from app.routers.report import _book_identities
    source = {"source_kind": "book_section", "author": "Quill, O.", "container_title": "Lanterns: Lanterns at sea"}
    container = {"is_monograph": True, "located_record": {"title": "Lanterns at Sea", "authors": ["Orla Quill"]}}
    assert _book_identities(source, container) == [("Lanterns at Sea", "Orla Quill"),
                                                   ("Lanterns: Lanterns at sea", "Quill, O.")]
    assert _book_identities(source, {"is_monograph": False, "located_record": {"title": "X"}}) == [
        ("Lanterns: Lanterns at sea", "Quill, O.")]


def test_sources_to_upload_names_the_located_book():
    from app.services.evidence_report import _render_upload_priorities
    member = {"reference_id": "r", "coverage_level": "unavailable",
              "source": {"raw_reference": "Quill, O. (2018). A chapter. In Lanterns: Lanterns at sea (pp. 1-9). Press.",
                         "author": "Quill, O.", "year": "2018", "title": "A chapter",
                         "container_title": "Lanterns: Lanterns at sea", "source_kind": "book_section"},
              "located_book": {"title": "Lanterns at Sea", "authors": ["Orla Quill"], "year": "2018"}}
    html = _render_upload_priorities([{"members": [member], "student_text": "x"}])
    assert "Lanterns at Sea" in html and "A chapter" not in html and "Quill, O. (2018)" in html


def test_without_a_catalogue_record_the_cited_book_title_is_named():
    from app.services.evidence_report import _located_single_authored_book
    assert _located_single_authored_book({"container_identity": {"is_monograph": True, "title": "Lanterns at Sea"}}) == {
        "title": "Lanterns at Sea", "authors": [], "year": ""}
    assert _located_single_authored_book({"container_identity": {"is_monograph": False, "title": "X"}}) is None


def test_a_repository_metadata_export_is_not_the_works_text():
    from app.services.source_resolver import metadata_record_xml
    mets = (b'<mets:mets OBJID="eprint_1" xmlns:mets="http://www.loc.gov/METS/" xmlns:mods="http://www.loc.gov/mods/v3">'
            b'<mods:title>Lanterns at Sea</mods:title></mets:mets>')
    assert metadata_record_xml(mets)
    assert metadata_record_xml(b'<?xml version="1.0"?><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/">x</oai_dc:dc>')
    jats = b'<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink"><body>text</body></article>'
    assert not metadata_record_xml(jats)
    assert not metadata_record_xml(b'<record><p>an ordinary page</p></record>')


def test_the_check_reads_the_sentences_before_and_unresolved_parts(monkeypatch):
    import json
    from app.services import judgment_runs
    from app.services.facet_evidence_judgment import _ANTECEDENT_CONTEXT_MISSING
    from app.services.verification_evidence import CandidateFacetFinding
    monkeypatch.setattr(judgment_runs.settings, "PAPER_LLM_PROCESSING_ENABLED", True)
    payload = {"claim": {"text": "It echoes the harbour-film tradition (Lane, 2020).",
                         "antecedent_context": [{"distance_before": 1, "text": "My film follows a lighthouse keeper."}]},
               "verification_candidates": {"candidates": [{"candidate_id": "c1",
                                                           "text": "It echoes the harbour-film tradition"}]}}
    finding = CandidateFacetFinding(candidate_id="c1", status="not_assessed", derived_outcome="not_assessed",
                                    evidence_coverage="absent", limitations=[_ANTECEDENT_CONTEXT_MISSING])
    seen = {}

    def call(system, user, **kwargs):
        seen.update(json.loads(user))
        return {"parts": [{"id": "c1", "about": OWN_WORK}]}

    cache = SimpleNamespace(coaching_lookup=lambda key: None, coaching_store=lambda key, value: None)
    view = {"citations": [{"citation_number": 3, "student_text": payload["claim"]["text"], "members": []}]}
    result = judgment_runs._writer_own_work(view, 3, "r1", payload, SimpleNamespace(items=[finding]), ROUTE, call, cache)
    assert seen["sentences_before"] == "My film follows a lighthouse keeper."
    assert [p["id"] for p in seen["parts"]] == ["c1"] and result["own_work"] == {"c1"}
