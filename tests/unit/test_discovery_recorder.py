"""The discovery recorder (discovery-recorder-v1, 2026-10-21): entries are checked when written,
caps degrade, and finishing changes nothing."""
import pytest

from app.services import discovery_recorder as dr
from app.services.discovery_recorder import DiscoveryRecordError, ModelAllowance
from app.services.reference_discovery import (
    ExpectedBibliographicFields, ReferenceDiscoveryRecord, ReferenceDiscoveryTrace,
    build_reference_discovery_candidate,
)
from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
from tests.unit.discovery_support import recorder_from

EXPECTED = ExpectedBibliographicFields(title="Harbour cranes and port labour", authors=["Writer, A."],
                                       year="2020", source_kind="journal_article")


def _recorder():
    return recorder_from(expected=EXPECTED, required={"academic_adapter"}, reference_id="ref-1",
                         submitted_reference="Writer, A. (2020). Harbour cranes and port labour.")


def _attempt(recorder, provider="crossref", found=True):
    """A real attempt recorded through the resolver's writer; its id."""
    result = RetrievalResult(source_name=provider, success=found, title=EXPECTED.title if found else None,
                             authors=["Writer, A."] if found else [], year="2020" if found else None,
                             error=None if found else "No results")
    token = _ACTIVE_DISCOVERY_TRACE.set(recorder)
    try:
        return SourceResolver._record_discovery_attempt(category="academic_adapter", provider=provider,
                                                        result=result, required=True)
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def _candidate(attempt_id, provider="crossref", key="record-1"):
    return build_reference_discovery_candidate(
        attempt_id=attempt_id, provider=provider, expected=EXPECTED, candidate_key=key,
        result=RetrievalResult(source_name=provider, success=True, title=EXPECTED.title))


def test_the_resolver_writer_returns_the_kept_attempt():
    recorder = _recorder()
    attempt_id = _attempt(recorder)
    assert attempt_id == recorder.attempts[0].attempt_id and len(recorder.candidates) == 1


def test_a_broken_link_is_refused_where_it_is_written():
    recorder = _recorder()
    attempt_id = _attempt(recorder)
    with pytest.raises(DiscoveryRecordError):
        recorder.add_candidate(_candidate("no-such-attempt"))
    with pytest.raises(DiscoveryRecordError):
        recorder.add_candidate(_candidate(attempt_id, provider="openalex"))
    with pytest.raises(DiscoveryRecordError):
        recorder.add_attempt(recorder.attempts[0])                      # the same id twice
    with pytest.raises(DiscoveryRecordError):
        recorder.add_attempt(recorder.attempts[0].model_copy(update={"attempt_id": "other",
                                                                     "query_ids": ["missing"]}))


def test_outside_tests_a_broken_link_is_dropped_and_noted(monkeypatch):
    monkeypatch.setattr(dr, "STRICT", False)
    recorder = _recorder()
    assert recorder.add_candidate(_candidate("no-such-attempt")) is False
    assert recorder.candidates == []
    assert "Discovery record refused an inconsistent entry." in recorder.limitations


def test_a_repeated_record_is_kept_once_when_asked():
    recorder = _recorder()
    attempt_id = _attempt(recorder)
    first = _candidate(attempt_id, key="record-2")
    assert recorder.add_candidate(first, unique=True)
    assert not recorder.add_candidate(_candidate(attempt_id, key="record-2"), unique=True)
    assert [c.candidate_id for c in recorder.candidates].count(first.candidate_id) == 1


def test_a_full_record_drops_the_extra_entry_and_notes_it(monkeypatch):
    monkeypatch.setitem(dr.CAPS, "attempts", 1)
    recorder = _recorder()
    _attempt(recorder)
    assert _attempt(recorder, provider="openalex") is None             # dropped, not raised
    assert len(recorder.attempts) == 1
    assert "Discovery record limit reached: further search attempts were not recorded." in recorder.limitations


def test_a_dropped_attempts_candidates_are_dropped_with_it(monkeypatch):
    monkeypatch.setitem(dr.CAPS, "attempts", 0)
    recorder = _recorder()
    assert _attempt(recorder) is None and recorder.candidates == []


def test_finishing_changes_nothing_and_keeps_the_stored_shape():
    recorder = _recorder()
    _attempt(recorder)
    recorder.native_script_route = {"policy_version": "native-script-route-v1", "status": "completed"}
    recorder.container_identity = {"status": "identified"}
    before = (list(recorder.limitations), len(recorder.attempts), len(recorder.candidates))
    first = recorder.finish(["An end-of-search limitation."])
    second = recorder.finish(["An end-of-search limitation."])
    stamp = lambda payload: {k: v for k, v in (payload or {}).items() if k != "created_at"}
    assert first.trace == second.trace and stamp(first.record) == stamp(second.record)
    assert (list(recorder.limitations), len(recorder.attempts), len(recorder.candidates)) == before
    ReferenceDiscoveryTrace.model_validate({k: v for k, v in first.trace.items() if k != "native_script_route"})
    assert first.trace["native_script_route"]["status"] == "completed"
    assert "An end-of-search limitation." in first.trace["limitations"]
    if first.record is not None:
        ReferenceDiscoveryRecord.model_validate({k: v for k, v in first.record.items()
                                                 if k not in {"native_script_route", "container_identity"}})
        assert first.record["container_identity"] == {"status": "identified"}


def test_a_preview_leaves_out_the_attachments():
    recorder = _recorder()
    _attempt(recorder)
    recorder.periodical_archive_match = {"publication": "Lanternland"}
    assert "periodical_archive_match" not in recorder.preview().trace
    assert recorder.finish().trace["periodical_archive_match"] == {"publication": "Lanternland"}


def test_the_model_allowance_is_shared_and_bounded():
    allowance = ModelAllowance()
    assert [allowance.reserve(2) for _ in range(3)] == [True, True, False]


def test_a_nested_search_shares_the_cited_references_allowance(monkeypatch):
    seen = []
    resolver = SourceResolver.__new__(SourceResolver)
    resolver.resolve = lambda **kwargs: (seen.append(_ACTIVE_DISCOVERY_TRACE.get().allowance)
                                         or RetrievalResult(source_name="test", success=True, full_text=b"text"))
    for name in ("_enrich_journal_registration", "_enrich_book_editions"):
        monkeypatch.setattr(resolver, name, lambda *args: None)
    monkeypatch.setattr(resolver, "_update_search_memo", lambda *args: None)
    from app.services.schemas import ParsedReference
    reference = ParsedReference(reference_id="ref-2", title="A nested work", author="Writer, A.", year="2020",
                                raw_ref="Writer (2020). A nested work.", source_kind="monograph")
    outer = _recorder()
    token = _ACTIVE_DISCOVERY_TRACE.set(outer)
    try:
        resolver.resolve_reference(reference)
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    resolver.resolve_reference(reference)
    assert seen[0] is outer.allowance and seen[1] is not outer.allowance


def _faulty_archive_trace():
    """A stored trace as the archive defect left it: the archive search found,
    with no candidate, so the record was dropped (fixed 2026-10-21)."""
    from app.services.retrieval import media_history_archive as archive
    recorder = _recorder()
    _attempt(recorder)
    archive_id = _attempt(recorder, provider=archive.PROVIDER)
    recorder.candidates = [c for c in recorder.candidates if c.attempt_id != archive_id]
    recorder.periodical_archive_match = {"publication": "Lanternland"}
    done = recorder.finish()
    assert done.record is None
    return done.trace


def test_a_stored_archive_result_gets_its_record_back_without_searching():
    from app.services.discovery_recorder import repaired_archive_record
    stored = _faulty_archive_trace()
    repaired = repaired_archive_record(stored)
    assert repaired is not None and repaired.record is not None
    assert repaired.record["periodical_archive_match"] == {"publication": "Lanternland"}
    archive_attempt = next(a for a in repaired.trace["attempts"] if a["provider"] == "media_history_archive")
    assert (archive_attempt["outcome"], archive_attempt["reason_code"]) == ("no_match", "archive_page_located")
    # Nothing to repair: a stored record, or no faulty archive attempt.
    assert repaired_archive_record(stored, {"outcome": "possible_match"}) is None
    assert repaired_archive_record(repaired.trace) is None


def test_reuse_hands_back_the_repaired_record():
    from app.services.schemas import ParsedReference
    from app.services.search.rerun_reuse import reference_signature, reused_result
    reference = ParsedReference(reference_id="ref-new", title=EXPECTED.title, author="Writer, A.", year="2020",
                                raw_ref="Writer, A. (2020). Harbour cranes and port labour.",
                                source_kind="journal_article")
    item = {"reference_id": "ref-1", "status": "durable_authorized",
            "reference_discovery_trace": _faulty_archive_trace(), "reference_discovery": None}
    index = {reference_signature(reference, identity_only=False): ("job-1", item)}
    reused = reused_result(index, reference, identity_only=False)
    assert reused["archive_record_repaired"] is True
    assert reused["reference_discovery"]["reference_id"] == "ref-new"
    assert item["reference_discovery"] is None          # the stored result is left as it was
