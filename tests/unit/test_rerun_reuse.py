"""paper-search-reuse-v1 (owner decision 2026-10-01): a re-run reuses unchanged references' results."""
import pytest
from types import SimpleNamespace

from app.services.search.rerun_reuse import reference_signature, reused_result
from app.services.source_reuse import earlier_result, refuse

REF = {"reference_id": "ref-1", "raw_ref": "Smith, J. (2020). A study. Example Books.", "source_kind": "monograph",
       "title": "A study", "author": "Smith, J.", "year": "2020", "container_title": "", "doi": ""}



def _reusable(item, reference=None):
    """The earlier-result decision, through source_reuse's public interface."""
    return earlier_result(item, reference).reusable

def test_only_complete_results_are_reusable():
    discovery = {"outcome": "unlocated_after_search", "created_at": "2026-10-01T00:00:00Z"}
    assert _reusable({"status": "unavailable", "reason_code": "full_text_unavailable", "reference_discovery": discovery})
    assert _reusable({"status": "durable_authorized", "representation_id": "r"})
    assert not _reusable({"status": "transient_authorized", "reference_discovery": discovery})
    assert not _reusable({"status": "unavailable", "reason_code": "full_text_search_incomplete",
                          "reference_discovery": discovery})
    assert not _reusable({"status": "unavailable", "reference_discovery": {"outcome": "search_incomplete"}})
    assert not _reusable({"status": "unavailable", "reference_discovery": discovery,
                          "retryable_provider_dependencies": ["brave"]})


def test_a_changed_parse_or_mode_is_not_reused_and_a_match_is_rekeyed():
    item = {"reference_id": "ref-old", "status": "unavailable", "reason_code": "full_text_unavailable",
            "reference_discovery": {"reference_id": "ref-old", "outcome": "unlocated_after_search", "created_at": "t0"}}
    index = {reference_signature(REF, identity_only=False): ("job-1", item)}
    same = SimpleNamespace(**dict(REF, reference_id="ref-new"))
    record = reused_result(index, same, identity_only=False)
    assert record["reference_id"] == "ref-new" and record["reference_discovery"]["reference_id"] == "ref-new"
    assert record["search_reuse"]["from_job_id"] == "job-1" and record["search_reuse"]["searched_at"] == "t0"
    assert item["reference_id"] == "ref-old"
    assert reused_result(index, SimpleNamespace(**dict(REF, source_kind="webpage")), identity_only=False) is None
    assert reused_result(index, same, identity_only=True) is None


def test_an_optional_route_failure_does_not_force_a_search_when_required_searches_completed(monkeypatch):
    import app.services.source_reuse as reuse
    item = {"status": "unavailable", "reason_code": "source_not_found",
            "reference_discovery": {"outcome": "search_incomplete"}}
    monkeypatch.setattr(reuse, "_required_searches_completed", lambda item, reference: reference is not None)
    assert _reusable(item, REF)
    assert not _reusable(item)
    # A provider's retry no longer forces a search when every required search
    # finished (2026-10-04, paper 10); unfinished required searches still do.
    assert _reusable(dict(item, retryable_provider_dependencies=["crossref"]), REF)
    monkeypatch.setattr(reuse, "_required_searches_completed", lambda item, reference: False)
    assert not _reusable(dict(item, retryable_provider_dependencies=["crossref"]), REF)


def test_a_network_failure_is_searched_again():
    discovery = {"outcome": "unlocated_after_search", "attempts": [
        {"route_category": "student_url", "outcome": "operational_failure"}]}
    assert not _reusable({"status": "unavailable", "reason_code": "source_not_found", "reference_discovery": discovery})
    down = {"outcome": "unlocated_after_search", "attempts": [
        {"route_category": "academic_adapter", "outcome": "operational_failure"},
        {"route_category": "academic_adapter", "outcome": "operational_failure"},
        {"route_category": "academic_adapter", "outcome": "no_match"}]}
    assert not _reusable({"status": "unavailable", "reason_code": "source_not_found", "reference_discovery": down})
    # A confirmed work whose student link failed is not searched again.
    confirmed = {"outcome": "confirmed", "attempts": [{"route_category": "student_url", "outcome": "operational_failure"}]}
    assert _reusable({"status": "abstract_only", "reason_code": "full_text_unavailable", "reference_discovery": confirmed})


def test_reused_link_checks_are_bound_to_the_new_reference():
    from app.services.schemas import ParsedReference
    from app.services.submitted_links import initial_observations
    old = ParsedReference(reference_id="ref-old", raw_ref="Smith, J. (2020). A study. https://doi.org/10.1/x",
                          author="Smith, J.", year="2020", title="A study", doi="10.1/x")
    new = old.model_copy(update={"reference_id": "ref-new"})
    rows = [row.model_dump(mode="json") for row in initial_observations(old)]
    item = {"reference_id": "ref-old", "status": "unavailable", "submitted_link_observations": rows,
            "reference_discovery": {"reference_id": "ref-old", "outcome": "unlocated_after_search"}}
    index = {reference_signature(new, identity_only=False): ("job-1", item)}
    record = reused_result(index, new, identity_only=False)
    bound = {row.reference_snapshot_sha256 for row in initial_observations(new)}
    assert all(row["reference_id"] == "ref-new" for row in record["submitted_link_observations"])
    assert {row["reference_snapshot_sha256"] for row in record["submitted_link_observations"]} == bound


def test_a_verified_open_access_copy_is_downloaded_again_instead_of_searched():
    # paper-search-reuse-v4, owner decision 2026-10-07.
    from app.services.source_reuse import public_copy
    access = {"version": "verified-public-source-access-v1", "href": "https://repository.example/ames.pdf",
              "content_sha256": "a" * 64}
    item = {"status": "transient_authorized", "source_name": "openalex", "public_source_access": access}
    assert _reusable(item) and public_copy(item)["href"] == access["href"]
    assert not _reusable({**item, "public_source_access": {**access, "href": "http://repository.example/ames.pdf"}})
    assert not _reusable({"status": "transient_authorized", "source_name": "openalex"})


def test_a_downloaded_copy_must_be_the_file_the_earlier_run_verified(monkeypatch):
    import hashlib
    from app.services.source_resolver import SourceResolver
    from app.services.retrieval.base import SourceRepresentation, RepresentationKind
    resolver = SourceResolver.__new__(SourceResolver)
    content = b"%PDF-1.7 the verified article"

    def acquire(self, result, **kwargs):
        result.set_representation(SourceRepresentation(kind=RepresentationKind.PDF, media_type="application/pdf",
                                                       content=content, source_url=result.locations[0].url))
        result.metadata = {**(result.metadata or {}), "identity_confidence": "high"}
        return True
    monkeypatch.setattr(SourceResolver, "_acquire_from_locations", acquire)
    reference = SimpleNamespace(title="Ports", author="Ames, A.", year="2023", doi="", source_kind="journal_article",
                                source_kind_confidence="high", source_kind_evidence=[], raw_ref="Ames, A. (2023). Ports.", url=None)
    same = resolver.refetch_public_source(reference, "https://repository.example/ames.pdf", provider="openalex",
                                          content_sha256=hashlib.sha256(content).hexdigest())
    assert same is not None and same.representation.content == content
    assert resolver.refetch_public_source(reference, "https://repository.example/ames.pdf", provider="openalex",
                                          content_sha256="b" * 64) is None


def test_a_copy_whose_admission_lapsed_mid_run_is_downloaded_again():
    # The long article, 2026-10-08: texts verified, then lost when a 58-minute run outlived their lease.
    from app.services.source_reuse import downloadable
    access = {"version": "verified-public-source-access-v1", "href": "https://repository.example/ames.pdf",
              "content_sha256": "a" * 64}
    lapsed = {"status": "unavailable", "reason_code": "transient_verification_run_unavailable",
              "source_name": "openalex", "public_source_access": access}
    assert downloadable(lapsed) and _reusable(lapsed)
    assert not _reusable({**lapsed, "public_source_access": None})
    assert not downloadable({**lapsed, "reason_code": "source_not_found"})


@pytest.mark.parametrize("same_bytes", [True, False])
def test_a_redownload_identical_to_the_verified_copy_keeps_its_identity(monkeypatch, same_bytes):
    # A re-download has no search record, so its provider identity is low; a
    # byte-identical file is the copy an earlier run verified (2026-10-09).
    import hashlib
    from types import SimpleNamespace
    import fitz
    import app.services.source_resolver as module
    from app.services.file_safety import SafetyVerdict
    from app.services.retrieval.base import RepresentationKind, RetrievalResult, SourceRepresentation
    from app.services.source_type import SourceKindAssessment
    doc = fitz.open(); doc.new_page().insert_text((72, 72), "Harbour cranes in river ports. 2020."); content = doc.tobytes()
    # No author on the page: a title page naming title and author is the work on its own (2026-10-14).
    resolver = object.__new__(module.SourceResolver)
    monkeypatch.setattr(resolver, "_durable_repository_active", lambda: True)
    monkeypatch.setattr(resolver, "_retain_provisional_candidate", lambda *a, **k: None)
    monkeypatch.setattr(module, "inspect_uploaded_pdf", lambda _: SimpleNamespace(verdict=SafetyVerdict.CLEAN, findings=[]))
    monkeypatch.setattr(module, "validate_retrieved_pdf", lambda *a, **k: SimpleNamespace(
        identity_confidence="medium", completeness="complete", text_quality="digital",
        reason="fixture", source_kind_verdict="compatible", observed_source_kind="journal_article"))
    earlier = hashlib.sha256(content if same_bytes else b"other").hexdigest()
    result = RetrievalResult(source_name="test", success=True, metadata={"earlier_verified_content_sha256": earlier},
        representation=SourceRepresentation(kind=RepresentationKind.PDF, media_type="application/pdf",
                                            content=content, source_url="https://ports.example/ames.pdf"))
    accepted, _, _ = resolver._preflight_acquired_representation(
        result, expected_doi=None, expected_title="Harbour cranes in river ports", expected_author="Ames",
        expected_year="2020", expected_source_kind=SourceKindAssessment("journal_article", "high"))
    assert accepted is same_bytes
    assert (result.metadata["identity_confidence"] == "high") is same_bytes


def test_another_paper_citing_the_same_work_in_its_own_format_reuses_the_result():
    from app.services.search.rerun_reuse import parsed_signature, reference_signature
    parsed = dict(source_kind="journal_article", title="Harbour cranes in river ports", author="Ames, A.",
                  year="2020", container_title="Port Studies", doi="10.1234/ports.1")
    mine = SimpleNamespace(raw_ref="Ames, A. (2020). Harbour cranes in river ports. Port Studies.",
                           reference_id="ref-0001-new", **parsed)
    theirs = {"raw_ref": "AMES A 2020 'Harbour cranes in river ports' Port Studies", **parsed}
    item = {"reference_id": "ref-0009-old", "status": "abstract_only", "reference_discovery": {}}
    key = reference_signature(theirs, identity_only=False)
    index = {key: ("job-1", item), parsed_signature(key): ("job-1", item)}
    reused = reused_result(index, mine, identity_only=False)
    assert reused["reference_id"] == "ref-0001-new" and reused["search_reuse"]["from_job_id"] == "job-1"
    # A thin parse does not name one work, so only identical reference text matches.
    thin = dict(parsed, doi="", year="")
    index = {parsed_signature(reference_signature({"raw_ref": "x", **thin}, identity_only=False)): ("job-1", item)}
    assert reused_result(index, SimpleNamespace(raw_ref="y", reference_id="r", **thin), identity_only=False) is None


def test_a_run_excluded_from_reuse_contributes_no_results(monkeypatch):
    import hashlib
    from datetime import datetime, timezone
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.models import Base
    from app.models.job import Job, JobStage, JobStatus
    from app.services.search import rerun_reuse

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    item = {"reference_id": "ref-1", "status": "unavailable", "reason_code": "full_text_unavailable",
            "reference_discovery": {"outcome": "unlocated_after_search", "created_at": "t0"}}

    def job(name, version):
        return Job(filename=name, status=JobStatus.COMPLETED, stage=JobStage.COMPLETED, paper_version_id=version,
                   scope_type="personal_owner", scope_id="owner-1", input_sha256=hashlib.sha256(b"x").hexdigest(),
                   input_media_type="application/pdf", input_byte_size=1, input_expires_at=datetime.now(timezone.utc),
                   extraction_payload={"references": [REF]}, source_results=[item])

    with Session(engine) as session:
        earlier, current = job("ports.pdf", "paper:1"), job("ports.pdf", "paper:2")
        session.add_all([earlier, current])
        session.commit()
        assert rerun_reuse.prior_results_index(session, current)
        monkeypatch.setattr(rerun_reuse.settings, "SEARCH_REUSE_EXCLUDED_JOBS", str(earlier.id))
        assert rerun_reuse.prior_results_index(session, current) == {}


def test_lantern_pages_stored_before_page_numbers_are_read_again_once():
    # Owner decision 2026-10-14: Lantern pages now carry printed page numbers,
    # so an earlier stored Lantern text is read again once (free route).
    from app.services.source_reuse import LANTERN_PAGES_VERSION as PAGES_VERSION
    stored = {"status": "durable_authorized", "representation_id": "r", "source_name": "lantern"}
    assert not _reusable(stored)
    assert _reusable({**stored, "lantern_pages": PAGES_VERSION})
    assert _reusable({"status": "durable_authorized", "representation_id": "r", "source_name": "crossref"})
