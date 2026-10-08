"""paper-search-reuse-v1 (owner decision 2026-10-01): a re-run reuses unchanged references' results."""
from types import SimpleNamespace

from app.services.search.rerun_reuse import _reusable, reference_signature, reused_result

REF = {"reference_id": "ref-1", "raw_ref": "Smith, J. (2020). A study. Example Books.", "source_kind": "monograph",
       "title": "A study", "author": "Smith, J.", "year": "2020", "container_title": "", "doi": ""}


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
    import app.services.search.rerun_reuse as reuse
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
    from app.services.search.rerun_reuse import public_copy
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
