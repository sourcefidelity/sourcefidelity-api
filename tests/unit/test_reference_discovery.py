"""Seven-state reference identity and search-completion contract."""

from datetime import datetime, timezone
import hashlib

import pytest

from app.services.reference_discovery import (
    BibliographicFieldComparison,
    ExpectedBibliographicFields,
    ReferenceDiscoveryCandidate,
    ReferenceDiscoveryTrace,
    ReferenceRouteAttempt,
    ReferenceSearchQuery,
    build_reference_discovery_candidate,
    assess_reference_discovery_trace,
    derive_reference_discovery_record,
    required_reference_discovery_routes,
)
from app.services.retrieval.base import RetrievalResult


NOW = datetime.now(timezone.utc)


def _query(category="academic_adapter"):
    value = "title:normalized scholarly work"
    return ReferenceSearchQuery(
        query_id=f"query-{category}",
        route_category=category,
        provider=f"provider-{category}",
        normalized_query=value,
        query_sha256=hashlib.sha256(value.encode()).hexdigest(),
    )


def _attempt(category="academic_adapter", outcome="no_match", *, required=True):
    return ReferenceRouteAttempt(
        attempt_id=f"attempt-{category}",
        route_category=category,
        provider=f"provider-{category}",
        required=required,
        permitted=outcome != "not_permitted",
        query_ids=[f"query-{category}"],
        outcome=outcome,
        started_at=NOW,
        completed_at=NOW if outcome in {"candidate_found", "no_match"} else None,
    )


def _comparison(field, outcome):
    digest = hashlib.sha256(field.encode()).hexdigest()
    return BibliographicFieldComparison(
        field_name=field,
        expected_sha256=digest,
        observed_sha256=digest,
        outcome=outcome,
        reason_code=f"{field}_{outcome}",
    )


def _record(*, expected=None, attempts=None, candidates=None, required=None):
    attempts = (
        attempts
        if attempts is not None
        else [_attempt(outcome="candidate_found" if candidates else "no_match")]
    )
    queries = [_query(attempt.route_category) for attempt in attempts]
    return derive_reference_discovery_record(
        reference_id="ref-0001",
        expected=expected or ExpectedBibliographicFields(
            title="A sufficiently specific scholarly work", authors=["Scholar"]
        ),
        required_route_categories=required or ["academic_adapter"],
        queries=queries,
        attempts=attempts,
        candidates=candidates or [],
        created_at=NOW,
    )


def test_authoritative_compatible_identifier_is_confirmed():
    candidate = ReferenceDiscoveryCandidate(
        candidate_id="candidate-1",
        attempt_id="attempt-academic_adapter",
        provider="provider-academic_adapter",
        authoritative_identifier_match=True,
        comparisons=[_comparison("doi", "agreement")],
    )
    record = _record(
        attempts=[_attempt(outcome="candidate_found")], candidates=[candidate]
    )
    assert record.outcome == "confirmed"
    assert record.contributes_to_neutral_pattern is False


@pytest.mark.parametrize('qualified', [True,False])
def test_catalog_projection_rejects_unqualified_year_without_changing_history(qualified):
    from app.services.reference_discovery import qualify_legacy_catalog_dates
    expected=ExpectedBibliographicFields(title='The collected films',authors=['Writer'],year='1991')
    candidate=build_reference_discovery_candidate(attempt_id='attempt-student_url',provider='student_url_html',expected=expected,
        result=RetrievalResult(source_name='student_url_html',success=False,title='The collected films',authors=['Writer'],year='2023',
            metadata={'web_identity':{'date_method':'catalog_publication_date' if qualified else 'metadata'}}))
    attempt=_attempt('student_url',outcome='candidate_found',required=False).model_copy(update={'provider':'student_url_html'})
    query=_query('student_url').model_copy(update={'provider':'student_url_html'})
    record=derive_reference_discovery_record(reference_id='ref',expected=expected,required_route_categories=[],
        queries=[query],attempts=[attempt],candidates=[candidate]).model_dump(mode='json')
    assert record['outcome']=='bibliographic_conflict'
    corrected=qualify_legacy_catalog_dates(record,'https://archive.org/details/catalog-book')
    assert record['outcome']=='bibliographic_conflict' and record['candidates'][0]['observed']['year']=='2023'
    assert corrected['outcome']==('bibliographic_conflict' if qualified else 'possible_match')
    if not qualified:
        assert corrected['candidates'][0]['observed']['year']==''
        assert corrected['contributes_to_neutral_pattern'] is False


def test_strong_candidate_with_minor_difference_is_confirmed_with_minor_differences():
    candidate = ReferenceDiscoveryCandidate(
        candidate_id="candidate-1",
        attempt_id="attempt-academic_adapter",
        provider="provider-academic_adapter",
        comparisons=[
            _comparison("title", "agreement"),
            _comparison("author", "agreement"),
            _comparison("year", "minor_difference"),
        ],
    )
    assert _record(candidates=[candidate]).outcome == "confirmed_with_minor_differences"


def test_credible_but_underresolved_candidate_is_possible_match():
    candidate = ReferenceDiscoveryCandidate(
        candidate_id="candidate-1",
        attempt_id="attempt-academic_adapter",
        provider="provider-academic_adapter",
        plausible_identity_match=True,
        comparisons=[_comparison("title", "unknown")],
    )
    assert _record(candidates=[candidate]).outcome == "possible_match"


def test_source_kind_agreement_is_not_independent_identity_support():
    candidate = ReferenceDiscoveryCandidate(
        candidate_id="candidate-1", attempt_id="attempt-academic_adapter",
        provider="provider-academic_adapter", plausible_identity_match=True,
        comparisons=[_comparison("title", "agreement"), _comparison("source_kind", "agreement")],
    )
    assert _record(candidates=[candidate]).outcome == "possible_match"


def test_same_year_and_kind_do_not_make_unrelated_title_credible():
    candidate = ReferenceDiscoveryCandidate(
        candidate_id="candidate-1", attempt_id="attempt-academic_adapter",
        provider="provider-academic_adapter",
        comparisons=[_comparison("title", "material_conflict"), _comparison("year", "agreement"), _comparison("source_kind", "agreement")],
    )
    assert not candidate.is_credible


def test_material_field_conflict_is_neutral_bibliographic_conflict():
    candidate = ReferenceDiscoveryCandidate(
        candidate_id="candidate-1",
        attempt_id="attempt-academic_adapter",
        provider="provider-academic_adapter",
        authoritative_identifier_match=True,
        comparisons=[
            _comparison("title", "agreement"),
            _comparison("year", "material_conflict"),
        ],
    )
    record = _record(candidates=[candidate])
    assert record.outcome == "bibliographic_conflict"
    assert record.contributes_to_neutral_pattern is True


def test_only_completed_required_routes_can_be_unlocated_after_search():
    attempts = [
        _attempt("academic_adapter"),
        _attempt("bounded_web"),
    ]
    record = _record(
        attempts=attempts,
        required=["academic_adapter", "bounded_web"],
    )
    assert record.outcome == "unlocated_after_search"
    assert record.contributes_to_neutral_pattern is True


@pytest.mark.parametrize(
    "attempts,required",
    [
        ([_attempt("academic_adapter", "operational_failure")], ["academic_adapter"]),
        ([_attempt("academic_adapter")], ["academic_adapter", "bounded_web"]),
        ([_attempt("library_metadata", "not_permitted")], ["library_metadata"]),
        ([_attempt("academic_adapter", "access_restricted")], ["academic_adapter"]),
    ],
)
def test_failed_unattempted_or_forbidden_required_route_is_search_incomplete(
    attempts, required
):
    record = _record(attempts=attempts, required=required)
    assert record.outcome == "search_incomplete"
    assert record.contributes_to_neutral_pattern is False


def test_optional_unpermitted_library_route_does_not_block_completed_search():
    attempts = [
        _attempt("academic_adapter"),
        _attempt("bounded_web"),
        _attempt("library_metadata", "not_permitted", required=False),
    ]
    record = _record(
        attempts=attempts,
        required=["academic_adapter", "bounded_web"],
    )
    assert record.outcome == "unlocated_after_search"


@pytest.mark.parametrize(
    "execution_outcome",
    [
        "timeout",
        "captcha",
        "operational_failure",
        "access_restricted",
        "rate_limited",
        "response_invalid",
        "budget_skipped",
    ],
)
def test_failed_required_web_query_execution_forces_search_incomplete(
    execution_outcome,
):
    query = _query("bounded_web").model_copy(
        update={
            "execution_provider": "public-search",
            "execution_outcome": execution_outcome,
            "result_count": 0,
        }
    )
    attempt = _attempt("bounded_web")
    record = derive_reference_discovery_record(
        reference_id="ref-web-failure",
        expected=ExpectedBibliographicFields(title="A Scholarly Work"),
        required_route_categories=["bounded_web"],
        queries=[query],
        attempts=[attempt],
        candidates=[],
        created_at=NOW,
    )

    assert record.outcome == "search_incomplete"


def test_unsearchable_reference_is_insufficient_metadata_even_after_route_failure():
    record = _record(
        expected=ExpectedBibliographicFields(title="Untitled"),
        attempts=[_attempt(outcome="operational_failure")],
    )
    assert record.outcome == "insufficient_metadata"
    assert record.contributes_to_neutral_pattern is False


def test_unknown_query_binding_fails_closed():
    attempt = _attempt()
    attempt = attempt.model_copy(update={"query_ids": ["missing-query"]})
    with pytest.raises(ValueError, match="unknown query"):
        derive_reference_discovery_record(
            reference_id="ref-0001",
            expected=ExpectedBibliographicFields(title="Specific scholarly work"),
            required_route_categories=["academic_adapter"],
            queries=[_query()],
            attempts=[attempt],
            candidates=[],
        )


def test_candidate_found_attempt_requires_bound_candidate_record():
    with pytest.raises(ValueError, match="candidate-found"):
        _record(attempts=[_attempt(outcome="candidate_found")])


def test_query_hash_mismatch_fails_closed():
    with pytest.raises(ValueError, match="query hash"):
        ReferenceSearchQuery(
            query_id="query-bad",
            route_category="academic_adapter",
            provider="crossref",
            normalized_query="title:work",
            query_sha256="0" * 64,
        )


def test_scholarly_route_policy_is_declared_before_adapters_execute():
    expected = ExpectedBibliographicFields(
        title="A Scholarly Work",
        source_kind="journal_article",
    )

    assert required_reference_discovery_routes(expected) == [
        "academic_adapter",
        "bounded_web",
    ]
    assert required_reference_discovery_routes(
        expected, library_metadata_enabled=True
    ) == ["academic_adapter", "bounded_web", "library_metadata"]


def test_non_scholarly_route_policy_remains_unavailable_not_falsely_complete():
    expected = ExpectedBibliographicFields(
        title="A Film Work",
        source_kind="traditional_media",
    )

    assert required_reference_discovery_routes(expected) == []


def test_trace_completion_refuses_missing_route_policy():
    completion = assess_reference_discovery_trace(
        ReferenceDiscoveryTrace(
            reference_id="ref-no-policy",
            expected=ExpectedBibliographicFields(
                title="A Film Work", source_kind="traditional_media"
            ),
        )
    )

    assert completion.ready is False
    assert completion.blocker_codes == ["no_route_policy"]


def test_trace_completion_accepts_complete_no_match_routes():
    queries = [_query("academic_adapter"), _query("bounded_web")]
    queries[1] = queries[1].model_copy(
        update={
            "execution_provider": "public-search",
            "execution_outcome": "no_results",
            "result_count": 0,
        }
    )
    completion = assess_reference_discovery_trace(
        ReferenceDiscoveryTrace(
            reference_id="ref-complete-miss",
            expected=ExpectedBibliographicFields(
                title="A Sufficiently Specific Scholarly Work",
                source_kind="journal_article",
            ),
            required_route_categories=["academic_adapter", "bounded_web"],
            queries=queries,
            attempts=[_attempt("academic_adapter"), _attempt("bounded_web")],
        ),
        created_at=NOW,
    )

    assert completion.ready is True
    assert completion.record is not None
    assert completion.record.outcome == "unlocated_after_search"


def test_trace_completion_refuses_unknown_web_execution_provenance():
    completion = assess_reference_discovery_trace(
        ReferenceDiscoveryTrace(
            reference_id="ref-web-unknown",
            expected=ExpectedBibliographicFields(
                title="A Sufficiently Specific Scholarly Work",
                source_kind="journal_article",
            ),
            required_route_categories=["bounded_web"],
            queries=[_query("bounded_web")],
            attempts=[_attempt("bounded_web")],
        )
    )

    assert completion.ready is False
    assert completion.blocker_codes == ["web_execution_provenance_incomplete"]


def test_resolver_candidate_normalizes_field_agreements_without_raw_values():
    candidate = build_reference_discovery_candidate(
        attempt_id="attempt-1",
        provider="crossref",
        expected=ExpectedBibliographicFields(
            title="A Scholarly Work",
            authors=["Scholar, Jane"],
            year="2024a",
            doi="https://doi.org/10.1234/WORK",
        ),
        result=RetrievalResult(
            source_name="crossref",
            success=True,
            title="A scholarly work",
            authors=["Jane Scholar"],
            year="2024",
            doi="10.1234/work",
        ),
    )

    assert candidate.authoritative_identifier_match is True
    assert {item.field_name: item.outcome for item in candidate.comparisons} == {
        "doi": "agreement",
        "title": "agreement",
        "author": "agreement",
        "year": "agreement",
    }
    assert all(
        item.expected_sha256 is None or len(item.expected_sha256) == 64
        for item in candidate.comparisons
    )


def test_resolver_candidate_preserves_material_title_and_year_conflicts():
    candidate = build_reference_discovery_candidate(
        attempt_id="attempt-1",
        provider="openalex",
        expected=ExpectedBibliographicFields(
            title="Specific Film Policy Study",
            authors=["Scholar"],
            year="2024",
        ),
        result=RetrievalResult(
            source_name="openalex",
            success=True,
            title="Unrelated Agricultural Study",
            authors=["Different Author"],
            year="2018",
        ),
    )

    comparisons = {item.field_name: item.outcome for item in candidate.comparisons}
    assert comparisons["title"] == "material_conflict"
    assert comparisons["author"] == "material_conflict"
    assert comparisons["year"] == "material_conflict"
    assert candidate.has_material_conflict is True


def test_location_candidate_binds_acquisition_result_without_raw_validation_reason():
    candidate = build_reference_discovery_candidate(
        attempt_id="attempt-web",
        provider="web_search",
        expected=ExpectedBibliographicFields(title="Expected Work"),
        result=RetrievalResult(
            source_name="web_search",
            success=True,
            title="A Different Work",
        ),
        candidate_key="ranked-location-1",
        location_url="https://example.org/different.pdf",
        acquisition_outcome="identity_rejected",
        validation_reason="content identity rejected",
    )

    assert candidate.location_available is True
    assert candidate.location_sha256 == hashlib.sha256(
        b"https://example.org/different.pdf"
    ).hexdigest()
    assert candidate.acquisition_outcome == "identity_rejected"
    assert candidate.validation_reason_sha256 == hashlib.sha256(
        b"content identity rejected"
    ).hexdigest()
    assert candidate.observed.title == "A Different Work"
    assert candidate.comparisons[0].outcome == "material_conflict"


def test_content_identity_rejection_overrides_plausible_search_title():
    candidate = build_reference_discovery_candidate(
        attempt_id="attempt-web",
        provider="web_search",
        expected=ExpectedBibliographicFields(title="Expected Work"),
        result=RetrievalResult(
            source_name="web_search",
            success=True,
            title="Expected Work",
        ),
        location_url="https://example.org/wrong.pdf",
        acquisition_outcome="identity_rejected",
    )

    assert candidate.plausible_identity_match is True
    assert candidate.is_credible is False
