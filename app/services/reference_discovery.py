"""Typed, fail-closed reference discovery and search-completion outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
import unicodedata
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator

from app.services.relevance import extract_surnames, score_title_relevance, verify_authors
from app.services.retrieval.base import RetrievalResult


ReferenceDiscoveryOutcome = Literal[
    "confirmed",
    "confirmed_with_minor_differences",
    "possible_match",
    "bibliographic_conflict",
    "unlocated_after_search",
    "insufficient_metadata",
    "search_incomplete",
]
RouteCategory = Literal[
    "durable_repository",
    "student_url",
    "academic_adapter",
    "library_metadata",
    "bounded_web",
]

_SCHOLARLY_DISCOVERY_KINDS = frozenset(
    {
        "journal_article",
        "book_review",
        "monograph",
        "edited_collection",
        "book_section",
        "conference_paper",
        "report",
        "thesis",
        "dataset",
        "software",
        "unknown",
    }
)
RouteOutcome = Literal[
    "candidate_found",
    "no_match",
    "operational_failure",
    "access_restricted",
    "unavailable",
    "not_permitted",
]
FieldComparisonOutcome = Literal[
    "agreement",
    "minor_difference",
    "material_conflict",
    "unknown",
]
SearchExecutionOutcome = Literal[
    "results",
    "no_results",
    "timeout",
    "captcha",
    "operational_failure",
    "access_restricted",
    "rate_limited",
    "response_invalid",
    "budget_skipped",
    "cooldown_skipped",
    "recovery_probe_in_progress",
    "unknown",
]
ReferenceDiscoveryCompletionBlocker = Literal[
    "no_route_policy",
    "candidate_binding_incomplete",
    "web_execution_provenance_incomplete",
    "web_candidate_provenance_incomplete",
]
CandidateAcquisitionOutcome = Literal[
    "metadata_only",
    "acquired",
    "acquired_fallback",
    "identity_rejected",
    "identity_unconfirmed",
    "completeness_rejected",
    "type_rejected",
    "type_unconfirmed",
    "transport_failure",
    "access_restricted",
    "unavailable",
    "not_attempted",
    "unknown",
]


class ExpectedBibliographicFields(BaseModel):
    title: str = Field(default="", max_length=1_000)
    authors: list[str] = Field(default_factory=list, max_length=64)
    year: str = Field(default="", max_length=40)
    doi: str = Field(default="", max_length=255)
    isbn: str = Field(default="", max_length=64)
    container_title: str = Field(default="", max_length=1_000)
    volume: str = Field(default="", max_length=80)
    issue: str = Field(default="", max_length=80)
    pages: str = Field(default="", max_length=100)
    source_kind: str = Field(default="unknown", max_length=100)
    edition_sensitive: bool = False

    def supports_identity_search(self) -> bool:
        if self.doi.strip() or self.isbn.strip():
            return True
        significant_title_words = [
            value for value in self.title.split() if len(value.strip(".,:;!?")) >= 3
        ]
        return len(significant_title_words) >= 2


def is_edition_sensitive_reference(expected: ExpectedBibliographicFields, raw_reference: str = "") -> bool:
    if expected.edition_sensitive or expected.source_kind in {"monograph", "edited_collection", "book_section"} or expected.isbn:
        return True
    # Older parsed book entries may have been labelled webpages because of a
    # trailing URL. A conventional place: publisher clause supplies independent
    # bibliographic structure; the hosting domain alone never establishes kind.
    raw = re.split(r"\bRetrieved\s+from\b|https?://", raw_reference, maxsplit=1, flags=re.I)[0].strip()
    return bool(re.search(r"\.\s+[A-Z][A-Za-z .,'’\-]{1,55}:\s+[A-Z][A-Za-z .,&'’\-]{1,70}\.?$", raw))


def _same_edition_isbn(expected: str, observed: str) -> bool:
    from app.services.book_metadata import normalize_isbn, isbn10_to_isbn13
    def canonical(value):
        value = normalize_isbn(value or "")
        return isbn10_to_isbn13(value) if value and len(value) == 10 else value
    return bool(canonical(expected) and canonical(expected) == canonical(observed))


def required_reference_discovery_routes(
    expected: ExpectedBibliographicFields,
    *,
    library_metadata_enabled: bool = False,
) -> list[RouteCategory]:
    """Declare the prototype's required routes independently of execution.

    The current bounded-web adapter is a scholarly-work discovery route. Web,
    social, audiovisual, traditional-media and archival identities need their
    own fit-for-purpose route policy before they can receive completed outcomes.
    """
    if expected.source_kind not in _SCHOLARLY_DISCOVERY_KINDS:
        return []
    routes: list[RouteCategory] = ["academic_adapter", "bounded_web"]
    if library_metadata_enabled:
        routes.append("library_metadata")
    return routes


class ReferenceSearchQuery(BaseModel):
    query_id: str = Field(min_length=1, max_length=128)
    route_category: RouteCategory
    provider: str = Field(min_length=1, max_length=100)
    execution_provider: str | None = Field(default=None, max_length=100)
    execution_outcome: SearchExecutionOutcome = "unknown"
    result_count: int | None = Field(default=None, ge=0)
    normalized_query: str = Field(min_length=1, max_length=2_000)
    query_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _query_hash_matches_text(self):
        digest = hashlib.sha256(self.normalized_query.encode("utf-8")).hexdigest()
        if digest != self.query_sha256:
            raise ValueError("Reference search query hash does not match")
        return self


class ReferenceRouteAttempt(BaseModel):
    attempt_id: str = Field(min_length=1, max_length=128)
    route_category: RouteCategory
    provider: str = Field(min_length=1, max_length=100)
    required: bool
    permitted: bool
    query_ids: list[str] = Field(default_factory=list, max_length=32)
    outcome: RouteOutcome
    reason_code: str | None = Field(default=None, max_length=100)
    started_at: datetime
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def _terminal_attempt_has_consistent_permission(self):
        if not self.permitted and self.outcome != "not_permitted":
            raise ValueError("An unpermitted route must record not_permitted")
        if self.permitted and self.outcome == "not_permitted":
            raise ValueError("A permitted route cannot record not_permitted")
        if self.outcome in {"candidate_found", "no_match"} and self.completed_at is None:
            raise ValueError("A completed search route requires completed_at")
        return self


class BibliographicFieldComparison(BaseModel):
    field_name: Literal[
        "title",
        "author",
        "year",
        "doi",
        "isbn",
        "container_title",
        "volume",
        "issue",
        "pages",
        "source_kind",
    ]
    expected_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    observed_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    outcome: FieldComparisonOutcome
    reason_code: str = Field(min_length=1, max_length=100)


class BookEditionMetadata(BaseModel):
    """Observed catalog record, never a claim that full book text was acquired."""
    volume_id: str = Field(min_length=1, max_length=255)
    publisher: str = Field(default="", max_length=1000)
    published_date: str = Field(default="", max_length=40)
    identifiers: list[str] = Field(default_factory=list, max_length=16)
    record_sha256: str = Field(min_length=64, max_length=64)
    edition_binding: Literal["same_isbn", "unresolved"] = "unresolved"


class ReferenceDiscoveryCandidate(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=100)
    discovery_provider: str | None = Field(default=None, max_length=100)
    observed: ExpectedBibliographicFields = Field(
        default_factory=ExpectedBibliographicFields
    )
    authoritative_identifier_match: bool = False
    plausible_identity_match: bool = False
    comparisons: list[BibliographicFieldComparison] = Field(
        default_factory=list, max_length=16
    )
    location_available: bool = False
    access_restricted: bool = False
    location_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    location_rank: int | None = Field(default=None, ge=1, le=10_000)
    origin_providers: list[str] = Field(default_factory=list, max_length=16)
    acquisition_outcome: CandidateAcquisitionOutcome = "metadata_only"
    disposition_reason_code: str | None = Field(default=None, max_length=100)
    validation_reason_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    edition_metadata: BookEditionMetadata | None = None

    @property
    def has_material_conflict(self) -> bool:
        return any(item.outcome == "material_conflict" for item in self.comparisons)

    @property
    def has_minor_difference(self) -> bool:
        return any(item.outcome == "minor_difference" for item in self.comparisons)

    @property
    def agreement_count(self) -> int:
        return sum(
            item.outcome == "agreement" and item.field_name in {"doi", "isbn", "title", "author", "year"}
            for item in self.comparisons
        )

    @property
    def is_credible(self) -> bool:
        if self.acquisition_outcome == "identity_rejected":
            return False
        return (
            self.authoritative_identifier_match
            or self.plausible_identity_match
            or (
                self.agreement_count >= 2
                and any(item.field_name == "title" and item.outcome == "agreement" for item in self.comparisons)
            )
        )


class ReferenceDiscoveryRecord(BaseModel):
    record_version: Literal["reference-discovery-v1"] = "reference-discovery-v1"
    reference_id: str = Field(min_length=1, max_length=255)
    created_at: datetime
    expected: ExpectedBibliographicFields
    required_route_categories: list[RouteCategory] = Field(
        default_factory=list, max_length=5
    )
    queries: list[ReferenceSearchQuery] = Field(default_factory=list, max_length=64)
    attempts: list[ReferenceRouteAttempt] = Field(default_factory=list, max_length=128)
    candidates: list[ReferenceDiscoveryCandidate] = Field(
        default_factory=list, max_length=128
    )
    outcome: ReferenceDiscoveryOutcome
    contributes_to_neutral_pattern: bool
    limitations: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def _pattern_flag_matches_outcome(self):
        expected = self.outcome in {
            "bibliographic_conflict",
            "unlocated_after_search",
        }
        if self.contributes_to_neutral_pattern != expected:
            raise ValueError("Pattern eligibility does not match discovery outcome")
        return self


class ReferenceDiscoveryTrace(BaseModel):
    """Pre-outcome resolver evidence; candidates/comparisons may still be open."""

    trace_version: Literal["reference-discovery-trace-v1"] = (
        "reference-discovery-trace-v1"
    )
    reference_id: str = Field(min_length=1, max_length=255)
    expected: ExpectedBibliographicFields
    required_route_categories: list[RouteCategory] = Field(
        default_factory=list, max_length=5
    )
    queries: list[ReferenceSearchQuery] = Field(default_factory=list, max_length=64)
    attempts: list[ReferenceRouteAttempt] = Field(default_factory=list, max_length=128)
    candidates: list[ReferenceDiscoveryCandidate] = Field(
        default_factory=list, max_length=128
    )
    candidates_complete: bool = False
    outcome_derived: bool = False
    limitations: list[str] = Field(default_factory=list, max_length=16)


class ReferenceDiscoveryCompletion(BaseModel):
    ready: bool
    record: ReferenceDiscoveryRecord | None = None
    blocker_codes: list[ReferenceDiscoveryCompletionBlocker] = Field(
        default_factory=list, max_length=8
    )

    @model_validator(mode="after")
    def _ready_requires_record(self):
        if self.ready != (self.record is not None and not self.blocker_codes):
            raise ValueError("Completion readiness does not match record/blockers")
        return self


def assess_reference_discovery_trace(
    trace: ReferenceDiscoveryTrace | dict,
    *,
    created_at: datetime | None = None,
) -> ReferenceDiscoveryCompletion:
    """Convert only provenance-complete traces into the seven-state contract."""
    parsed = (
        trace
        if isinstance(trace, ReferenceDiscoveryTrace)
        else ReferenceDiscoveryTrace.model_validate(trace)
    )
    blockers: list[ReferenceDiscoveryCompletionBlocker] = []
    # Exhaustive-search policy gates absence, not affirmative source identity.
    # A directly cited webpage can be confirmed from observed fields even
    # while a completed-negative policy for that source kind is unsupported.
    if not parsed.required_route_categories and not any(
        candidate.is_credible for candidate in parsed.candidates
    ):
        blockers.append("no_route_policy")

    candidate_attempt_ids = {candidate.attempt_id for candidate in parsed.candidates}
    found_attempt_ids = {
        attempt.attempt_id
        for attempt in parsed.attempts
        if attempt.outcome == "candidate_found"
    }
    if candidate_attempt_ids != found_attempt_ids:
        blockers.append("candidate_binding_incomplete")

    attempts_by_id = {attempt.attempt_id: attempt for attempt in parsed.attempts}
    queries_by_id = {query.query_id: query for query in parsed.queries}
    web_attempts = [
        attempt
        for attempt in parsed.attempts
        if attempt.route_category == "bounded_web"
    ]
    for attempt in web_attempts:
        bound_queries = [
            queries_by_id.get(query_id) for query_id in attempt.query_ids
        ]
        if not bound_queries or any(
            query is None
            or query.execution_provider in {None, "", "unknown"}
            or query.execution_outcome == "unknown"
            for query in bound_queries
        ):
            blockers.append("web_execution_provenance_incomplete")
            break

    for candidate in parsed.candidates:
        attempt = attempts_by_id.get(candidate.attempt_id)
        if attempt and attempt.route_category == "bounded_web" and (
            not candidate.location_sha256
            or candidate.discovery_provider in {None, "", "unknown"}
        ):
            blockers.append("web_candidate_provenance_incomplete")
            break

    blockers = list(dict.fromkeys(blockers))
    if blockers:
        return ReferenceDiscoveryCompletion(
            ready=False,
            blocker_codes=blockers,
        )
    try:
        record = derive_reference_discovery_record(
            reference_id=parsed.reference_id,
            expected=parsed.expected,
            required_route_categories=parsed.required_route_categories,
            queries=parsed.queries,
            attempts=parsed.attempts,
            candidates=parsed.candidates,
            created_at=created_at,
        )
    except ValueError:
        return ReferenceDiscoveryCompletion(
            ready=False,
            blocker_codes=["candidate_binding_incomplete"],
        )
    return ReferenceDiscoveryCompletion(ready=True, record=record)


def build_reference_discovery_candidate(
    *,
    attempt_id: str,
    provider: str,
    expected: ExpectedBibliographicFields,
    result: RetrievalResult,
    candidate_key: str | None = None,
    location_url: str | None = None,
    acquisition_outcome: CandidateAcquisitionOutcome = "metadata_only",
    validation_reason: str | None = None,
    access_restricted: bool = False,
    discovery_provider: str | None = None,
    location_rank: int | None = None,
    origin_providers: list[str] | None = None,
    disposition_reason_code: str | None = None,
) -> ReferenceDiscoveryCandidate:
    """Normalize one resolver candidate into inspectable field comparisons."""
    comparisons: list[BibliographicFieldComparison] = []
    observed_isbn = str((result.metadata or {}).get("isbn") or "")
    edition_match = _same_edition_isbn(expected.isbn, observed_isbn)
    edition_metadata = None
    if (result.metadata or {}).get("book_edition_metadata"):
        edition_metadata = BookEditionMetadata.model_validate(result.metadata["book_edition_metadata"])
        edition_metadata = edition_metadata.model_copy(update={
            "edition_binding": "same_isbn" if edition_match else "unresolved",
        })
    if expected.isbn or observed_isbn:
        comparisons.append(_comparison("isbn", expected.isbn, observed_isbn,
            "agreement" if edition_match else "unknown",
            "exact_edition_isbn_match" if edition_match else "edition_isbn_unresolved"))
    expected_doi = _normalize_doi(expected.doi)
    observed_doi = _normalize_doi(result.doi or "")
    authoritative_identifier_match = bool(
        expected_doi and observed_doi and expected_doi == observed_doi
    )
    if expected_doi or observed_doi:
        doi_conflict = bool(expected_doi and observed_doi and not authoritative_identifier_match)
        comparisons.append(
            _comparison(
                "doi",
                expected.doi,
                result.doi or "",
                "agreement"
                if authoritative_identifier_match
                else "material_conflict"
                if doi_conflict
                else "unknown",
                "exact_doi_match"
                if authoritative_identifier_match
                else "doi_conflict"
                if doi_conflict
                else "doi_missing_on_one_side",
            )
        )

    title_relevant = False
    if expected.title or result.title:
        if expected.title and result.title:
            if _normalize_text(expected.title) == _normalize_text(result.title):
                title_outcome: FieldComparisonOutcome = "agreement"
                title_reason = "normalized_title_match"
                title_relevant = True
            else:
                relevance = score_title_relevance(expected.title, result.title)
                title_relevant = relevance.is_relevant
                title_outcome = (
                    "minor_difference" if relevance.is_relevant else "material_conflict"
                )
                title_reason = (
                    "compatible_title_variant"
                    if relevance.is_relevant
                    else "title_identity_conflict"
                )
        else:
            title_outcome = "unknown"
            title_reason = "title_missing_on_one_side"
        comparisons.append(
            _comparison(
                "title",
                expected.title,
                result.title or "",
                title_outcome,
                title_reason,
            )
        )

    if expected.authors or result.authors:
        if expected.authors and result.authors:
            passes, _score, _detail = verify_authors(expected.authors, result.authors)
            expected_surnames = {
                surname
                for author in expected.authors
                for surname in extract_surnames(author)
            }
            observed_surnames = {
                surname
                for author in result.authors
                for surname in extract_surnames(author)
            }
            if expected_surnames and expected_surnames == observed_surnames:
                author_outcome: FieldComparisonOutcome = "agreement"
                author_reason = "normalized_author_match"
            elif passes:
                author_outcome = "minor_difference"
                author_reason = "compatible_author_variant"
            else:
                author_outcome = "material_conflict"
                author_reason = "author_identity_conflict"
        else:
            author_outcome = "unknown"
            author_reason = "author_missing_on_one_side"
        comparisons.append(
            _comparison(
                "author",
                " | ".join(expected.authors),
                " | ".join(result.authors),
                author_outcome,
                author_reason,
            )
        )

    if expected.year or result.year:
        expected_year = _year(expected.year)
        observed_year = _year(result.year or "")
        if expected_year and observed_year:
            year_outcome: FieldComparisonOutcome = (
                "agreement" if expected_year == observed_year else "material_conflict"
            )
            year_reason = (
                "normalized_year_match"
                if expected_year == observed_year
                else "year_identity_conflict"
            )
        else:
            year_outcome = "unknown"
            year_reason = "year_missing_on_one_side"
        if (result.metadata or {}).get("web_identity", {}).get("date_method") == "catalog_publication_date" and observed_year:
            year_reason = "catalog_publication_year_" + year_outcome
        if year_outcome == "material_conflict" and is_edition_sensitive_reference(expected) and not edition_match:
            year_outcome, year_reason = "unknown", "book_edition_year_unresolved"
        comparisons.append(
            _comparison(
                "year",
                expected.year,
                result.year or "",
                year_outcome,
                year_reason,
            )
        )

    metadata = result.metadata or {}
    observed_kind = str(
        metadata.get("observed_source_kind")
        or metadata.get("source_kind")
        or "unknown"
    )
    if expected.source_kind != "unknown" or observed_kind != "unknown":
        if expected.source_kind == "unknown" or observed_kind == "unknown":
            kind_outcome: FieldComparisonOutcome = "unknown"
            kind_reason = "source_kind_missing_on_one_side"
        elif expected.source_kind == observed_kind:
            kind_outcome = "agreement"
            kind_reason = "source_kind_match"
        else:
            kind_outcome = (
                "material_conflict"
                if metadata.get("source_kind_verdict") == "incompatible"
                else "unknown"
            )
            kind_reason = (
                "source_kind_conflict"
                if kind_outcome == "material_conflict"
                else "source_kind_compatibility_unresolved"
            )
        comparisons.append(
            _comparison(
                "source_kind",
                expected.source_kind,
                observed_kind,
                kind_outcome,
                kind_reason,
            )
        )

    identity_seed = "\x1f".join(
        (
            provider,
            attempt_id,
            result.doi or "",
            result.title or "",
            result.year or "",
            "|".join(result.authors),
            candidate_key or "",
        )
    )
    return ReferenceDiscoveryCandidate(
        candidate_id=f"candidate-{hashlib.sha256(identity_seed.encode('utf-8')).hexdigest()[:24]}",
        attempt_id=attempt_id,
        provider=provider,
        discovery_provider=discovery_provider,
        observed=ExpectedBibliographicFields(
            title=result.title or "",
            authors=list(result.authors),
            year=result.year or "",
            doi=result.doi or "",
            isbn=observed_isbn,
            source_kind=observed_kind,
        ),
        edition_metadata=edition_metadata,
        authoritative_identifier_match=authoritative_identifier_match,
        plausible_identity_match=authoritative_identifier_match or title_relevant,
        comparisons=comparisons,
        location_available=bool(
            location_url
            or result.locations
            or result.full_text_url
            or result.representation
        ),
        access_restricted=access_restricted,
        location_sha256=(
            hashlib.sha256(location_url.encode("utf-8")).hexdigest()
            if location_url
            else None
        ),
        location_rank=location_rank,
        origin_providers=list(dict.fromkeys(origin_providers or [provider])),
        acquisition_outcome=acquisition_outcome,
        disposition_reason_code=disposition_reason_code,
        validation_reason_sha256=(
            hashlib.sha256(validation_reason.encode("utf-8")).hexdigest()
            if validation_reason
            else None
        ),
    )


def _comparison(
    field_name: str,
    expected: str,
    observed: str,
    outcome: FieldComparisonOutcome,
    reason_code: str,
) -> BibliographicFieldComparison:
    return BibliographicFieldComparison(
        field_name=field_name,
        expected_sha256=_value_hash(expected) if expected else None,
        observed_sha256=_value_hash(observed) if observed else None,
        outcome=outcome,
        reason_code=reason_code,
    )


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_doi(value: str) -> str:
    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip()
    return normalized


def _year(value: str) -> str:
    match = re.search(r"(?:19|20)\d{2}", value)
    return match.group() if match else ""


def _value_hash(value: str) -> str:
    return hashlib.sha256(_normalize_text(value).encode("utf-8")).hexdigest()


def qualify_legacy_catalog_dates(discovery: dict | None, reference_url: str | None) -> dict | None:
    """Quarantine old inferred catalog-page years without inventing publication dates.

    This is a new report projection over preserved discovery provenance, not
    a new search result or a rewrite of the immutable evidence records.
    """
    parsed = urlsplit(reference_url or "")
    if not discovery or parsed.hostname not in {"archive.org", "www.archive.org"} or not parsed.path.startswith("/details/"):
        return discovery
    record = ReferenceDiscoveryRecord.model_validate(discovery)
    corrected = []
    changed = False
    for candidate in record.candidates:
        comparisons = list(candidate.comparisons)
        if candidate.provider == "student_url_html":
            for index, comparison in enumerate(comparisons):
                if (comparison.field_name == "year" and comparison.outcome != "unknown"
                        and not comparison.reason_code.startswith("catalog_publication_year_")):
                    comparisons[index] = comparison.model_copy(update={
                        "outcome": "unknown", "observed_sha256": None,
                        "reason_code": "catalog_page_year_not_publication_qualified",
                    })
                    candidate = candidate.model_copy(update={
                        "observed": candidate.observed.model_copy(update={"year": ""}),
                    })
                    changed = True
        corrected.append(candidate.model_copy(update={"comparisons": comparisons}))
    if not changed:
        return discovery
    result = derive_reference_discovery_record(
        reference_id=record.reference_id,expected=record.expected,
        required_route_categories=record.required_route_categories,queries=record.queries,
        attempts=record.attempts,candidates=corrected,created_at=record.created_at,
    )
    # Removing an unreliable year cannot establish a newly confirmed edition.
    if result.outcome in {"confirmed", "confirmed_with_minor_differences", "unlocated_after_search"}:
        result = result.model_copy(update={"outcome":"possible_match", "contributes_to_neutral_pattern":False})
    return {**result.model_dump(mode="json"), "limitations": [*result.limitations,
        "The catalog webpage year was not verified as the work's publication year; that date comparison is not assessed."]}


def qualify_book_edition_years(discovery: dict | None, raw_reference: str = "") -> dict | None:
    """Withhold unbound edition-year conflicts, retaining observed values/provenance."""
    if not discovery or not any(
        item.get("field_name") == "year" and item.get("outcome") == "material_conflict"
        for candidate in discovery.get("candidates") or [] for item in candidate.get("comparisons") or []
    ):
        return discovery
    record = ReferenceDiscoveryRecord.model_validate(discovery)
    if not is_edition_sensitive_reference(record.expected, raw_reference):
        return discovery
    changed, candidates = False, []
    for candidate in record.candidates:
        comparisons = []
        for comparison in candidate.comparisons:
            if (comparison.field_name == "year" and comparison.outcome == "material_conflict"
                    and not _same_edition_isbn(record.expected.isbn, candidate.observed.isbn)):
                comparison = comparison.model_copy(update={"outcome": "unknown", "reason_code": "book_edition_year_unresolved"})
                changed = True
            comparisons.append(comparison)
        candidates.append(candidate.model_copy(update={"comparisons": comparisons}))
    if not changed:
        return discovery
    result = derive_reference_discovery_record(reference_id=record.reference_id, expected=record.expected,
        required_route_categories=record.required_route_categories, queries=record.queries,
        attempts=record.attempts, candidates=candidates, created_at=record.created_at)
    if result.outcome in {"confirmed", "confirmed_with_minor_differences", "unlocated_after_search"}:
        result = result.model_copy(update={"outcome": "possible_match", "contributes_to_neutral_pattern": False})
    return {**result.model_dump(mode="json"), "limitations": list(dict.fromkeys([
        *record.limitations, *result.limitations,
        "Located book records have different years, but the edition used has not been established. Compare its publication page and ISBN; this is not a confirmed reference error.",
    ]))[:16]}


def derive_reference_discovery_record(
    *,
    reference_id: str,
    expected: ExpectedBibliographicFields,
    required_route_categories: list[RouteCategory],
    queries: list[ReferenceSearchQuery],
    attempts: list[ReferenceRouteAttempt],
    candidates: list[ReferenceDiscoveryCandidate],
    created_at: datetime | None = None,
) -> ReferenceDiscoveryRecord:
    """Derive exactly one outcome without treating failed search as absence."""
    _validate_bindings(required_route_categories, queries, attempts, candidates)
    limitations: list[str] = []
    if any(candidate.edition_metadata and candidate.edition_metadata.edition_binding == "unresolved"
           for candidate in candidates):
        limitations.append(
            "Located book dates describe cataloged editions; the edition used has not been bound by an ISBN. "
            "Compare publication, copyright and reprint information in the copy used before changing its year."
        )
    if not expected.supports_identity_search():
        outcome: ReferenceDiscoveryOutcome = "insufficient_metadata"
        limitations.append("The supplied fields do not support a meaningful identity search.")
    else:
        credible = [candidate for candidate in candidates if candidate.is_credible]
        compatible = [candidate for candidate in credible if not candidate.has_material_conflict]
        conflicts = [candidate for candidate in credible if candidate.has_material_conflict]
        exact = [
            candidate
            for candidate in compatible
            if candidate.authoritative_identifier_match
            and not candidate.has_minor_difference
            and bool(candidate.comparisons)
            and all(
                comparison.outcome == "agreement"
                for comparison in candidate.comparisons
            )
        ]
        strong = [candidate for candidate in compatible if candidate.agreement_count >= 2
                  and (not expected.isbn or _same_edition_isbn(expected.isbn, candidate.observed.isbn))
                  and not any(c.reason_code == "book_edition_year_unresolved" for c in candidate.comparisons)]
        if exact:
            outcome = "confirmed"
        elif strong:
            outcome = (
                "confirmed_with_minor_differences"
                if any(candidate.has_minor_difference for candidate in strong)
                else "confirmed"
            )
        elif conflicts:
            outcome = "bibliographic_conflict"
        elif compatible:
            outcome = "possible_match"
        elif _search_is_incomplete(required_route_categories, attempts, queries):
            outcome = "search_incomplete"
            limitations.append(
                "At least one required search route failed, was unavailable, or did not complete."
            )
        else:
            outcome = "unlocated_after_search"
            limitations.append(
                "Every required and permitted route completed without a credible match."
            )
    return ReferenceDiscoveryRecord(
        reference_id=reference_id,
        created_at=created_at or datetime.now(timezone.utc),
        expected=expected,
        required_route_categories=required_route_categories,
        queries=queries,
        attempts=attempts,
        candidates=candidates,
        outcome=outcome,
        contributes_to_neutral_pattern=outcome
        in {"bibliographic_conflict", "unlocated_after_search"},
        limitations=limitations,
    )


def _validate_bindings(
    required: list[RouteCategory],
    queries: list[ReferenceSearchQuery],
    attempts: list[ReferenceRouteAttempt],
    candidates: list[ReferenceDiscoveryCandidate],
) -> None:
    if len(required) != len(set(required)):
        raise ValueError("Required route categories must be unique")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Reference search query IDs must be unique")
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("Reference route attempt IDs must be unique")
    if any(set(attempt.query_ids) - set(query_ids) for attempt in attempts):
        raise ValueError("Reference route attempt refers to an unknown query")
    queries_by_id = {query.query_id: query for query in queries}
    for attempt in attempts:
        for query_id in attempt.query_ids:
            query = queries_by_id[query_id]
            if (
                query.route_category != attempt.route_category
                or query.provider != attempt.provider
            ):
                raise ValueError("Reference route attempt/query binding is inconsistent")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Reference candidate IDs must be unique")
    attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    candidate_attempt_ids = {candidate.attempt_id for candidate in candidates}
    found_attempt_ids = {
        attempt.attempt_id for attempt in attempts if attempt.outcome == "candidate_found"
    }
    if candidate_attempt_ids != found_attempt_ids:
        raise ValueError("Candidate records must match candidate-found route attempts")
    if any(
        candidate.provider != attempts_by_id[candidate.attempt_id].provider
        for candidate in candidates
    ):
        raise ValueError("Candidate provider does not match its route attempt")


def _search_is_incomplete(
    required: list[RouteCategory],
    attempts: list[ReferenceRouteAttempt],
    queries: list[ReferenceSearchQuery],
) -> bool:
    completed = {attempt.route_category for attempt in attempts if attempt.required}
    if set(required) - completed:
        return True
    failure_outcomes = {
        "operational_failure",
        "access_restricted",
        "unavailable",
        "not_permitted",
    }
    if any(
        attempt.required and attempt.outcome in failure_outcomes for attempt in attempts
    ):
        return True
    incomplete_execution_outcomes = {
        "timeout",
        "captcha",
        "operational_failure",
        "access_restricted",
        "rate_limited",
        "response_invalid",
        "budget_skipped",
        "cooldown_skipped",
        "recovery_probe_in_progress",
        "unknown",
    }
    required_attempt_ids = {
        attempt.attempt_id for attempt in attempts if attempt.required
    }
    required_query_ids = {
        query_id
        for attempt in attempts
        if attempt.attempt_id in required_attempt_ids
        for query_id in attempt.query_ids
    }
    return any(
        query.query_id in required_query_ids
        and query.execution_provider is not None
        and query.execution_outcome in incomplete_execution_outcomes
        for query in queries
    )
