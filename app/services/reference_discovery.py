"""Typed, fail-closed reference discovery and search-completion outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import html
import re
import unicodedata
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator, model_serializer

from app.services.schemas import bound_text_fields, declared_max_length, note_bounded

from app.services.relevance import extract_surnames, score_title_relevance, verify_authors
from app.services.bibliographic_scripts import cross_script_comparison_unresolved
from app.services.retrieval.base import OBSERVED_AUTHOR_LIMIT, RetrievalResult
from app.services.search.policy import API_FIRST_SEARCH_POLICY, LEGACY_SEARCH_POLICY, REQUIRED_API_PROVIDERS
from app.services.search.transient import TransientSearchAudit
from app.services.search.candidate_audit import SearchCandidateAudit
from app.services.journal_discovery import JournalCheck
from app.services.metadata_identity import MetadataIdentityObservation, build_observation
from app.services.reference_review_scope import ReviewScreen

SearchPolicyVersion = Literal["configured-search-v1", "api-first-search-v2"]

# These observations preserve a successful HTTP lookup without inventing an
# empty registry or completed identity adjudication of discarded records.
INCOMPLETE_METADATA_SEARCH_REASONS = frozenset({
    'metadata_candidates_filtered', 'metadata_empty_response_unqualified',
})


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
    "candidates_processed",
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
    # The application chose not to call this route (for example, an index that
    # cannot hold the kind of work). Not an outcome of any request.
    "declined",
    "recovery_probe_in_progress",
    # This application raised, not the provider. Recorded distinctly so a bug
    # in our own code never reads as a completed observation about a source.
    "internal_error",
    "unknown",
]
ReferenceDiscoveryCompletionBlocker = Literal[
    "transient_audit_incomplete",
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
    author_normalization_policy_version: Literal['author-etal-comparison-v1'] | None = None

    # Names of fields the application truncated; serialized only when
    # non-empty so an untouched record and its stored hashes are unchanged.
    bounded_fields: list[str] = Field(default_factory=list, max_length=32)

    @model_serializer(mode='wrap')
    def _preserve_legacy_author_policy(self, handler):
        payload = handler(self)
        if 'author_normalization_policy_version' not in self.model_fields_set:
            payload.pop('author_normalization_policy_version', None)
        if not self.bounded_fields:
            payload.pop('bounded_fields', None)
        return payload

    reference_parse_review: bool = False
    title: str = Field(default="", max_length=1_000)
    authors: list[str] = Field(default_factory=list, max_length=OBSERVED_AUTHOR_LIMIT)

    @model_validator(mode="before")
    @classmethod
    def _bound_declared_fields(cls, data):
        """Bound every declared field at its own limit, and say which.

        Authors were bounded first because a real reference overran them, but
        the sibling string fields carried the same arrangement: a hard limit
        and no truncation, so one malformed provider record fails the whole
        paper at persist time. A prefix preserves what identity matching uses -
        these values are compared by token overlap, not by equality - and a
        value past these limits is malformed rather than merely long. Physics
        and genomics papers routinely list hundreds of authors, and the leading
        authors are the identifying ones. Altered fields are recorded in
        `bounded_fields`, so the provider-side caps that used to be
        load-bearing are gone.
        """
        data = bound_text_fields(
            cls, data,
            ("title", "container_title", "publisher", "year", "doi", "isbn",
             "volume", "issue", "pages"),
        )
        if not isinstance(data, dict):
            return data
        authors = data.get("authors")
        if isinstance(authors, list) and len(authors) > OBSERVED_AUTHOR_LIMIT:
            data = dict(data)
            data["authors"] = authors[:OBSERVED_AUTHOR_LIMIT]
            note_bounded(data, "authors")
        return data
    year: str = Field(default="", max_length=40)
    doi: str = Field(default="", max_length=255)
    isbn: str = Field(default="", max_length=64)
    container_title: str = Field(default="", max_length=1_000)
    publisher: str = Field(default="", max_length=300)
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
    bounded_review_screen: ReviewScreen | None = None

    @model_serializer(mode='wrap')
    def _preserve_old_screen_absence(self, handler):
        payload = handler(self)
        if 'bounded_review_screen' not in self.model_fields_set:
            payload.pop('bounded_review_screen', None)
        return payload
    query_id: str = Field(min_length=1, max_length=128)
    route_category: RouteCategory
    provider: str = Field(min_length=1, max_length=100)
    execution_provider: str | None = Field(default=None, max_length=100)
    execution_engine_group: str | None = Field(default=None, max_length=200)
    execution_outcome: SearchExecutionOutcome = "unknown"
    result_count: int | None = Field(default=None, ge=0)
    required: bool | None = None  # Legacy records inherit their parent attempt.
    provider_calls: int | None = Field(default=None, ge=0)
    latency_seconds: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    credits_remaining: float | None = Field(default=None, ge=0)
    cache_hit: bool = False
    reason_code: str | None = Field(default=None, max_length=100)
    normalized_query: str = Field(min_length=1, max_length=2_000)
    query_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _query_hash_matches_text(self):
        digest = hashlib.sha256(self.normalized_query.encode("utf-8")).hexdigest()
        if digest != self.query_sha256:
            raise ValueError("Reference search query hash does not match")
        return self


ROUTE_ERROR_CODES = Literal[
    "connect_error",
    "network_error",
    "timeout",
    "rate_limited",
    "access_restricted",
    "circuit_open",
    "provider_unavailable",
    "http_client_error",
    "http_server_error",
    "invalid_response",
    "unclassified",
]


class SearchMemoReuse(BaseModel):
    """This bounded-web attempt reuses an earlier completed search in this scope.

    `search-reuse-memo-v1` (owner decision 2026-09-29). The queries bound to
    the attempt are the earlier search's own operational records; they were
    not sent again. `searched_at` is when that search ran.
    """
    model_config = {"extra": "forbid"}
    policy_version: Literal["search-reuse-memo-v1"] = "search-reuse-memo-v1"
    memo_id: str = Field(min_length=1, max_length=64)
    searched_at: datetime
    expires_at: datetime
    original_outcome: ReferenceDiscoveryOutcome


class ReferenceRouteAttempt(BaseModel):
    transient_search_audits: list[TransientSearchAudit] = Field(default_factory=list, max_length=8)
    # Development audit only (SEARCH_CANDIDATE_AUDIT_URLS). Never Brave, and
    # never serialized when absent, so ordinary records are byte-identical.
    candidate_audit: SearchCandidateAudit | None = None
    # Set only on an attempt that reused a completed-search memo; never
    # serialized when absent.
    search_memo: SearchMemoReuse | None = None

    @model_serializer(mode='wrap')
    def _omit_absent_candidate_audit(self, handler):
        payload = handler(self)
        if self.candidate_audit is None:
            payload.pop('candidate_audit', None)
        if self.search_memo is None:
            payload.pop('search_memo', None)
        return payload
    attempt_id: str = Field(min_length=1, max_length=128)
    route_category: RouteCategory
    provider: str = Field(min_length=1, max_length=100)
    required: bool
    permitted: bool
    query_ids: list[str] = Field(default_factory=list, max_length=32)
    outcome: RouteOutcome
    reason_code: str | None = Field(default=None, max_length=100)
    # Why the route failed, from a fixed vocabulary. `reason_code` says which
    # branch classified the attempt; this says what actually went wrong, which
    # is the difference between "this machine lost the network" and "this
    # provider is broken". A whole run of failures once carried no such
    # distinction and looked like a provider incident.
    error_code: ROUTE_ERROR_CODES | None = None
    started_at: datetime
    completed_at: datetime | None = None
    timing_version: Literal['retrieval-operation-timing-v1'] | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _terminal_attempt_has_consistent_permission(self):
        if not self.permitted and self.outcome != "not_permitted":
            raise ValueError("An unpermitted route must record not_permitted")
        if self.permitted and self.outcome == "not_permitted":
            raise ValueError("A permitted route cannot record not_permitted")
        if self.outcome in {"candidate_found", "no_match", "candidates_processed"} and self.completed_at is None:
            raise ValueError("A completed search route requires completed_at")
        if self.outcome == "candidates_processed" and not self.transient_search_audits:
            raise ValueError("Transient candidate completion requires balanced audit accounting")
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
        # Publisher joins title, author and year as an identity field: the
        # combination is what distinguishes one work from another with a
        # similar title. The comparison has been built since the merge rule
        # started scoring it; this enum had not been extended to accept it.
        "publisher",
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
    submitted_identifier_location_match: bool | None = None
    metadata_identity: MetadataIdentityObservation | None = None
    # Independently returned registration metadata, not a search-result snippet.
    registration_record_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    registration_observation_sha256: str | None = Field(default=None, min_length=64, max_length=64)

    @model_serializer(mode='wrap')
    def _preserve_historical_registration_absence(self, handler):
        payload = handler(self)
        for field in ('registration_record_sha256', 'registration_observation_sha256', 'metadata_identity', 'submitted_identifier_location_match'):
            if field not in self.model_fields_set:
                payload.pop(field, None)
        return payload
    identity_evidence_kind: Literal["source_representation", "landing_page_metadata", "landing_page_observation", "pdf_front_matter_metadata", "pdf_front_matter_observation"] | None = None
    location_provenance: Literal["discovery_result", "independently_acquired_content"] = "discovery_result"
    validated_identity_content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
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
    # Retired 2026-09-29 and never populated: the application keeps no
    # search-result links. The field stays so stored records remain readable;
    # the development audit of Exa/Tavily candidates is `candidate_audit` on
    # the route attempt (SEARCH_CANDIDATE_AUDIT_URLS).
    development_location_url: str | None = Field(default=None, max_length=2000)
    # Why this candidate reported no title. "We read the page and it names no
    # work" and "we never read the page" are different facts, and the bounded
    # review may only act on the first.
    title_absence_reason: str | None = Field(default=None, max_length=64)
    location_rank: int | None = Field(default=None, ge=1, le=10_000)
    origin_providers: list[str] = Field(default_factory=list, max_length=16)
    acquisition_outcome: CandidateAcquisitionOutcome = "metadata_only"
    disposition_reason_code: str | None = Field(default=None, max_length=100)
    validation_reason_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    edition_metadata: BookEditionMetadata | None = None

    @model_validator(mode="after")
    def _landing_identity_requires_independent_binding(self):
        if self.identity_evidence_kind in {"landing_page_observation", "pdf_front_matter_observation"}:
            if (not self.validated_identity_content_sha256
                    or self.location_provenance != "independently_acquired_content"
                    or not self.location_sha256 or not self.observed.title or not self.observed.authors):
                raise ValueError("Landing observation lacks independent bibliographic binding")
        if self.identity_evidence_kind in {"landing_page_metadata", "pdf_front_matter_metadata"}:
            if (not self.validated_identity_content_sha256
                    or self.location_provenance != "independently_acquired_content"
                    or not self.location_sha256
                    or self.has_material_conflict
                    or not (self.authoritative_identifier_match or self.agreement_count >= 2)
                    or self.has_unresolved_supplied_identity_fields):
                raise ValueError("Landing-page identity lacks independently bound bibliographic agreement")
        return self

    @property
    def has_unresolved_supplied_identity_fields(self) -> bool:
        """The parser's no-date marker is not a supplied publication year.

        Keep its original hash and unknown comparison for audit; it supplies
        neither an agreement nor a conflict. Other missing fields stay closed.
        """
        return any(
            c.expected_sha256
            and c.outcome not in {"agreement", "minor_difference"}
            and not (
                c.field_name == "year"
                and c.expected_sha256 == _value_hash("n.d.")
                and c.outcome == "unknown"
            )
            for c in self.comparisons
        )

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
            bool(self.validated_identity_content_sha256)
            or self.authoritative_identifier_match
            or self.plausible_identity_match
            or (
                self.agreement_count >= 2
                and any(item.field_name == "title" and item.outcome == "agreement" for item in self.comparisons)
            )
        )


class ReferenceDiscoveryRecord(BaseModel):
    search_retention_policy: Literal["brave-operational-transient-v1"] | None = None
    search_policy_version: SearchPolicyVersion = LEGACY_SEARCH_POLICY
    required_web_providers: list[str] = Field(default_factory=list, max_length=8)
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
    candidate_ranking_policy_version: Literal['candidate-purpose-ranking-v1'] | None = None
    search_capacity_policy_version: Literal['search-inspection-capacity-v3', 'search-inspection-capacity-v4'] | None = None
    metadata_screen_resolution_policy_version: Literal['screened-metadata-work-binding-v1'] | None = None
    bounded_review_policy_version: Literal['bounded-reference-review-v1','bounded-reference-review-v2','bounded-reference-review-v3','bounded-reference-review-v4','bounded-reference-review-v5','bounded-reference-review-v6','bounded-reference-review-v7','bounded-reference-review-v8','bounded-reference-review-v9'] | None = None
    metadata_identity_policy_version: Literal['corroborated-metadata-identity-v1'] | None = None
    credibility_policy_version: Literal['reference-credibility-v2', 'reference-credibility-v3', 'reference-credibility-v4'] | None = None
    credibility_reference_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    search_retention_policy: Literal["brave-operational-transient-v1"] | None = None
    """Pre-outcome resolver evidence; candidates/comparisons may still be open."""

    trace_version: Literal["reference-discovery-trace-v1"] = (
        "reference-discovery-trace-v1"
    )
    search_policy_version: SearchPolicyVersion = LEGACY_SEARCH_POLICY
    required_web_providers: list[str] = Field(default_factory=list, max_length=8)
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
    journal_checks: list[JournalCheck] = Field(default_factory=list, max_length=3)

    @model_serializer(mode="wrap")
    def _preserve_legacy_trace(self, handler):
        payload = handler(self)
        for key in ('candidate_ranking_policy_version', 'search_capacity_policy_version'):
            if key not in self.model_fields_set:
                payload.pop(key, None)
        if 'metadata_screen_resolution_policy_version' not in self.model_fields_set:
            payload.pop('metadata_screen_resolution_policy_version', None)
        if 'bounded_review_policy_version' not in self.model_fields_set:
            payload.pop('bounded_review_policy_version', None)
        if 'metadata_identity_policy_version' not in self.model_fields_set:
            payload.pop('metadata_identity_policy_version', None)
        if "credibility_policy_version" not in self.model_fields_set:
            payload.pop("credibility_policy_version", None)
        if "credibility_reference_sha256" not in self.model_fields_set:
            payload.pop("credibility_reference_sha256", None)
        if "journal_checks" not in self.model_fields_set:
            payload.pop("journal_checks", None)
        return payload


class ReferenceDiscoveryCompletion(BaseModel):
    ready: bool
    record: ReferenceDiscoveryRecord | None = None
    blocker_codes: list[ReferenceDiscoveryCompletionBlocker] = Field(
        default_factory=list, max_length=8
    )
    # What actually failed, when the blocker code is a category rather than a
    # cause. A bounded slug, never raw exception text.
    blocker_detail: str | None = Field(default=None, max_length=60)

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
        if parsed.search_retention_policy and any(
            q and q.execution_provider == "brave" and q.execution_outcome == "results"
            for q in bound_queries
        ):
            audits = attempt.transient_search_audits
            independent_count = sum(
                c.attempt_id == attempt.attempt_id
                and c.discovery_provider == "brave"
                and c.location_provenance == "independently_acquired_content"
                and c.identity_evidence_kind not in {"landing_page_observation", "pdf_front_matter_observation"}
                and bool(c.validated_identity_content_sha256)
                for c in parsed.candidates
            )
            if (not audits or sum(a.candidate_count for a in audits) == 0
                    or sum(a.identity_established for a in audits) != independent_count):
                blockers.append("transient_audit_incomplete")
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
            search_policy_version=parsed.search_policy_version,
            search_retention_policy=parsed.search_retention_policy,
            created_at=created_at,
        )
    except ValueError as exc:
        # This was a catch-all: every failure of the record builder reported as
        # incomplete candidate binding. A duplicate Google Books volume id
        # therefore presented as a binding problem and sent a diagnosis to the
        # wrong provider entirely. Name what actually failed.
        reason = re.sub(r"[^a-z0-9]+", "_", str(exc).casefold()).strip("_")[:60]
        return ReferenceDiscoveryCompletion(
            ready=False,
            blocker_codes=["candidate_binding_incomplete"],
            blocker_detail=reason or None,
        )
    return ReferenceDiscoveryCompletion(ready=True, record=record)


# Reasons that mean a page was read and found to name no work. Anything else,
# including an unknown reason, leaves the absence unexplained.
OBSERVED_TITLE_ABSENCE = frozenset({
    "uniform_typography", "front_matter_label_only", "illegible_title",
    "no_title_element", "navigation_listing_page",
})


_MAX_OBSERVED_AUTHORS = 64


def _title_absence_reason(result, acquisition_outcome: str) -> str | None:
    """Explain an empty observed title, preferring what the reader actually saw."""
    if result.title:
        return None
    metadata = result.metadata or {}
    declared = metadata.get("title_reason") or (
        (metadata.get("web_identity") or {}).get("title_reason")
        if isinstance(metadata.get("web_identity"), dict) else None
    )
    if declared:
        return None if declared == "title_observed" else str(declared)[:64]
    if acquisition_outcome == "not_attempted":
        return "lead_not_visited"
    if acquisition_outcome in {"access_restricted", "transport_failure"}:
        return "lead_not_reached"
    representation = getattr(result, "representation", None)
    content = getattr(representation, "content", None)
    if content and str(getattr(representation, "kind", "")).endswith("PDF"):
        from app.services.pdf_verifier import describe_first_page_title

        reason = describe_first_page_title(content).reason
        return None if reason == "title_observed" else reason
    return None


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
    if (result.metadata or {}).get("candidate_budget_skipped"):
        seed = f"{attempt_id}\x1f{provider}\x1f{candidate_key or ''}"
        return ReferenceDiscoveryCandidate(
            candidate_id="candidate-" + hashlib.sha256(seed.encode()).hexdigest()[:24],
            attempt_id=attempt_id, provider=provider,
            observed=ExpectedBibliographicFields(title=result.title or "", authors=list(result.authors or []),
                year=result.year or "", doi=result.doi or ""),
            acquisition_outcome="not_attempted", disposition_reason_code="candidate_budget_exhausted",
            location_available=bool(location_url or result.locations),
            location_sha256=hashlib.sha256(location_url.encode()).hexdigest() if location_url else None,
            discovery_provider=discovery_provider, location_rank=location_rank,
            origin_providers=list(dict.fromkeys(origin_providers or [provider])),
        )
    comparisons: list[BibliographicFieldComparison] = []
    def comparison_authors(values):
        if expected.author_normalization_policy_version != 'author-etal-comparison-v1':
            return values
        return [re.sub(r'(?i),?\s+et\.?\s+al\.?\s*$', '', value).strip() for value in values]
    expected_author_names = comparison_authors(expected.authors)
    observed_author_names = comparison_authors(result.authors)
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
            if cross_script_comparison_unresolved(expected.title, result.title):
                title_outcome = "unknown"
                title_reason = "cross_script_title_unresolved"
            elif _normalize_text(expected.title) == _normalize_text(result.title):
                title_outcome: FieldComparisonOutcome = "agreement"
                title_reason = "normalized_title_match"
                title_relevant = True
            else:
                relevance = score_title_relevance(expected.title, result.title)
                # Semantic proximity finds candidates; it is not bibliographic
                # identity. Only a corroborated subtitle variant qualifies here.
                left, right = _normalize_text(expected.title), _normalize_text(result.title)
                author_match = bool(expected.authors and result.authors and
                    verify_authors(expected_author_names, observed_author_names)[0])
                subtitle_variant = any(
                    ":" in full and _normalize_text(full.split(":", 1)[0]) == short
                    and len(short.split()) >= 4
                    for full, short in ((expected.title, right), (result.title, left))
                )
                title_relevant = author_match and subtitle_variant
                title_outcome = (
                    "minor_difference" if title_relevant else "material_conflict"
                )
                title_reason = (
                    "compatible_title_variant"
                    if title_relevant
                    else "topical_title_only" if relevance.is_relevant else "title_identity_conflict"
                )
                if expected.reference_parse_review:
                    title_outcome = 'unknown'
                    title_reason = 'review_only_title_completeness_unresolved'
                    title_relevant = False
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
            passes, _score, _detail = verify_authors(expected_author_names, observed_author_names)
            expected_surnames = {
                surname
                for author in expected_author_names
                for surname in extract_surnames(author)
            }
            observed_surnames = {
                surname
                for author in observed_author_names
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
    container = metadata.get('container_title') or ''
    pages = metadata.get('pages') or metadata.get('page') or ''
    # The imprint is one of the four identity fields, so the trace has to show
    # whether it agreed. Measured over 9,935 stored candidate records, not one
    # retained a publisher, which is why no merge could ever evidence it.
    from app.services.retrieval.canonical_work import _publisher_from_result
    observed_publisher = _publisher_from_result(result) or ''
    observed_publisher = observed_publisher if len(observed_publisher) <= 300 else ''
    # Crossref's existing adapter retains the deposited message. Only scalar
    # or single-container observations belong in this bounded field contract.
    message = metadata.get('message')
    if not container and isinstance(message, dict):
        titles = message.get('container-title')
        if isinstance(titles, list) and len(titles) == 1 and isinstance(titles[0], str):
            container = titles[0]
    container = container if isinstance(container, str) and len(container) <= 1000 else ''
    pages = pages if isinstance(pages, str) and len(pages) <= 100 else ''
    volume = metadata.get('volume') or ''
    issue = metadata.get('issue') or ''
    for field, observed_value in [('container_title', container), ('pages', pages),
                                  ('volume', volume), ('issue', issue),
                                  ('publisher', observed_publisher)]:
        submitted_value = getattr(expected, field)
        if submitted_value or observed_value:
            agrees = bool(submitted_value and observed_value and _normalize_text(submitted_value) == _normalize_text(observed_value))
            if agrees:
                outcome, reason = 'agreement', 'normalized_component_field_match'
            elif _field_materially_conflicts(field, submitted_value, observed_value):
                # Previously every non-match was recorded as `unknown`, which
                # made `material_conflict` unreachable for these fields and
                # the field-conflict finding unable to fire at all.
                outcome, reason = 'material_conflict', 'component_field_values_disagree'
            else:
                outcome, reason = 'unknown', 'component_field_missing_or_unresolved'
            comparisons.append(_comparison(field, submitted_value, observed_value,
                                           outcome, reason))
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
    candidate = ReferenceDiscoveryCandidate(
        candidate_id=f"candidate-{hashlib.sha256(identity_seed.encode('utf-8')).hexdigest()[:24]}",
        attempt_id=attempt_id,
        provider=provider,
        discovery_provider=discovery_provider,
        observed=ExpectedBibliographicFields(
            title=result.title or "",
            authors=list(result.authors or []),
            year=result.year or "",
            doi=result.doi or "",
            isbn=observed_isbn,
            container_title=container,
            pages=pages,
            publisher=observed_publisher,
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
        title_absence_reason=_title_absence_reason(result, acquisition_outcome),
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
    metadata_identity = build_observation(provider, result)
    if location_url:
        from urllib.parse import unquote
        candidate.submitted_identifier_location_match = bool(
            (expected.doi and _normalize_doi(expected.doi) in unquote(location_url).casefold())
            or (expected.isbn and expected.isbn.replace('-', '') in unquote(location_url).replace('-', '')))
    if metadata_identity is not None:
        candidate.metadata_identity = metadata_identity
    # The existing Crossref adapter retains its deposited record. Bind only
    # observations actually parsed from it; caller labels cannot certify data.
    if provider == 'crossref' and result.source_name == 'crossref' and result.success and isinstance(message, dict):
        from app.services.retrieval.crossref import CrossrefRetriever
        try:
            registered = CrossrefRetriever()._parse_message(message)
        except (TypeError, ValueError, AttributeError, KeyError, IndexError):
            # Malformed optional provenance must not break discovery or certify
            # observations that the adapter cannot reproduce.
            return candidate
        if (registered.title, registered.authors, registered.year, registered.doi) == (
                result.title, result.authors, result.year, result.doi):
            candidate.registration_record_sha256 = hashlib.sha256(
                json.dumps(message, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            candidate.registration_observation_sha256 = hashlib.sha256(
                candidate.observed.model_dump_json().encode()).hexdigest()
    return candidate


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


_CONFLICT_STOPWORDS = frozenset({"the", "of", "and", "for", "a", "an", "in", "on", "de", "la"})

# Fields whose disagreement may be reported. Each exclusion below was measured
# over 3,687 stored candidates on 2026-09-23, not assumed.
#
#   pages      -- fired 20 times, every sampled firing an Elsevier article
#                 number ("102305") compared against a range ("1-11").
#   publisher  -- fired 34 times, every sampled firing a DataCite
#                 "Unpublished" placeholder or a mis-parsed journal string.
#   container_title -- fired once, and wrongly: a reference gave "Economics
#                 and Law" where Crossref records "Ekonomia i Prawo". That is
#                 one journal under its English and Polish titles. Nothing in
#                 the strings separates a translated title from a genuinely
#                 different journal, and the provider recorded only one of the
#                 pair, so no multi-title comparison rescues it either.
#
# A volume or issue is a number. It has no abbreviation, translation or
# subtitle form, which is what made the other three unsafe. It has also not
# yet been observed on a live run: stored candidates predate the Crossref
# volume/issue work, so no measurement of its precision exists. Treat a first
# live run as the verification this pair still needs.
_MATERIAL_CONFLICT_FIELDS = frozenset({"volume", "issue"})


def _conflict_tokens(value: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", value.lower())
            if t and t not in _CONFLICT_STOPWORDS]


def _abbreviation_compatible(left: str, right: str) -> bool:
    """True when one title is plausibly the other abbreviated or extended.

    "J. Econ. Perspect." against "Journal of Economic Perspectives" is a
    citation style, and a subtitle present on one side only is an indexing
    choice. Neither is a disagreement about which journal was cited.
    """
    a, b = _conflict_tokens(left), _conflict_tokens(right)
    if not a or not b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    index = 0
    for token in short:
        while index < len(long_):
            other = long_[index]
            if (other.startswith(token[:max(3, len(token))])
                    or token.startswith(other[:max(3, len(other))])):
                break
            index += 1
        if index >= len(long_):
            return False
        index += 1
    return True


def _field_materially_conflicts(field: str, submitted: str, observed: str) -> bool:
    """Whether two present values disagree in a way a reader could act on.

    Absence is never a conflict: a provider that does not record a volume has
    not contradicted the one the student wrote.
    """
    if field not in _MATERIAL_CONFLICT_FIELDS or not submitted or not observed:
        return False
    left, right = str(submitted).strip(), str(observed).strip()
    if field in {"volume", "issue"}:
        a, b = re.sub(r"\D", "", left), re.sub(r"\D", "", right)
        return bool(a and b and a != b)
    # Retained and exercised by tests: the title comparison is correct on
    # abbreviation, entity encoding and subtitles, and is one measured case --
    # translated journal titles -- away from being usable.
    if _conflict_tokens(left) == _conflict_tokens(right):
        return False
    return not _abbreviation_compatible(left, right)


def _normalize_text(value: str) -> str:
    # Crossref deposits can carry HTML entities: "Media &amp; Policy" left the
    # journal unresolved and held back Chalaby's DOI finding (2026-09-29).
    normalized = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_doi(value: str) -> str:
    """Compare DOIs by identity, not by the encoding a reference happens to use.

    A DOI copied out of a URL keeps its percent-encoded punctuation, so
    ``10.21066/carcl.libri.2015-04%2802%29.0001`` and
    ``10.21066/carcl.libri.2015-04(02).0001`` are the same DOI written two
    ways. Treating them as different produced a field conflict against a
    reference that was correct.
    """
    from urllib.parse import unquote

    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    return unquote(normalized)


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
        search_policy_version=record.search_policy_version,
        search_retention_policy=record.search_retention_policy,
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
        attempts=record.attempts, candidates=candidates, created_at=record.created_at,
        search_policy_version=record.search_policy_version,
        search_retention_policy=record.search_retention_policy)
    if result.outcome in {"confirmed", "confirmed_with_minor_differences", "unlocated_after_search"}:
        result = result.model_copy(update={"outcome": "possible_match", "contributes_to_neutral_pattern": False})
    return {**result.model_dump(mode="json"), "limitations": list(dict.fromkeys([
        *record.limitations, *result.limitations,
        "Located book records have different years, but the edition used has not been established. Compare its publication page and ISBN; this is not a confirmed reference error.",
    ]))[:16]}


def credible_field_candidates(record: ReferenceDiscoveryRecord, fields=("title", "author", "year")):
    """Current-field-bound, independently observed work metadata for flags.

    This is not source admission or a declaration of edition equivalence.
    Rejected candidates, failed/unpermitted attempts and stale comparisons do
    not neutralize a finding or supply evidence for one.
    """
    for candidate in record.candidates:
        if not candidate.is_credible or candidate.has_material_conflict:
            continue
        if not any(a.attempt_id == candidate.attempt_id and a.provider == candidate.provider
                   and a.permitted and a.completed_at
                   and a.outcome in {"candidate_found", "candidates_processed"} for a in record.attempts):
            continue
        if not (candidate.edition_metadata or candidate.validated_identity_content_sha256
                or candidate.authoritative_identifier_match):
            continue
        comparisons = {c.field_name: c for c in candidate.comparisons}
        valid = True
        for field in fields:
            old = ' | '.join(record.expected.authors) if field == 'author' else getattr(record.expected, field, '')
            new = ' | '.join(candidate.observed.authors) if field == 'author' else getattr(candidate.observed, field, '')
            c = comparisons.get(field)
            if not old or not new or c is None or c.expected_sha256 != _value_hash(old) or c.observed_sha256 != _value_hash(new):
                valid = False
                break
        if valid and all(comparisons[f].outcome in {'agreement', 'minor_difference'} for f in ('title', 'author')):
            yield candidate


def derive_reference_discovery_record(
    *,
    reference_id: str,
    expected: ExpectedBibliographicFields,
    required_route_categories: list[RouteCategory],
    queries: list[ReferenceSearchQuery],
    attempts: list[ReferenceRouteAttempt],
    candidates: list[ReferenceDiscoveryCandidate],
    search_policy_version: SearchPolicyVersion = LEGACY_SEARCH_POLICY,
    search_retention_policy: Literal["brave-operational-transient-v1"] | None = None,
    created_at: datetime | None = None,
) -> ReferenceDiscoveryRecord:
    """Derive exactly one outcome without treating failed search as absence."""
    _validate_bindings(required_route_categories, queries, attempts, candidates)
    limitations: list[str] = []
    if search_retention_policy:
        limitations.append("Brave discovery details were transient. This audit retains operational counts and independent source evidence, not a replayable search-result list.")
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
        strong = [candidate for candidate in compatible if not expected.reference_parse_review and candidate.agreement_count >= 2
                  and (not expected.isbn or _same_edition_isbn(expected.isbn, candidate.observed.isbn))
                  and not any(c.reason_code == "book_edition_year_unresolved" for c in candidate.comparisons)]
        if exact and not expected.reference_parse_review:
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
        elif expected.reference_parse_review:
            # An incomplete search premise cannot establish source absence,
            # including under the legacy route policy. Historical records are
            # not re-derived when read.
            outcome = "search_incomplete"
            limitations.append(
                "The reference could not be parsed completely; source identity remains unresolved."
            )
        elif _search_is_incomplete(required_route_categories, attempts, queries, search_policy_version):
            outcome = "search_incomplete"
            limitations.append(
                "At least one required search route failed, was unavailable, or did not complete."
            )
        elif search_policy_version == API_FIRST_SEARCH_POLICY and any(
            (c.acquisition_outcome in {"transport_failure", "access_restricted", "unavailable",
                                      "not_attempted", "unknown", "identity_unconfirmed",
                                      "completeness_rejected", "type_rejected", "type_unconfirmed", "acquired", "acquired_fallback"}
             or (c.acquisition_outcome == "metadata_only" and not any(
                 comparison.field_name in {"title", "doi"} and comparison.outcome == "material_conflict"
                 for comparison in c.comparisons)))
            and c.discovery_provider not in {"searxng", "tavily"}
            for c in candidates
        ):
            outcome = "search_incomplete"
            limitations.append("Some candidate identities could not be checked; acquisition failure is not identity absence.")
        else:
            outcome = "unlocated_after_search"
            limitations.append(
                "Every required and permitted route completed without a credible match."
            )
    return ReferenceDiscoveryRecord(
        search_retention_policy=search_retention_policy,
        search_policy_version=search_policy_version,
        required_web_providers=list(REQUIRED_API_PROVIDERS) if search_policy_version == API_FIRST_SEARCH_POLICY else [],
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


# Query outcomes a provider recovering could plausibly cure. Excluded:
# `budget_skipped` (our elapsed budget, not the provider), `internal_error`
# (our own code), `unknown`, and the metadata reasons in
# INCOMPLETE_METADATA_SEARCH_REASONS (the provider answered; what it returned
# was not accepted as a certified empty result).
_PROVIDER_CURABLE_EXECUTION_OUTCOMES = frozenset({
    "timeout", "captcha", "operational_failure", "access_restricted",
    "rate_limited", "response_invalid", "cooldown_skipped",
    "recovery_probe_in_progress",
})


def search_blocking_providers(
    required: list[RouteCategory],
    attempts: list[ReferenceRouteAttempt],
    queries: list[ReferenceSearchQuery],
    search_policy_version: SearchPolicyVersion = LEGACY_SEARCH_POLICY,
) -> list[str]:
    """Providers whose failure is what holds this search incomplete.

    Mirrors the provider-attributable branches of `_search_is_incomplete`
    directly below and must change with it: a required attempt that failed,
    a required web API that never executed, and a required query that did
    not complete. A provider named here is one whose recovery could let the
    reference complete. Unresolved web leads and never-attempted categories
    are not attributable to one provider and name none.

    Until 2026-09-24 the recovery trigger listed every provider with any
    incomplete query, required or not. All four re-run waves on 2026-09-23 --
    192 reference re-runs -- were triggered by SearXNG, which cannot complete or
    invalidate a search under `api-first-search-v2`, and they changed
    potentially-fabricated-reference flags back and forth across report
    versions without changing any completion gate.
    """
    api_first = search_policy_version == API_FIRST_SEARCH_POLICY
    failure_outcomes = {"operational_failure", "access_restricted", "unavailable", "not_permitted"}
    blocking = {
        attempt.provider for attempt in attempts
        if attempt.required and attempt.outcome in failure_outcomes
        and not (api_first and attempt.route_category == "bounded_web")
        # Our own budget ran out; the provider did nothing wrong.
        and attempt.reason_code != "route_elapsed_budget_exhausted"
    }
    required_attempt_ids = {attempt.attempt_id for attempt in attempts if attempt.required}
    required_query_ids = {
        query_id for attempt in attempts
        if attempt.attempt_id in required_attempt_ids
        for query_id in attempt.query_ids
    }
    if api_first:
        web_queries = [q for q in queries
                       if q.route_category == "bounded_web" and q.query_id in required_query_ids]
        blocking |= set(REQUIRED_API_PROVIDERS) - {q.execution_provider for q in web_queries}
        required_query_ids -= {q.query_id for q in web_queries
                               if q.execution_provider not in REQUIRED_API_PROVIDERS}
    blocking |= {
        query.execution_provider for query in queries
        if query.query_id in required_query_ids and query.execution_provider
        and query.execution_outcome in _PROVIDER_CURABLE_EXECUTION_OUTCOMES
    }
    return sorted(str(p).strip().casefold() for p in blocking if p)


# A required full-text search query that did not run to an answer. Unlike
# `_PROVIDER_CURABLE_EXECUTION_OUTCOMES`, `budget_skipped` counts: a call
# ceiling or elapsed budget left the search unfinished even though the
# provider did nothing wrong, and "full text was not retrieved" must not read
# as a completed search.
_FULL_TEXT_INCOMPLETE_EXECUTION_OUTCOMES = frozenset({
    "timeout", "captcha", "operational_failure", "access_restricted",
    "rate_limited", "response_invalid", "budget_skipped", "cooldown_skipped",
    "recovery_probe_in_progress", "internal_error", "unknown",
})
# Our own per-reference candidate-inspection bound, reached after the maximum
# number of candidates were inspected. The search ran to its bound; this is
# not an unfinished search.
_FULL_TEXT_COMPLETED_BOUND_REASONS = frozenset({"candidate_inspection_capacity_exhausted"})
_ROUTE_FAILURE_OUTCOMES = frozenset({"operational_failure", "access_restricted", "unavailable"})


def full_text_search_incompleteness(discovery: dict | None) -> dict | None:
    """Whether a required full-text web search for a reference did not finish.

    Reads a stored discovery trace or record (dicts, never re-validated so an
    older record is still readable). Returns None when every required web
    provider answered, or when no web route was required (for example, a kind
    academic indexes do not hold). Otherwise returns ``{"providers": [...]}``
    naming the required providers that were skipped (call ceiling, cooldown,
    elapsed budget), failed operationally or never ran. ``providers`` may be
    empty when the web route itself never ran or failed as a whole; that is
    still incomplete but names no provider whose recovery would cure it.
    """
    if not isinstance(discovery, dict):
        return None
    attempts = [a for a in discovery.get("attempts") or [] if isinstance(a, dict)]
    queries = {q.get("query_id"): q for q in discovery.get("queries") or [] if isinstance(q, dict)}
    required_categories = set(discovery.get("required_route_categories") or [])
    web_attempts = [a for a in attempts if a.get("route_category") == "bounded_web" and a.get("required")]
    if not web_attempts:
        if "bounded_web" in required_categories:
            return {"providers": []}
        return None
    api_first = discovery.get("search_policy_version") == API_FIRST_SEARCH_POLICY
    required_providers = [str(p).casefold() for p in (
        discovery.get("required_web_providers") or REQUIRED_API_PROVIDERS)] if api_first else []
    web_queries = [queries[qid] for attempt in web_attempts
                   for qid in attempt.get("query_ids") or [] if qid in queries]
    incomplete: set[str] = set()
    route_failed = False

    def unfinished(query: dict) -> bool:
        return (query.get("execution_outcome") in _FULL_TEXT_INCOMPLETE_EXECUTION_OUTCOMES
                and query.get("reason_code") not in _FULL_TEXT_COMPLETED_BOUND_REASONS)

    if api_first:
        for provider in required_providers:
            own = [q for q in web_queries if str(q.get("execution_provider") or "").casefold() == provider]
            if not own or any(unfinished(q) for q in own):
                incomplete.add(provider)
    else:
        for query in web_queries:
            provider = str(query.get("execution_provider") or "").casefold()
            if provider and unfinished(query):
                incomplete.add(provider)
        route_failed = any(a.get("outcome") in _ROUTE_FAILURE_OUTCOMES for a in web_attempts)
    if incomplete or route_failed:
        return {"providers": sorted(incomplete)}
    return None


def _search_is_incomplete(
    required: list[RouteCategory],
    attempts: list[ReferenceRouteAttempt],
    queries: list[ReferenceSearchQuery],
    search_policy_version: SearchPolicyVersion = LEGACY_SEARCH_POLICY,
) -> bool:
    api_first = search_policy_version == API_FIRST_SEARCH_POLICY
    if any(a.unresolved or a.not_attempted or a.identity_established
           for attempt in attempts for a in attempt.transient_search_audits):
        # Affirmative independently bound candidates are selected before this
        # negative gate. A count alone cannot establish or rule out identity.
        return True
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
        attempt.required and attempt.outcome in failure_outcomes
        and not (api_first and attempt.route_category == "bounded_web")
        for attempt in attempts
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
        "internal_error",
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
    if api_first:
        web_queries = [q for q in queries if q.route_category == "bounded_web" and q.query_id in required_query_ids]
        if set(REQUIRED_API_PROVIDERS) - {q.execution_provider for q in web_queries}:
            return True
        # Optional SearXNG/Tavily cannot satisfy or invalidate required APIs.
        required_query_ids -= {q.query_id for q in web_queries if q.execution_provider not in REQUIRED_API_PROVIDERS}
    return any(
        query.query_id in required_query_ids
        and query.execution_provider is not None
        and (query.execution_outcome in incomplete_execution_outcomes
             or query.reason_code in INCOMPLETE_METADATA_SEARCH_REASONS)
        for query in queries
    )
