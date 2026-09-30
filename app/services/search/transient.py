"""Brave operational data is not a durable search-result archive."""

from typing import Literal
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer
from app.services.reference_review_scope import ReviewScreen

from app.services.retrieval.base import RetrievalResult

BRAVE_TRANSIENT_POLICY = "brave-operational-transient-v1"
_ATTEMPTED_URLS = ContextVar("transient_candidate_attempted_urls", default=None)
_INSPECTIONS = ContextVar("transient_candidate_inspections", default=None)


@contextmanager
def transient_acquisition_scope(attempted_urls: set[str], inspections: dict | None = None):
    """Operation-local deduplication must not become durable provenance."""
    token = _ATTEMPTED_URLS.set(attempted_urls)
    inspection_token = _INSPECTIONS.set(inspections)
    try:
        yield
    finally:
        _ATTEMPTED_URLS.reset(token)
        _INSPECTIONS.reset(inspection_token)

_DISPOSITIONS = frozenset({"acquired", "acquired_fallback", "identity_rejected", "identity_unconfirmed",
    "completeness_rejected", "type_rejected", "type_unconfirmed", "transport_failure",
    "access_restricted", "unavailable", "not_attempted", "metadata_only", "unknown"})


class TransientSearchAudit(BaseModel):
    bounded_review_screen: ReviewScreen | None = None

    @model_serializer(mode='wrap')
    def _preserve_old_review_absence(self, handler):
        payload = handler(self)
        if 'bounded_review_screen' not in self.model_fields_set:
            payload.pop('bounded_review_screen', None)
        return payload
    model_config = ConfigDict(extra="forbid")
    retention_policy: Literal["brave-operational-transient-v1"] = BRAVE_TRANSIENT_POLICY
    provider: Literal["brave"] = "brave"
    candidate_count: int = Field(ge=0)
    identity_established: int = Field(default=0, ge=0)
    identity_rejected: int = Field(default=0, ge=0)
    unresolved: int = Field(default=0, ge=0)
    not_attempted: int = Field(default=0, ge=0)
    disposition_counts: dict[str, int] = Field(default_factory=dict)
    details_discarded: Literal[True] = True

    @model_validator(mode="after")
    def counts_balance(self):
        if self.bounded_review_screen and (
                self.bounded_review_screen.candidate_count != self.candidate_count
                or self.bounded_review_screen.observations is not None):
            raise ValueError('Invalid transient review accounting')
        if self.candidate_count != sum((self.identity_established, self.identity_rejected,
                                        self.unresolved, self.not_attempted)):
            raise ValueError("Transient candidate accounting does not balance")
        if self.disposition_counts and (set(self.disposition_counts) - _DISPOSITIONS
                or any(v < 0 for v in self.disposition_counts.values())
                or sum(self.disposition_counts.values()) != self.candidate_count):
            raise ValueError("Invalid aggregate acquisition dispositions")
        return self


def is_transient_brave(result: RetrievalResult) -> bool:
    return (result.metadata or {}).get("search_retention_policy") == BRAVE_TRANSIENT_POLICY


def finalize_transient_brave(result: RetrievalResult) -> None:
    """Release discovery details before any durable sink or resolver return.

    Only content-bound identity validations may retain independent source URLs.
    Search titles, snippets, ranks, failed URLs and URL hashes never survive.
    Idempotent so both admission and exception-finally paths can enforce it.
    """
    if not is_transient_brave(result):
        return
    metadata = result.metadata
    if metadata.get("transient_details_discarded"):
        return
    records = {location.url: {"outcome": "not_attempted"} for location in result.locations}
    scope_by_url = {location.url: location.metadata.get('bounded_review_scope', 'unknown')
                    for location in result.locations}
    review_counts = dict(metadata.get('transient_unselected_review_counts') or {})
    if sum(review_counts.values()) != int(metadata.get('transient_unselected_count', 0)):
        review_counts = {'unknown': int(metadata.get('transient_unselected_count', 0))}
    for attempt in [*metadata.pop("_reused_location_attempts", []), *metadata.get("location_attempts", [])]:
        if attempt.get("url"):
            records[attempt["url"]] = attempt
            seen = _ATTEMPTED_URLS.get()
            if seen is not None and attempt.get("outcome") != "not_attempted":
                seen.add(attempt["url"])
            inspections = _INSPECTIONS.get()
            if inspections is not None and attempt.get("outcome") != "not_attempted":
                # Memory-only, reference-local handoff. Failed Brave URLs and
                # search fields never enter this operation's durable trace.
                inspections[attempt["url"]] = {key: attempt[key] for key in (
                    "outcome", "reason_code", "landing_metadata_identity", "landing_metadata_observation", "landing_page_observation",
                    "validated_identity_content_sha256", "independent_source_url") if key in attempt}
    counts = {"identity_established": 0, "identity_rejected": 0,
              "unresolved": 0, "not_attempted": int(metadata.get("transient_unselected_count", 0))}
    independent = []
    dispositions = {"not_attempted": counts["not_attempted"]} if counts["not_attempted"] else {}
    for url, attempt in records.items():
        outcome = attempt.get("outcome")
        code = outcome if outcome in _DISPOSITIONS else "unknown"
        dispositions[code] = dispositions.get(code, 0) + 1
        content_hash = attempt.get("validated_identity_content_sha256")
        landing = attempt.get("landing_metadata_identity") if outcome == "unavailable" else None
        observation = attempt.get('landing_metadata_observation')
        if observation:
            independent.append(dict(url=observation['source_url'], discovery_provider='brave',
                outcome=outcome, landing_metadata_observation=observation,
                provenance_basis='independently_acquired_content'))
        if landing or (content_hash and outcome in {"acquired", "acquired_fallback", "completeness_rejected"}):
            review_decision = 'credible'
            counts["identity_established"] += 1
            record = {"url": landing["source_url"] if landing else attempt.get("independent_source_url") or url,
                "discovery_provider": "brave", "outcome": outcome,
                "validated_identity_content_sha256": landing["content_sha256"] if landing else content_hash,
                "provenance_basis": "independently_acquired_content"}
            if landing:
                record["landing_metadata_identity"] = landing
            independent.append(record)
        elif outcome == "identity_rejected":
            review_decision = 'resolved_different'
            counts["identity_rejected"] += 1
        elif outcome == "not_attempted":
            review_decision = scope_by_url.get(url, 'unknown')
            counts["not_attempted"] += 1
        else:
            review_decision = scope_by_url.get(url, 'unknown')
            counts["unresolved"] += 1
        if review_decision not in {'outside_bound','material','unknown','resolved_different','credible'}:
            review_decision = 'unknown'
        review_counts[review_decision] = review_counts.get(review_decision, 0) + 1
    review = {}
    if metadata.get('bounded_review_input_sha256'):
        review['bounded_review_screen'] = ReviewScreen(
            input_sha256=metadata['bounded_review_input_sha256'], candidate_count=sum(counts.values()), **review_counts)
    audit = TransientSearchAudit(candidate_count=sum(counts.values()), disposition_counts=dispositions, **counts, **review)
    metadata["transient_search_audits"] = [audit.model_dump(mode="json")]
    metadata["location_attempts"] = independent
    metadata.pop("candidate_dispositions", None)
    metadata.pop("transient_unselected_count", None)
    metadata.pop('transient_unselected_review_counts', None)
    metadata.pop('bounded_review_input_sha256', None)
    metadata["transient_details_discarded"] = True
    result.locations.clear()
    # A search lead is not the provenance of an acquired representation.
    accepted_urls = {item["url"] for item in independent}
    if not result.representation or result.representation.source_url not in accepted_urls:
        result.full_text_url = None
    else:
        result.full_text_url = result.representation.source_url
    for representation in (result.representation, result.parent_representation):
        if representation and representation.source_url not in accepted_urls:
            representation.source_url = None
