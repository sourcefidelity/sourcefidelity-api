"""Canonical-work aggregation for structured retrieval-provider evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.services.retrieval.base import (
    AcquisitionLocation,
    RepresentationKind,
    RetrievalResult,
    SourceRepresentation,
)
from app.services.source_type import (
    SourceKindAssessment,
    classify_provider_source_kind,
    compare_source_kinds,
    normalize_source_kind,
)


_CONFIDENCE_RANK = {"rejected": -1, "low": 0, "medium": 1, "high": 2}
_TRACKING_QUERY_PREFIXES = ("utm_",)


@dataclass(frozen=True)
class IdentityAssessment:
    confidence: str
    reason: str

    @property
    def accepted(self) -> bool:
        return _CONFIDENCE_RANK.get(self.confidence, -1) >= 1


@dataclass
class CanonicalWorkGraph:
    """Merge provider records only after deterministic work-identity checks."""

    expected_doi: str | None = None
    expected_title: str | None = None
    expected_author: str | None = None
    expected_year: str | None = None
    expected_source_kind: str = "unknown"
    expected_source_kind_confidence: str = "unknown"
    expected_source_kind_evidence: tuple[str, ...] = ()
    accepted_results: list[RetrievalResult] = field(default_factory=list)
    identity_evidence: list[dict] = field(default_factory=list)
    rejected_candidates: list[dict] = field(default_factory=list)
    _locations: dict[str, AcquisitionLocation] = field(default_factory=dict)

    def add(self, result: RetrievalResult) -> IdentityAssessment:
        assessment = assess_work_identity(
            result,
            expected_doi=self.expected_doi,
            expected_title=self.expected_title,
            expected_author=self.expected_author,
            expected_year=self.expected_year,
            expected_source_kind=self.expected_source_kind,
            expected_source_kind_confidence=self.expected_source_kind_confidence,
            expected_source_kind_evidence=self.expected_source_kind_evidence,
        )
        provider_kind = classify_provider_source_kind(result.metadata)
        evidence = {
            "provider": result.source_name,
            "confidence": assessment.confidence,
            "reason": assessment.reason,
            "doi": result.doi,
            "title": result.title,
            "year": result.year,
            "expected_source_kind": normalize_source_kind(self.expected_source_kind),
            "observed_source_kind": provider_kind.kind,
            "observed_source_kind_confidence": provider_kind.confidence,
        }
        if not assessment.accepted:
            self.rejected_candidates.append(evidence)
            return assessment

        self.identity_evidence.append(evidence)
        self.accepted_results.append(result)
        for location in result.locations:
            self._merge_location(location)
        return assessment

    @property
    def locations(self) -> list[AcquisitionLocation]:
        return list(self._locations.values())

    def to_result(self) -> RetrievalResult:
        if not self.accepted_results:
            return RetrievalResult(
                source_name="canonical_work_graph",
                success=False,
                metadata={
                    "canonical_work": {
                        "accepted_providers": [],
                        "identity_evidence": [],
                        "rejected_candidates": self.rejected_candidates,
                    }
                },
                error="No provider record passed canonical-work identity checks",
            )

        representation_result = self._best_representation_result()
        abstract_result = self._best_abstract_result()
        providers = [result.source_name for result in self.accepted_results]
        provider_metadata = {
            result.source_name: result.metadata
            for result in self.accepted_results
            if result.metadata is not None
        }
        publisher = next(
            filter(None, map(_publisher_from_result, self.accepted_results)),
            None,
        )
        metadata = {
            "canonical_work": {
                "accepted_providers": providers,
                "identity_evidence": self.identity_evidence,
                "rejected_candidates": self.rejected_candidates,
                "metadata_conflicts": self._metadata_conflicts(),
                "location_count": len(self._locations),
                "abstract_providers": [
                    result.source_name
                    for result in self.accepted_results
                    if result.abstract
                ],
                "representation_providers": [
                    result.source_name
                    for result in self.accepted_results
                    if result.representation
                ],
            },
            "provider_metadata": provider_metadata,
        }
        if publisher:
            metadata["publisher"] = publisher
        if representation_result:
            metadata["selected_representation_provider"] = representation_result.source_name

        locations = self.locations
        source_name = (
            representation_result.source_name
            if representation_result else "canonical_work_graph"
        )
        return RetrievalResult(
            source_name=source_name,
            success=True,
            metadata=metadata,
            representation=(
                representation_result.representation if representation_result else None
            ),
            full_text_url=locations[0].url if locations else None,
            locations=locations,
            abstract=abstract_result.abstract if abstract_result else None,
            doi=self.expected_doi or _consensus_value(self.accepted_results, "doi"),
            title=self.expected_title or _consensus_value(self.accepted_results, "title"),
            year=self.expected_year or _consensus_value(self.accepted_results, "year"),
            authors=_best_authors(self.accepted_results, self.expected_author),
        )

    def _merge_location(self, location: AcquisitionLocation) -> None:
        key = canonicalize_location_url(location.url)
        existing = self._locations.get(key)
        if existing is None:
            metadata = dict(location.metadata)
            metadata["providers"] = [location.provider]
            self._locations[key] = replace(location, metadata=metadata)
            return

        providers = list(existing.metadata.get("providers", [existing.provider]))
        if location.provider not in providers:
            providers.append(location.provider)
        metadata = {**existing.metadata, **location.metadata, "providers": providers}
        self._locations[key] = replace(
            existing,
            media_type=existing.media_type or location.media_type,
            representation_kind=(
                existing.representation_kind or location.representation_kind
            ),
            landing_page_url=existing.landing_page_url or location.landing_page_url,
            host_type=existing.host_type or location.host_type,
            version=existing.version or location.version,
            license=existing.license or location.license,
            access_type=existing.access_type or location.access_type,
            intended_application=(
                existing.intended_application or location.intended_application
            ),
            is_best=existing.is_best or location.is_best,
            metadata=metadata,
        )

    def _best_representation_result(self) -> RetrievalResult | None:
        candidates = [
            result for result in self.accepted_results if result.representation is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=_representation_result_score)

    def _best_abstract_result(self) -> RetrievalResult | None:
        candidates = [result for result in self.accepted_results if result.abstract]
        if not candidates:
            return None
        return max(candidates, key=lambda result: len(result.abstract or ""))

    def _metadata_conflicts(self) -> dict[str, list[dict]]:
        conflicts: dict[str, list[dict]] = {}
        for field_name in ("doi", "title", "year"):
            observed: dict[str, list[str]] = {}
            display: dict[str, str] = {}
            for result in self.accepted_results:
                value = getattr(result, field_name)
                if not value or value == "n.d.":
                    continue
                normalized = _normalize_metadata_value(field_name, value)
                display.setdefault(normalized, value)
                observed.setdefault(normalized, []).append(result.source_name)
            if len(observed) > 1:
                conflicts[field_name] = [
                    {"value": display[value], "providers": providers}
                    for value, providers in observed.items()
                ]
        return conflicts


def assess_work_identity(
    result: RetrievalResult,
    *,
    expected_doi: str | None,
    expected_title: str | None,
    expected_author: str | None,
    expected_year: str | None,
    expected_source_kind: str = "unknown",
    expected_source_kind_confidence: str = "unknown",
    expected_source_kind_evidence: tuple[str, ...] = (),
) -> IdentityAssessment:
    """Assess provider-record identity before its locations enter the graph."""
    normalized_expected_doi = _normalize_doi(expected_doi)
    normalized_result_doi = _normalize_doi(result.doi)
    if normalized_expected_doi and normalized_result_doi:
        if normalized_expected_doi == normalized_result_doi:
            return IdentityAssessment("high", "exact DOI match")
        return IdentityAssessment(
            "rejected",
            f"DOI conflict: expected {normalized_expected_doi}, provider returned {normalized_result_doi}",
        )

    expected_kind = SourceKindAssessment(
        normalize_source_kind(expected_source_kind),
        expected_source_kind_confidence,
        expected_source_kind_evidence,
    )
    observed_kind = classify_provider_source_kind(result.metadata)
    kind_compatibility = compare_source_kinds(expected_kind, observed_kind)
    if kind_compatibility.verdict == "incompatible":
        return IdentityAssessment(
            "rejected",
            f"bibliographic type conflict: {kind_compatibility.reason}",
        )

    if not expected_title or not result.title:
        return IdentityAssessment("low", "insufficient DOI/title identity evidence")

    expected_tokens = _significant_tokens(expected_title)
    result_tokens = _significant_tokens(result.title)
    if not expected_tokens or not result_tokens:
        return IdentityAssessment("low", "title did not contain comparable tokens")
    overlap = len(expected_tokens & result_tokens) / len(expected_tokens)

    supporting_fields: list[str] = []
    if expected_author and result.authors:
        surname = _author_surname(expected_author)
        if surname and any(surname in author.lower() for author in result.authors):
            supporting_fields.append("author")
    if expected_year and result.year and _year(expected_year) == _year(result.year):
        supporting_fields.append("year")

    if overlap >= 0.8 and supporting_fields:
        return IdentityAssessment(
            "high",
            f"title overlap={overlap:.2f} with {'+'.join(supporting_fields)} support",
        )
    if overlap >= 0.75:
        return IdentityAssessment("medium", f"strong title overlap={overlap:.2f}")
    if overlap >= 0.6 and supporting_fields:
        return IdentityAssessment(
            "medium",
            f"title overlap={overlap:.2f} with {'+'.join(supporting_fields)} support",
        )
    return IdentityAssessment(
        "low", f"provider record did not meet merge threshold (title overlap={overlap:.2f})"
    )


def canonicalize_location_url(url: str) -> str:
    """Canonicalize for deduplication without removing access-bearing parameters."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return url.strip()
    host = (parts.hostname or "").lower()
    if port and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    query = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not name.lower().startswith(_TRACKING_QUERY_PREFIXES)
    ]
    return urlunsplit(
        (parts.scheme.lower(), host, parts.path or "/", urlencode(sorted(query)), "")
    )


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized or None


def _significant_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if len(token) >= 3
    }


def _author_surname(value: str) -> str:
    if "," in value:
        return value.split(",", 1)[0].strip().lower()
    parts = [part for part in re.findall(r"[A-Za-z]+", value) if part]
    return parts[-1].lower() if parts else ""


def _year(value: str | None) -> str | None:
    match = re.search(r"\d{4}", value or "")
    return match.group() if match else None


def _normalize_metadata_value(field_name: str, value: str) -> str:
    if field_name == "doi":
        return _normalize_doi(value) or ""
    if field_name == "title":
        return " ".join(sorted(_significant_tokens(value)))
    if field_name == "year":
        return _year(value) or value.strip().lower()
    return value.strip().lower()


def _consensus_value(results: list[RetrievalResult], field_name: str) -> str | None:
    values: dict[str, tuple[str, int]] = {}
    for result in results:
        value = getattr(result, field_name)
        if not value or value == "n.d.":
            continue
        normalized = _normalize_metadata_value(field_name, value)
        display, count = values.get(normalized, (value, 0))
        values[normalized] = (display, count + 1)
    if not values:
        return None
    return max(values.values(), key=lambda item: item[1])[0]


def _best_authors(
    results: list[RetrievalResult], expected_author: str | None
) -> list[str]:
    candidates = [result.authors for result in results if result.authors]
    if candidates:
        return max(candidates, key=len)
    return [expected_author] if expected_author else []


def _representation_result_score(result: RetrievalResult) -> tuple[int, int, int]:
    representation: SourceRepresentation = result.representation  # type: ignore[assignment]
    completeness_score = {
        "complete": 3,
        "likely_complete": 2,
        "not_assessed": 1,
        "partial": 0,
    }.get(representation.completeness, 1)
    kind_score = {
        RepresentationKind.PDF: 5,
        RepresentationKind.PLAIN_TEXT: 4,
        RepresentationKind.XML: 3,
        RepresentationKind.HTML: 2,
        RepresentationKind.EPUB: 1,
    }.get(representation.kind, 0)
    return completeness_score, kind_score, len(representation.content)


def _publisher_from_result(result: RetrievalResult) -> str | None:
    metadata = result.metadata or {}
    publisher = metadata.get("publisher")
    if publisher:
        return str(publisher)
    message = metadata.get("message") or {}
    if isinstance(message, dict) and message.get("publisher"):
        return str(message["publisher"])
    return None
