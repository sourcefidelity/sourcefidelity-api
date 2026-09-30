"""Canonical-work aggregation for structured retrieval-provider evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
import unicodedata
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
# A title this short cannot carry a merge on word overlap alone.
_MIN_TOKENS_FOR_UNSUPPORTED_MERGE = 4
_TRACKING_QUERY_PREFIXES = ("utm_",)
_TITLE_TAIL_BOUNDARY = re.compile(r"^\s*[-:;.,/?!|\u2013\u2014(\[]")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def _normalized_locator(value: str) -> str:
    """Compare submitted and observed addresses without their decoration."""
    text = (value or "").strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    text = text.split("#", 1)[0].split("?", 1)[0]
    return text.rstrip("/")


def _result_locators(result: RetrievalResult) -> set[str]:
    values = {result.full_text_url or ""}
    values.update(location.url or "" for location in (result.locations or []))
    metadata = result.metadata or {}
    values.add(str(metadata.get("url") or ""))
    return {_normalized_locator(value) for value in values if value}


def _normalized_title(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", (value or "").strip().lower())
    return _LEADING_ARTICLE.sub("", collapsed)


@dataclass(frozen=True)
class IdentityAssessment:
    confidence: str
    reason: str
    # "part" when the cited title matched, "container" when only the work that
    # contains it did. A container match identifies the book, never the part.
    title_basis: str = ""
    # "title_alone" when a distinctive title was the only agreeing field, and
    # "corroborated" when author, year, publisher or an identifier agreed too.
    # A distinctive title is good enough to show a reader what was probably
    # cited, but it is one signal: the record may still be a different edition,
    # a reprint, or another work sharing the phrase. So a title-alone merge
    # identifies for display and is barred from contributing a discrepancy
    # against the student's reference -- see corroborated_results.
    corroboration: str = "corroborated"

    @property
    def accepted(self) -> bool:
        return _CONFIDENCE_RANK.get(self.confidence, -1) >= 1


@dataclass
class CanonicalWorkGraph:
    """Merge provider records only after deterministic work-identity checks."""

    expected_doi: str | None = None
    expected_title: str | None = None
    expected_container_title: str = ""
    expected_publisher: str = ""
    expected_url: str = ""
    expected_author: str | None = None
    expected_year: str | None = None
    expected_source_kind: str = "unknown"
    expected_source_kind_confidence: str = "unknown"
    expected_source_kind_evidence: tuple[str, ...] = ()
    accepted_results: list[RetrievalResult] = field(default_factory=list)
    identity_evidence: list[dict] = field(default_factory=list)
    rejected_candidates: list[dict] = field(default_factory=list)
    # Accepted results whose only agreeing field was a distinctive title.
    _title_alone_results: list[RetrievalResult] = field(default_factory=list)
    _locations: dict[str, AcquisitionLocation] = field(default_factory=dict)

    def add(self, result: RetrievalResult) -> IdentityAssessment:
        assessment = assess_work_identity(
            result,
            expected_doi=self.expected_doi,
            expected_title=self.expected_title,
            expected_container_title=self.expected_container_title,
            expected_publisher=self.expected_publisher,
            expected_url=self.expected_url,
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
            "title_basis": assessment.title_basis,
            "corroboration": assessment.corroboration,
            "rejection_code": (
                "doi_registers_a_different_title"
                if assessment.reason.startswith("the supplied DOI resolves")
                else None),
        }
        if not assessment.accepted:
            self.rejected_candidates.append(evidence)
            return assessment

        self.identity_evidence.append(evidence)
        self.accepted_results.append(result)
        if assessment.corroboration == "title_alone":
            self._title_alone_results.append(result)
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
                # Identified for display, barred from supplying a discrepancy.
                "title_alone_providers": [
                    result.source_name for result in self._title_alone_results],
                "corroborated_providers": [
                    result.source_name for result in self.corroborated_results],
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
            # Where the reference left a field blank, the merged record fills
            # it. A corroborated record answers first: a title-alone match may
            # be a different edition, and its year would then be presented as
            # this work's. When only title-alone records exist, theirs is the
            # best available answer and is still shown.
            doi=self.expected_doi or _consensus_value(self._display_results, "doi"),
            title=self.expected_title or _consensus_value(self._display_results, "title"),
            year=self.expected_year or _consensus_value(self._display_results, "year"),
            authors=_best_authors(self._display_results, self.expected_author),
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

    @property
    def corroborated_results(self) -> list[RetrievalResult]:
        """Accepted records that agreed on more than the title alone.

        A distinctive title is enough to identify a work for a reader, but it
        is a single signal: a reprint, a later edition, or another work sharing
        the phrase can satisfy it. Letting such a record disagree with the
        student's year or DOI would report a discrepancy that rests entirely on
        the merge being right, which nothing here established. Title-alone
        records stay in the graph, stay visible, and contribute locations and
        text; they do not supply a disagreement.
        """
        title_alone = {id(result) for result in self._title_alone_results}
        return [r for r in self.accepted_results if id(r) not in title_alone]

    @property
    def _display_results(self) -> list[RetrievalResult]:
        """Corroborated records where there are any, otherwise all of them."""
        return self.corroborated_results or self.accepted_results

    def _metadata_conflicts(self) -> dict[str, list[dict]]:
        conflicts: dict[str, list[dict]] = {}
        for field_name in ("doi", "title", "year"):
            observed: dict[str, list[str]] = {}
            display: dict[str, str] = {}
            for result in self.corroborated_results:
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
    expected_container_title: str = "",
    expected_publisher: str = "",
    expected_url: str = "",
    expected_source_kind: str = "unknown",
    expected_source_kind_confidence: str = "unknown",
    expected_source_kind_evidence: tuple[str, ...] = (),
) -> IdentityAssessment:
    """Assess provider-record identity before its locations enter the graph.

    Identity rests on the combination of title, author and year rather than on
    title text alone: a record agreeing on several of them is unlikely to be a
    different work, and a record agreeing on none of them is not this one
    however many words its title happens to share.
    """
    normalized_expected_doi = _normalize_doi(expected_doi)
    normalized_result_doi = _normalize_doi(result.doi)
    doi_matches = False
    if normalized_expected_doi and normalized_result_doi:
        if normalized_expected_doi != normalized_result_doi:
            return IdentityAssessment(
                "rejected",
                f"DOI conflict: expected {normalized_expected_doi}, provider returned {normalized_result_doi}",
            )
        # A matching DOI is strong evidence, not a bypass. A fabricated or
        # partly fabricated reference frequently carries a real DOI lifted from
        # another work, so resolving that DOI and accepting whatever it names
        # would confirm the invention and hand the reader a stranger's text.
        # The title still has to agree; a DOI that resolves to a different
        # title is itself the finding, and is reported as one below.
        doi_matches = True
        if not expected_title:
            # Nothing to contradict: the reference supplied an identifier and
            # no title, so the registration is the only identity evidence
            # there is, and it outranks the coarse provider type taxonomy.
            return IdentityAssessment("high", "exact DOI match")

    expected_kind = SourceKindAssessment(
        normalize_source_kind(expected_source_kind),
        expected_source_kind_confidence,
        expected_source_kind_evidence,
    )
    observed_kind = classify_provider_source_kind(result.metadata)
    kind_compatibility = compare_source_kinds(expected_kind, observed_kind)
    if kind_compatibility.verdict == "incompatible" and not doi_matches:
        return IdentityAssessment(
            "rejected",
            f"bibliographic type conflict: {kind_compatibility.reason}",
        )

    def _title_agrees(expected: str, observed: str) -> bool:
        """Equal, or one extends the other at a subtitle boundary, or near-equal.

        Bare containment is what merged "The Studio System" into "A Fine
        Romance: Adapting Broadway to Hollywood in the Studio System Era".
        """
        left, right = _normalized_title(expected), _normalized_title(observed)
        if not left or not right:
            return False
        if left == right:
            return True
        longer, shorter = (right, left) if len(right) > len(left) else (left, right)
        if longer.startswith(shorter) and _TITLE_TAIL_BOUNDARY.match(longer[len(shorter):]):
            return True
        left_tokens, right_tokens = _significant_tokens(expected), _significant_tokens(observed)
        if not left_tokens or not right_tokens:
            return False
        shared = left_tokens & right_tokens
        return (len(shared) / len(left_tokens) >= 0.8
                and len(shared) / len(right_tokens) >= 0.8)

    agreements: list[str] = []
    title_basis = ""
    if expected_title and result.title and _title_agrees(expected_title, result.title):
        agreements.append("title")
        title_basis = "part"
    elif (expected_container_title and result.title
            and _title_agrees(expected_container_title, result.title)):
        # The containing work a miscited part names is still named correctly.
        agreements.append("title")
        title_basis = "container"

    if expected_author and result.authors:
        surname = _author_surname(expected_author)
        if surname and any(surname in _fold_diacritics(author).lower()
                           for author in result.authors):
            agreements.append("author")
    if expected_year and result.year and _year(expected_year) == _year(result.year):
        agreements.append("year")
    observed_publisher = _publisher_from_result(result) or ""
    if expected_publisher and observed_publisher:
        expected_imprint = _significant_tokens(expected_publisher)
        observed_imprint = _significant_tokens(observed_publisher)
        # "McGraw Hill" and "McGraw-Hill Education" are the same imprint.
        if expected_imprint and observed_imprint and (
                expected_imprint <= observed_imprint or observed_imprint <= expected_imprint):
            agreements.append("publisher")
    # The identifier the student supplied. An exact DOI match has already
    # returned above; a submitted URL the provider also lists is the same kind
    # of evidence, once both are stripped of scheme, www and query decoration.
    if doi_matches:
        agreements.append("identifier")
    elif expected_url:
        submitted = _normalized_locator(expected_url)
        if submitted and submitted in _result_locators(result):
            agreements.append("identifier")

    distinctive = _significant_tokens(
        (expected_title if title_basis != "container" else expected_container_title) or "")
    summary = f"{'+'.join(agreements) or 'nothing'} agree"
    if title_basis == "container":
        summary += " (title matched the containing work, not the cited part)"

    # A distinctive title can still stand alone; anything shorter needs the
    # combination the owner's rule describes — title plus one other field, or
    # three fields agreeing — because a work matching that much is unlikely to
    # be a different one.
    if "title" in agreements and len(distinctive) >= _MIN_TOKENS_FOR_UNSUPPORTED_MERGE:
        corroborated = len(agreements) >= 2
        return IdentityAssessment(
            "high" if corroborated else "medium", summary, title_basis,
            "corroborated" if corroborated else "title_alone")
    if "title" in agreements and len(agreements) >= 2:
        return IdentityAssessment("high" if len(agreements) >= 3 else "medium",
                                  summary, title_basis)
    # The title is the anchor. Author, year and publisher agreeing without it
    # describes "a book by this author, that year, from that imprint", which a
    # prolific author with a regular publisher can satisfy more than once — and
    # the cost of being wrong is the app showing a different work's text as the
    # cited source. Such a record is retained as a rejected candidate for
    # inspection rather than merged.
    if doi_matches and "title" not in agreements:
        # The condition worth reporting rather than merging: the identifier
        # resolves, and it resolves to something else.
        return IdentityAssessment(
            "low",
            f"the supplied DOI resolves to a different title: submitted "
            f"{(expected_title or '')[:120]!r}, registered {(result.title or '')[:120]!r}",
            title_basis)
    return IdentityAssessment(
        "low",
        f"insufficient agreement: {summary}"
        + ("; the title did not agree" if "title" not in agreements else ""),
        title_basis)


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


def _fold_diacritics(value: str) -> str:
    """Compare names written with and without their marks.

    Measured on stored candidates: "Kir, E., & Akyuz, A." against OpenAlex's
    "Elif Kir, Asli Akyuz" scored as no author agreement because the surnames
    differ only by a dotless i and a diaeresis, leaving a same-work record
    resting on its title alone.
    """
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("\u0131", "i").replace("\u0130", "i")


def _author_surname(value: str) -> str:
    folded = _fold_diacritics(value)
    if "," in folded:
        return folded.split(",", 1)[0].strip().lower()
    parts = [part for part in re.findall(r"[A-Za-z]+", folded) if part]
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
