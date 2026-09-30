"""Pydantic schemas for LLM structured output validation.

These schemas define the expected structure for LLM responses,
ensuring type safety and catching malformed output.
"""

from enum import Enum
import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator


def declared_max_length(model, field_name):
    """Read a field's own ``max_length`` so a limit is never written twice.

    Bounding logic that repeats the number drifts from the field it guards.
    Reading the declaration keeps one source of truth per field.
    """
    field = model.model_fields.get(field_name)
    for item in getattr(field, "metadata", ()) or ():
        limit = getattr(item, "max_length", None)
        if isinstance(limit, int):
            return limit
    return None


def note_bounded(data: dict, field_name: str) -> None:
    """Record that the application altered a field, on the record itself."""
    noted = list(data.get("bounded_fields") or [])
    if field_name not in noted:
        noted.append(field_name)
    data["bounded_fields"] = noted


def bound_text_fields(model, data, field_names) -> dict:
    """Truncate each named text field at its own declared limit and say so.

    Used from a model-level ``mode="before"`` validator, because only there
    can the alteration be written back onto the record: a field validator
    sees one value and cannot annotate its neighbours. The owner's rule is
    that an unexpectedly long field must not crash a run; the project's rule
    is that findings expose their limitations. Both hold only if the
    truncation is recorded where a reviewer will see it.
    """
    if not isinstance(data, dict):
        return data
    out = None
    for name in field_names:
        value = data.get(name)
        limit = declared_max_length(model, name)
        if isinstance(value, str) and limit and len(value) > limit:
            if out is None:
                out = dict(data)
            out[name] = value[:limit]
            note_bounded(out, name)
    return out if out is not None else data


class BoundedFieldsMixin(BaseModel):
    """Names of fields the application truncated or degraded on this record.

    Serialized only when non-empty, so an untouched record dumps exactly as it
    did before and every stored hash of one is unchanged; a record that was
    altered carries the fact, and its hash changes with it, as it should.
    """

    bounded_fields: list[str] = Field(default_factory=list, max_length=32)

    @model_serializer(mode="wrap")
    def _omit_empty_bounded_fields(self, handler):
        payload = handler(self)
        if isinstance(payload, dict) and not self.bounded_fields:
            payload.pop("bounded_fields", None)
        return payload


class ReferenceURLRepair(BaseModel):
    """Auditable, extraction-bound locator reconstruction; not source admission."""

    version: str = "reference-url-lines-v1"
    layout_sha256: str
    raw_reference_sha256: str
    original_url: str
    observed_span: str
    span_start: int
    span_end: int


class SubmittedHyperlinkBinding(BaseModel):
    """Original-document locator provenance, never source identity evidence."""

    version: Literal["submitted-hyperlink-binding-v1"] = "submitted-hyperlink-binding-v1"
    paper_sha256: str
    reference_text_sha256: str
    paragraph_indexes: list[int]


class ParsedReference(BoundedFieldsMixin):
    """A single parsed reference from the LLM.

    This is the canonical parsed reference format used across the system.
    """

    reference_id: str = Field(
        default="",
        description=(
            "Stable paper-local application identity. Unlike citation_key, this "
            "is unique within a paper and is used for joins."
        ),
    )

    author: str = Field(
        default="",
        description="Author names as they appear in the reference",
    )
    year: str = Field(
        default="n.d.",
        description="4-digit year or 'n.d.' if no date",
    )
    title: str = Field(
        default="",
        description="Title of the work",
    )
    doi: str = Field(
        default="",
        description="Plain DOI only, without https://doi.org/ prefix",
    )
    url: str = Field(
        default="",
        description="URL for the work, prefers DOI URL if available",
    )
    raw_ref: str = Field(
        default="",
        description="Original input reference string",
    )
    container_title: str = Field(default="", max_length=1000)
    # The fourth identity field alongside title, author and year: a book
    # naming the same publisher is unlikely to be a different book.
    publisher: str = Field(default="", max_length=300)
    # An article's volume and issue distinguish it the way a book's publisher
    # does. They were absent here while `ExpectedBibliographicFields` already
    # carried them, so every volume and issue comparison resolved `unknown`
    # for want of a submitted value, and a wrong volume could not be seen.
    volume: str = Field(default="", max_length=80)
    issue: str = Field(default="", max_length=80)
    pages: str = Field(default="", max_length=100)

    @model_validator(mode="before")
    @classmethod
    def _bound_text(cls, data):
        """A malformed parse must not crash the run, and must say what it lost.

        These carry whatever the parser recovered from the reference string, so
        a mis-split entry can put an entire paragraph into `container_title`.
        Owner decision, 2026-09-23: an unexpectedly long field must not crash a
        paper run - losing detail on one reference is acceptable where losing
        the whole submission is not. A prefix keeps what matching uses, since
        these are compared by token overlap rather than equality. The altered
        field names are recorded in `bounded_fields`.
        """
        return bound_text_fields(
            cls, data, ("container_title", "publisher", "volume", "issue", "pages")
        )
    url_repair: Optional[ReferenceURLRepair] = None
    submitted_hyperlink_binding: Optional[SubmittedHyperlinkBinding] = None
    citation_key: str = Field(
        default="",
        description="First author surname plus year, e.g. 'Smith2020'",
    )
    is_media_source: bool = Field(
        default=False,
        description="True for film, TV, podcast, song, video, game, or similar media",
    )
    source_kind: str = Field(
        default="unknown",
        description=(
            "Bibliographic work type expected from the reference, independent "
            "of representation format"
        ),
    )
    source_kind_confidence: str = Field(
        default="unknown",
        description="Confidence in source_kind: high, medium, low, or unknown",
    )
    source_kind_evidence: List[str] = Field(
        default_factory=list,
        description="Bounded inspectable signals supporting source_kind",
    )
    needs_review: bool = Field(
        default=False,
        description=(
            "True when the reference could not be reliably parsed by the LLM "
            "(e.g. malformed entry defeated the model, or it was recovered via "
            "regex fallback / JSON salvage). Surface to the instructor for "
            "manual verification rather than treating as a clean parse."
        ),
    )
    extraction_method: str = Field(
        default="regex",
        description=(
            'How the structured fields were extracted: "regex" (deterministic '
            'pattern match, most reliable) | "llm" (LLM per-reference fallback '
            'used for edge cases — flagged needs_review) | "fallback" (both '
            'regex and LLM failed, fields empty, needs_review=True).'
        ),
    )

    @model_validator(mode="after")
    def classify_source_kind(self):
        """Derive a conservative expected work type from the raw reference."""
        from app.services.source_type import (
            classify_reference_source_kind,
            normalize_source_kind,
        )

        normalized = normalize_source_kind(self.source_kind)
        if normalized != "unknown":
            self.source_kind = normalized
            if self.source_kind_confidence not in {"high", "medium", "low"}:
                self.source_kind_confidence = "medium"
            self.source_kind_evidence = self.source_kind_evidence[:4]
            return self

        assessment = classify_reference_source_kind(
            self.raw_ref,
            title=self.title,
            url=self.url,
        )
        self.source_kind = assessment.kind
        self.source_kind_confidence = assessment.confidence
        self.source_kind_evidence = list(assessment.evidence[:4])
        if assessment.kind in {"video", "podcast_episode", "traditional_media"}:
            self.is_media_source = True
        return self

    @field_validator("doi", mode="before")
    @classmethod
    def clean_doi(cls, v: str) -> str:
        """Remove a doi.org prefix and decode a percent-encoded DOI."""
        from app.services.ref_field_extractor import decode_doi
        if isinstance(v, str):
            for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
                if v.startswith(prefix):
                    v = v[len(prefix):]
                    break
            return decode_doi(v)
        return v or ""

    @field_validator("year", mode="before")
    @classmethod
    def clean_year(cls, v: str) -> str:
        """Preserve an explicit year range or year with citation suffix."""
        if not v:
            return "n.d."
        v = str(v).strip()
        if re.fullmatch(r"(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}", v):
            return v
        if re.fullmatch(r"(?:19|20)\d{2}[a-z]?", v, re.IGNORECASE):
            return v.lower()
        # Preserve APA/Harvard disambiguation suffixes such as 2020a.
        match = re.search(r"\b(?:19|20)\d{2}[a-z]?\b", v, re.IGNORECASE)
        if match:
            return match.group(0).lower()
        return "n.d."


class CitationMarkerMember(BoundedFieldsMixin):
    """One exact source marker inside a possibly grouped citation unit."""

    text: str = Field(min_length=1, max_length=2_000)
    local_start: int = Field(ge=0)
    local_end: int = Field(gt=0)
    reference_ids: List[str] = Field(default_factory=list)
    marker_type: str = Field(default="unknown")

    @model_validator(mode="before")
    @classmethod
    def _bound_text_with_its_span(cls, data):
        """Keep the marker's leading text and shorten its span to match.

        The text and the offsets must agree, so this clamps both together: the
        highlight then covers a real prefix of the citation rather than a span
        the text cannot support. Unlike `CitationSourceBinding`, this model has
        no unresolved state to fall back to, and a partial highlight of a real
        marker is honest - it starts in the right place and simply ends early.

        Owner decision, 2026-09-23: an unexpectedly long field must not crash a
        paper run. Observed markers reach 328 characters against this 2,000
        limit, so this is a guard rather than a live repair.
        """
        if not isinstance(data, dict):
            return data
        text = data.get("text")
        limit = declared_max_length(cls, "text")
        if not (isinstance(text, str) and limit and len(text) > limit):
            return data
        data = dict(data)
        data["text"] = text[:limit]
        start = data.get("local_start")
        if isinstance(start, int):
            data["local_end"] = start + limit
        note_bounded(data, "text")
        return data

    @model_validator(mode="after")
    def validate_local_span(self):
        if self.local_end <= self.local_start:
            raise ValueError("Citation marker member span is empty")
        if self.local_end - self.local_start != len(self.text):
            raise ValueError("Citation marker member span does not match its text")
        return self


class InTextCitation(BaseModel):
    """An in-text citation extracted from a student paper's body text.

    Represents a passage (quotation or paraphrase) attributed to a cited source.
    Links to ParsedReference via citation_key.
    """

    reference_ids: List[str] = Field(
        default_factory=list,
        description="Application-owned reference IDs linked to this marker member",
    )
    candidate_reference_ids: List[str] = Field(
        default_factory=list,
        description="Candidate IDs retained when deterministic linking is ambiguous",
    )
    link_status: str = Field(
        default="linked",
        description='"linked", "ambiguous", or "missing_reference"',
    )

    text: str = Field(default="", description="The extracted passage (quotation or paraphrase)")
    claim_type: str = Field(
        default="paraphrase",
        description='"quotation" (exact words, in quote marks) or "paraphrase" (student\'s own words)',
    )
    citation_key: str = Field(
        default="",
        description="Links to ParsedReference.citation_key (e.g. 'Smith2020')",
    )
    citation_marker: str = Field(
        default="",
        description='The raw marker, e.g. "(Smith, 2020)" or "Elsaesser (1998) argues"',
    )
    citation_markers: List[CitationMarkerMember] = Field(
        default_factory=list,
        description=(
            "Exact marker spans and source membership for grouped citation units; "
            "empty for legacy single-marker extractions"
        ),
    )
    marker_member: str = Field(
        default="",
        description="The individual member parsed from a compound citation marker",
    )
    marker_type: str = Field(
        default="parenthetical",
        description='"parenthetical" or "narrative"',
    )
    page_number: str = Field(default="", description="Page number if specified (MLA / APA page-specific)")
    paragraph_index: int = Field(default=0, description="Paragraph position in the paper (for reporting)")
    sentence_index: int = Field(default=0, description="Sentence position within the paragraph")
    marker_start: int = Field(default=-1, description="Marker start offset within the paragraph")
    marker_end: int = Field(default=-1, description="Marker end offset within the paragraph")
    passage_start: int = Field(default=-1, description="Passage start offset in normalized paper body")
    passage_end: int = Field(default=-1, description="Passage end offset in normalized paper body")
    is_secondary: bool = Field(
        default=False,
        description='True for "as cited in" secondary citations',
    )
    original_author: str = Field(
        default="",
        description="For secondary citations: who the idea originally belongs to",
    )
    confidence: str = Field(
        default="high",
        description="Extraction confidence: high (regex) / medium (LLM) / low (ambiguous)",
    )
    drop_reason: Optional[str] = Field(
        default=None,
        description=(
            "If set, this citation FAILED validation and should be treated as "
            "dropped (not a real citation). Values: 'hallucinated_key' (key not "
            "in reference list and re-attribution failed), 'text_not_in_original' "
            "(cited text not findable in the original body — likely fabricated/"
            "altered), or 'ambiguous_text_location' (more than one acceptable "
            "body span). Callers should filter "
            "out drop_reason != None. Kept in the list for audit/reporting."
        ),
    )


class ParsedReferenceBatch(BaseModel):
    """A batch of parsed references from the LLM.

    Used for validation and to ensure the LLM returns the correct count.
    """

    references: List[ParsedReference] = Field(
        default_factory=list,
        description="List of parsed references",
    )

    def __len__(self) -> int:
        return len(self.references)

    def __iter__(self):
        return iter(self.references)

    def __getitem__(self, index: int) -> ParsedReference:
        return self.references[index]


class ReferenceParseResult(BaseModel):
    """Result of parsing a batch of references.

    Includes both the parsed references and metadata about the parsing.
    """

    references: List[ParsedReference]
    total_count: int = Field(description="Total number of references in batch")
    parsed_count: int = Field(description="Number successfully parsed")
    failed_count: int = Field(description="Number that failed parsing")
    from_cache: int = Field(default=0, description="Number retrieved from cache")
    llm_calls: int = Field(default=0, description="Number of LLM API calls made")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


# ---------------------------------------------------------------------------
# Subject-identification pass (Phase 3.8 pre-analysis)
#
# One LLM call over the body text + reference list that emits four outputs
# (PLAN.md §3.1, "Subject identification pass (pre-analysis)"):
#   1. The paper's primary subject (what it analyzes)
#   2. Which references are primary sources vs secondary scholarship
#   3. Per-paragraph structure zoning (intro / body / conclusion)
#   4. Topic keywords (stored for the §5 ablation; not consumed by the
#      citation extractor in this step)
# Downstream consumers: citation extraction (subject context), verification
# (primary-vs-secondary classification, structure zoning for section-based
# verification), reporting (keywords, missing-primary-source note).
# ---------------------------------------------------------------------------


class ParagraphRole(str, Enum):
    """Structural role of a paragraph, used for section-based verification
    (PLAN.md §3.1 "Section-based verification"). The three labels are the
    only roles the design specifies."""

    INTRODUCTION = "introduction"
    BODY = "body"
    CONCLUSION = "conclusion"


class ParagraphStructure(BaseModel):
    """The structure-zoning output for a single paragraph."""

    index: int = Field(description="Paragraph index, 0-based, matching the paragraph order passed in")
    role: ParagraphRole = Field(
        default=ParagraphRole.BODY,
        description="intro / body / conclusion (the structure-zoning label)",
    )
    role_rationale: str = Field(
        default="",
        description="Short LLM-given reason for the label (for audit / debugging)",
    )


class ReferenceClassification(BaseModel):
    """Per-reference primary-vs-secondary classification."""

    citation_key: str = Field(
        description="Links to ParsedReference.citation_key (e.g. 'Smith2020')",
    )
    is_primary_source: bool = Field(
        default=False,
        description=(
            "True if this reference is the primary source (object of study: "
            "the film, law, etc.), False if it is secondary scholarship."
        ),
    )
    role_rationale: str = Field(
        default="",
        description="Short LLM-given reason for the classification (for audit / debugging)",
    )


# Allowed primary-source subject types (PLAN.md §3.1, "Primary source
# classification is broad"). Used to normalize the LLM's subject_type output;
# any value not in this set defaults to "other".
PRIMARY_SUBJECT_TYPES = {
    "film", "novel", "play", "poem", "law", "regulation", "court_ruling",
    "government_report", "website", "social_media", "platform", "dataset",
    "software", "other",
}


class FilmAnalysisCandidate(BaseModel):
    """Bound model proposal, not an accepted analysis or omission finding."""
    model_config = {'extra': 'forbid'}
    title: str
    title_start: int
    title_end: int
    passage_start: int
    passage_end: int
    passage_sha256: str
    proposed_role: Literal['substantive_analysis', 'incidental_mention', 'uncertain']


class FilmAnalysisPreflight(BaseModel):
    version: Literal['film-analysis-candidates-v1'] = 'film-analysis-candidates-v1'
    status: Literal['not_requested', 'unavailable', 'invalid', 'bound_candidates', 'no_candidates'] = 'not_requested'
    body_sha256: str = ''
    reference_inventory_sha256: str = ''
    candidates: List[FilmAnalysisCandidate] = Field(default_factory=list)
    rejected_count: int = 0
    # Binding is mechanical; semantic scope and reference absence remain unassessed.
    automatic_findings_enabled: Literal[False] = False


class MediaAnalysisCandidate(FilmAnalysisCandidate):
    title_occurrences: List[tuple[int, int]] = Field(default_factory=list)
    # Legacy observations have no accepted bibliographic role.
    title_role: Literal['work_title', 'publication_title', 'uncertain'] = 'uncertain'
    media_type: Literal['film', 'tv_series', 'tv_episode', 'book', 'album', 'song',
                        'radio_program', 'radio_episode', 'podcast', 'podcast_episode',
                        'play', 'poem', 'video', 'video_game', 'article', 'periodical', 'other', 'unknown']

    @model_validator(mode='after')
    def validate_publication_role(self):
        if self.title_role == 'publication_title' and self.media_type not in {'periodical', 'unknown'}:
            raise ValueError('Publication titles cannot identify an article or another component')
        return self


class MediaAnalysisPreflight(FilmAnalysisPreflight):
    version: Literal['media-analysis-candidates-v1', 'media-analysis-candidates-v2', 'media-analysis-candidates-v3'] = 'media-analysis-candidates-v2'
    candidates: List[MediaAnalysisCandidate] = Field(default_factory=list)


class SubjectIdentification(BaseModel):
    """Top-level result of the subject-identification pass.

    A single object holding all four pre-analysis outputs. Callers (the future
    orchestrator, the citation extractor, the verification engine, reporting)
    compose this with the paper's references and citations — it does NOT mutate
    ParsedReference or InTextCitation.
    """

    primary_subject: str = Field(
        default="",
        description=(
            "What the paper analyzes, e.g. 'film: Example Film (1931)' or "
            "'law: municipal recycling restrictions (2020)'"
        ),
    )
    film_analysis_preflight: FilmAnalysisPreflight = Field(default_factory=FilmAnalysisPreflight)
    media_analysis_preflight: MediaAnalysisPreflight = Field(default_factory=MediaAnalysisPreflight)
    subject_type: str = Field(
        default="other",
        description=(
            "Category of the primary subject. One of: "
            "film/novel/play/poem/law/regulation/court_ruling/"
            "government_report/website/social_media/platform/dataset/"
            "software/other."
        ),
    )
    primary_subject_in_references: bool = Field(
        default=True,
        description=(
            "False triggers the missing-primary-source check: the paper "
            "appears to analyze a subject not present in its reference list."
        ),
    )
    missing_primary_source_note: str = Field(
        default="",
        description=(
            "Populated when primary_subject_in_references is False, e.g. "
            "'This paper appears to analyze [subject] which is not in the "
            "reference list.' Empty when the subject IS referenced."
        ),
    )
    paragraphs: List[ParagraphStructure] = Field(
        default_factory=list,
        description="Per-paragraph structure zoning (intro/body/conclusion)",
    )
    references: List[ReferenceClassification] = Field(
        default_factory=list,
        description="Per-reference primary-vs-secondary classification",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description=(
            "5-10 topic keywords characterizing the paper. Stored for the "
            "§5 ablation (config 5 tests keywords-alone, config 6 the full "
            "pass); not consumed by the citation extractor in this step."
        ),
    )
    model: str = Field(
        default="",
        description="LLM model identity that produced this result (R15 audit/reproducibility)",
    )
    llm_call_succeeded: bool = Field(
        default=True,
        description=(
            "False when the LLM call failed and a safe-default result was "
            "returned. Downstream code should treat a False result as "
            "low-confidence (all paragraphs BODY, no primary-source classification)."
        ),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_llm_reference_array(response_text: str, expected_count: int) -> List[dict]:
    """Validate the LLM response as a JSON array of references.

    Args:
        response_text: Raw JSON text from LLM.
        expected_count: Expected number of references in the array.

    Returns:
        List of validated reference dictionaries.

    Raises:
        ValueError: If response is invalid or count mismatches.
    """
    import json

    text = response_text.strip()

    # Remove any markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last line if they're code block markers
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from LLM: {e}\nResponse: {text[:500]}")

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    if len(data) != expected_count:
        raise ValueError(
            f"Expected {expected_count} references, got {len(data)}. "
            f"This may indicate merged or omitted references."
        )

    # Validate each reference has required fields.
    # NOTE: raw_ref is NOT required here — the LLM doesn't return it (removed
    # from the prompt to cut output size). It's injected post-validation in
    # _parse_batch_with_llm from the regex-split input.
    required_fields = {"author", "year", "title", "doi", "url", "citation_key", "is_media_source"}
    for i, ref in enumerate(data):
        if not isinstance(ref, dict):
            raise ValueError(f"Reference {i} is not an object: {type(ref).__name__}")
        missing = required_fields - set(ref.keys())
        if missing:
            raise ValueError(f"Reference {i} missing fields: {missing}")

    return data
