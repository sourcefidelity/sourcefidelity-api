"""Pydantic schemas for LLM structured output validation.

These schemas define the expected structure for LLM responses,
ensuring type safety and catching malformed output.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ParsedReference(BaseModel):
    """A single parsed reference from the LLM.

    This is the canonical parsed reference format used across the system.
    """

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
    citation_key: str = Field(
        default="",
        description="First author surname plus year, e.g. 'Smith2020'",
    )
    is_media_source: bool = Field(
        default=False,
        description="True for film, TV, podcast, song, video, game, or similar media",
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

    @field_validator("doi", mode="before")
    @classmethod
    def clean_doi(cls, v: str) -> str:
        """Remove https://doi.org/ prefix if present."""
        if isinstance(v, str) and v.startswith("https://doi.org/"):
            return v.replace("https://doi.org/", "")
        return v or ""

    @field_validator("year", mode="before")
    @classmethod
    def clean_year(cls, v: str) -> str:
        """Ensure year is 4 digits or 'n.d.'."""
        if not v:
            return "n.d."
        v = str(v).strip()
        if v.isdigit() and len(v) == 4:
            return v
        # Try to extract 4-digit year
        import re
        match = re.search(r"\b(19|20)\d{2}\b", v)
        if match:
            return match.group(0)
        return "n.d."


class InTextCitation(BaseModel):
    """An in-text citation extracted from a student paper's body text.

    Represents a passage (quotation or paraphrase) attributed to a cited source.
    Links to ParsedReference via citation_key.
    """

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
    marker_type: str = Field(
        default="parenthetical",
        description='"parenthetical" or "narrative"',
    )
    page_number: str = Field(default="", description="Page number if specified (MLA / APA page-specific)")
    paragraph_index: int = Field(default=0, description="Paragraph position in the paper (for reporting)")
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
            "altered, the R27 prompt-injection defense). Callers should filter "
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
            "What the paper analyzes, e.g. 'film: Dracula (1931)' or "
            "'law: MRF import restrictions (China, 1990)'"
        ),
    )
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