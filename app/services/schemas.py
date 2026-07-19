"""Pydantic schemas for LLM structured output validation.

These schemas define the expected structure for LLM responses,
ensuring type safety and catching malformed output.
"""

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

    # Validate each reference has required fields
    required_fields = {"author", "year", "title", "doi", "url", "raw_ref", "citation_key", "is_media_source"}
    for i, ref in enumerate(data):
        if not isinstance(ref, dict):
            raise ValueError(f"Reference {i} is not an object: {type(ref).__name__}")
        missing = required_fields - set(ref.keys())
        if missing:
            raise ValueError(f"Reference {i} missing fields: {missing}")

    return data