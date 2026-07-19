"""LLM prompts for reference parsing and other tasks.

This module contains the system and user prompts used for LLM calls.
Reference parsing prompts for LLM-based extraction.
"""

from typing import List
import json


# ---------------------------------------------------------------------------
# Reference Parsing Prompts
# ---------------------------------------------------------------------------

# Shared scaffolding for all formats — field descriptions vary by format.
# NOTE: {reference_count} is left as a literal placeholder for the caller
# to fill via .format(reference_count=N).
_REFERENCE_PARSE_TEMPLATE = """You are a reference extraction system. Extract structured information from each reference according to {format_name}.

The input contains exactly {{reference_count}} references.

You MUST output a single JSON array containing exactly {{reference_count}} objects.
Do NOT output a single object.
Do NOT output text before or after the array.
Do NOT use Markdown.
Preserve the same order as the input references.
Do not merge references.
Do not omit references.

Each object must have exactly these fields:
{field_descriptions}

If a field cannot be extracted, use "" for strings and false for boolean."""


_APA_FIELD_DESCRIPTIONS = """- "author": string. Format as "Surname, Initial." (e.g. "Bordwell, D."). Use "&" before the final author. For institutional authors, use the full name (e.g. "Ministry of Commerce.").
- "year": string, 4-digit year or "n.d." The year appears in parentheses after the author(s), e.g. "(2021)".
- "title": string. Sentence case for articles, Title Case for books.
- "doi": string, plain DOI only. Remove "https://doi.org/" if present.
- "url": string. If DOI exists, use "https://doi.org/[DOI]"; otherwise the original URL if present; otherwise "".
- "raw_ref": string, the original input reference string
- "citation_key": string, first author surname plus year, e.g. "Smith2020"
- "is_media_source": boolean, true only for film, TV, podcast, song, video, game, or similar media"""


_MLA_FIELD_DESCRIPTIONS = """- "author": string. Format as "Lastname, Firstname." with the full first name (e.g. "Bordwell, David."). MLA does NOT use initials. For three or more authors, use "Lastname, Firstname, et al."
- "year": string, 4-digit year or "n.d." In MLA the year appears near the END of the reference (typically after the publisher). Look for it there, not after the author.
- "title": string. For articles/essays/chapters: the title is in quotation marks (e.g. "The History of Film"). For books/journals/films/websites: the title is italicised. Preserve the distinction in the title string.
- "doi": string, plain DOI only, or "" if none. MLA rarely uses DOIs — most entries use URLs instead.
- "url": string. MLA commonly uses URLs or permalinks. Include the full URL if present, otherwise "".
- "raw_ref": string, the original input reference string
- "citation_key": string, first author surname plus year, e.g. "Bordwell2006". If no author, use the first significant word of the title plus year, e.g. "History2023".
- "is_media_source": boolean, true only for film, TV, podcast, song, video, game, or similar media"""


def build_reference_parse_system_prompt(format_hint: str = "apa") -> str:
    """Build the system prompt for reference parsing, format-aware.

    Returns a template string with ``{reference_count}`` still embedded,
    ready for ``.format(reference_count=N)`` by the caller.

    Args:
        format_hint: ``"apa"`` or ``"mla"`` (case-insensitive).

    Returns:
        Format-aware system prompt template.
    """
    fmt = format_hint.lower()
    if fmt == "mla":
        format_name = "MLA 9th edition"
        field_descriptions = _MLA_FIELD_DESCRIPTIONS
    else:
        format_name = "APA 7th edition"
        field_descriptions = _APA_FIELD_DESCRIPTIONS

    return _REFERENCE_PARSE_TEMPLATE.format(
        format_name=format_name,
        field_descriptions=field_descriptions,
    )


# Backward-compatible constant — old callers that import
# REFERENCE_PARSE_SYSTEM_PROMPT still work (APA default).
REFERENCE_PARSE_SYSTEM_PROMPT = build_reference_parse_system_prompt("apa")


def build_reference_parse_user_prompt(references: List[str]) -> str:
    """Build the user prompt for reference parsing.

    Args:
        references: List of raw reference strings.

    Returns:
        Formatted user prompt with the reference count and JSON array.
    """
    ref_count = len(references)
    batch_json = json.dumps(references, ensure_ascii=False, indent=2)

    return f"""The input contains exactly {ref_count} references.

Input raw references as a JSON array of strings:
{batch_json}

Output only the JSON array."""


# ---------------------------------------------------------------------------
# APA Format Checking Prompts (Phase 3.12)
# ---------------------------------------------------------------------------


APA_CHECK_SYSTEM_PROMPT = """You are an APA 7th edition format checker. Analyze the provided reference and identify any format errors.

Check for:
1. Author format: "Surname, Initial." with commas and ampersands correctly placed
2. Year: In parentheses after authors
3. Title: Italicized for books/reports, sentence case for articles
4. Journal name: Italicized, Title Case
5. Volume: Italicized
6. Issue: In parentheses, not italicized
7. Page range: pp. for books, just numbers for journals
8. DOI: As "https://doi.org/xxxxx" format (not "doi:xxxxx")
9. Alphabetization: Should be in author surname order

Output a JSON object with:
- "is_correct": boolean
- "errors": array of strings describing each error found
- "corrections": array of suggested corrections"""


def build_apa_check_user_prompt(reference: str) -> str:
    """Build the user prompt for APA format checking.

    Args:
        reference: A single reference string to check.

    Returns:
        Formatted user prompt.
    """
    return f"""Check this reference for APA 7th edition format errors:

{reference}

Output only the JSON object."""


# ---------------------------------------------------------------------------
# Fabricated Paraphrase Detection Prompts (Phase 13.1)
# ---------------------------------------------------------------------------


FABRICATION_CHECK_SYSTEM_PROMPT = """You are an academic integrity verification system. Your job is to determine if claims attributed to a source are actually supported by that source.

You will receive:
1. A claim or paraphrase from a student paper
2. The full text of the cited source

Determine:
1. Does the source contain information that directly supports this claim?
2. Does the source contradict the claim?
3. Is the claim a fabrication (invented by an LLM or student)?

Output a JSON object with:
- "is_supported": boolean - does the source support the claim?
- "is_contradicted": boolean - does the source contradict the claim?
- "is_fabricated": boolean - does the claim appear invented?
- "confidence": float between 0 and 1
- "evidence": string - the exact passage from the source that relates to the claim, or "" if none
- "explanation": string - brief explanation of your judgment"""


def build_fabrication_check_user_prompt(claim: str, source_text: str, max_chars: int = 50000) -> str:
    """Build the user prompt for fabrication detection.

    Args:
        claim: The claim or paraphrase to verify.
        source_text: The full text of the cited source.
        max_chars: Maximum characters to include from source (to avoid token limits).

    Returns:
        Formatted user prompt.
    """
    # Truncate source text if needed
    truncated_source = source_text[:max_chars]
    if len(source_text) > max_chars:
        truncated_source += f"\n\n[...TRUNCATED - {len(source_text) - max_chars:,} characters omitted...]"

    return f"""Verify this claim against the source:

CLAIM:
{claim}

SOURCE TEXT:
{truncated_source}

Output only the JSON object."""