"""Reference parsing service – orchestrator.

Detects the citation format of a paper and dispatches to the
appropriate format-specific parser.
"""

import json
import logging
from typing import List, Optional

from app.config import settings
from app.services.parsers import detect_format
from app.services.parsers.base_parser import BaseParser
from app.services.llm_service import chat_completion_json
from app.services.schemas import ParsedReference, validate_llm_reference_array
from app.services.prompts import (
    REFERENCE_PARSE_SYSTEM_PROMPT,
    build_reference_parse_system_prompt,
    build_reference_parse_user_prompt,
)
from app.services import doi_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_reference_section(
    text: str,
    format_hint: Optional[str] = None,
) -> Optional[str]:
    """Extract the reference / bibliography section from *text*.

    Parameters
    ----------
    text:
        Full text of the paper.
    format_hint:
        Optional format name (``"apa"``, ``"mla"``, …).  If omitted,
        the format is auto-detected.

    Returns
    -------
    The reference section as a single string, or ``None``.
    """
    parser = _resolve_parser(text, format_hint)
    section = parser.extract_reference_section(text)

    if section:
        logger.info(
            "Extracted reference section (%d chars) using %s",
            len(section),
            parser.__name__,
        )
    else:
        logger.warning("No reference section found using %s", parser.__name__)

    return section


def split_references(
    raw_text: str,
    format_hint: Optional[str] = None,
) -> List[str]:
    """Split a raw reference section into individual reference strings.

    NOTE: This uses regex-based splitting. For better handling of
    student formatting edge cases, use extract_and_parse_references()
    with use_llm_split=True.

    Parameters
    ----------
    raw_text:
        The reference section extracted from the paper.
    format_hint:
        Optional format name.  If omitted the format is auto-detected
        from *raw_text*.

    Returns
    -------
    List of cleaned reference strings.
    """
    if not raw_text:
        return []

    parser = _resolve_parser(raw_text, format_hint)
    refs = parser.split_references(raw_text)

    logger.info("Split into %d references using %s", len(refs), parser.__name__)
    return refs


def extract_and_parse_references(
    raw_text: str,
    format_hint: Optional[str] = None,
    use_llm_split: bool = True,
) -> List[ParsedReference]:
    """Extract and parse references using LLM-first approach.

    This handles student formatting edge cases better than regex:
    - Multi-line references
    - Missing punctuation
    - Inconsistent formatting
    - Mixed formats

    Parameters
    ----------
    raw_text:
        The reference section text.
    format_hint:
        Optional format name (``"apa"``, ``"mla"``).
    use_llm_split:
        If True, use LLM to split AND parse in one call (recommended).
        If False, use regex to split, then LLM to parse.

    Returns
    -------
    List of ParsedReference objects.

    Raises
    ------
    RuntimeError:
        If LLM is not configured or parsing fails.
    """
    if not raw_text:
        return []

    if use_llm_split:
        return _split_and_parse_with_llm(raw_text, format_hint)
    else:
        # Legacy approach: regex split, then LLM parse
        refs = split_references(raw_text, format_hint)
        return parse_reference_batch(refs, format_hint=format_hint)


def _split_and_parse_with_llm(
    raw_text: str,
    format_hint: Optional[str] = None,
) -> List[ParsedReference]:
    """Use LLM to split AND parse references in one call.

    Args:
        raw_text: The raw reference section text.
        format_hint: Optional format hint for the LLM.

    Returns:
        List of ParsedReference objects.
    """
    fmt = format_hint.lower() if format_hint else "apa"
    if fmt == "mla":
        format_name = "MLA 9th edition"
    else:
        format_name = "APA 7th edition"

    system_prompt = f"""You are an expert at parsing academic citations in {format_name} format.

Your task:
1. Identify ALL individual references in the text
2. Parse each reference into structured fields

Extract these fields for each reference:
- author: Authors in the correct {format_name} format
- year: Publication year (4 digits) or "n.d." if no date. In APA, year follows the author. In MLA, year appears near the end after the publisher.
- title: The work title
- doi: DOI without "https://doi.org/" prefix, or empty string if none
- url: Full URL if no DOI
- citation_key: First author surname + year (e.g., "Smith2020"). If no author, use first significant word of title + year.
- is_media_source: true if film, TV show, album, painting, or other non-academic media
- raw_ref: The original reference text

Return a JSON object with a "references" array containing all references found.

Be thorough - do not miss any references. Handle variations in formatting gracefully.
If a reference spans multiple lines, merge them into one."""

    user_prompt = f"""Parse all references from this reference section:

{raw_text}

Return a JSON object with a "references" array."""

    try:
        response_data = chat_completion_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=settings.LLM_MODEL,
            temperature=0.0,
            max_tokens=settings.LLM_MAX_TOKENS,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    # Extract references array
    if isinstance(response_data, dict) and "references" in response_data:
        ref_list = response_data["references"]
    elif isinstance(response_data, list):
        ref_list = response_data
    else:
        raise RuntimeError(f"Unexpected LLM response structure: {type(response_data)}")

    logger.info("LLM extracted %d references from raw text", len(ref_list))

    # Convert to ParsedReference objects
    parsed_refs = []
    for i, ref_data in enumerate(ref_list):
        try:
            parsed = ParsedReference(**ref_data)
        except Exception as e:
            logger.warning("Failed to validate ref %d: %s", i, e)
            parsed = ParsedReference(
                raw_ref=ref_data.get("raw_ref", ""),
                author=ref_data.get("author", ""),
                year=ref_data.get("year", "n.d."),
                title=ref_data.get("title", ""),
                doi=ref_data.get("doi", ""),
                url=ref_data.get("url", ""),
                citation_key=ref_data.get("citation_key", ""),
                is_media_source=ref_data.get("is_media_source", False),
            )
        parsed_refs.append(parsed)

    # Cache results
    if settings.CACHE_ENABLED:
        for parsed in parsed_refs:
            if parsed.doi or parsed.title:
                doi_cache.cache_reference(
                    data=parsed.model_dump(),
                    doi=parsed.doi,
                    title=parsed.title,
                )

    return parsed_refs


def parse_reference_batch(
    refs: List[str],
    batch_size: Optional[int] = None,
    use_cache: bool = True,
    format_hint: Optional[str] = None,
) -> List[ParsedReference]:
    """Parse raw reference strings into structured data using LLM.

    This function:
    1. Checks cache for already-parsed references
    2. Sends uncached references to LLM in batches
    3. Validates LLM output
    4. Caches results for future use

    Parameters
    ----------
    refs:
        List of raw reference strings.
    batch_size:
        Number of references per LLM call (default from settings).
    use_cache:
        Whether to use caching (default True).
    format_hint:
        Optional format name ("apa", "mla") for format-aware prompting.

    Returns
    -------
    List of ParsedReference objects in the same order as input.

    Raises
    ------
    RuntimeError:
        If LLM is not configured or all parsing fails.
    """
    if not refs:
        return []

    batch_size = batch_size or settings.LLM_BATCH_SIZE
    results: List[Optional[ParsedReference]] = [None] * len(refs)

    # Step 1: Check cache for already-parsed references
    cached_count = 0
    uncached_indices: List[int] = []

    if use_cache and settings.CACHE_ENABLED:
        for i, ref in enumerate(refs):
            cached_data = doi_cache.get_cached_reference(
                doi=doi_cache._extract_doi_heuristic(ref),
                title=doi_cache._extract_title_heuristic(ref),
            )
            if cached_data:
                try:
                    results[i] = ParsedReference(**cached_data)
                    cached_count += 1
                except Exception as e:
                    logger.warning("Cached data invalid for ref %d: %s", i, e)
                    uncached_indices.append(i)
            else:
                uncached_indices.append(i)
        logger.info("Cache hits: %d/%d references", cached_count, len(refs))
    else:
        uncached_indices = list(range(len(refs)))

    # Step 2: If all cached, return early
    if not uncached_indices:
        logger.info("All %d references retrieved from cache", len(refs))
        return results  # type: ignore

    # Step 3: Process uncached references in batches
    llm_call_count = 0
    errors: List[str] = []

    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(uncached_indices))
        batch_indices = uncached_indices[batch_start:batch_end]
        batch_refs = [refs[i] for i in batch_indices]

        logger.info(
            "Processing batch %d-%d (%d references)",
            batch_start + 1,
            batch_end,
            len(batch_refs),
        )

        try:
            batch_results = _parse_batch_with_llm(batch_refs, format_hint=format_hint)
            llm_call_count += 1

            # Step 4: Store results and cache
            for idx, parsed in zip(batch_indices, batch_results):
                results[idx] = parsed

                # Cache for future use
                if use_cache and settings.CACHE_ENABLED:
                    doi_cache.cache_reference(
                        data=parsed.model_dump(),
                        doi=parsed.doi,
                        title=parsed.title,
                    )

        except Exception as e:
            error_msg = f"Batch {batch_start}-{batch_end} failed: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

            # Mark failed references with empty data
            for idx in batch_indices:
                results[idx] = ParsedReference(
                    author="",
                    year="n.d.",
                    title="",
                    doi="",
                    url="",
                    raw_ref=refs[idx],
                    citation_key="",
                    is_media_source=False,
                )

    # Step 5: Log summary
    successful = sum(1 for r in results if r is not None)
    logger.info(
        "Parsed %d/%d references (%d from cache, %d via LLM, %d failed)",
        successful,
        len(refs),
        cached_count,
        successful - cached_count,
        len(refs) - successful,
    )

    if errors:
        logger.warning("Parsing errors: %s", "; ".join(errors))

    # Return all results (None becomes empty ParsedReference)
    return [r or ParsedReference(raw_ref=refs[i]) for i, r in enumerate(results)]


def _parse_batch_with_llm(
    refs: List[str],
    format_hint: Optional[str] = None,
) -> List[ParsedReference]:
    """Send a batch of references to the LLM for parsing.

    Args:
        refs: List of raw reference strings.
        format_hint: Optional format name ("apa", "mla") for format-aware
            prompting. Defaults to "apa".

    Returns:
        List of ParsedReference objects.

    Raises:
        RuntimeError: If LLM call or validation fails.
    """
    # Build prompts (format-aware)
    system_prompt_template = build_reference_parse_system_prompt(format_hint or "apa")
    system_prompt = system_prompt_template.format(reference_count=len(refs))
    user_prompt = build_reference_parse_user_prompt(refs)

    # Call LLM
    try:
        response_data = chat_completion_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    # Validate response
    # The LLM may return {"references": [...]} or just [...]
    if isinstance(response_data, dict) and "references" in response_data:
        ref_list = response_data["references"]
    elif isinstance(response_data, list):
        ref_list = response_data
    else:
        raise RuntimeError(f"Unexpected LLM response structure: {type(response_data)}")

    # Validate count and structure
    validated_data = validate_llm_reference_array(
        json.dumps(ref_list),
        expected_count=len(refs),
    )

    # Convert to ParsedReference objects
    parsed_refs = []
    for i, ref_data in enumerate(validated_data):
        try:
            parsed = ParsedReference(**ref_data)
        except Exception as e:
            logger.warning("Failed to validate ref %d: %s", i, e)
            # Fallback with raw_ref preserved
            parsed = ParsedReference(
                raw_ref=refs[i],
                author=ref_data.get("author", ""),
                year=ref_data.get("year", "n.d."),
                title=ref_data.get("title", ""),
                doi=ref_data.get("doi", ""),
                url=ref_data.get("url", ""),
                citation_key=ref_data.get("citation_key", ""),
                is_media_source=ref_data.get("is_media_source", False),
            )
        parsed_refs.append(parsed)

    return parsed_refs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_parser(text: str, hint: Optional[str] = None) -> type[BaseParser]:
    """Return the parser class to use.

    If *hint* is given, map it to the matching parser.  Otherwise
    auto-detect from *text*.
    """
    if hint is not None:
        parser = _lookup_by_name(hint)
        if parser is not None:
            return parser
        logger.warning("Unknown format hint %r – falling back to auto-detect", hint)
    return detect_format(text)


_NAME_MAP: dict[str, type[BaseParser]] = {}


def _lookup_by_name(name: str) -> Optional[type[BaseParser]]:
    """Map a short name (``"apa"``, ``"mla"``) to a parser class."""
    if not _NAME_MAP:
        from app.services.parsers.apa_parser import ApaParser
        from app.services.parsers.mla_parser import MlaParser

        _NAME_MAP["apa"] = ApaParser
        _NAME_MAP["mla"] = MlaParser
    return _NAME_MAP.get(name.lower())


# ---------------------------------------------------------------------------
# Identifier completeness reporting
# ---------------------------------------------------------------------------


def count_missing_identifiers(references: List[ParsedReference]) -> dict:
    """Count references lacking a DOI or URL, for instructor reporting.

    Instructors are encouraged to require DOIs and URLs in student references.
    This function summarises which references are missing identifiers so the
    instructor can penalise accordingly and so the resolver knows which
    references will fall through to the lower-precision title-search path.

    Not every source *should* have a URL/DOI, however. The count distinguishes:

      - **Traditional media** (films, TV shows, albums, artwork, live
        performances): these predate the web and are identified by
        title/creator/year/studio per citation convention. Missing identifiers
        here are NOT penalised — informational only.
      - **Physical archives** (manuscripts, special collections, theses,
        dissertations cited by repository + box/folder): legitimately lack a
        URL/DOI and cannot be verified automatically. NOT penalised.
      - **All other sources** (books, journal articles, websites, social media,
        podcasts, digital-only videos): these should have a DOI or URL
        (electronic versions and library catalogue pages provide them).
        Missing identifiers here ARE penalisable.

    Args:
        references: Parsed references from ``extract_and_parse_references``.

    Returns:
        dict with keys:
            total: total reference count
            with_doi: references with a DOI
            with_url: references with a URL (but no DOI)
            missing_traditional_media: missing identifiers, but the source is
                traditional media (film/TV/album/artwork) — no penalty
            missing_archive: missing identifiers, but the source is a physical
                archive / thesis / manuscript — no penalty, unverifiable
            missing_should_have: missing identifiers, source should have one
                (article/website/podcast/digital) — penalisable
            missing_examples: up to 10 raw_ref snippets of missing-should-have
                              refs, for the instructor to review
    """
    import re

    # A URL may live in the structured ``url`` field or embedded in raw_ref.
    _url_re = re.compile(r"https?://|www\.", re.IGNORECASE)

    # Source-type detection uses the shared module (canonical regexes, kept in
    # sync with the resolver routing).
    from app.services.source_type import is_traditional_media, is_archive_source

    with_doi = 0
    with_url = 0
    missing_traditional = 0
    missing_archive = 0
    missing_should_have: list[str] = []

    for r in references:
        has_doi = bool(r.doi and r.doi.strip())
        has_url = bool(r.url and r.url.strip()) or bool(
            r.raw_ref and _url_re.search(r.raw_ref)
        )
        if has_doi:
            with_doi += 1
        elif has_url:
            with_url += 1
        else:
            raw = (r.raw_ref or "")
            if is_traditional_media(raw):
                missing_traditional += 1
            elif is_archive_source(raw):
                missing_archive += 1
            else:
                snippet = (raw or r.title or "(no text)").strip()
                if len(snippet) > 100:
                    snippet = snippet[:97] + "…"
                missing_should_have.append(snippet)

    return {
        "total": len(references),
        "with_doi": with_doi,
        "with_url": with_url,
        "missing_traditional_media": missing_traditional,
        "missing_archive": missing_archive,
        "missing_should_have": len(missing_should_have),
        "missing_examples": missing_should_have[:10],
    }


