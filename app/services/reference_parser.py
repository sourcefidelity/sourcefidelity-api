"""Reference parsing service – orchestrator.

Detects the citation format of a paper and dispatches to the
appropriate format-specific parser.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    use_regex_first: bool = True,
) -> List[ParsedReference]:
    """Extract and parse references into structured ParsedReference objects.

    Three extraction strategies, selected by flags:

    **Regex-first** (``use_regex_first=True``, default — RECOMMENDED):
        Regex splits references, regex extracts fields, LLM per-reference
        fallback for edge cases. No JSON — eliminates truncation. Fast,
        model-neutral, robust. Use this for all production paths.

    **LLM split** (``use_llm_split=True, use_regex_first=False``):
        One-shot LLM call splits AND parses into JSON. Higher quality when it
        works but truncates on large reference sections. Use only for small
        reference lists (< 10 refs).

    **Legacy two-step** (``use_llm_split=False, use_regex_first=False``):
        Regex splits, then LLM parses in batches of 10 via JSON. The previous
        default. Retained for backward compatibility and ablation testing.

    Parameters
    ----------
    raw_text:
        The reference section text.
    format_hint:
        Optional format name (``"apa"``, ``"mla"``).
    use_llm_split:
        If True and use_regex_first is False, use LLM to split AND parse in
        one call. Ignored when use_regex_first is True.
    use_regex_first:
        If True (default), use the regex-first flow with LLM per-reference
        fallback. This is the recommended path — no JSON, model-neutral,
        robust to truncation.

    Returns
    -------
    List of ParsedReference objects.

    Raises
    ------
    RuntimeError:
        If parsing fails entirely (rare — the regex-first path degrades
        gracefully via fallback).
    """
    if not raw_text:
        return []

    # ── Regex-first flow (recommended default) ───────────────────────────
    if use_regex_first:
        fmt = (format_hint or "").lower()
        # Split: APA uses regex (works well, 99.3% recall). MLA uses LLM cleanup
        # (text in/text out) because MLA double-spacing defeats regex-based line
        # merging — the LLM understands reference structure, regex heuristics
        # can't distinguish period-within-reference from period-at-end-of-reference.
        if fmt == "mla":
            try:
                raw_refs = _mla_cleanup_split(raw_text)
            except Exception:
                # Native MLA splitter is the fallback (may over-count on
                # double-spaced papers, but better than nothing when LLM is down)
                raw_refs = split_references(raw_text, format_hint)
        else:
            raw_refs = split_references(raw_text, format_hint)

        return _extract_fields_regex_first(raw_refs, format_hint)

    # ── Legacy flows (retained for backward compat / ablation) ───────────
    if use_llm_split:
        return _split_and_parse_with_llm(raw_text, format_hint)
    else:
        refs = split_references(raw_text, format_hint)
        return parse_reference_batch(refs, format_hint=format_hint)


def _regex_fallback_references(
    raw_text: str,
    format_hint: Optional[str] = None,
) -> List[ParsedReference]:
    """Last-resort reference extraction when the LLM fails entirely.

    Uses the regex-based parser to split the reference section into individual
    entries, then creates minimal ParsedReference objects with needs_review=True.
    Structured fields (author, year, title, doi) are left empty — the raw text
    is preserved so retrieval can still attempt title/DOI extraction, and the
    instructor sees the reference flagged for manual verification in the report.

    This is the graceful-degradation path: a malformed reference that defeats
    the LLM must not abort the whole paper. The paper still gets processed;
    only the structured-field extraction is marked unreliable for these refs.
    """
    try:
        raw_refs = split_references(raw_text, format_hint)
    except Exception:
        raw_refs = [raw_text] if raw_text.strip() else []

    parsed_refs = [
        ParsedReference(raw_ref=raw.strip(), needs_review=True, extraction_method="fallback")
        for raw in raw_refs
        if raw.strip()
    ]
    logger.info(
        "Regex fallback produced %d references (all flagged needs_review)", len(parsed_refs)
    )
    return parsed_refs


# ---------------------------------------------------------------------------
# Regex-first field extraction with LLM per-reference fallback (no JSON)
#
# The new architecture: regex extracts fields (fast, deterministic), LLM handles
# edge cases only (one reference at a time, plain-text output, no truncation).
# This replaces the JSON-batch approach for high-volume extraction.
# ---------------------------------------------------------------------------


def _extract_fields_with_llm(
    ref: str,
    format_hint: Optional[str] = None,
) -> ParsedReference:
    """LLM per-reference field extraction fallback (plain-text, NOT JSON).

    Sends a single reference to the LLM and asks for labeled plain-text output
    (Author: ..., Year: ..., Title: ...). The response is parsed by
    ref_field_extractor.extract_fields_from_llm_response.

    This is the edge-case handler: tiny output (~50 tokens), no truncation risk,
    works on any model including small local models. The result is flagged
    needs_review=True because LLM extraction is less reliable than regex.

    Args:
        ref: A single reference string.
        format_hint: Optional format name (unused in prompt — the LLM reads the ref).

    Returns:
        ParsedReference with extraction_method="llm", needs_review=True.
    """
    from app.services.llm_service import chat_completion
    from app.services.prompts import (
        PER_REFERENCE_EXTRACT_SYSTEM_PROMPT,
        build_per_reference_extract_user_prompt,
    )

    try:
        response = chat_completion(
            system_prompt=PER_REFERENCE_EXTRACT_SYSTEM_PROMPT,
            user_prompt=build_per_reference_extract_user_prompt(ref),
            model=settings.LLM_MODEL,
            temperature=0.0,
            max_tokens=2000,  # ample for the 5-line answer
            # Low-stakes per-reference extraction — disable thinking to avoid
            # reasoning-budget empties and gain ~8x speedup. Subject-ID test
            # (Aug 9) showed thinking-off is safe for structured extraction.
            disable_thinking=True,
        )
        from app.services.ref_field_extractor import extract_fields_from_llm_response
        return extract_fields_from_llm_response(response, ref)
    except Exception as e:
        logger.warning("LLM per-reference fallback failed: %s", str(e)[:80])
        return ParsedReference(
            raw_ref=ref, needs_review=True, extraction_method="fallback"
        )


def _extract_fields_regex_first(
    raw_refs: List[str],
    format_hint: Optional[str] = None,
    use_llm_fallback: bool = True,
) -> List[ParsedReference]:
    """Regex-first field extraction with optional LLM per-reference fallback.

    For each reference:
    1. Try regex field extraction (format-specific patterns).
    2. If regex fails (no author or no title), try LLM per-reference fallback.
    3. If both fail, mark as fallback (empty fields, needs_review).

    The LLM fallback calls run concurrently — each is independent and I/O-bound.

    Args:
        raw_refs: List of individual reference strings (post-split).
        format_hint: "apa" or "mla" — selects the regex pattern set.
        use_llm_fallback: If True, refs that fail regex go to the LLM. If False,
            they're marked "fallback" directly (no LLM dependency).

    Returns:
        List of ParsedReference objects in the same order as input.
    """
    from app.services.ref_field_extractor import extract_fields_apa, extract_fields_mla

    fmt = (format_hint or "apa").lower()
    extractor = extract_fields_mla if fmt == "mla" else extract_fields_apa

    # Phase 1: regex extraction on all refs
    results: list[ParsedReference | None] = [None] * len(raw_refs)
    failed_indices: list[int] = []
    for i, ref in enumerate(raw_refs):
        ref = ref.strip()
        if not ref:
            results[i] = ParsedReference(raw_ref="", extraction_method="fallback")
            continue
        parsed = extractor(ref)
        if parsed is not None:
            results[i] = parsed
        else:
            failed_indices.append(i)

    regex_success = len(raw_refs) - len(failed_indices)
    logger.info(
        "Regex field extraction: %d/%d succeeded, %d need %s",
        regex_success,
        len(raw_refs),
        len(failed_indices),
        "LLM fallback" if use_llm_fallback else "flagging",
    )

    # Phase 2: LLM fallback for failed refs (concurrent)
    if failed_indices and use_llm_fallback:
        def _fallback_one(idx: int) -> tuple[int, ParsedReference]:
            return idx, _extract_fields_with_llm(raw_refs[idx], format_hint)

        max_workers = min(len(failed_indices), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fallback_one, i) for i in failed_indices]
            for fut in as_completed(futures):
                idx, parsed = fut.result()
                results[idx] = parsed
    elif failed_indices:
        # No LLM fallback — mark as fallback directly
        for i in failed_indices:
            results[i] = ParsedReference(
                raw_ref=raw_refs[i], needs_review=True, extraction_method="fallback"
            )

    # Cache successful results
    if settings.CACHE_ENABLED:
        for parsed in results:
            if parsed and (parsed.doi or parsed.title):
                doi_cache.cache_reference(
                    data=parsed.model_dump(),
                    doi=parsed.doi,
                    title=parsed.title,
                )

    return results  # type: ignore


def _mla_cleanup_split(raw_text: str) -> List[str]:
    """Split MLA reference section using LLM cleanup (text in / text out, NOT JSON).

    MLA splitting is harder to regex than APA. Instead of the one-shot JSON
    approach (which truncates), this uses a plain-text cleanup call: the LLM
    normalizes the reference list to one-per-line, then simple line-splitting
    handles the rest. No structured output, no truncation risk.

    Falls back to regex splitting if the LLM call fails.

    Args:
        raw_text: The raw MLA reference section text.

    Returns:
        List of individual reference strings.
    """
    from app.services.llm_service import chat_completion
    from app.services.prompts import MLA_CLEANUP_SYSTEM_PROMPT, build_mla_cleanup_user_prompt

    # Reasoning-effort fix (Aug 9, measured): the empty-response flakiness was
    # reasoning-budget EXHAUSTION, not random noise. Controlled test on the
    # Digital paper (3 runs each) with reasoning_content + usage captured:
    #   - default (no effort control): reasoning ran to ~32K chars, hit the
    #     max_tokens cap (finish_reason=length), produced 0 output. 0/3 correct.
    #   - reasoning_effort="high": same runaway risk — 2/3, 1 run reproduced empty.
    #   - reasoning_effort="low": 3/3 correct, 3/3 non-empty. Reasoning stayed
    #     bounded (~10-15K chars), finished cleanly (stop), correct count.
    #   - thinking OFF: 3/3 non-empty but 0/3 correct — over-splits 12->24
    #     (reasoning is genuinely needed to judge reference vs line-wrap boundaries).
    # "low" is the sweet spot: enough reasoning for correct boundaries, bounded
    # enough to never exhaust the budget. Prior approaches that failed: retry-
    # on-empty (failures are correlated — 3 empties in a row observed, retries
    # don't help); max_tokens=16384 (already at model cap, output is ~300 tokens
    # so budget isn't the lever); local model (llama3.1:8b echoes input + adds
    # preamble, 1/10 correct — does not merge wrapped lines at all).
    #
    # CAVEAT (per DeepSeek docs): the effort string maps per-model. "low" on
    # deepseek-v4-flash = low effort (what was tested); the SAME "low" string on
    # deepseek-v4-pro maps to "high" effort. If LLM_MODEL changes to v4-pro, the
    # effective effort level silently changes — re-validate if switching models.
    # This is a §5 ablation candidate: confirm "low" on the full MLA corpus.
    MAX_EMPTY_RETRIES = 2  # defense-in-depth; catches rare low-effort empties

    cleaned = ""
    try:
        for attempt in range(MAX_EMPTY_RETRIES + 1):
            cleaned = chat_completion(
                system_prompt=MLA_CLEANUP_SYSTEM_PROMPT,
                user_prompt=build_mla_cleanup_user_prompt(raw_text),
                model=settings.LLM_MODEL,
                temperature=0.0,
                max_tokens=settings.LLM_MAX_TOKENS,  # reasoning models need room for thinking + output
                reasoning_effort="low",  # caps reasoning to prevent budget exhaustion (see note above)
            )
            if cleaned.strip():
                break
            if attempt < MAX_EMPTY_RETRIES:
                logger.info("MLA cleanup returned empty (attempt %d/%d) — retrying",
                            attempt + 1, MAX_EMPTY_RETRIES + 1)
        # Split cleaned text into lines, filter empty
        refs = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if refs:
            logger.info("MLA cleanup-split produced %d references (after %d attempt(s))",
                        len(refs), attempt + 1)
            return refs
        logger.warning("MLA cleanup returned empty after %d attempts — falling back to line split",
                       MAX_EMPTY_RETRIES + 1)
    except Exception as e:
        logger.warning("MLA cleanup-split failed (%s) — falling back to line split", str(e)[:60])

    # MLA's split_references() raises NotImplementedError, so fall back to
    # simple line-splitting: split on blank lines and numbered entries.
    import re as _re
    lines = raw_text.strip().split("\n")
    refs = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                refs.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        refs.append(" ".join(current))
    logger.info("MLA line-split fallback produced %d references", len(refs))
    return refs


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
        # Graceful degradation: the LLM failed entirely (after retries + salvage).
        # This happens on malformed references that defeat the model — e.g.
        # merged author names, broken formatting. Rather than aborting the whole
        # paper, fall back to regex splitting (which is robust to formatting
        # quirks) and flag every reference as needing manual review. The
        # instructor sees the references in the report and can verify them by
        # hand. The paper still gets processed for retrieval/verification; only
        # the structured-field extraction is marked unreliable.
        logger.warning(
            "LLM reference parsing failed (%s); falling back to regex split with "
            "needs_review flag on all %d-char reference section",
            str(e)[:120],
            len(raw_text),
        )
        return _regex_fallback_references(raw_text, format_hint)

    # Extract references array
    if isinstance(response_data, dict) and "references" in response_data:
        ref_list = response_data["references"]
    elif isinstance(response_data, list):
        ref_list = response_data
    else:
        raise RuntimeError(f"Unexpected LLM response structure: {type(response_data)}")

    logger.info("LLM extracted %d references from raw text", len(ref_list))

    # Detect possible truncation: if the regex splitter finds significantly more
    # references than the LLM returned, the LLM likely truncated (salvage recovered
    # only the leading subset). Flag the returned refs for review so the instructor
    # knows the parse may be incomplete, and append the unparsed remainder via regex
    # so no references are silently dropped.
    try:
        regex_refs = split_references(raw_text, format_hint)
    except Exception:
        regex_refs = []
    truncated = len(regex_refs) > len(ref_list) + 2  # +2 tolerance for LLM merges
    if truncated:
        logger.warning(
            "Possible LLM truncation: regex found %d refs, LLM returned %d — "
            "flagging returned refs needs_review and appending regex-only refs",
            len(regex_refs),
            len(ref_list),
        )

    # Convert to ParsedReference objects
    parsed_refs = []
    seen_raw = set()
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
        if truncated:
            parsed.needs_review = True
        parsed_refs.append(parsed)
        if parsed.raw_ref:
            seen_raw.add(parsed.raw_ref.strip().lower())

    # If truncation detected, append references the regex splitter found that the
    # LLM missed — flagged needs_review so the instructor checks them manually.
    if truncated:
        for raw in regex_refs:
            if raw.strip() and raw.strip().lower() not in seen_raw:
                parsed_refs.append(
                    ParsedReference(
                        raw_ref=raw.strip(),
                        needs_review=True,
                    )
                )
                seen_raw.add(raw.strip().lower())

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

    # Step 3: Process uncached references in batches (concurrently).
    # Batches are independent (different reference slices), so they can run in
    # parallel. LLM calls are I/O-bound (network wait), so threads give a real
    # speedup: N batches finish in ~1 batch's wall-time instead of N×.
    llm_call_count = 0
    errors: List[str] = []

    # Build the batch list first
    batch_jobs: list[tuple[list[int], list[str]]] = []
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(uncached_indices))
        batch_indices = uncached_indices[batch_start:batch_end]
        batch_refs = [refs[i] for i in batch_indices]
        batch_jobs.append((batch_indices, batch_refs))

    def _process_batch(
        batch_indices: list[int], batch_refs: list[str]
    ) -> tuple[list[int], list, str | None]:
        """Process one batch. Returns (indices, parsed_refs, error_or_None)."""
        logger.info(
            "Processing batch of %d references", len(batch_refs)
        )
        try:
            parsed = _parse_batch_with_llm(batch_refs, format_hint=format_hint)
            return batch_indices, parsed, None
        except Exception as e:
            return batch_indices, [], str(e)

    # Run batches concurrently. Cap workers at the number of batches — no point
    # spinning up more threads than there are batches.
    max_workers = min(len(batch_jobs), 4) if batch_jobs else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_process_batch, indices, refs_list)
            for indices, refs_list in batch_jobs
        ]
        for fut in as_completed(futures):
            batch_indices, batch_results, err = fut.result()
            llm_call_count += 1
            batch_start = batch_indices[0]
            batch_end = batch_indices[-1] + 1

            if err is not None:
                error_msg = f"Batch {batch_start}-{batch_end} failed: {err}"
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
            else:
                # Store results and cache
                for idx, parsed in zip(batch_indices, batch_results):
                    results[idx] = parsed
                    if use_cache and settings.CACHE_ENABLED:
                        doi_cache.cache_reference(
                            data=parsed.model_dump(),
                            doi=parsed.doi,
                            title=parsed.title,
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

    # Convert to ParsedReference objects.
    # NOTE: raw_ref is NOT requested from the LLM (removed from the prompt to
    # cut output size ~40% and reduce truncation). We inject it here from the
    # input refs list — the parsed objects are in the same order (enforced by
    # validate_llm_reference_array with expected_count), so refs[i] is the
    # original text for validated_data[i].
    parsed_refs = []
    for i, ref_data in enumerate(validated_data):
        # Inject raw_ref from the input — the LLM doesn't echo it back anymore
        ref_data.setdefault("raw_ref", refs[i] if i < len(refs) else "")
        try:
            parsed = ParsedReference(**ref_data)
        except Exception as e:
            logger.warning("Failed to validate ref %d: %s", i, e)
            # Fallback with raw_ref preserved
            parsed = ParsedReference(
                raw_ref=refs[i] if i < len(refs) else "",
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


