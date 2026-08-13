"""Subject-identification pass — Phase 3.8 pre-analysis.

One LLM call over a paper's body text + reference list that produces four
outputs used by every downstream stage (PLAN.md §3.1, "Subject identification
pass (pre-analysis)"):

  1. Primary subject (what the paper analyzes) + subject type
  2. Per-reference primary-vs-secondary classification
  3. Per-paragraph structure zoning (intro / body / conclusion)
  4. Topic keywords (5-10) — stored for the §5 ablation; not consumed by the
     citation extractor in this step

This pass is the input to:
  - Citation extraction (subject context distinguishes student analysis of the
    primary text from citations of secondary scholarship — R21)
  - Verification (structure zoning drives section-based verification;
    primary-vs-secondary classification decides which refs to verify against)
  - Reporting (missing-primary-source note, keywords, section labels)

Low-volume (1 call per paper), so JSON output is safe — opposite regime from
reference parsing, where JSON was dropped to avoid truncation on high-volume
calls. Uses ``chat_completion_json`` to inherit JSON-mode enforcement,
truncation salvage, and failure-aware retry.

PII (R22): PII stripping is designed to run inside this pass (PLAN.md line
790) but is implemented in a separate cross-cutting module that is not yet
built. Until then, callers must pass already-deidentified text. See
``# TODO(R22)`` at the call site below.
"""

import logging
from typing import Any

from app.config import settings
from app.services.llm_service import chat_completion_json
from app.services.prompts import (
    SUBJECT_IDENTIFICATION_SYSTEM_PROMPT,
    build_subject_identification_user_prompt,
)
from app.services.schemas import (
    PRIMARY_SUBJECT_TYPES,
    ParagraphRole,
    ParagraphStructure,
    ParsedReference,
    ReferenceClassification,
    SubjectIdentification,
)
from app.services.sentence_splitter import split_paragraphs_and_sentences

logger = logging.getLogger(__name__)

# Generous output cap — the response is metadata only (no full text echoed),
# but large papers with many paragraphs/references can produce sizable JSON.
# DeepSeek's reasoning models need headroom; the reference-extraction lesson
# (Aug 6) was that small max_tokens silently swallowed the whole response.
_MAX_TOKENS = 6000


def identify_subject(
    body_text: str,
    references: list[ParsedReference],
    format_hint: str = "apa",
) -> SubjectIdentification:
    """Run the subject-identification LLM pass on one paper.

    Args:
        body_text: The paper body text with the reference section already
            stripped. Paragraphs are assumed separated by blank lines
            (the contract ``text_extractor`` and ``sentence_splitter`` use).
        references: Parsed references from the reference list.
        format_hint: "apa" or "mla" (informational only; currently unused
            by the prompt but kept for symmetry with other services and
            future format-specific tuning).

    Returns:
        A :class:`SubjectIdentification`. On LLM failure, returns a safe
        default object with ``llm_call_succeeded=False`` (all paragraphs
        BODY, no primary-source classifications, empty keywords) so callers
        never crash. Treat a False result as low-confidence downstream.
    """
    if not body_text or not body_text.strip():
        logger.debug("identify_subject called with empty body text")
        return _failed_result(0, references)

    paragraphs = split_paragraphs_and_sentences(body_text)
    paragraph_count = len(paragraphs)
    if paragraph_count == 0:
        logger.debug("identify_subject: no paragraphs after splitting")
        return _failed_result(0, references)

    user_prompt = build_subject_identification_user_prompt(
        body_text=body_text,
        references=references,
        paragraph_count=paragraph_count,
    )

    # TODO(R22): route body_text through the PII stripper before this call.
    # PII stripping is a cross-cutting module not yet built; until it exists,
    # callers must pass already-deidentified text (the test harness uses
    # deidentified papers). This is the single LLM call site for this pass.
    try:
        raw: Any = chat_completion_json(
            system_prompt=SUBJECT_IDENTIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=_MAX_TOKENS,
            # Low-stakes structured output — disable thinking to avoid the
            # reasoning-phase empty-response problem on large inputs (the
            # Moral paper flakiness). The verification judge keeps thinking ON.
            disable_thinking=True,
        )
    except Exception as e:
        # chat_completion_json raises RuntimeError on total failure. Degrade
        # to a safe default rather than crashing the caller — the subject-ID
        # pass is pre-analysis, not load-bearing for correctness (downstream
        # stages default to "verify everything" when no subject info exists).
        logger.warning("Subject-identification LLM call failed: %s", e)
        return _failed_result(paragraph_count, references)

    if not isinstance(raw, dict) or not raw:
        logger.warning(
            "Subject-identification returned %s, expected dict — degrading to defaults",
            type(raw).__name__,
        )
        return _failed_result(paragraph_count, references)

    return _build_result(raw, paragraph_count, references)


# ---------------------------------------------------------------------------
# Normalization helpers — turn the raw LLM JSON into a validated SubjectIdentification.
# Defensive: the LLM may return fewer/more paragraph entries than actual,
# miss citation_keys, or invent subject_type values. We coerce, not crash.
# ---------------------------------------------------------------------------


def _build_result(
    raw: dict,
    paragraph_count: int,
    references: list[ParsedReference],
) -> SubjectIdentification:
    """Normalize raw LLM JSON into a SubjectIdentification.

    Always returns a complete object (no missing paragraphs/refs). Defaults
    are applied for any field the LLM got wrong or omitted.
    """
    primary_subject = str(raw.get("primary_subject", "")).strip()

    subject_type = str(raw.get("subject_type", "other")).strip().lower()
    if subject_type not in PRIMARY_SUBJECT_TYPES:
        subject_type = "other"

    primary_in_refs = _coerce_bool(raw.get("primary_subject_in_references", True))

    missing_note = str(raw.get("missing_primary_source_note", "")).strip()
    # Backfill the note when the flag says missing but the LLM left it blank,
    # and clear it when the flag says present (keeps the two fields consistent).
    if not primary_in_refs and not missing_note and primary_subject:
        missing_note = (
            f"This paper appears to analyze {primary_subject} which is not "
            f"in the reference list."
        )
    elif primary_in_refs:
        missing_note = ""

    paragraphs = _normalize_paragraph_roles(
        raw.get("paragraphs", []),
        expected_count=paragraph_count,
    )
    ref_classifications = _normalize_reference_classifications(
        raw.get("references", []),
        references,
    )

    keywords = _normalize_keywords(raw.get("keywords", []))

    return SubjectIdentification(
        primary_subject=primary_subject,
        subject_type=subject_type,
        primary_subject_in_references=primary_in_refs,
        missing_primary_source_note=missing_note,
        paragraphs=paragraphs,
        references=ref_classifications,
        keywords=keywords,
        model=settings.LLM_MODEL,
        llm_call_succeeded=True,
    )


def _normalize_paragraph_roles(
    raw_paragraphs: Any,
    expected_count: int,
) -> list[ParagraphStructure]:
    """Turn the LLM's paragraphs array into a complete list of ParagraphStructure.

    Fills any missing indices with BODY and drops out-of-range extras so the
    output always has exactly ``expected_count`` entries indexed 0..N-1.
    """
    by_index: dict[int, ParagraphStructure] = {}
    if isinstance(raw_paragraphs, list):
        for entry in raw_paragraphs:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            # Clamp into range; skip negatives.
            if idx < 0:
                continue
            role = _parse_role(entry.get("role", "body"))
            rationale = str(entry.get("role_rationale", "")).strip()
            by_index[idx] = ParagraphStructure(
                index=idx, role=role, role_rationale=rationale,
            )

    # Build the complete list, filling gaps with BODY.
    result: list[ParagraphStructure] = []
    for i in range(expected_count):
        if i in by_index:
            result.append(by_index[i])
        else:
            result.append(
                ParagraphStructure(index=i, role=ParagraphRole.BODY, role_rationale="")
            )
    return result


def _normalize_reference_classifications(
    raw_refs: Any,
    references: list[ParsedReference],
) -> list[ReferenceClassification]:
    """Match the LLM's reference classifications to actual citation_keys.

    Drops any LLM entry whose citation_key isn't in the real reference list
    (the model may invent or miskey). Returns one entry per real reference,
    defaulting to is_primary_source=False when the LLM didn't classify it.
    """
    if not isinstance(raw_refs, list):
        raw_refs = []

    by_key: dict[str, ReferenceClassification] = {}
    for entry in raw_refs:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("citation_key", "")).strip()
        if not key:
            continue
        is_primary = _coerce_bool(entry.get("is_primary_source", False))
        rationale = str(entry.get("role_rationale", "")).strip()
        by_key[key] = ReferenceClassification(
            citation_key=key,
            is_primary_source=is_primary,
            role_rationale=rationale,
        )

    result: list[ReferenceClassification] = []
    for ref in references:
        key = getattr(ref, "citation_key", "") or ""
        if key and key in by_key:
            result.append(by_key[key])
        else:
            result.append(
                ReferenceClassification(
                    citation_key=key,
                    is_primary_source=False,
                    role_rationale="",
                )
            )
    return result


def _normalize_keywords(raw_keywords: Any, cap: int = 10) -> list[str]:
    """Coerce the keywords field into a clean list of lowercase strings (≤ cap)."""
    if not isinstance(raw_keywords, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for kw in raw_keywords:
        kw = str(kw).strip().lower()
        if not kw or kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
        if len(out) >= cap:
            break
    return out


def _parse_role(value: Any) -> ParagraphRole:
    """Map an LLM role string to a ParagraphRole, defaulting to BODY."""
    s = str(value).strip().lower()
    if s.startswith("intro"):
        return ParagraphRole.INTRODUCTION
    if s.startswith("conclu"):
        return ParagraphRole.CONCLUSION
    return ParagraphRole.BODY


def _coerce_bool(value: Any) -> bool:
    """Best-effort bool coercion for untrusted LLM JSON values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in {"true", "yes", "1", "t", "y"}


def _failed_result(
    paragraph_count: int,
    references: list[ParsedReference],
) -> SubjectIdentification:
    """Construct a safe-default SubjectIdentification for the failure path.

    Every paragraph is BODY, every reference defaults to secondary (False),
    keywords empty. ``llm_call_succeeded=False`` lets downstream code mark
    results as low-confidence.
    """
    return SubjectIdentification(
        primary_subject="",
        subject_type="other",
        primary_subject_in_references=True,
        missing_primary_source_note="",
        paragraphs=[
            ParagraphStructure(index=i, role=ParagraphRole.BODY, role_rationale="")
            for i in range(paragraph_count)
        ],
        references=[
            ReferenceClassification(
                citation_key=(getattr(r, "citation_key", "") or ""),
                is_primary_source=False,
                role_rationale="",
            )
            for r in references
        ],
        keywords=[],
        model=settings.LLM_MODEL,
        llm_call_succeeded=False,
    )
