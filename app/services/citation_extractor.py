"""In-text citation extraction from student paper body text.

Extracts quotations and paraphrases from the body text and links each to its
cited reference. Handles both APA and MLA citation styles.

Three-stage hybrid approach:
  Stage 1: Structural pre-processing (extract body text, split paragraphs/sentences)
  Stage 2: Citation marker detection (regex — finds all citation markers)
  Stage 3: LLM boundary extraction (determines which sentences belong to which source)

Output: list of InTextCitation objects, each with the extracted text span,
the claim type (quotation/paraphrase), and a link to the cited reference.
"""

import logging
import re

from app.services.schemas import InTextCitation, ParsedReference
from app.services.sentence_splitter import split_paragraphs_and_sentences

logger = logging.getLogger(__name__)

# ── Citation marker regexes ────────────────────────────────────────────

# APA parenthetical: (Author, Year) or (Author, Year, p. N) or (Author & Author, Year)
# Also handles multi-citation: (Author, Year; Author, Year)
APA_PAREN_RE = re.compile(
    r"\(([A-Z][A-Za-z\-']+(?:\s+(?:&|and)\s+[A-Z][A-Za-z\-']+|"
    r"\s+et\s+al\.?)?),"  # author(s)
    r"\s*(\d{4}[a-z]?)"  # year
    r"(?:,\s*(?:p|pp)\.\s*\d+(?:-\d+)?)?"  # optional page
    r"(?:;\s*[A-Z][A-Za-z\-']+,[^)]+)?"  # optional second citation
    r"\)"
)

# APA narrative: Author (Year) or Author (Year) + verb
APA_NARRATIVE_RE = re.compile(
    r"([A-Z][A-Za-z\-']+(?:\s+(?:&|and)\s+[A-Z][A-Za-z\-']+|"
    r"\s+et\s+al\.?)?)\s*\((\d{4}[a-z]?)\)"
)

# MLA parenthetical: (Author PageNum) — no year, no comma before page
# (Author) — author only, no page
MLA_PAREN_RE = re.compile(
    r"\(([A-Z][A-Za-z\-']+)"
    r"(?:\s+(\d{1,4}(?:-\d+)?))?"  # optional page number
    r"\)"
)

# MLA narrative: "Author argues/notes/writes/states/claims/suggests/observes..."
# No parenthetical — the name + verb signals attribution
MLA_NARRATIVE_VERBS = (
    "argues", "notes", "writes", "states", "claims", "suggests",
    "observes", "asserts", "contends", "maintains", "believes",
    "points out", "points to", "explains", "describes", "discusses",
    "finds", "concludes", "reports", "shows", "demonstrates",
)
MLA_NARRATIVE_RE = re.compile(
    r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?)"
    r"\s+(" + "|".join(MLA_NARRATIVE_VERBS) + r")\b",
    re.IGNORECASE,
)

# Quotation detection: text in double or curly quotes (3+ chars)
QUOTE_RE = re.compile(r'["\u201c]([^"\u201d]{3,})["\u201d]')

# Secondary citation: (Author, Year, as cited in Author, Year)
SECONDARY_RE = re.compile(
    r"\(([A-Z][A-Za-z\-']+),\s*(\d{4}),\s*as\s+cited\s+in\s+"
    r"([A-Z][A-Za-z\-']+),\s*(\d{4})\)",
    re.IGNORECASE,
)


def extract_citations(
    body_text: str,
    references: list[ParsedReference],
    format_hint: str = "apa",
    use_llm_boundaries: bool = False,
) -> list[InTextCitation]:
    """Extract in-text citations from body text and link to references.

    Args:
        body_text: The paper body text (excluding reference section).
        references: Parsed references from the reference list.
        format_hint: "apa" or "mla".
        use_llm_boundaries: If True, use the LLM to refine citation boundaries
            for multi-sentence paraphrases and implicit continuations. Slower
            (one LLM call per ~1500-word chunk) but more accurate for complex
            papers. If False, uses sentence-level extraction only (faster).

    Returns:
        List of InTextCitation objects.
    """
    if not body_text or not body_text.strip():
        return []

    # Build a lookup: surname → citation_key(s)
    ref_by_surname = _build_surname_index(references)

    # Stage 1: split into paragraphs and sentences
    paragraphs = split_paragraphs_and_sentences(body_text)

    # Stage 2: find all citation markers (regex)
    citations: list[InTextCitation] = []

    for para_idx, sentences in enumerate(paragraphs):
        para_text = " ".join(sentences)
        para_citations = _find_citations_in_paragraph(
            para_text, para_idx, sentences, ref_by_surname, format_hint
        )
        citations.extend(para_citations)

    # Stage 3: LLM full-body extraction (one call, full paper as input,
    # metadata-only output: sentence numbers + citation keys, no full text).
    # This handles multi-sentence paraphrases, implicit continuations, and
    # narrative citations that the regex misses. Replaces the regex results
    # when successful.
    if use_llm_boundaries:
        llm_citations = _extract_citations_with_llm(paragraphs, references)
        if llm_citations:
            # Stage 4: Two-pass — find implicit continuations in unattributed sentences.
            # Pass 1 found explicit citations. Pass 2 checks which remaining sentences
            # continue discussing a previously-cited source (topic continuation).
            continued = _find_implicit_continuations(llm_citations, paragraphs)
            return llm_citations + continued

    return citations


def _build_surname_index(references: list[ParsedReference]) -> dict[str, list[ParsedReference]]:
    """Build a surname → reference lookup from the reference list."""
    index: dict[str, list[ParsedReference]] = {}
    for ref in references:
        # Extract surname from author field
        surnames = _extract_ref_surnames(ref)
        for surname in surnames:
            key = surname.lower()
            index.setdefault(key, []).append(ref)
    return index


def _extract_ref_surnames(ref: ParsedReference) -> list[str]:
    """Extract surname(s) from a ParsedReference for matching."""
    surnames: list[str] = []
    author = ref.author.strip()
    if not author:
        return surnames

    # Split on common multi-author separators
    authors = re.split(r"(?:,?\s*(?:&|and)\s*|;\s*|,\s*(?=[A-Z]))", author)
    for a in authors:
        a = a.strip().strip(".")
        if not a:
            continue
        # "Surname, Initials" format
        if "," in a:
            surname = a.split(",")[0].strip()
        else:
            # "First Last" format
            parts = a.split()
            surname = parts[-1] if parts else a
        surname = re.sub(r"[^A-Za-z\-']", "", surname)
        if len(surname) > 1:
            surnames.append(surname)
    return surnames


def _find_citations_in_paragraph(
    para_text: str,
    para_idx: int,
    sentences: list[str],
    ref_index: dict[str, list[ParsedReference]],
    format_hint: str,
) -> list[InTextCitation]:
    """Find all citation markers in a paragraph and extract attributed text."""
    citations: list[InTextCitation] = []

    # Check for secondary citations first ("as cited in")
    for m in SECONDARY_RE.finditer(para_text):
        original_author = m.group(1)
        cited_refs = ref_index.get(original_author.lower(), [])
        actual_refs = ref_index.get(m.group(3).lower(), [])
        if actual_refs:
            ref = actual_refs[0]
            citations.append(InTextCitation(
                text=sentences[0] if sentences else para_text,
                claim_type="paraphrase",
                citation_key=ref.citation_key,
                citation_marker=m.group(0),
                marker_type="parenthetical",
                paragraph_index=para_idx,
                is_secondary=True,
                original_author=original_author,
                confidence="high",
            ))

    if format_hint == "mla":
        # MLA parenthetical: (Author) or (Author PageNum)
        for m in MLA_PAREN_RE.finditer(para_text):
            surname = m.group(1)
            page = m.group(2) or ""
            refs = ref_index.get(surname.lower(), [])
            if refs:
                ref = refs[0]
                # Extract the attributed text (sentence(s) containing this citation)
                text = _extract_attributed_text(para_text, m.start(), m.end(), sentences, "parenthetical")
                claim_type = _detect_claim_type(text)
                citations.append(InTextCitation(
                    text=text,
                    claim_type=claim_type,
                    citation_key=ref.citation_key,
                    citation_marker=m.group(0),
                    marker_type="parenthetical",
                    page_number=page,
                    paragraph_index=para_idx,
                ))

        # MLA narrative: "Author argues..."
        for m in MLA_NARRATIVE_RE.finditer(para_text):
            surname = m.group(1)
            refs = ref_index.get(surname.lower(), [])
            if refs:
                ref = refs[0]
                text = _extract_attributed_text(para_text, m.start(), m.end(), sentences, "narrative")
                claim_type = _detect_claim_type(text)
                citations.append(InTextCitation(
                    text=text,
                    claim_type=claim_type,
                    citation_key=ref.citation_key,
                    citation_marker=m.group(0),
                    marker_type="narrative",
                    paragraph_index=para_idx,
                ))

    else:  # APA
        # APA parenthetical: (Author, Year)
        for m in APA_PAREN_RE.finditer(para_text):
            surname = m.group(1).split()[0]  # first word = surname
            year = m.group(2)
            refs = ref_index.get(surname.lower(), [])
            # Filter by year if available
            if refs and year:
                year_refs = [r for r in refs if year in (r.year or "")]
                refs = year_refs or refs
            if refs:
                ref = refs[0]
                text = _extract_attributed_text(para_text, m.start(), m.end(), sentences, "parenthetical")
                claim_type = _detect_claim_type(text)
                citations.append(InTextCitation(
                    text=text,
                    claim_type=claim_type,
                    citation_key=ref.citation_key,
                    citation_marker=m.group(0),
                    marker_type="parenthetical",
                    paragraph_index=para_idx,
                ))

        # APA narrative: Author (Year)
        for m in APA_NARRATIVE_RE.finditer(para_text):
            surname = m.group(1).split()[0]
            year = m.group(2)
            refs = ref_index.get(surname.lower(), [])
            if refs and year:
                year_refs = [r for r in refs if year in (r.year or "")]
                refs = year_refs or refs
            if refs:
                ref = refs[0]
                text = _extract_attributed_text(para_text, m.start(), m.end(), sentences, "narrative")
                claim_type = _detect_claim_type(text)
                citations.append(InTextCitation(
                    text=text,
                    claim_type=claim_type,
                    citation_key=ref.citation_key,
                    citation_marker=m.group(0),
                    marker_type="narrative",
                    paragraph_index=para_idx,
                ))

    return citations


def _extract_attributed_text(
    para_text: str,
    marker_start: int,
    marker_end: int,
    sentences: list[str],
    marker_type: str,
) -> str:
    """Extract the text attributed to a citation.

    For parenthetical citations: the sentence containing the citation marker.
    (Multi-sentence backward extension is handled by the LLM stage.)

    For narrative citations: the sentence containing the citation marker.
    (Forward extension is handled by the LLM stage.)
    """
    # For now (stages 1+2), return the sentence containing the marker.
    # The LLM stage will refine boundaries for multi-sentence paraphrases.
    for sent in sentences:
        # Check if this sentence contains the marker (by character offset is
        # tricky after joining; use a simpler substring check)
        if sent in para_text:
            sent_start = para_text.find(sent)
            sent_end = sent_start + len(sent)
            if sent_start <= marker_start < sent_end:
                return sent

    # Fallback: return the whole paragraph (conservative — favor recall)
    return para_text


def _detect_claim_type(text: str) -> str:
    """Determine if the text is a quotation or paraphrase.

    A quotation contains text in quote marks. A paraphrase does not.
    """
    if QUOTE_RE.search(text):
        return "quotation"
    return "paraphrase"


# ── Stage 3: LLM Boundary Refinement ───────────────────────────────────


_LLM_SYSTEM_PROMPT = """You are an academic citation analysis assistant. You are given paragraphs from a student paper and a list of cited references. Your task is to identify which sentences in each paragraph are attributed to which cited source.

A single citation may cover MULTIPLE sentences:
- For parenthetical citations (Author, Year) at the END of a passage, the cited content extends BACKWARD to the beginning of the paraphrase.
- For narrative citations Author (Year) at the START, the cited content extends FORWARD until the next citation or a topic shift.
- An author may be discussed across several sentences after the initial citation ("He argues...", "Dyer notes...", "She states...") — all belong to the same source.

Output a JSON array. Each element describes one attributed passage:
{
  "first_sentence": "the first 80 characters of the attributed passage (for matching)",
  "sentence_count": number of sentences attributed to this source,
  "citation_key": "the citation key from the reference list (e.g. 'Smith2020')",
  "author_surname": "the author surname as it appears in the text",
  "claim_type": "quotation" or "paraphrase",
  "marker_type": "parenthetical" or "narrative",
  "page_number": "page number if present, else empty string"
}

IMPORTANT: Keep "first_sentence" to MAXIMUM 80 CHARACTERS. The system uses it only to locate the full passage. Keep the total output as small as possible."""


def _extract_citations_with_llm(
    paragraphs: list[list[str]],
    references: list[ParsedReference],
) -> list[InTextCitation]:
    """Extract citations using LLM over the full body text.

    Sends the body (with numbered sentences) as input and asks the LLM to
    return metadata only (paragraph + sentence indices + citation keys).
    The full text is then looked up from the original paragraphs.

    Handles multi-sentence paraphrases, implicit continuations, and narrative
    citations that regex misses. Uses batching for very long papers (>8000 words)
    to respect the LLM's output limit. Uses token-based batching threshold.
    """
    from app.services.llm_service import chat_completion_json

    if not paragraphs:
        return []

    # Reference list summary — include titles so the LLM can match
    # continuation sentences (e.g., "This reflects the era's fascination")
    # to a source by TOPIC, not just by author surname.
    ref_list = "\n".join(
        f"- {r.citation_key}: {r.author} ({r.year}). {r.title}"
        for r in references if r.title.strip()
    ) or "\n".join(f"- {r.citation_key}: {r.author} ({r.year})" for r in references)

    # Extract author surnames from the reference list to feed to the LLM.
    # This significantly improves MLA narrative citation detection (e.g.,
    # "Dawson claims..." with no parenthetical marker) and implicit
    # continuations ("He argues..." after an initial citation).
    surname_list = ", ".join(sorted(set(
        surname for ref in references
        for surname in _extract_ref_surnames(ref)
    )))
    if surname_list:
        surname_list = f"\n\nAuthor surnames from the reference list (any sentence mentioning these names is likely a citation): {surname_list}"

    # Estimate token count of the numbered text to decide on batching.
    # Uses the provider's input_batch_tokens setting — DeepSeek is limited
    # to ~1500 tokens per call; GPT-4/Claude can handle 100K+.
    from app.services.providers import get_provider_config
    config = get_provider_config()

    total_chars = sum(len(f"[P{pi}S{si}] {s}") for pi, sents in enumerate(paragraphs) for si, s in enumerate(sents))
    estimated_tokens = total_chars // 4  # rough: 4 chars ≈ 1 token
    tokens_per_batch = config.input_batch_tokens

    if estimated_tokens <= tokens_per_batch:
        # Short enough for a single call
        return _llm_extract_batch(paragraphs, 0, len(paragraphs), ref_list, surname_list)

    # Calculate paragraph count per batch to stay under the token limit
    avg_chars_per_para = total_chars / max(1, len(paragraphs))
    avg_tokens_per_para = avg_chars_per_para / 4
    batch_size = max(3, int(tokens_per_batch / max(1, avg_tokens_per_para)))
    all_citations: list[InTextCitation] = []
    for start in range(0, len(paragraphs), batch_size):
        end = min(start + batch_size, len(paragraphs))
        batch_citations = _llm_extract_batch(paragraphs, start, end, ref_list, surname_list)
        all_citations.extend(batch_citations)
        logger.info("LLM batch %d-%d: %d citations", start, end, len(batch_citations))

    return all_citations


def _llm_extract_batch(
    paragraphs: list[list[str]],
    start_para: int,
    end_para: int,
    ref_list: str,
    surname_hint: str = "",
) -> list[InTextCitation]:
    """Process one batch of paragraphs through the LLM.

    Sentence numbering uses GLOBAL paragraph indices (start_para..end_para)
    so the results can be looked up in the original paragraphs list.
    """
    from app.services.llm_service import chat_completion_json

    # Build numbered sentences for this batch (using global paragraph indices)
    numbered = []
    for pi in range(start_para, end_para):
        if pi >= len(paragraphs):
            break
        for si, sent in enumerate(paragraphs[pi]):
            numbered.append(f"[P{pi}S{si}] {sent}")

    if not numbered:
        return []

    full_numbered = "\n".join(numbered)

    # JSON output format: some providers (DeepSeek, OpenAI) require a top-level
    # JSON object, not a bare array. We always use {"citations": [...]} to be
    # safe across all providers. Short keys (p, s, e, k, t) save output space
    # for providers with tighter output limits.
    system = """You are a citation extraction assistant. Given a student paper's sentences (numbered by paragraph and sentence) and a list of cited references, identify which sentences contain claims attributed to a cited source.

A sentence is "attributed" if it:
- Contains a citation marker like (Author, Year) or Author (Year)
- Paraphrases or quotes a source
- Continues discussing a previously-cited source (e.g., "He argues...", "This reflects...")
- Matches the TOPIC of a source title (use titles to connect continuation sentences to their source)

When in doubt about whether a sentence continues a cited source, INCLUDE it — it is better to check an extra sentence than to miss a claim.

For each attributed passage, report the paragraph number, start sentence, end sentence (inclusive), the citation key from the reference list, and whether it's a quotation or paraphrase.

Return a JSON object like: {"citations": [{"p": 0, "s": 2, "e": 2, "k": "Browning2017", "t": "quotation", "c": "high"}]}
where p=paragraph, s=start sentence, e=end sentence, k=citation key, t=type, c=confidence.

Confidence levels for the "c" field:
- "high": the sentence has an explicit citation marker (Author, Year) or directly quotes a source
- "medium": the sentence is immediately adjacent to an explicit citation (1 sentence before/after) and clearly continues the same argument
- "low": the sentence is 2+ sentences away from any marker, or was matched by topic/title only (inferred continuation)

This confidence level is critical: high-confidence citations can be penalized if they don't match. Low-confidence citations should NOT be penalized — the attribution itself is uncertain."""

    user = f"""References:
{ref_list}{surname_hint}

Sentences:
{full_numbered}"""

    try:
        result = chat_completion_json(
            system_prompt=system,
            user_prompt=user,
            max_tokens=8192,
        )
        # The response is a JSON object with a "citations" key (not a bare array)
        items = result.get("citations", []) if isinstance(result, dict) else (
            result if isinstance(result, list) else []
        )
    except Exception as e:
        logger.warning("LLM citation extraction failed for batch %d-%d: %s", start_para, end_para, e)
        return []

    citations: list[InTextCitation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pi = item.get("p", item.get("para", -1))
        si = item.get("s", item.get("start_sent", -1))
        ei = item.get("e", item.get("end_sent", si))
        cite_key = item.get("k", item.get("citation_key", ""))
        claim_type = item.get("t", item.get("claim_type", "paraphrase"))
        if claim_type not in ("quotation", "paraphrase"):
            claim_type = "paraphrase"

        # Extraction confidence — critical for fair reporting.
        # High = explicit marker; Medium = adjacent continuation;
        # Low = inferred from topic/title only. Only high/medium
        # citations should be penalized if they don't match the source.
        confidence_raw = item.get("c", item.get("confidence", "medium")).lower()
        if confidence_raw not in ("high", "medium", "low"):
            confidence_raw = "medium"

        if pi < 0 or pi >= len(paragraphs):
            continue
        sentences = paragraphs[pi]
        if si < 0 or si >= len(sentences):
            continue
        ei = min(ei, len(sentences) - 1)

        text = " ".join(sentences[si:ei + 1])
        if not text or len(text) < 10:
            continue

        citations.append(InTextCitation(
            text=text,
            claim_type=claim_type,
            citation_key=cite_key,
            citation_marker=cite_key,
            marker_type="llm_detected",
            paragraph_index=pi,
            confidence=confidence_raw,
        ))

    logger.info("LLM extracted %d citations from paragraphs %d-%d", len(citations), start_para, end_para)
    return citations


# ── Stage 4: Implicit continuation detection (two-pass) ────────────────


def _find_implicit_continuations(
    found_citations: list[InTextCitation],
    paragraphs: list[list[str]],
) -> list[InTextCitation]:
    """Two-pass: find unattributed sentences that continue a previously-cited source.

    Pass 1 (done above) found explicit citations. This function identifies
    sentences that were NOT attributed in pass 1 but continue discussing a
    source from an adjacent cited sentence — the implicit continuation case.

    Strategy: for each paragraph, find sentences between two citations (or
    after the last citation) that have no attribution. Send these "gap"
    sentences to the LLM with the preceding cited source and ask:
    "Does this sentence continue discussing the same source?"

    This catches "This reflects...", "Similarly...", "The addition of..."
    — sentences with no surname, no marker, but topically connected.
    """
    from app.services.llm_service import chat_completion_json
    from app.services.providers import get_provider_config

    if not found_citations or not paragraphs:
        return []

    config = get_provider_config()

    # Build a map: which (paragraph, sentence) pairs are already attributed?
    attributed: set[tuple[int, int]] = set()
    for cite in found_citations:
        # Mark all sentences in this citation's range
        pi = cite.paragraph_index
        # We need to find the sentence range from the text — but we stored
        # the text, not the indices. Instead, find which sentences contain
        # parts of the citation text.
        if pi < len(paragraphs):
            for si, sent in enumerate(paragraphs[pi]):
                if sent[:30] in cite.text or cite.text[:30] in sent:
                    attributed.add((pi, si))

    # Find "gap" sentences: unattributed sentences in paragraphs that have citations
    new_citations: list[InTextCitation] = []

    for para_idx, sentences in enumerate(paragraphs):
        # Does this paragraph have any citations?
        para_cites = [c for c in found_citations if c.paragraph_index == para_idx]
        if not para_cites:
            continue

        # Find unattributed sentences
        gaps: list[tuple[int, str]] = []
        for si, sent in enumerate(sentences):
            if (para_idx, si) not in attributed:
                gaps.append((si, sent))

        if not gaps:
            continue

        # Build context: the cited sentences in this paragraph + the gap sentences
        gap_numbered = [f"[S{si}] {s}" for si, s in gaps]
        cite_context = " | ".join(c.text[:100] for c in para_cites[:3])

        # Batch gaps if too many
        gap_text = "\n".join(gap_numbered)
        if len(gap_text) // 4 > config.input_batch_tokens:
            # Too many gaps — process first 15
            gap_numbered = gap_numbered[:15]
            gap_text = "\n".join(gap_numbered)

        system = """You are a citation continuation assistant. Given sentences that were NOT attributed to any source in pass 1, determine which ones continue discussing a source that WAS cited earlier in the same paragraph.

A sentence continues a source if:
- It refers to the same topic/argument ("This reflects...", "Similarly...", "As a result...")
- It uses pronouns referring to the cited author's ideas ("He argues...", "This approach...")
- It elaborates on or extends the preceding cited point without introducing a new source

Return a JSON object: {"continuations": [{"s": sentence_index, "k": "citation_key"}]}
Only include sentences that clearly continue a cited source. If unsure, exclude."""

        user = f"""Previously cited in this paragraph:
{cite_context}

Unattributed sentences to check:
{gap_text}"""

        try:
            result = chat_completion_json(
                system_prompt=system,
                user_prompt=user,
                max_tokens=2000,
            )
            items = result.get("continuations", []) if isinstance(result, dict) else (
                result if isinstance(result, list) else []
            )

            for item in items:
                if not isinstance(item, dict):
                    continue
                si = item.get("s", -1)
                cite_key = item.get("k", "")
                if si < 0 or si >= len(sentences):
                    continue
                sent_text = sentences[si]
                if len(sent_text) < 5:
                    continue

                # Find the matching citation to inherit its key
                if cite_key:
                    new_citations.append(InTextCitation(
                        text=sent_text,
                        claim_type="paraphrase",
                        citation_key=cite_key,
                        citation_marker="implicit_continuation",
                        marker_type="implicit",
                        paragraph_index=para_idx,
                        confidence="low",  # implicit — lower confidence
                    ))
        except Exception as e:
            logger.debug("Implicit continuation detection failed for para %d: %s", para_idx, e)

    if new_citations:
        logger.info("Two-pass: found %d implicit continuations", len(new_citations))
    return new_citations
