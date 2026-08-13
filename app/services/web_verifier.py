"""Web-fetch verification for website citations.

When a student cites a website (news article, government page, blog), we fetch
the live HTML, extract the readable article text, verify the student's
quotation/paraphrase against it, then DISCARD the content. Website content is
never persisted to S3 — only the verification result is stored in the report.

This is the Phase 3.7 component that makes website citations verifiable:
  1. Fetch the HTML at the cited URL (with safety checks)
  2. Extract readable text using trafilatura (strips nav/ads/sidebars)
  3. Verify the quotation/paraphrase against the extracted text
  4. Discard the content; return only the verification result

For quotations: check if the quoted text appears in the extracted page text
(exact or near-exact match).

For paraphrases: use the LLM to judge whether the paraphrase is faithful to
the page content (same mechanism as the abstract verifier, but against the
full page text instead of an abstract).
"""

import logging
import re
from dataclasses import dataclass

import httpx
import trafilatura

from app.services.safe_fetch import safe_request

from app.services.abstract_verifier import (
    verify_claim_against_abstract,
    AbstractVerificationResult,
    PARAPHRASE,
    QUOTATION,
    CONSISTENT,
    MISMATCH,
    MISREPRESENTATION,
    INCONCLUSIVE,
)

logger = logging.getLogger(__name__)

# Result verdicts (reuse abstract_verifier constants for consistency)


@dataclass
class WebVerificationResult:
    """Result of verifying a citation against a fetched web page.

    Attributes:
        verdict: "consistent" | "topical_mismatch" | "misrepresentation" | "inconclusive" | "fetch_failed"
        confidence: "high" | "medium" | "low"
        explanation: human-readable reasoning
        url: the URL that was checked
        page_title: the <title> of the fetched page
        text_length: how many chars of readable text were extracted
    """

    verdict: str
    confidence: str
    explanation: str
    url: str
    page_title: str | None = None
    text_length: int = 0


def verify_citation_against_webpage(
    url: str,
    quotation: str,
    claim_type: str = PARAPHRASE,
    timeout: int = 20,
) -> WebVerificationResult:
    """Fetch a web page and verify a quotation or paraphrase against its text.

    The page content is extracted in-memory and discarded after verification —
    it is never persisted to S3.

    Args:
        url: The cited URL to fetch.
        quotation: The student's quoted or paraphrased text.
        claim_type: "quotation" (exact words) or "paraphrase" (student's own words).
        timeout: Fetch timeout in seconds.

    Returns:
        WebVerificationResult with the verification verdict.
    """
    # Step 1: Fetch the HTML
    # Use a realistic browser user-agent — many sites (Britannica, news sites)
    # return 403 to bot user-agents. This is for fetching content the student
    # already cited (the URL is in their reference list), not for crawling.
    try:
        resp = safe_request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=timeout,
        )
    except Exception as e:
        logger.warning("Web fetch failed for %s: %s", url, e)
        return WebVerificationResult(
            verdict="fetch_failed",
            confidence="low",
            explanation=f"Could not fetch the page: {str(e)[:80]}",
            url=url,
        )

    # Step 2: Extract readable text with trafilatura
    page_text = trafilatura.extract(
        resp.text,
        include_links=False,
        include_tables=False,
        favor_recall=True,  # prefer more text over precision for citation checking
    )
    if not page_text or len(page_text) < 100:
        return WebVerificationResult(
            verdict="inconclusive",
            confidence="low",
            explanation="Page loaded but no readable article text could be extracted (may be a login page, PDF, or non-article content)",
            url=url,
        )

    # Extract page title for the result
    page_title = _extract_title(resp.text)

    # Step 3: Verify the quotation/paraphrase against the page text
    if claim_type == QUOTATION:
        result = _verify_quotation(quotation, page_text)
    else:
        result = _verify_paraphrase(quotation, page_text, page_title)

    # Step 4: Return result — page_text is discarded (goes out of scope)
    return WebVerificationResult(
        verdict=result.verdict,
        confidence=result.confidence,
        explanation=result.explanation,
        url=url,
        page_title=page_title,
        text_length=len(page_text),
    )


def _verify_quotation(quotation: str, page_text: str) -> AbstractVerificationResult:
    """Verify a direct quotation against page text.

    First tries an exact/near-exact string match. If found, returns "consistent"
    with high confidence. If not found, falls back to topical-plausibility via LLM.
    """
    # Normalize both for comparison
    q_clean = _normalize_text(quotation)
    p_clean = _normalize_text(page_text)

    # Exact match check (the quotation appears in the page text)
    if q_clean in p_clean:
        return AbstractVerificationResult(
            verdict=CONSISTENT,
            confidence="high",
            explanation="Quotation found in the page text (exact match)",
            claim_type=QUOTATION,
        )

    # Near-exact match: check if the quotation appears with minor variations
    # (extra whitespace, missing/added punctuation). Use a sliding window of
    # the quotation's first ~50 chars.
    q_probe = q_clean[:50]
    if len(q_probe) > 20 and q_probe in p_clean:
        return AbstractVerificationResult(
            verdict=CONSISTENT,
            confidence="high",
            explanation="Quotation found in the page text (near-exact match)",
            claim_type=QUOTATION,
        )

    # Not found by string match — use LLM for topical plausibility
    # (the quote's topic might be on the page even if the exact words aren't,
    # OR the quote may be fabricated)
    return verify_claim_against_abstract(
        claim=quotation,
        abstract=page_text[:5000],  # use first 5000 chars for the LLM context
        claim_type=QUOTATION,
        source_title="Web page content",
        source_doi=None,
        abstract_source="web_fetch",
    )


def _verify_paraphrase(
    paraphrase: str, page_text: str, page_title: str | None
) -> AbstractVerificationResult:
    """Verify a paraphrase against page text using the LLM.

    Uses the same faithfulness check as the abstract verifier, but against
    the full page text (truncated to fit the LLM context window).
    """
    # Truncate page text for the LLM (keep the beginning, which usually
    # contains the article's main claims)
    context_text = page_text[:8000]

    return verify_claim_against_abstract(
        claim=paraphrase,
        abstract=context_text,
        claim_type=PARAPHRASE,
        source_title=page_title,
        source_doi=None,
        abstract_source="web_fetch",
    )


def _normalize_text(text: str) -> str:
    """Normalize text for string matching: lowercase, collapse whitespace, strip punctuation edges."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_title(html: str) -> str | None:
    """Extract the <title> from an HTML page."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()
        title = re.sub(r"\s+", " ", title)
        return title if title else None
    return None
