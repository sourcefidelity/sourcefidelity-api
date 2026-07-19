"""Abstract-based verification for paywalled sources.

When a cited source has no open-access full-text PDF, we can still retrieve its
abstract (from OpenAlex, Crossref, or Semantic Scholar) and use it to perform
two checks:

1. **Source-existence check** (high confidence): Does the cited source exist?
   Does its title/author/year match? The abstract confirms the paper is real
   and establishes what it claims to be about.

2. **Topical-plausibility / faithfulness check** (medium-low confidence):
   Given the abstract defines the paper's scope and claims, is the student's
   citation consistent with the source?

   For QUOTATIONS: judge topical plausibility — does the quoted text's topic
   fit the scope the abstract describes? (Cannot verify exact words.)

   For PARAPHRASES (more common): judge both topical plausibility AND
   faithfulness — does the paraphrase accurately represent what the source
   claims, or does it misrepresent/overstate the source's findings?

This never produces "confirmed exact quotation" — only the full-text path can.
The result is marked "abstract-only, lower confidence."
"""

import logging
from dataclasses import dataclass

from app.services.llm_service import chat_completion_json

logger = logging.getLogger(__name__)

# Verdict constants
CONSISTENT = "consistent"
MISMATCH = "topical_mismatch"
MISREPRESENTATION = "misrepresentation"
INCONCLUSIVE = "inconclusive"

# Claim type constants
QUOTATION = "quotation"
PARAPHRASE = "paraphrase"


@dataclass
class AbstractVerificationResult:
    """Result of verifying a citation against a source abstract.

    Attributes:
        verdict: "consistent" | "topical_mismatch" | "misrepresentation" | "inconclusive"
        confidence: "high" (source confirmed to exist) | "medium" | "low"
        explanation: human-readable LLM reasoning
        abstract_source: which API provided the abstract
        claim_type: "quotation" or "paraphrase"
    """

    verdict: str
    confidence: str
    explanation: str
    abstract_source: str | None = None
    claim_type: str = "paraphrase"


_QUOTATION_PROMPT = """You are an academic citation verification assistant. You are given:
1. A direct quotation that a student has attributed to a source.
2. The abstract of that source (retrieved from an academic database).

Your task: judge whether the quotation is TOPOCALLY CONSISTENT with what the
source is about, based on its abstract. You CANNOT verify the exact words
(only the full text could do that) — you are judging topical plausibility only.

Respond in JSON:
{
  "verdict": "consistent" | "topical_mismatch" | "inconclusive",
  "explanation": "one or two sentences explaining your judgment"
}

- "consistent": the quotation's topic fits within the scope the abstract
  describes. This does NOT confirm the quotation is real — only that it's
  plausible given what the paper is about.
- "topical_mismatch": the quotation is about a topic clearly unrelated to the
  source's subject (as established by the abstract). This suggests the
  quotation may be fabricated or misattributed.
- "inconclusive": the abstract is too short or generic to make a judgment,
  or the quotation is too generic to assess topically.
"""


_PARAPHRASE_PROMPT = """You are an academic citation verification assistant. You are given:
1. A paraphrased claim that a student has attributed to a source (the student's
   own words, NOT a direct quotation).
2. The abstract of that source (retrieved from an academic database).

Your task: judge two things based on the abstract:
(a) Is the paraphrase TOPOCALLY CONSISTENT with what the source is about?
(b) Does the paraphrase FAITHFULLY represent the source's claims, or does it
    misrepresent or overstate them?

You CANNOT verify the exact content (only the full text could do that) — but
the abstract establishes the source's scope and main claims, so you can judge
whether the paraphrase is plausible and faithful.

Respond in JSON:
{
  "verdict": "consistent" | "topical_mismatch" | "misrepresentation" | "inconclusive",
  "explanation": "one or two sentences explaining your judgment"
}

- "consistent": the paraphrase's topic fits the source's scope AND does not
  obviously misrepresent the source's claims. This does NOT confirm the
  paraphrase is accurate — only that it's plausible given the abstract.
- "topical_mismatch": the paraphrase is about a topic clearly unrelated to the
  source's subject. This suggests the citation may be fabricated or misattributed.
- "misrepresentation": the topic is related, but the paraphrase overstates,
  distorts, or contradicts what the abstract says the source claims. For
  example: the source found a weak correlation but the paraphrase claims a
  strong causal effect; or the source discusses a limitation but the paraphrase
  presents it as a definitive finding.
- "inconclusive": the abstract is too short or generic, or the paraphrase is
  too vague to assess.
"""


def verify_claim_against_abstract(
    claim: str,
    abstract: str,
    claim_type: str = PARAPHRASE,
    source_title: str | None = None,
    source_doi: str | None = None,
    abstract_source: str | None = None,
) -> AbstractVerificationResult:
    """Judge whether a quotation or paraphrase is consistent with a source abstract.

    Args:
        claim: The student's quoted or paraphrased text.
        abstract: The source's abstract text.
        claim_type: "quotation" (exact words) or "paraphrase" (student's own words).
        source_title: Optional title of the cited source (for context).
        source_doi: Optional DOI (confirms source identity).
        abstract_source: Which API provided the abstract (e.g. "openalex").

    Returns:
        AbstractVerificationResult with verdict and explanation.
    """
    context_parts = []
    if source_title:
        context_parts.append(f"Source title: {source_title}")
    if source_doi:
        context_parts.append(f"Source DOI: {source_doi}")
    context = "\n".join(context_parts) if context_parts else "(no metadata available)"

    label = "Direct quotation" if claim_type == QUOTATION else "Paraphrased claim"

    system_prompt = _QUOTATION_PROMPT if claim_type == QUOTATION else _PARAPHRASE_PROMPT

    user_prompt = f"""{context}

{label} attributed to this source:
\"\"\"
{claim}
\"\"\"

Abstract of the source:
\"\"\"
{abstract}
\"\"\"

Judge whether this {claim_type} is consistent with the source's abstract."""

    try:
        result = chat_completion_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
        )
        verdict = result.get("verdict", "inconclusive").strip().lower()
        explanation = result.get("explanation", "")

        # Normalize verdict
        valid_verdicts = (CONSISTENT, MISMATCH, MISREPRESENTATION, INCONCLUSIVE)
        if claim_type == QUOTATION:
            # Quotation prompt doesn't produce "misrepresentation"
            valid_verdicts = (CONSISTENT, MISMATCH, INCONCLUSIVE)
        if verdict not in valid_verdicts:
            verdict = INCONCLUSIVE

        # Confidence: source existence is confirmed (we got an abstract).
        if verdict in (MISMATCH, MISREPRESENTATION):
            confidence = "high"  # strong negative signal
        elif verdict == CONSISTENT:
            confidence = "medium"  # plausible but unconfirmed
        else:
            confidence = "low"

        return AbstractVerificationResult(
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            abstract_source=abstract_source,
            claim_type=claim_type,
        )
    except Exception as e:
        logger.warning("Abstract verification LLM call failed: %s", e)
        return AbstractVerificationResult(
            verdict=INCONCLUSIVE,
            confidence="low",
            explanation=f"Verification failed: {e}",
            abstract_source=abstract_source,
            claim_type=claim_type,
        )


# Backward-compatible alias for the original function name.
def verify_quotation_against_abstract(
    quotation: str,
    abstract: str,
    source_title: str | None = None,
    source_doi: str | None = None,
    abstract_source: str | None = None,
) -> AbstractVerificationResult:
    """Backward-compatible wrapper for quotation verification."""
    return verify_claim_against_abstract(
        claim=quotation,
        abstract=abstract,
        claim_type=QUOTATION,
        source_title=source_title,
        source_doi=source_doi,
        abstract_source=abstract_source,
    )
