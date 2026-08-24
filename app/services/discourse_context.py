"""Deterministic local discourse dependencies for cited continuations.

Only exact bounded context already retained on the claim is inspected.  The
current prototype recognizes a high-precision question→answer continuation;
it does not infer arbitrary cross-sentence discourse relations.
"""

from __future__ import annotations

import re

from app.services.verification_evidence import (
    ClaimDiscourseDependency,
    ClaimEvidence,
)


DISCOURSE_DEPENDENCY_VERSION = "local-exact-discourse-dependency-v1"
_ANSWER_LIKE_START = re.compile(
    r"^\s*(?:this|these|those|they|their|them|it|its|such)\b",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^\s*(?:what|which|who|where|when|why|how|are|is|do|does|did|can|could|"
    r"would|should)\b",
    re.IGNORECASE,
)


def resolve_claim_discourse_dependencies(claim: ClaimEvidence) -> ClaimEvidence:
    """Attach one exact preceding-question scope when the continuation is clear."""
    nearest = next(
        (
            context
            for context in claim.antecedent_context
            if context.distance_before == 1
        ),
        None,
    )
    if (
        nearest is None
        or not nearest.text.rstrip().endswith("?")
        or not _QUESTION_START.match(nearest.text)
        or not _ANSWER_LIKE_START.match(claim.text)
    ):
        return claim.model_copy(update={"discourse_dependencies": []})
    dependency = ClaimDiscourseDependency(
        relation="answers_preceding_question",
        context_index=nearest.context_index,
        context_text=nearest.text,
        context_paper_start=nearest.paper_start,
        context_paper_end=nearest.paper_end,
        confidence="high",
        method=DISCOURSE_DEPENDENCY_VERSION,
    )
    return claim.model_copy(update={"discourse_dependencies": [dependency]})
