"""Stable paper-local identities for parsed bibliography entries."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from app.services.schemas import ParsedReference


def _normalized_identity_text(reference: ParsedReference) -> str:
    source = reference.raw_ref or "|".join(
        (reference.author, reference.year, reference.title, reference.doi, reference.url)
    )
    source = unicodedata.normalize("NFKC", source).casefold()
    return re.sub(r"\s+", " ", source).strip()


def assign_reference_ids(
    references: list[ParsedReference],
    *,
    paper_version_id: str = "",
) -> list[ParsedReference]:
    """Assign deterministic, unique IDs in bibliography order.

    The ordinal represents the ordered paper-local reference span. The digest
    detects accidental mismatches while avoiding raw reference text in IDs.
    IDs are deliberately overwritten because cached parse results may have
    originated in another paper.
    """
    for index, reference in enumerate(references):
        seed = "\x1f".join(
            (paper_version_id.strip(), str(index), _normalized_identity_text(reference))
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        reference.reference_id = f"ref-{index + 1:04d}-{digest}"
    return references
