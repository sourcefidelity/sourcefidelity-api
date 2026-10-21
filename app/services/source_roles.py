"""What each kind of source text is for: the one table of passage roles.

Retrieval, the evidence-sentence selector, patchwriting, whole-source plans
and the acceptance tooling each kept their own list of roles that are not
evidence, and they drifted (rules audit 2026-10-18: patchwriting still
flagged contents lines and headings that retrieval had stopped offering).
Every module reads its set from here.
"""
from __future__ import annotations

# Source text that is never evidence for a student's statement.
NON_EVIDENCE_ROLES = frozenset({
    "reference_list",
    "citation_notes",
    "publication_metadata",
    "page_furniture",
    "author_biography",
    # A book's contents or index, and a heading or title standing alone
    # (2026-10-18).
    "document_navigation",
    "heading",
})

# Retrieval admits an explanatory note through its own channel; the selector
# may show it, as the judge reads it. A title/byline block stays identity
# context for the judge and is never listed as evidence.
SELECTION_EXCLUDED_ROLES = (NON_EVIDENCE_ROLES - {"citation_notes"}) | {"document_metadata"}

# Wording shared with these is a title, note or furniture, not the source's prose.
PATCHWRITING_UNFLAGGED_ROLES = NON_EVIDENCE_ROLES | {"document_metadata"}
