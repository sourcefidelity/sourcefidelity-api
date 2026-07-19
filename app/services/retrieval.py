"""Adaptive paragraph retrieval from full text.

Adaptive paragraph retrieval for source matching.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def retrieve_relevant_paragraph(full_text: str, claim: dict) -> Optional[str]:
    """
    Find the most relevant paragraph in the full text for a given citation claim.

    Args:
        full_text: The full text of the cited work.
        claim: Dict with citation metadata.

    Returns:
        The most relevant paragraph text, or None.
    """
    # Placeholder – will be implemented in Phase 3.9
    raise NotImplementedError("Phase 3.9")
