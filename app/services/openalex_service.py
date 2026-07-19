"""OpenAlex API integration.

OpenAlex API service.
"""

import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def lookup_doi(doi: str) -> Optional[dict]:
    """
    Look up a DOI on OpenAlex.

    Args:
        doi: The DOI string (with or without 'doi.org/' prefix).

    Returns:
        OpenAlex work dict, or None if not found.
    """
    # Placeholder – will be implemented in Phase 3.6
    raise NotImplementedError("Phase 3.6")


async def search_by_title(title: str) -> Optional[dict]:
    """
    Search OpenAlex by paper title.

    Args:
        title: The paper title.

    Returns:
        Best matching OpenAlex work dict, or None.
    """
    # Placeholder – will be implemented in Phase 3.6
    raise NotImplementedError("Phase 3.6")
