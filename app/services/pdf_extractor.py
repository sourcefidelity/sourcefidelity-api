"""PDF/HTML full-text retrieval and extraction.

PDF text extraction module.
Handles PDF download, HTML parsing, and fallback logic.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def download_pdf(url: str) -> Optional[bytes]:
    """
    Download a PDF from a URL.

    Args:
        url: Direct PDF URL.

    Returns:
        PDF content as bytes, or None if download fails.
    """
    # Placeholder – will be implemented in Phase 3.7
    raise NotImplementedError("Phase 3.7")


async def extract_html_text(url: str) -> Optional[str]:
    """
    Download and extract text from an HTML page.

    Args:
        url: The webpage URL.

    Returns:
        Extracted plain text, or None.
    """
    # Placeholder – will be implemented in Phase 3.7
    raise NotImplementedError("Phase 3.7")
