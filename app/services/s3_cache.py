"""S3/MinIO cache for full-text storage.

S3 storage cache for retrieved documents.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def upload_text(key: str, text: str) -> bool:
    """
    Upload text to S3/MinIO cache.

    Args:
        key: Object key (e.g., DOI-based filename).
        text: Text content to store.

    Returns:
        True if successful.
    """
    # Placeholder – will be implemented in Phase 3.5
    raise NotImplementedError("Phase 3.5")


async def download_text(key: str) -> Optional[str]:
    """
    Download text from S3/MinIO cache.

    Args:
        key: Object key.

    Returns:
        Text content, or None if not found.
    """
    # Placeholder – will be implemented in Phase 3.5
    raise NotImplementedError("Phase 3.5")
