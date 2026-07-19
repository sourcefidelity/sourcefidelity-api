"""DOI and title-hash caching layer.

Caches parsed references by DOI or title hash to avoid re-parsing
and re-validating the same references across jobs.

Cache structure:
- Primary key: DOI (normalized)
- Fallback key: Title hash (SHA-256 of normalized title)
- Value: Parsed reference data
"""

import hashlib
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# In-memory cache (for single-process development)
# In production, this would use Redis or database-backed cache
_cache: Dict[str, Dict[str, Any]] = {}
_cache_hits = 0
_cache_misses = 0


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_doi(doi: str) -> str:
    """Normalize a DOI to canonical form.

    Handles:
    - doi:10.xxxx/yyy
    - https://doi.org/10.xxxx/yyy
    - 10.xxxx/yyy

    Returns:
        Lowercase DOI without prefix: "10.xxxx/yyy"
    """
    if not doi:
        return ""

    doi = doi.strip().lower()

    # Remove common prefixes
    prefixes = ["https://doi.org/", "http://doi.org/", "doi:"]
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break

    # Validate format (basic check)
    if not doi.startswith("10."):
        logger.warning("Invalid DOI format: %s", doi)
        return ""

    return doi


def normalize_title(title: str) -> str:
    """Normalize a title for hashing.

    Removes:
    - Leading/trailing whitespace
    - Multiple spaces
    - Punctuation variations

    Returns:
        Normalized title for hashing.
    """
    if not title:
        return ""

    # Lowercase and strip
    title = title.lower().strip()

    # Replace multiple spaces with single space
    import re
    title = re.sub(r"\s+", " ", title)

    # Remove common punctuation variations
    title = re.sub(r"['\"—–-]", "", title)

    return title


def compute_title_hash(title: str) -> str:
    """Compute SHA-256 hash of normalized title.

    Args:
        title: Raw title string.

    Returns:
        Hex string of SHA-256 hash.
    """
    normalized = normalize_title(title)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------


def get_cached_reference(doi: Optional[str] = None, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a cached reference by DOI or title.

    Args:
        doi: DOI string (will be normalized).
        title: Title string (will be hashed).

    Returns:
        Cached reference data if found, None otherwise.

    Note:
        DOI lookup is preferred. Title hash is used as fallback.
    """
    global _cache_hits, _cache_misses

    # Try DOI first
    if doi:
        normalized_doi = normalize_doi(doi)
        if normalized_doi and normalized_doi in _cache:
            _cache_hits += 1
            logger.debug("Cache HIT (DOI): %s", normalized_doi)
            return _cache[normalized_doi]

    # Fallback to title hash
    if title:
        title_hash = compute_title_hash(title)
        cache_key = f"title:{title_hash}"
        if cache_key in _cache:
            _cache_hits += 1
            logger.debug("Cache HIT (title hash): %s", title_hash[:16])
            return _cache[cache_key]

    _cache_misses += 1
    logger.debug("Cache MISS (DOI: %s, title: %s)", doi, title[:50] if title else None)
    return None


def cache_reference(
    data: Dict[str, Any],
    doi: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Cache a parsed reference by DOI and/or title.

    Args:
        data: Parsed reference data.
        doi: DOI string (will be normalized).
        title: Title string (will be hashed).

    Note:
        If both DOI and title are provided, caches under both keys.
    """
    if not data:
        return

    # Add timestamp
    data["_cached_at"] = datetime.utcnow().isoformat()

    # Cache by DOI
    if doi:
        normalized_doi = normalize_doi(doi)
        if normalized_doi:
            _cache[normalized_doi] = data
            logger.debug("Cached (DOI): %s", normalized_doi)

    # Cache by title hash
    if title:
        title_hash = compute_title_hash(title)
        cache_key = f"title:{title_hash}"
        _cache[cache_key] = data
        logger.debug("Cached (title hash): %s", title_hash[:16])


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dictionary with hits, misses, size, and hit rate.
    """
    total = _cache_hits + _cache_misses
    hit_rate = _cache_hits / total if total > 0 else 0.0

    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "size": len(_cache),
        "hit_rate": f"{hit_rate:.1%}",
    }


def clear_cache() -> None:
    """Clear the entire cache."""
    global _cache, _cache_hits, _cache_misses
    _cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    logger.info("Cache cleared")


def get_cache_keys() -> list:
    """Get all cache keys (for debugging)."""
    return list(_cache.keys())


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


def batch_get_cached(
    references: list,
) -> tuple[list, list]:
    """Check which references are already cached.

    Args:
        references: List of raw reference strings.

    Returns:
        Tuple of (cached_indices, uncached_indices).
        - cached_indices: Indices of references found in cache
        - uncached_indices: Indices of references not in cache
    """
    cached_indices = []
    uncached_indices = []

    for i, ref in enumerate(references):
        # Try to find by DOI or title from the raw string
        # (This is a simple heuristic - full parsing would be better)
        cached = get_cached_reference(
            doi=_extract_doi_heuristic(ref),
            title=_extract_title_heuristic(ref),
        )
        if cached:
            cached_indices.append(i)
        else:
            uncached_indices.append(i)

    return cached_indices, uncached_indices


def _extract_doi_heuristic(text: str) -> Optional[str]:
    """Quick heuristic to extract DOI from text.

    Not a full parser - just looks for DOI patterns.
    """
    import re

    # Look for DOI patterns
    patterns = [
        r"10\.\d{4,}/[^\s]+",  # Standard DOI
        r"doi[:\s]+(10\.\d{4,}/[^\s]+)",  # doi: prefix
        r"doi\.org/(10\.\d{4,}/[^\s]+)",  # URL form
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.lastindex else match.group(0)
            return normalize_doi(doi)

    return None


def _extract_title_heuristic(text: str) -> Optional[str]:
    """Quick heuristic to extract title from text.

    Not a full parser - just a fallback for cache lookup.
    """
    # This is intentionally simple - the LLM will do proper parsing
    # Just extract text between first period and year
    import re

    # Try to find title-like text after author and year
    match = re.search(r"\(\d{4}\)\.\s*([^.]+(?:\.[^.]+)?)", text)
    if match:
        return match.group(1).strip()

    return None