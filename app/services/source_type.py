"""Source-type detection for routing references to the right resolution path.

Different source types need different retrieval strategies:

  - Academic works (articles, books): → academic databases (OpenAlex, CORE,
    S2, Crossref) + S3 cache. These have DOIs/ISBNs or searchable titles.
  - Traditional media (films, TV, albums, artwork): NOT in academic databases
    in a useful form. Routing them to title search produces false-positive
    keyword matches (e.g. "Rain Man" → papers *about* the film). These should
    skip academic DBs entirely.
  - Websites / digital-native: → live web-fetch path (Phase 3.7). These live
    at a URL, not in an academic index.
  - Physical archives: unverifiable automatically. Skip all retrieval.

This module provides the detection used by both the source resolver (for
routing) and count_missing_identifiers (for the penalty policy).
"""

import re

# Traditional-media markers. These sources are cited by title/creator/year per
# convention and do not belong in academic-database title search.
TRADITIONAL_MEDIA_RE = re.compile(
    r"\[(?:film|motion picture|tv series|television series|album|"
    r"recording|painting|sculpture|play|performance|photograph|"
    r"dvd|blu-?ray|cd|lp|ep|videorecording|video recording|"
    r"video game|game|opera|ballet|musical|concert|television episode|"
    r"tv episode|podcast episode)\]",
    re.IGNORECASE,
)

# Director/artist/creator credits also signal traditional media.
DIRECTOR_RE = re.compile(
    r"\b(?:director|dir\.|performer|perf\.|artist|creator|choreographer|"
    r"conductor|host|narrator)\b",
    re.IGNORECASE,
)

# Physical-archive markers. Unverifiable automatically.
ARCHIVE_RE = re.compile(
    r"\b(?:archive|archives|collection|manuscript|ms\.|mss\.|"
    r"special collections|box\s+\d|folder\s+\d|repository|"
    r"unpublished manuscript)\b",
    re.IGNORECASE,
)


def is_traditional_media(raw_ref: str) -> bool:
    """Return True if the reference cites traditional media (film, TV, album, etc.).

    Such references should NOT be sent to academic-database title search —
    they produce false-positive keyword matches (papers *about* the work)
    rather than the work itself.
    """
    return bool(TRADITIONAL_MEDIA_RE.search(raw_ref) or DIRECTOR_RE.search(raw_ref))


def is_archive_source(raw_ref: str) -> bool:
    """Return True if the reference cites a physical archive / manuscript.

    These cannot be verified automatically (no API to a physical archive).
    """
    return bool(ARCHIVE_RE.search(raw_ref))
