"""Where a report's DOI links point (owner decision 2026-10-19).

The user may set a DOI resolver address (`DOI_RESOLVER_URL`) so DOI links open
wherever that resolver sends them; unset, they go to https://doi.org/. Links
only: the application itself follows public DOIs through doi.org.
"""
from urllib.parse import quote, urlsplit

from app.config import settings

PUBLIC_DOI_BASE = "https://doi.org/"


def doi_link_base() -> str:
    base = (settings.DOI_RESOLVER_URL or "").strip()
    if base and urlsplit(base).scheme in {"http", "https"} and urlsplit(base).netloc:
        return base
    return PUBLIC_DOI_BASE


def doi_href(doi: str, *, safe: str = "/") -> str:
    """The link for a DOI, quoted for an address."""
    return doi_link_base() + quote(doi, safe=safe)
