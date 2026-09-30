"""Publisher PDF URL construction for campus-network paywalled retrieval.

On a university network, publishers (Springer, Taylor & Francis, Wiley,
Elsevier, etc.) grant IP-based access to subscribed content. But a DOI
resolves to an HTML landing page, not a PDF. To download the actual PDF,
we construct the direct PDF URL using publisher-specific patterns.

This module maps DOIs to publisher PDF URLs. When SourceFidelity runs on a campus
network, these URLs will serve the full PDF (the publisher recognizes the
institution's IP). Off-campus, they'll redirect to a login/paywall page.

Usage: given a DOI and the publisher (from Crossref metadata), construct the
most likely PDF URL and try to download it. If it's a real PDF (magic bytes),
cache it; if it's HTML (paywall), fall back to abstract-only verification.
"""

import logging
import re
from urllib.parse import unquote, urlsplit

import httpx

from app.log_safety import private_value_id

from app.services.safe_fetch import safe_fetch_bytes
from app.services.candidate_budget import require_source_candidate

logger = logging.getLogger(__name__)

# Magic bytes for PDF
PDF_MAGIC = b"%PDF-"

# Timeout for PDF download attempts
_PDF_TIMEOUT = 30

# Headers that make us look like a browser (some publishers reject bot UAs)
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,*/*",
}


def construct_pdf_url(doi: str, publisher: str | None = None) -> str | None:
    """Construct the most likely direct PDF URL for a DOI.

    Args:
        doi: The DOI (e.g. "10.1080/10509208.2019.1660132").
        publisher: The publisher name (from Crossref metadata), used to pick
            the URL pattern. If None, all known patterns are tried.

    Returns:
        The most likely PDF URL, or None if no pattern is known.
    """
    doi_clean = doi.strip()
    pub_lower = (publisher or "").lower()

    # Springer / Nature (link.springer.com)
    if "springer" in pub_lower or "nature" in pub_lower:
        return f"https://link.springer.com/content/pdf/{doi_clean}.pdf"

    # Taylor & Francis (tandfonline.com)
    if "taylor" in pub_lower or "routledge" in pub_lower or "t & f" in pub_lower:
        return f"https://www.tandfonline.com/doi/pdf/{doi_clean}?download=true"

    # Wiley (onlinelibrary.wiley.com)
    if "wiley" in pub_lower:
        return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi_clean}?download=true"

    # SAGE (journals.sagepub.com)
    if "sage" in pub_lower:
        return f"https://journals.sagepub.com/doi/pdf/{doi_clean}"

    # Oxford University Press (academic.oup.com)
    if "oxford" in pub_lower or "oup" in pub_lower:
        # OUP PDF URLs use the DOI path but are less predictable
        return f"https://academic.oup.com/doi/pdf/{doi_clean}"

    # Cambridge University Press (cambridge.org)
    if "cambridge" in pub_lower:
        return f"https://www.cambridge.org/core/services/aop-cambridge-core/content/view/{doi_clean}.pdf"

    # Elsevier/ScienceDirect is complex — needs PII not DOI, so we can't
    # construct a direct PDF URL without parsing the landing page.
    # Return None; the caller will try the landing page instead.
    if "elsevier" in pub_lower or "sciencedirect" in pub_lower:
        return None

    # Unknown publisher — return None; caller falls back to OA URL or abstract
    return None


def try_download_publisher_pdf(
    doi: str,
    publisher: str | None = None,
    oa_url: str | None = None,
) -> bytes | None:
    """Attempt to download a publisher PDF (works on campus networks).

    Tries, in order:
    1. The constructed publisher PDF URL (if publisher is known)
    2. The OA URL from OpenAlex/S2 (if provided)

    Returns the PDF bytes if successful (magic bytes check), None otherwise.
    On a campus network, the publisher PDF URL will succeed for subscribed
    content. Off-campus, it'll return a paywall page (which fails the magic
    byte check).

    Args:
        doi: The DOI.
        publisher: Publisher name (from Crossref metadata).
        oa_url: An open-access PDF URL from OpenAlex/S2, if one was found.

    Returns:
        PDF bytes, or None if no PDF could be downloaded.
    """
    # Build the list of URLs to try
    urls_to_try: list[str] = []

    pdf_url = construct_pdf_url(doi, publisher)
    if pdf_url:
        urls_to_try.append(pdf_url)

    if oa_url and oa_url not in urls_to_try:
        urls_to_try.append(oa_url)

    if not urls_to_try:
        return None

    for url in urls_to_try:
        require_source_candidate(url)
        try:
            data = safe_fetch_bytes(
                url,
                usage_label="publisher PDF",
                accept_content_types=("application/pdf",),
                timeout=_PDF_TIMEOUT,
                headers=_BROWSER_HEADERS,
            )
            if data.startswith(PDF_MAGIC):
                logger.info(
                    "Downloaded publisher PDF from %s (%d bytes)",
                    private_value_id("url", url),
                    len(data),
                )
                return data
        except Exception as e:
            logger.debug(
                "Publisher PDF download failed for %s: %s",
                private_value_id("url", url),
                type(e).__name__,
            )

    return None


# ── MDPI official file server (owner approved 2026-09-29) ─────────────────
# MDPI (DOI prefix 10.3390) answers automated requests to www.mdpi.com with 403
# for both the article page and its /pdf route, while its own file host
# mdpi-res.com serves the same open-access PDF. When an MDPI location is
# blocked, one plain request to the file host is tried; the downloaded PDF
# then passes the ordinary acquisition, safety and identity gates. No browser
# user agent is sent and nothing else is attempted.
MDPI_DOI_PREFIX = "10.3390/"
MDPI_FILE_SERVER_ROUTE = "mdpi_file_server"
MDPI_BLOCKED_STATUSES = frozenset({401, 403, 451})
_MDPI_HOSTS = frozenset({"mdpi.com", "www.mdpi.com"})
_DOI_HOSTS = frozenset({"doi.org", "dx.doi.org"})
# Journal code, then volume, a two-digit issue and the article number: four
# zero-padded digits, or five once a volume passes article 9999
# (10.3390/literature5020007 is Literature 5(2), article 7).
_MDPI_SUFFIX = re.compile(r"([a-z]+)(\d{7,10})")
_MAX_MDPI_ISSUE = 24


def _normalize_doi(doi: str | None) -> str:
    value = unquote(str(doi or "")).strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            return value[len(prefix):].strip()
    return value


def is_mdpi_doi(doi: str | None) -> bool:
    return _normalize_doi(doi).startswith(MDPI_DOI_PREFIX)


def mdpi_file_server_urls(doi: str | None) -> list[str]:
    """File-server PDF addresses for an MDPI article DOI, most likely first.

    Built only from the DOI. The URL uses the journal's slug, which for most
    journals equals the DOI code (``literature``) but for some does not
    (``su`` is Sustainability); a constructed address that does not exist
    answers 404 and is simply unavailable. A digit string that reads two ways
    yields both readings; anything that is not an article DOI yields none.
    """
    normalized = _normalize_doi(doi)
    if not normalized.startswith(MDPI_DOI_PREFIX):
        return []
    match = _MDPI_SUFFIX.fullmatch(normalized[len(MDPI_DOI_PREFIX):])
    if not match:
        return []
    code, digits = match.groups()
    urls: list[str] = []
    for article_digits in (4, 5):
        volume = digits[: -(2 + article_digits)]
        issue = digits[-(2 + article_digits):-article_digits]
        article = digits[-article_digits:]
        if not volume or volume.startswith("0") or len(volume) > 3:
            continue
        if not 1 <= int(issue) <= _MAX_MDPI_ISSUE:
            continue
        if article_digits == 5 and (article.startswith("0") or int(volume) < 10):
            # Five digits appear only once a volume passes article 9999, which
            # only long-running mega-journals reach.
            continue
        if int(article) == 0:
            continue
        stem = f"{code}-{int(volume):02d}-{int(article):05d}"
        url = f"https://mdpi-res.com/d_attachment/{code}/{stem}/article_deploy/{stem}.pdf"
        if url not in urls:
            urls.append(url)
    return urls


def is_mdpi_location(url: str | None) -> bool:
    """Whether a location is MDPI's own site or a DOI link to an MDPI work."""
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return False
    host = (parts.hostname or "").casefold()
    if host in _MDPI_HOSTS:
        return True
    return host in _DOI_HOSTS and is_mdpi_doi(parts.path.lstrip("/"))


def mdpi_request_headers() -> dict[str, str]:
    """A plain request that names the application; no browser user agent."""
    from app.config import settings

    return {
        "User-Agent": f"SourceFidelity/{settings.APP_VERSION} (academic source verification)",
        "Accept": "application/pdf,*/*",
    }
