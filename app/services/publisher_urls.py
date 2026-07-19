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

import httpx

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
        try:
            resp = httpx.get(url, headers=_BROWSER_HEADERS, timeout=_PDF_TIMEOUT, follow_redirects=True)
            if resp.status_code == 200 and resp.content.startswith(PDF_MAGIC):
                logger.info("Downloaded publisher PDF from %s (%d bytes)", url[:60], len(resp.content))
                return resp.content
            else:
                logger.debug(
                    "Publisher PDF URL %s returned status=%d, content-type=%s (not a PDF)",
                    url[:60], resp.status_code, resp.headers.get("content-type", "?"),
                )
        except Exception as e:
            logger.debug("Publisher PDF download failed for %s: %s", url[:60], e)

    return None
