"""Link validation for reference URLs.

Checks every URL in a reference list and categorizes the result for the
instructor report. This is a pre-verification step: before we attempt to
retrieve full text and check quotations, we first establish whether the
student's cited links are reachable and point to the right content.

Categories:
  - OK:           page loads (200)
  - MATCH:        page loads AND its title matches the cited reference
  - MISMATCH:     page loads but title doesn't match (wrong URL or page changed)
  - DEAD:         404/410 — page doesn't exist
  - REDIRECT:     301/302 — page moved (redirect target reported)
  - PAYWALL:      401/403 — requires authentication
  - SERVER_ERROR: 500/502/503 — server problem (may be temporary)
  - TIMEOUT:      no response within timeout (network-restricted region / dead host / slow — can't distinguish)
  - DNS_ERROR:    host doesn't exist or connection refused
  - WRONG_TYPE:   200 but content isn't HTML or PDF (image, download, etc.)

Note: a TIMEOUT is ambiguous. If our server and the target host are in
different network-reachability regions (e.g. cross-region fetches), we
cannot determine whether the cause is regional network filtering, a
dead host, or a slow server. We report it as "unreachable from our network"
without claiming the link is broken.
"""

import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# Categories
OK = "ok"
MATCH = "content_match"
MISMATCH = "content_mismatch"
DEAD = "dead"
REDIRECT = "redirect"
PAYWALL = "paywall"
SERVER_ERROR = "server_error"
TIMEOUT = "timeout"
DNS_ERROR = "dns_error"
WRONG_TYPE = "wrong_type"

# HTTP status → category mapping
_STATUS_CATEGORIES = {
    200: OK,
    301: REDIRECT,
    302: REDIRECT,
    303: REDIRECT,
    307: REDIRECT,
    308: REDIRECT,
    401: PAYWALL,
    403: PAYWALL,
    404: DEAD,
    410: DEAD,
    451: PAYWALL,  # "Unavailable for legal reasons"
    500: SERVER_ERROR,
    502: SERVER_ERROR,
    503: SERVER_ERROR,
    504: SERVER_ERROR,
}

_TIMEOUT_SECONDS = 15
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass
class LinkCheckResult:
    """Result of checking a single reference URL.

    Attributes:
        url: the URL that was checked
        category: one of the category constants above
        status_code: HTTP status code (or None if no response)
        detail: human-readable explanation
        page_title: the <title> of the page if it loaded (for match checking)
        redirect_url: final URL after redirects (if redirected)
        cited_title: the title from the reference (for match comparison)
        title_match: whether the page title matches the cited title
    """

    url: str
    category: str
    status_code: int | None = None
    detail: str = ""
    page_title: str | None = None
    redirect_url: str | None = None
    cited_title: str | None = None
    title_match: bool | None = None


def check_link(
    url: str,
    cited_title: str | None = None,
    timeout: int = _TIMEOUT_SECONDS,
) -> LinkCheckResult:
    """Check a single URL and categorize its status.

    Args:
        url: The URL to check.
        cited_title: The title from the student's reference (for content matching).
        timeout: Request timeout in seconds.

    Returns:
        LinkCheckResult with the categorized status.
    """
    if not url or not url.strip():
        return LinkCheckResult(url=url, category=DEAD, detail="Empty URL")

    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        return LinkCheckResult(
            url=url, category=WRONG_TYPE, detail=f"Invalid URL scheme (not http/https)"
        )

    headers = {
        "User-Agent": "SourceFidelity/0.1.0 (source-verification-bot; +https://github.com/sourcefidelity/sourcefidelity-api)",
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
    }

    try:
        resp = httpx.get(
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        status = resp.status_code
        category = _STATUS_CATEGORIES.get(status, OK if 200 <= status < 300 else SERVER_ERROR)

        # Extract page title and check content match for successful responses
        page_title = None
        title_match = None
        if category in (OK, MATCH, MISMATCH) and status == 200:
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" in content_type or "application/xhtml" in content_type:
                page_title = _extract_title(resp.text)
                # Check if redirect happened
                final_url = str(resp.url)
                redirect_url = final_url if final_url != url else None

                if cited_title and page_title:
                    title_match = _titles_match(cited_title, page_title)
                    if title_match:
                        category = MATCH
                    else:
                        category = MISMATCH
                elif redirect_url:
                    category = REDIRECT

                return LinkCheckResult(
                    url=url,
                    category=category,
                    status_code=status,
                    page_title=page_title,
                    redirect_url=redirect_url if 'redirect_url' in dir() else None,
                    cited_title=cited_title,
                    title_match=title_match,
                    detail=_detail_for(category, page_title, cited_title),
                )
            elif "application/pdf" not in content_type and "text/plain" not in content_type:
                return LinkCheckResult(
                    url=url,
                    category=WRONG_TYPE,
                    status_code=status,
                    detail=f"Content type is {content_type} (not HTML or PDF)",
                )

        return LinkCheckResult(
            url=url,
            category=category,
            status_code=status,
            page_title=page_title,
            cited_title=cited_title,
            detail=_detail_for(category),
        )

    except httpx.TimeoutException:
        return LinkCheckResult(
            url=url,
            category=TIMEOUT,
            detail=f"No response within {timeout}s (may be network-restricted, dead host, or slow server — cannot distinguish)",
        )
    except httpx.ConnectError as e:
        return LinkCheckResult(
            url=url,
            category=DNS_ERROR,
            detail=f"Connection failed: {str(e)[:80]}",
        )
    except httpx.HTTPError as e:
        return LinkCheckResult(
            url=url,
            category=SERVER_ERROR,
            detail=f"HTTP error: {str(e)[:80]}",
        )
    except Exception as e:
        logger.warning("Link check failed for %s: %s", url, e)
        return LinkCheckResult(
            url=url,
            category=SERVER_ERROR,
            detail=f"Unexpected error: {str(e)[:80]}",
        )


def check_reference_links(references) -> dict:
    """Check all URLs in a reference list and return a categorized summary.

    Args:
        references: List of ParsedReference objects (from extract_and_parse_references).

    Returns:
        dict with:
            total: total references checked
            total_with_urls: how many had URLs
            results: list of LinkCheckResult for each URL
            summary: dict mapping category -> count
    """
    import re as _re

    _url_re = _re.compile(r"https?://|www\.", _re.IGNORECASE)
    results: list[LinkCheckResult] = []

    for ref in references:
        # Extract URL from structured field or raw_ref
        url = ref.url.strip() if ref.url else ""
        if not url:
            m = _url_re.search(ref.raw_ref or "")
            if m:
                # Extract the full URL from raw_ref
                url_match = _re.search(r"https?://[^\s]+", ref.raw_ref or "")
                if url_match:
                    url = url_match.group(0).rstrip(".,);]")
        if not url:
            continue

        result = check_link(url, cited_title=ref.title or None)
        results.append(result)

    summary: dict[str, int] = {}
    for r in results:
        summary[r.category] = summary.get(r.category, 0) + 1

    return {
        "total": len(references),
        "total_with_urls": len(results),
        "results": results,
        "summary": summary,
    }


def _extract_title(html: str) -> str | None:
    """Extract the <title> from an HTML page."""
    m = _TITLE_RE.search(html)
    if m:
        title = m.group(1).strip()
        # Clean common HTML entities and whitespace
        title = re.sub(r"\s+", " ", title)
        return title if title else None
    return None


def _titles_match(cited_title: str, page_title: str) -> bool:
    """Check whether a cited title plausibly matches a page title.

    Uses word overlap — the titles don't need to be identical, but the
    significant words should largely overlap.
    """
    cited_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", cited_title))
    page_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", page_title))
    if not cited_words:
        return False
    overlap = len(cited_words & page_words) / len(cited_words)
    return overlap >= 0.5


def _detail_for(category: str, page_title: str | None = None, cited_title: str | None = None) -> str:
    """Generate a human-readable detail string for a category."""
    details = {
        OK: "Page loads successfully",
        MATCH: f"Page loads and title matches the cited reference",
        MISMATCH: f"Page loads but title doesn't match the cited reference (may be wrong URL or page changed)",
        DEAD: "Page does not exist (404/410)",
        REDIRECT: "Page redirected (target reported)",
        PAYWALL: "Page requires authentication (paywall/login)",
        SERVER_ERROR: "Server error (may be temporary)",
        TIMEOUT: "Unreachable from our network (may be network-restricted, dead host, or slow — cannot distinguish)",
        DNS_ERROR: "Host does not exist or connection refused",
        WRONG_TYPE: "Link points to non-HTML content (image, download, etc.)",
    }
    return details.get(category, "Unknown status")
