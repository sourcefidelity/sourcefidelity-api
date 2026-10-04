"""Evidence for verifying a cited web page (`webpage-verification-v1`, owner decision 2026-10-03).

Bibliographic indexes do not hold web pages, so their silence proves nothing.
The student's own link, the Internet Archive and web search do. A web page
can be reported as **Cannot be verified** only when all of these hold:

1. its submitted link fails tellingly: the page is gone (404/410), the site
   answers with a page about something else (a title conflict), or the link
   is only the site's home page. A refused, blocked or unreachable link
   proves nothing and leaves the page unassessed;
2. the Internet Archive holds no capture of that exact address showing the
   cited title;
3. completed web searches (Brave and Exa) found no page with that title;
4. the reference gives a link at all.

A link or archived capture that shows the cited title verifies the page.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

POLICY = "webpage-verification-v1"
WEB_PAGE_KINDS = frozenset({"webpage", "blog_post", "news_article"})


_STOP = frozenset("the a an of and in on to for with by at from as is are was film".split())


def _title_words(value: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]{3,}", str(value or "").casefold()) if w not in _STOP}


def near_title(cited: str, page: str) -> bool:
    """Most of the cited title's words appear in the page title (GradeSaver's
    "… Study Guide: Analysis" for "… Analysis. [Web log post]")."""
    words = _title_words(re.sub(r"\[[^\]]*\]", " ", cited or ""))
    return len(words) >= 2 and len(words & _title_words(page)) >= 0.6 * len(words)


def link_state(observations: list[dict] | None, url: str | None,
               title: str = "") -> tuple[str, str]:
    """('confirmed' | 'near' | 'telling' | 'not_telling', reason) from the submitted-link check."""
    parsed = urlsplit(str(url or ""))
    if parsed.scheme in {"http", "https"} and parsed.path.strip("/") == "" and not parsed.query:
        return "telling", "site_home_page_only"
    requests = [q for row in observations or [] if isinstance(row, dict) and row.get("kind") == "url"
                for q in row.get("requests") or [] if isinstance(q, dict)]
    if not requests:
        return "not_telling", "link_not_checked"
    last = max(requests, key=lambda q: str(q.get("completed_at") or ""))
    status, outcome = last.get("http_status"), last.get("outcome")
    if last.get("destination_identity") == "confirmed":
        return "confirmed", "submitted_page_confirmed"
    if outcome in {"not_found", "removed"}:
        return "telling", "submitted_page_missing"
    if (outcome == "response" and isinstance(status, int) and 200 <= status < 300
            and last.get("destination_identity") == "bibliographic_conflict"
            and "title" in (last.get("identity_fields") or [])):
        from app.services.source_resolver import interstitial_page_title
        destinations = [d.get("destination") for d in last.get("identity_differences") or [] if d.get("field") == "title"]
        if destinations and not any(interstitial_page_title(str(d or "")) for d in destinations):
            if any(near_title(title, str(d or "")) for d in destinations):
                return "near", "submitted_page_near_title"
            return "telling", "submitted_page_about_another_work"
    if last.get("page_observation") == "site_homepage":
        return "telling", "site_home_page_only"
    return "not_telling", f"link_{outcome or 'unknown'}"


def archive_check(url: str, title: str) -> str:
    """'matched' when the closest Internet Archive capture shows the cited
    title, 'unmatched' when a capture shows something else, 'none' otherwise."""
    from app.services.retrieval.wayback import archived_snapshot
    from app.services.safe_fetch import safe_request
    from app.services.source_resolver import _extract_html_titles, _html_title_matches, interstitial_page_title
    from app.services.reference_discovery import _web_title_furniture_only
    if not url or not title:
        return "none"
    snapshot = archived_snapshot(url)
    if snapshot is None:
        return "none"
    try:
        response = safe_request(snapshot["snapshot_url"], timeout=20, raise_on_status=False,
                                usage_label="archived copy")
        titles = [t for t in _extract_html_titles(response.text) if t and not interstitial_page_title(t)]
    except Exception:  # noqa: BLE001 - an unreadable capture is not a capture of the page
        return "none"
    if not titles:
        return "none"
    if _html_title_matches(title, titles) or any(_web_title_furniture_only(title, t, author_match=False) for t in titles):
        return "matched"
    return "unmatched"


def web_page_check(reference, observations: list[dict] | None) -> dict | None:
    """The stored evidence for a web-page reference with a link; None otherwise."""
    kind = str(getattr(reference, "source_kind", "") or "")
    url = str(getattr(reference, "url", "") or "")
    title = str(getattr(reference, "title", "") or "")
    if kind not in WEB_PAGE_KINDS or not re.match(r"(?i)^https?://", url):
        return None
    if len(_title_words(title)) < 2 or re.match(r"(?i)\s*(?:retrieved|available|accessed)\b", title):
        return None   # no usable title to search for or compare
    state, reason = link_state(observations, url, title)
    check = {"policy_version": POLICY, "link": state, "link_reason": reason, "archive": "not_checked"}
    if state == "telling":
        check["archive"] = archive_check(url, str(getattr(reference, "title", "") or ""))
    return check
