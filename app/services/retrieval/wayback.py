"""Archived copies of a submitted web page (`wayback-snapshot-v1`, owner decision 2026-10-02).

A cited page that refuses automated requests, has moved or no longer exists is
often preserved by the Internet Archive. When the student's own link fails,
the closest archived capture of that exact address is requested through the
Wayback availability API and read like any fetched page: the ordinary identity,
type and completeness checks decide whether it is the cited work.

Boundaries (ARCHITECTURE §4): only the student's submitted address is looked
up, never a search result; the submitted-link observation keeps the original
failure, so the archived copy never silently corrects the student's link; the
capture's timestamp and address are kept as provenance; requests use the same
fetch-safety controls. This is a public archive of the page, not a way around
a login, paywall or consent wall: a capture of such a wall fails the same
checks a live one does.
"""
import re
from urllib.parse import quote

from app.services.processing_metrics import record_provider_request

POLICY = "wayback-snapshot-v1"
AVAILABILITY_URL = "https://archive.org/wayback/available"
# "id_" asks for the page as captured, without the archive's own toolbar.
SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{url}"
# The failures a capture can stand in for: refused, missing, unreachable or
# unreadable pages. A page that was read and named a different work is not one.
RETRY_REASONS = frozenset({"access_restricted", "transport_failure", "fetch_unavailable",
                           "readable_text_unavailable"})


def archived_snapshot(url: str, *, timeout: float = 15.0) -> dict | None:
    """{snapshot_url, timestamp} for the closest archived capture, or None."""
    from app.services.safe_fetch import safe_request
    if not re.match(r"^https?://", str(url or ""), re.IGNORECASE):
        return None
    record_provider_request("wayback")
    try:
        response = safe_request(f"{AVAILABILITY_URL}?url={quote(url, safe='')}", timeout=timeout,
                                usage_label="archived copy lookup", raise_on_status=False)
        payload = response.json() if response.status_code == 200 else {}
    except Exception:  # noqa: BLE001 - an archive lookup failure is only a missing copy
        return None
    closest = ((payload or {}).get("archived_snapshots") or {}).get("closest") or {}
    timestamp = str(closest.get("timestamp") or "")
    if not closest.get("available") or str(closest.get("status")) != "200" or not re.fullmatch(r"\d{8,14}", timestamp):
        return None
    return {"snapshot_url": SNAPSHOT_URL.format(timestamp=timestamp, url=url), "timestamp": timestamp,
            "policy_version": POLICY}
