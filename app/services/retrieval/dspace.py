"""DSpace 7+ repository items (owner request 2026-09-28).

A DSpace 7 landing page is a JavaScript application: the server sends an empty
shell (`<ds-app>`), and the item's title, authors and files exist only in the
repository's REST API. Many university thesis and article repositories run
it. Given the landing page, this module asks that API, on the same host, for
the item (by handle or item id), reads its bibliographic metadata and lists the
PDF files in its ORIGINAL bundle. Nothing here decides identity: the caller
passes the metadata and file locations to the ordinary acquisition path, whose
identity gates apply to the file itself.
"""
from __future__ import annotations

import re
from urllib.parse import quote, urlsplit

MAX_FILES = 2
_HANDLE = re.compile(r"/handle/(\d+(?:\.\d+)*/[\w.\-]+)")
_ITEM = re.compile(r"/items/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_SHELL = re.compile(r"<ds-app\b", re.I)


def is_dspace_shell(html: str) -> bool:
    return bool(_SHELL.search(html or ""))


def item_api_url(page_url: str) -> str | None:
    """The REST address of the item a landing page shows, or None."""
    parts = urlsplit(page_url)
    if parts.scheme != "https" or not parts.hostname:
        return None
    origin = f"https://{parts.netloc}"
    handle = _HANDLE.search(parts.path)
    if handle:
        return f"{origin}/server/api/pid/find?id={quote(handle.group(1), safe='/')}"
    item = _ITEM.search(parts.path)
    if item:
        return f"{origin}/server/api/core/items/{item.group(1).lower()}"
    return None


def _values(metadata: dict, key: str) -> list[str]:
    return [str(row.get("value") or "").strip() for row in metadata.get(key) or [] if row.get("value")]


def _same_host(url: str, host: str) -> bool:
    parts = urlsplit(url or "")
    return parts.scheme == "https" and parts.hostname == host


def repository_item(page_url: str, fetch_json) -> dict | None:
    """{title, authors, year, files: [{url, name, size}]} for a DSpace 7 landing page.

    `fetch_json(url)` performs one guarded request and returns parsed JSON.
    Every followed link must stay on the landing page's host.
    """
    api = item_api_url(page_url)
    if api is None:
        return None
    host = urlsplit(page_url).hostname
    item = fetch_json(api)
    if not isinstance(item, dict) or item.get("type") != "item":
        return None
    metadata = item.get("metadata") or {}
    bundles_url = ((item.get("_links") or {}).get("bundles") or {}).get("href")
    files = []
    if _same_host(bundles_url, host):
        bundles = ((fetch_json(bundles_url) or {}).get("_embedded") or {}).get("bundles") or []
        original = next((b for b in bundles if b.get("name") == "ORIGINAL"), None)
        streams_url = (((original or {}).get("_links") or {}).get("bitstreams") or {}).get("href")
        if _same_host(streams_url, host):
            streams = ((fetch_json(streams_url) or {}).get("_embedded") or {}).get("bitstreams") or []
            for stream in streams:
                name = str(stream.get("name") or "")
                mime = _values(stream.get("metadata") or {}, "dc.format.mimetype")
                content = ((stream.get("_links") or {}).get("content") or {}).get("href")
                if (name.lower().endswith(".pdf") or "application/pdf" in mime) and _same_host(content, host):
                    files.append({"url": content, "name": name, "size": stream.get("sizeBytes")})
    year = next((m.group(0) for m in (re.search(r"\b(1[5-9]|20)\d{2}\b", v)
                                      for v in _values(metadata, "dc.date.issued")) if m), None)
    return {"title": next(iter(_values(metadata, "dc.title")), None),
            "authors": _values(metadata, "dc.contributor.author"),
            "year": year, "files": files[:MAX_FILES]}
