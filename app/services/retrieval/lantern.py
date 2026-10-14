"""Magazine and trade-paper pages from the Media History Digital Library (`lantern-pages-v1`).

Owner decision 2026-10-14. Students cite fan magazines and trade papers by a
Lantern page link (lantern.mediahist.org/catalog/<volume>_<leaf>). The link
names one scanned leaf of a bound volume, often not the article's first page
(a Screenland link to a Mary Pickford caption for "Meet the new blonde",
which is printed page 26 of an earlier issue in the same volume). This route:

1. reads the volume and leaf from the link;
2. maps printed page numbers to leaves with the Internet Archive page-number
   file of that volume (`<volume>_page_numbers.json`);
3. among the leaves printed with the cited first page (one per bound issue),
   plus the linked leaf, takes the one whose OCR text carries the cited title;
4. reads the OCR text of the cited pages from Lantern.

A range longer than `MAX_CONTIGUOUS_PAGES` is read as its first two pages and
its last page (an article continued at the back of the issue) and is not
complete. The text is OCR and recorded as such; nothing is guessed when the
title is not found.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

POLICY = "lantern-pages-v1"
LANTERN = "https://lantern.mediahist.org"
ARCHIVE = "https://archive.org"
MAX_CONTIGUOUS_PAGES = 8
MAX_CANDIDATE_LEAVES = 8
_LINK = re.compile(r"^https?://lantern\.mediahist\.org/catalog/([A-Za-z0-9._-]+?)_(\d{1,5})/?(?:[?#].*)?$")
_OCR = re.compile(r'(?s)<div class="card card-body ">(.*?)</div>')
_PAGES = re.compile(r"(?:pp?\.\s*)?\b(\d{1,4})\s*[-–—]\s*(\d{1,4})\b|(?:\bpp?\.\s*)(\d{1,4})\b")


@dataclass
class LanternArticle:
    volume: str
    magazine: str
    volume_dates: str
    leaves: list[int]
    printed_pages: list[str]
    text: str
    complete: bool
    first_leaf_url: str
    title_found: bool = True
    notes: list[str] = field(default_factory=list)


def lantern_target(url: str | None) -> tuple[str, int] | None:
    """(volume identifier, leaf number) of a Lantern page link, or None."""
    match = _LINK.match(str(url or "").strip())
    return (match.group(1), int(match.group(2))) if match else None


def cited_pages(raw_ref: str | None) -> tuple[int, int] | None:
    """The cited page range from the reference text (the last range given)."""
    found = None
    for match in _PAGES.finditer(str(raw_ref or "")):
        if match.group(1):
            start, end = int(match.group(1)), int(match.group(2))
            if 0 < start <= end and end - start < 400:
                found = (start, end)
        elif match.group(3):
            found = (int(match.group(3)),) * 2
    return found


def _words(value: str) -> list[str]:
    return re.sub(r"[^\w]+", " ", str(value or "")).casefold().split()


def title_in_text(title: str | None, text: str, *, window: int = 1500) -> bool:
    """At least 80% of the cited title's words (three or more), in order, near
    the start of the page text. OCR breaks display headlines ("The Girl Who
    Changed th s TARS"), so an exact phrase would miss real titles; only the
    few leaves printed with the cited first page are ever compared."""
    wanted = [word for word in _words(title) if len(word) > 1]
    if len(" ".join(wanted)) < 8:
        return False
    head = _words(text[:window])
    position, matched = 0, 0
    for word in wanted:
        try:
            position = head.index(word, position) + 1
            matched += 1
        except ValueError:
            continue
    return matched >= 3 and matched >= 0.8 * len(wanted)


def year_in_dates(year: str | None, dates: str) -> bool:
    """The cited year falls within the volume's date span ("Nov 1930-Apr 1931")."""
    if not re.fullmatch(r"\d{4}", str(year or "")):
        return False
    years = [int(value) for value in re.findall(r"\b(1[89]\d\d|20\d\d)\b", dates or "")]
    return bool(years) and min(years) <= int(year) <= max(years)


def magazine_matches(container: str | None, magazine: str) -> bool:
    wanted, have = _words(container), _words(magazine)
    return bool(wanted and have) and (" ".join(wanted) in " ".join(have) or " ".join(have) in " ".join(wanted))


def _page_text(get, volume: str, leaf: int) -> str:
    response = get(f"{LANTERN}/catalog/{volume}_{leaf:04d}")
    response.raise_for_status()
    match = _OCR.search(response.text)
    return html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))).strip() if match else ""


def _volume_record(get, volume: str, leaf: int) -> tuple[str, str]:
    response = get(f"{LANTERN}/catalog/{volume}_{leaf:04d}.json")
    response.raise_for_status()
    data = response.json()
    attributes = (data.get("data") or data).get("attributes") or {}

    def value(name: str) -> str:
        item = attributes.get(name)
        return str((item.get("attributes") or {}).get("value") or "") if isinstance(item, dict) else str(item or "")
    return value("title"), value("dateStart")


def _page_map(get_archive, volume: str) -> dict[int, str]:
    response = get_archive(f"{ARCHIVE}/download/{volume}/{volume}_page_numbers.json")
    response.raise_for_status()
    pages = response.json().get("pages") or []
    return {int(page["leafNum"]): str(page.get("pageNumber") or "") for page in pages
            if isinstance(page, dict) and isinstance(page.get("leafNum"), int)}


def read_article(url: str, *, title: str | None, raw_ref: str | None, get, get_archive) -> LanternArticle | None:
    """The cited article's OCR text, or None when the link is not a Lantern page
    or the cited title is on none of the candidate pages. `get`/`get_archive`
    are paced GET callables for Lantern and the Internet Archive."""
    target = lantern_target(url)
    if target is None:
        return None
    volume, linked_leaf = target
    magazine, dates = _volume_record(get, volume, linked_leaf)
    pages = cited_pages(raw_ref)
    page_map: dict[int, str] = {}
    try:
        page_map = _page_map(get_archive, volume)
    except Exception:  # noqa: BLE001 - without the map only the linked leaf and its neighbours are tried
        page_map = {}
    candidates: list[int] = []
    if pages and page_map:
        candidates = [leaf for leaf, printed in sorted(page_map.items()) if printed == str(pages[0])]
    candidates = [linked_leaf] + [leaf for leaf in candidates if leaf != linked_leaf]
    if not page_map:
        candidates += [linked_leaf + offset for offset in (1, 2, -1)]
    first_leaf, first_text = None, ""
    for leaf in candidates[:MAX_CANDIDATE_LEAVES]:
        if leaf < 1:
            continue
        text = _page_text(get, volume, leaf)
        if title_in_text(title, text):
            first_leaf, first_text = leaf, text
            break
    if first_leaf is None:
        return None
    start, end = pages if pages else (None, None)
    leaves, notes, complete = [first_leaf], [], True
    if start is not None and end is not None and end > start:
        if end - start + 1 > MAX_CONTIGUOUS_PAGES:
            wanted = [start + 1, end]
            complete = False
            notes.append("non_contiguous_page_range")
        else:
            wanted = list(range(start + 1, end + 1))
        for page in wanted:
            leaf = _leaf_for_page(page_map, first_leaf, page, start)
            if leaf is None:
                complete = False
                notes.append("cited_page_not_mapped")
                continue
            leaves.append(leaf)
    elif start is None:
        complete = False
        notes.append("no_cited_pages")
    texts = [first_text] + [_page_text(get, volume, leaf) for leaf in leaves[1:]]
    if any(not text for text in texts):
        complete = False
        notes.append("page_text_missing")
    return LanternArticle(
        volume=volume, magazine=magazine, volume_dates=dates, leaves=leaves,
        printed_pages=[page_map.get(leaf, "") for leaf in leaves], text="\n\n".join(texts),
        complete=complete, first_leaf_url=f"{LANTERN}/catalog/{volume}_{first_leaf:04d}", notes=notes)


def _leaf_for_page(page_map: dict[int, str], first_leaf: int, page: int, start: int) -> int | None:
    """The leaf printed with `page` in the same issue as the article's first leaf."""
    if page_map:
        # The same issue: the nearest leaf after the first one carrying that number.
        for leaf in sorted(leaf for leaf in page_map if leaf > first_leaf):
            if page_map[leaf] == str(page):
                return leaf if leaf - first_leaf <= (page - start) + 12 else None
        return None
    return first_leaf + (page - start)
