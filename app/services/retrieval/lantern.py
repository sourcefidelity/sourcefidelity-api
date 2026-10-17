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
4. reads the OCR text of the cited pages from Lantern, one page per form
   feed, each labelled with its printed number where the page shows it
   (`page_labels`, owner decision 2026-10-14).

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
# A single page after the magazine, before the link or the end: "Movie Classic, 41."
_BARE_PAGE = re.compile(r",\s*(\d{1,4})\.?\s*(?:https?://\S*\s*)?$")


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


def page_labels(article: LanternArticle) -> list[str | None]:
    """Each page's printed number, from the archive's page-number file, kept
    only where the page itself prints it on a line of its own at its top or
    bottom (owner decision 2026-10-14); pages are separated by form feeds."""
    labels = []
    for printed, text in zip(article.printed_pages, article.text.split("\f")):
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        edges = lines[:3] + lines[-3:]
        labels.append(printed if printed and printed.isdigit() and printed in edges else None)
    return labels


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
    bare = None if found else _BARE_PAGE.search(str(raw_ref or "").strip())
    if bare and not re.fullmatch(r"1[6-9]\d\d|20\d\d", bare.group(1)):
        found = (int(bare.group(1)),) * 2
    return found


# The page list that ends a magazine reference: "Fanmag, 50-51, 90-92." or
# "Fanmag, 49(10), 30-31, 74-75." (an article continued at the back).
_PAGE_LIST = re.compile(
    r",\s*(?:pp?\.\s*)?(\d{1,4}(?:\s*[-–—]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)*)"
    r"\.?\s*(?:https?://\S*\s*)?$")


def cited_page_ranges(raw_ref: str | None) -> list[tuple[int, int]]:
    """Every cited page range of a magazine article, in order (2026-10-16).

    The article starts at the first; later ranges are its continuation at the
    back of the issue. A lone year ("Photoplay, 1932.") is not a page.
    """
    match = _PAGE_LIST.search(str(raw_ref or "").strip())
    if not match:
        return []
    ranges = []
    for part in match.group(1).split(","):
        numbers = [int(value) for value in re.findall(r"\d+", part)]
        start, end = numbers[0], numbers[-1]
        if len(numbers) == 1 and re.fullmatch(r"1[6-9]\d\d|20\d\d", str(start)):
            return []
        if not 0 < start <= end or end - start >= 400:
            return []
        ranges.append((start, end))
    return ranges


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
    # "1946b": an APA letter for two works by one author in a year (2026-10-16).
    if not re.fullmatch(r"\d{4}[a-z]?", str(year or "")):
        return False
    years = [int(value) for value in re.findall(r"\b(1[89]\d\d|20\d\d)\b", dates or "")]
    return bool(years) and min(years) <= int(str(year)[:4]) <= max(years)


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


def read_article(url: str, *, title: str | None, raw_ref: str | None, get, get_archive,
                 searched_author: str | None = None) -> LanternArticle | None:
    """The cited article's OCR text, or None when the link is not a Lantern page
    or the cited title is on none of the candidate pages. `get`/`get_archive`
    are paced GET callables for Lantern and the Internet Archive.
    `searched_author` marks a page the archive search found and bound
    (`searched_article_on_spread`); a student's link keeps `title_on_spread`."""
    target = lantern_target(url)
    if target is None:
        return None
    volume, linked_leaf = target
    magazine, dates = _volume_record(get, volume, linked_leaf)
    ranges = cited_page_ranges(raw_ref)
    pages = ranges[0] if ranges else cited_pages(raw_ref)
    later_ranges = ranges[1:]
    page_map: dict[int, str] = {}
    try:
        page_map = _page_map(get_archive, volume)
    except Exception:  # noqa: BLE001 - without the map only the linked leaf and its neighbours are tried
        page_map = {}
    cache: dict[int, str] = {}

    def text_of(leaf: int) -> str:
        if leaf not in cache:
            cache[leaf] = _page_text(get, volume, leaf) if leaf >= 1 else ""
        return cache[leaf]

    start, end = pages if pages else (None, None)
    first_leaf = _first_leaf_on_linked_spread(title, linked_leaf, start, page_map, text_of,
                                              searched_author=searched_author)
    if first_leaf is None and searched_author is not None and later_ranges and page_map:
        # The first page carries no OCR text; the continuation page names the
        # article ("Ask the Boss (Continued from page 51)").
        if not text_of(linked_leaf).strip() or text_of(linked_leaf).strip().casefold() == _NO_OCR:
            continuation = _leaf_for_page(page_map, linked_leaf, later_ranges[0][0], start or 0)
            if continuation is not None and title_in_order(title, text_of(continuation)[:1500]):
                first_leaf = linked_leaf
    if first_leaf is None:
        # Other issues bound in the volume: the strict single-page match.
        # Nearest after the linked leaf first: a volume binds a dozen issues,
        # and the link (or a contents page the search found) sits in or just
        # before the cited one (2026-10-16).
        candidates = sorted((leaf for leaf, printed in page_map.items()
                             if start is not None and printed == str(start)
                             and leaf not in {linked_leaf, linked_leaf - 1}),
                            key=lambda leaf: (leaf < linked_leaf, abs(leaf - linked_leaf)))
        if not page_map:
            candidates = [linked_leaf + 1, linked_leaf + 2]
        first_leaf = next((leaf for leaf in candidates[:MAX_CANDIDATE_LEAVES]
                           if leaf >= 1 and title_in_text(title, text_of(leaf))), None)
    if first_leaf is None:
        return None
    first_text = text_of(first_leaf)
    leaves, notes, complete = [first_leaf], [], True
    # No cited pages: only the page carrying the title, not known complete (the
    # next page of Hampton's spread is another article).
    if start is not None and end is not None and (end > start or later_ranges):
        if end - start + 1 > MAX_CONTIGUOUS_PAGES:
            wanted = [start + 1, end]
            complete = False
            notes.append("non_contiguous_page_range")
        else:
            wanted = list(range(start + 1, end + 1))
        for later_start, later_end in later_ranges:
            if later_end - later_start + 1 > MAX_CONTIGUOUS_PAGES:
                wanted += [later_start, later_end]
                complete = False
                notes.append("non_contiguous_page_range")
            else:
                wanted += list(range(later_start, later_end + 1))
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
    texts = [first_text] + [text_of(leaf) for leaf in leaves[1:]]
    if any(not text for text in texts):
        complete = False
        notes.append("page_text_missing")
    return LanternArticle(
        volume=volume, magazine=magazine, volume_dates=dates, leaves=leaves,
        printed_pages=[page_map.get(leaf, "") for leaf in leaves], text="\f".join(texts),
        complete=complete, first_leaf_url=f"{LANTERN}/catalog/{volume}_{first_leaf:04d}", notes=notes)


def article_first_leaf(volume: str, found_leaf: int, start_page: int | None, get_archive) -> int:
    """The leaf to read a searched-for article from (media-history-archive-v1).

    A search finds the article on its own page or on the issue's contents
    page. The first leaf at or after the found one printed with the cited
    first page, within one issue's span, is read as a student's link to that
    page would be; without a page map or a cited page, the found leaf.
    """
    if start_page is None:
        return found_leaf
    try:
        page_map = _page_map(get_archive, volume)
    except Exception:  # noqa: BLE001 - without the map the found leaf is read
        return found_leaf
    following = [leaf for leaf, printed in sorted(page_map.items())
                 if printed == str(start_page) and found_leaf <= leaf <= found_leaf + _ISSUE_SPAN_LEAVES]
    return following[0] if following else found_leaf


_ISSUE_SPAN_LEAVES = 120


# Words a headline match does not count.
HEADLINE_STOPWORDS = frozenset({"a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "at", "by",
                                "is", "it", "as", "with", "from", "has", "her", "his", "how", "why"})
_NO_OCR = "no ocr text available"


def title_in_order(title: str | None, text: str, *, share: float = 0.66) -> bool:
    """The title's distinctive words, in order and close together, the way a
    headline reads (media-history-archive-v1). Measured 2026-10-16 over 26
    Lantern articles: about 1 wrong page in 650 at 75-80%, against 100 for
    `title_on_spread`, which counts words anywhere on two pages."""
    from app.services.lookalike_letters import fold_lookalikes
    wanted = [w for w in _words(fold_lookalikes(title or "")) if len(w) >= 3 and w not in HEADLINE_STOPWORDS]
    if len(wanted) < 2:
        return False
    need = max(2, -(-int(share * 100) * len(wanted) // 100))
    tokens = _words(text)
    # Two words must stand together ("exciting week for a woman" is no headline).
    span = 3 if len(wanted) == 2 else 2 * len(wanted) + 4
    for start, token in enumerate(tokens):
        if not _same_scanned_word(wanted[0], token) and not any(_same_scanned_word(w, token) for w in wanted):
            continue
        window, position, matched = tokens[start:start + span], 0, 0
        for word in wanted:
            for index in range(position, len(window)):
                if _same_scanned_word(word, window[index]):
                    matched, position = matched + 1, index + 1
                    break
        if matched >= need:
            return True
    return False


def _same_scanned_word(word: str, token: str) -> bool:
    if word == token or (len(word) >= 4 and word in token and len(token) <= len(word) + 8):
        return True
    if len(word) >= 5 and len(token) >= 3 and abs(len(token) - len(word)) <= 3:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, word, token).ratio() >= 0.75
    return False


def searched_article_on_spread(title: str | None, author_surname: str | None, left: str, right: str) -> bool:
    """A searched article's first page (media-history-archive-v1, owner request
    2026-10-16). The search already bound title, author and page on the
    issue's contents line or the page itself; the spread must carry the title
    as a headline, or the cited author's byline (a graphic headline OCR cannot
    read: Vidor's "Exciting Woman")."""
    spread = f"{left}\n{right}"
    # Not `title_on_spread`: words anywhere on two pages matched 100 of 650
    # wrong pairings in the 2026-10-16 measurement; this rule matched 52, and
    # only confirms a page the search already bound to the cited issue and page.
    if title_in_order(title, spread):
        return True
    surname = _words(author_surname or "")
    return bool(surname) and len(surname[-1]) >= 3 and surname[-1] in set(_words(spread))


def title_on_spread(title: str | None, left: str, right: str) -> bool:
    """At least 80% of the title's words (three or more) anywhere on a two-page
    spread. Headlines often run across both pages of a magazine spread, and OCR
    reads columns out of order; used only for the leaf the student linked."""
    wanted = [word for word in _words(title) if len(word) > 1]
    if len(" ".join(wanted)) < 8:
        return False
    tokens = set(_words(left)) | set(_words(right))
    matched = sum(1 for word in wanted if _scanned_word_present(word, tokens))
    return matched >= 3 and matched >= 0.8 * len(wanted)


def _scanned_word_present(word: str, tokens: set[str]) -> bool:
    """A title word on a scanned page: exact; fused into a longer word
    ("HappyGive!", four letters or more); or a damaged reading ("ressed" for
    "dressed", "nity" for "dignity", five letters or more, 70% alike)."""
    if word in tokens:
        return True
    if len(word) >= 4 and any(word in token for token in tokens if len(token) > len(word)):
        return True
    if len(word) >= 5:
        from difflib import SequenceMatcher
        return any(SequenceMatcher(None, word, token).ratio() >= 0.7
                   for token in tokens if len(token) >= 3 and abs(len(token) - len(word)) <= 4)
    return False


def _first_leaf_on_linked_spread(title, linked_leaf: int, start: int | None, page_map: dict[int, str],
                                 text_of, *, searched_author: str | None = None) -> int | None:
    """The article's first leaf when the student's link points into it: the
    linked leaf or the one before it (a link to an article's second page),
    printed with the cited first page when both are known, whose two-page
    spread carries the title."""
    for first in (linked_leaf, linked_leaf - 1):
        if first < 1:
            continue
        if start is not None and page_map and page_map.get(first) != str(start):
            continue
        if (searched_article_on_spread(title, searched_author, text_of(first), text_of(first + 1))
                if searched_author is not None else title_on_spread(title, text_of(first), text_of(first + 1))):
            return first
    return None


def _leaf_for_page(page_map: dict[int, str], first_leaf: int, page: int, start: int) -> int | None:
    """The leaf printed with `page` in the same issue as the article's first leaf."""
    if page_map:
        # The same issue: the nearest leaf after the first one carrying that number.
        for leaf in sorted(leaf for leaf in page_map if leaf > first_leaf):
            if page_map[leaf] == str(page):
                return leaf if leaf - first_leaf <= (page - start) + 12 else None
        return None
    return first_leaf + (page - start)
