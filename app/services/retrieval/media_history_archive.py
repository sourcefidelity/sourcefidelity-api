"""Find a cited magazine article in the Media History Digital Library.

The library's Lantern search reads the OCR text of digitised film, radio and
fan magazines (Photoplay, Screenland, Modern Screen, ...). Lantern's public
JSON interface (https://lantern.mediahist.org/api) needs no key; the library
asks for no more than ten requests a second.

A page is accepted only when all of these hold:

* it belongs to the cited magazine (the whole name, not a word of it: there
  is no student link to anchor the search), in a volume whose dates cover the
  cited year (Lantern's year filter does not filter, so the volume's own
  range is read, as `lantern-pages-v1` reads it);
* its text contains a phrase of the cited title; and
* the cited author's surname or first page number stands beside that phrase,
  unless the whole title, of four or more distinctive words, matched.

Coverage is partial, so an empty answer is never evidence that the article
does not exist (`media-history-archive-v1`). The page found is then read
like a student's own Lantern link (`lantern-pages-v1`, the resolver's
`_try_lantern`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import time
import unicodedata

import httpx

from app.services.lookalike_letters import fold_lookalikes
from app.services.processing_metrics import record_provider_request
from app.services.retrieval.lantern import year_in_dates

logger = logging.getLogger(__name__)

VERSION = "media-history-archive-v1"
PROVIDER = "media_history_archive"
_SEARCH = "https://lantern.mediahist.org/catalog.json"
_PUBLICATIONS = "https://mediahist.org/data/list-publications.php"
_PUBLICATIONS_TTL_SECONDS = 24 * 3600
_publications_cache: tuple[float, tuple[str, ...]] | None = None
_PAGE = "https://lantern.mediahist.org/catalog/{page_id}/raw.json"
_PACE_SECONDS = 0.15
_MAX_REQUESTS = 10
_RESULT_PAGES = 2
_WINDOW = 200
from app.services.retrieval.lantern import HEADLINE_STOPWORDS as _STOPWORDS  # noqa: E402


@dataclass(frozen=True)
class ArchivePageMatch:
    page_id: str
    publication: str
    volume_dates: str
    read_url: str
    excerpt: str
    phrase: str
    full_title: bool
    author_beside: bool
    first_page_beside: bool


@dataclass(frozen=True)
class ArchiveSearch:
    # results | no_results_low_coverage | declined (magazine not held) | timeout
    # | rate_limited | operational_failure | response_invalid
    outcome: str
    queries: tuple[str, ...] = ()
    match: ArchivePageMatch | None = None
    requests: int = 0
    attempts: tuple[dict, ...] = field(default_factory=tuple)


def page_link(page_id: str) -> str:
    """The Lantern page link a student would cite, read by `lantern-pages-v1`."""
    return f"https://lantern.mediahist.org/catalog/{page_id}"


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", fold_lookalikes(text)).casefold()
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = re.sub(r"['’]", "", folded)
    return re.findall(r"[a-z0-9]+", folded)


def _distinctive(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]


def title_phrases(title: str) -> list[list[str]]:
    """The cited title, then shorter runs of it that still carry two distinctive words."""
    words = _tokens(title)
    phrases = [words]
    if len(words) >= 3:
        phrases += [words[:-1], words[1:]]
    for size in range(len(words) - 2, 1, -1):
        for start in range(len(words) - size + 1):
            phrases.append(words[start:start + size])
    seen, kept = set(), []
    for phrase in phrases:
        key = tuple(phrase)
        if key in seen or len(_distinctive(phrase)) < (1 if len(words) <= 2 else 2):
            continue
        seen.add(key)
        kept.append(phrase)
    return kept[:5]


def _publication_name(label: str) -> str:
    """"Photoplay (Jan-Jun 1947)" -> "Photoplay"."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", label or "").strip()


def _same_publication(cited: str, found: str) -> bool:
    cited_tokens = [t for t in _tokens(cited) if t not in {"magazine", "the"}]
    return bool(cited_tokens) and cited_tokens == [t for t in _tokens(found) if t not in {"magazine", "the"}]


def held_publications(get) -> tuple[str, ...]:
    """The library's publication titles, read once a day."""
    global _publications_cache
    now = time.monotonic()
    if _publications_cache and now - _publications_cache[0] < _PUBLICATIONS_TTL_SECONDS:
        return _publications_cache[1]
    payload = get(_PUBLICATIONS)
    if not isinstance(payload, list):
        raise ValueError("invalid response")
    names = tuple(str(name).strip() for name in payload if isinstance(name, str) and name.strip())
    _publications_cache = (now, names)
    return names


def _phrase_at(body_tokens: list[str], phrase: list[str]) -> int | None:
    for start in range(len(body_tokens) - len(phrase) + 1):
        if body_tokens[start:start + len(phrase)] == phrase:
            return start
    return None


def find_periodical_article(title: str, container: str, year: str, author_surname: str = "",
                            first_page: str = "", *, timeout: float = 20.0) -> ArchiveSearch:
    """Search Lantern for the cited article; see the module docstring for acceptance."""
    cited_year = int(year[:4]) if re.match(r"^\d{4}", year or "") else None
    words = _tokens(title)
    if not words or not container.strip() or cited_year is None:
        return ArchiveSearch("response_invalid")
    surname = _tokens(author_surname)[-1:] if author_surname else []
    page = re.sub(r"\D", "", first_page or "")
    requests, queries, attempts = 0, [], []

    def get(url: str, params: dict | None = None) -> dict:
        nonlocal requests
        requests += 1
        record_provider_request(PROVIDER)
        time.sleep(_PACE_SECONDS)
        response = httpx.get(url, params=params, timeout=timeout, follow_redirects=True,
                             headers={"User-Agent": "SourceFidelity/1.0 (academic source checking)"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, (dict, list)):
            raise ValueError("invalid response")
        return payload

    bodies: dict[str, tuple[str, str]] = {}  # page id -> (OCR text, reader link)
    try:
        publications = [name for name in held_publications(get) if _same_publication(container, name)][:2]
        if not publications:
            return ArchiveSearch("declined", (), None, requests, ())
        for phrase in title_phrases(title):
            for publication in publications:
                if requests >= _MAX_REQUESTS:
                    return ArchiveSearch("no_results_low_coverage", tuple(queries), None, requests, tuple(attempts))
                query = '"' + " ".join(phrase) + '"'
                queries.append(f"{query} in {publication}")
                hits = []
                for result_page in range(1, _RESULT_PAGES + 1):
                    payload = get(_SEARCH, {"q": query, "f[title][]": publication, "page": result_page})
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if not isinstance(data, list):
                        raise ValueError("invalid response")
                    hits += [item for item in data if isinstance(item, dict)]
                    pages = ((payload.get("meta") or {}).get("pages") or {})
                    if not pages.get("next_page") or requests >= _MAX_REQUESTS:
                        break
                attempts.append({"query": f"{query} in {publication}",
                                 "outcome": "results" if hits else "no_results", "result_count": len(hits)})
                for hit in hits:
                    page_id = str(hit.get("id") or "")
                    label = str((hit.get("attributes") or {}).get("title") or "")
                    name = _publication_name(label)
                    if (not page_id or not _same_publication(container, name)
                            or not year_in_dates(str(cited_year), label)):
                        continue
                    if page_id not in bodies:
                        if requests >= _MAX_REQUESTS:
                            break
                        record = get(_PAGE.format(page_id=page_id))
                        text = record.get("body") if isinstance(record, dict) else None
                        text = text[0] if isinstance(text, list) and text else text
                        bodies[page_id] = (text if isinstance(text, str) else "",
                                           str(record.get("read") or "") if isinstance(record, dict) else "")
                    body, read_url = bodies[page_id]
                    body_tokens = _tokens(body)
                    at = _phrase_at(body_tokens, phrase)
                    if at is None:
                        continue
                    window = body_tokens[max(0, at - 25):at + len(phrase) + 25]
                    author_beside = bool(surname) and surname[0] in window
                    page_beside = bool(page) and page in window
                    full = phrase == words
                    if not (author_beside or page_beside or (full and len(_distinctive(words)) >= 4)):
                        continue
                    flat = " ".join(body.split())
                    first = re.search(r"\s+".join(map(re.escape, phrase[:2])), flat, re.I)
                    offset = first.start() if first else 0
                    return ArchiveSearch("results", tuple(queries), ArchivePageMatch(
                        page_id=page_id[:200], publication=name[:200], volume_dates=label[:200],
                        read_url=read_url[:500],
                        excerpt=flat[max(0, offset - 40):offset + _WINDOW][:300],
                        phrase=" ".join(phrase), full_title=full,
                        author_beside=author_beside, first_page_beside=page_beside,
                    ), requests, tuple(attempts))
    except httpx.TimeoutException:
        return ArchiveSearch("timeout", tuple(queries), None, requests, tuple(attempts))
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        return ArchiveSearch("rate_limited" if status == 429 else "operational_failure",
                             tuple(queries), None, requests, tuple(attempts))
    except (ValueError, httpx.HTTPError) as exc:
        logger.debug("Media history archive search failed (type=%s)", type(exc).__name__)
        return ArchiveSearch("response_invalid" if isinstance(exc, ValueError) else "operational_failure",
                             tuple(queries), None, requests, tuple(attempts))
    return ArchiveSearch("no_results_low_coverage", tuple(queries), None, requests, tuple(attempts))
