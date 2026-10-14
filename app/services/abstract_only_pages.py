"""Pages whose usable content is an abstract, not a text of the work.

Two owner decisions (2026-10-13) read an acquired web page as an abstract
rather than as limited text of the cited work:

* ``landing-abstract-back-matter-v1``: an access-restricted publisher landing
  page whose readable text is the abstract followed only by back matter (notes,
  references, acknowledgements, appendices, access prompts, "similar content",
  publication metadata) and no body section. The notes and references are not
  the work's argument, so only the abstract is kept, with the page as its
  provenance.
* ``doi-abstract-over-short-page-v1``: when an academic adapter confirmed the
  cited DOI and holds that DOI's abstract, a short acquired page that cannot be
  the whole work (fewer words than the cited kind's minimum, and below the
  stated-page minimum) yields to the abstract. Full text, a complete page and
  substantial limited text (a several-page preview) are unchanged.

Both tests are deterministic and deliberately conservative: a page that shows
any body section, a substantial paragraph outside its abstract and back matter,
or no access prompt is left as it was.
"""
from __future__ import annotations

import hashlib
import re

from app.services.retrieval.base import RepresentationKind, RetrievalResult

LANDING_POLICY = "landing-abstract-back-matter-v1"
DOI_ABSTRACT_POLICY = "doi-abstract-over-short-page-v1"

_ABSTRACT_HEADINGS = frozenset({"abstract", "summary", "abstract and keywords"})

_BACK_MATTER_HEADINGS = frozenset({
    "notes", "note", "footnotes", "endnotes", "end notes",
    "references", "reference", "bibliography", "works cited", "literature cited", "cited literature",
    "acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment",
    "funding", "funding information", "author information", "authors and affiliations",
    "affiliations", "corresponding author", "ethics declarations", "conflict of interest",
    "conflicts of interest", "competing interests", "additional information",
    "rights and permissions", "about this article", "cite this article", "copyright information",
    "publisher's note", "publishers note", "keywords", "key words", "jel classification",
    "jel codes", "data availability", "data availability statement", "change history",
    "similar content being viewed by others", "explore related subjects", "related articles",
    "recommended articles", "cited by", "citations", "metrics", "article metrics",
    "access this article", "access options", "buy article pdf", "purchase", "subscribe",
    "log in via an institution", "institutional subscriptions", "get access",
    "article information", "article details", "publication history", "share this article",
    "appendix", "appendices", "online appendix", "supplementary information",
    "supplementary material", "electronic supplementary material",
})
_APPENDIX_HEADING = re.compile(r"^appendix\s+[a-z0-9]{1,3}\b")
# Back matter made of numbered or short entries (notes, references, appendices):
# inside these a numbered or short line is an entry, not a heading.
_LIST_SECTIONS = frozenset({
    "notes", "note", "footnotes", "endnotes", "end notes", "references", "reference", "bibliography",
    "works cited", "literature cited", "cited literature", "appendix", "appendices", "online appendix",
})

_BODY_HEADINGS = frozenset({
    "introduction", "background", "method", "methods", "methodology", "materials and methods",
    "data", "data and methods", "data and methodology", "results", "discussion",
    "results and discussion", "conclusion", "conclusions", "concluding remarks",
    "literature review", "theory", "theoretical framework", "the model", "model",
    "empirical strategy", "analysis", "findings", "case study", "overview", "preface",
    "related literature", "related work", "empirical analysis", "estimation", "robustness",
})
# "1 introduction", "2.3 data and sources", "iv. the model": a number, then words
# only ("1 unit = 1 article" in an access box is not a section).
_NUMBERED_HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|[ivx]{1,5}\.)\s+[^\W\d_][\w'’-]*(?:\s+[^\W\d_][\w'’,-]*){0,7}$")
_ACCESS_PROMPT = re.compile(
    r"\b(?:access this article|buy article pdf|purchase (?:this )?(?:article|pdf|access)|"
    r"rent this article|log in via an institution|institutional subscriptions|subscribe (?:to|now)|"
    r"get access|instant access to the full article|sign in to (?:read|access|view)|"
    r"access through your institution|check access|price includes vat)\b",
    re.IGNORECASE,
)

_HEADING_MAX_WORDS = 8
_ABSTRACT_MIN_WORDS, _ABSTRACT_MAX_WORDS = 30, 600
_OTHER_MAX_WORDS = 250
_OTHER_LINE_MAX_WORDS = 40
# Other back matter (acknowledgements, access box, article information) is short.
_SHORT_BACK_MAX_WORDS = 400
_SHORT_BACK_LINE_MAX_WORDS = 80


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _heading_key(line: str) -> str | None:
    """The normalized heading a short line could be, else None."""
    if _words(line) > _HEADING_MAX_WORDS or re.search(r"[.;,!?]\s*$", line):
        return None
    key = line.replace("’", "'").rstrip(": ")
    return re.sub(r"\s+", " ", key).strip().casefold()


def landing_abstract_and_back_matter(text: str) -> dict | None:
    """The abstract of a landing page that carries only an abstract and back matter.

    Returns ``None`` (page unchanged) unless all of these hold: an access prompt
    appears; an "Abstract" heading opens 30-600 words of abstract; at least one
    back-matter heading follows; no body heading (Introduction, a numbered
    section, Methods ...) appears outside the back matter; and the text outside
    the abstract and back matter is short, with no paragraph of 40+ words.
    """
    if not text or not _ACCESS_PROMPT.search(text):
        return None
    state = "front"
    in_list = False
    abstract: list[str] = []
    back_sections: list[str] = []
    other_words = back_words = short_back_words = 0
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        key = _heading_key(line)
        if key is not None:
            if key in _ABSTRACT_HEADINGS and not abstract:
                state = "abstract"
                continue
            if key in _BACK_MATTER_HEADINGS or _APPENDIX_HEADING.match(key):
                state = "back"
                in_list = key in _LIST_SECTIONS or bool(_APPENDIX_HEADING.match(key))
                if key not in back_sections:
                    back_sections.append(key)
                continue
            if key in _BODY_HEADINGS:
                return None
            if not (state == "back" and in_list) and _NUMBERED_HEADING.match(key):
                return None
            if (state == "abstract" and abstract) or (state == "back" and not in_list):
                # An unrecognised heading after the abstract or inside short
                # back matter opens a section of unknown kind; its text counts
                # as outside the abstract and back matter.
                state = "other"
        words = _words(line)
        if state == "abstract":
            abstract.append(line)
        elif state == "back":
            back_words += words
            if not in_list:
                short_back_words += words
                if words >= _SHORT_BACK_LINE_MAX_WORDS or short_back_words > _SHORT_BACK_MAX_WORDS:
                    return None
        else:
            if words >= _OTHER_LINE_MAX_WORDS:
                return None
            other_words += words
            if other_words > _OTHER_MAX_WORDS:
                return None
    abstract_text = "\n\n".join(abstract)
    abstract_words = _words(abstract_text)
    if not back_sections or not (_ABSTRACT_MIN_WORDS <= abstract_words <= _ABSTRACT_MAX_WORDS):
        return None
    return {
        "policy": LANDING_POLICY,
        "abstract": abstract_text,
        "abstract_words": abstract_words,
        "abstract_sha256": hashlib.sha256(abstract_text.encode("utf-8")).hexdigest(),
        "back_matter_sections": back_sections[:24],
        "back_matter_words": back_words,
        "other_words": other_words,
    }


# A note or reference entry: numbered as a note ("3. ", "[3] "), or opening
# with a name and giving a year soon after ("Kee, Hiau Looi & Nicita, A., 2009.").
# A section number run into its paragraph ("1 The feminist imperative") is not.
_NOTE_LINE = re.compile(r"^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s+\S")
_NAMED_ENTRY_LINE = re.compile(
    r"^\s*[^\W\d_][\w'’-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+)[^\n]{0,200}?\b(?:1[89]|20)\d{2}\b")
_INDEX_MARK = re.compile(r"google scholar|crossref|pubmed|web of science|doi\.org", re.IGNORECASE)
_LISTING_HEADINGS = frozenset({
    "references listed on ideas", "most related items", "more about this item", "statistics",
    "corrections", "download full text from publisher", "citations", "cited by", "related works",
    "similar items", "lists of references", "referenced works",
})


def body_prose_words(text: str) -> int:
    """Words of a page's own prose: lines outside back matter and listings that
    are not note, reference or listing entries. A landing page made of an
    abstract, numbered notes and "Cited by" lists has few, a whole article many
    (owner decision 2026-10-13: such pages are not complete texts)."""
    words = 0
    in_back = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        key = _heading_key(line)
        if key is not None and (key in _LIST_SECTIONS or key in _LISTING_HEADINGS
                                or _APPENDIX_HEADING.match(key)):
            in_back = True
            continue
        if key is not None:
            # Any other heading (Keywords, Introduction, "2 Data") ends a list.
            in_back = False
            continue
        if in_back or _NOTE_LINE.match(line) or _NAMED_ENTRY_LINE.match(line) or _INDEX_MARK.search(line):
            continue
        words += _words(line)
    return words


NOTES_PAGE_POLICY = "notes-and-listings-not-a-whole-work-v1"
_PROSE_SHARE = 0.4


def mostly_notes_or_listings(text: str) -> bool:
    """A page whose own prose is short and a small share of its text: an
    abstract or landing page carrying the work's notes, references or a
    "Cited by" list (2026-10-13). Never a whole work, whatever its length."""
    from app.services.other_language import dominant_script
    from app.services.web_completeness import STATED_PAGE_MIN_WORDS
    if dominant_script(text or "") in {"han", "kana", "hangul"}:
        # Words are counted by spaces, which these scripts do not use; a page
        # of continuous Han prose would read as almost no words (2026-10-13).
        return False
    total = _words(text)
    if not total:
        return False
    prose = body_prose_words(text)
    return prose < STATED_PAGE_MIN_WORDS and prose < _PROSE_SHARE * total


def landing_abstract_provenance(observation: dict, *, provider: str, content_sha256: str | None,
                                doi: str | None) -> dict:
    """Bounded provenance for an abstract kept from a landing page.

    Content-bound only: no address or address hash, so a page found through
    transient search leaves nothing of its discovery behind.
    """
    return {
        "policy": LANDING_POLICY,
        "basis": "publisher_landing_page",
        "provider": str(provider or "")[:80],
        "doi": doi or None,
        "page_content_sha256": content_sha256,
        "abstract_sha256": observation["abstract_sha256"],
        "abstract_words": observation["abstract_words"],
        "back_matter_sections": list(observation["back_matter_sections"]),
    }


def short_page_words(result: RetrievalResult | None, cited_kind: str | None) -> int | None:
    """Words of an acquired web page too short to be the whole cited work.

    ``None`` when the result holds no page, the page is complete, it is not a
    web page (a PDF preview keeps its own rule), or it is long enough to be
    the work: at least the stated-page minimum or the cited kind's minimum,
    whichever is lower.
    """
    from app.services.web_completeness import KIND_WORD_RANGES, STATED_PAGE_MIN_WORDS
    representation = getattr(result, "representation", None)
    if representation is None or not representation.content:
        return None
    if str(representation.completeness or "").casefold() == "complete":
        return None
    if RepresentationKind.HTML not in (representation.kind, representation.original_kind):
        return None
    text = representation.content.decode("utf-8", "replace")
    # Notes and reference lists are not the work: only its prose counts.
    words = body_prose_words(text) if mostly_notes_or_listings(text) else _words(text)
    low = (KIND_WORD_RANGES.get(str(cited_kind or "")) or (STATED_PAGE_MIN_WORDS, None))[0]
    return words if words < min(STATED_PAGE_MIN_WORDS, low) else None


# Prose a third-party page may hold beyond the publisher's abstract and still
# yield to it: its abstract restated with metadata, keywords or listings.
PUBLISHER_ABSTRACT_EXTRA_PROSE_WORDS = 150


def publisher_abstract_covers_page(result: RetrievalResult | None, landing: RetrievalResult | None) -> int | None:
    """The prose words of an incomplete web page that adds little to the
    abstract the publisher's own page gave (owner decision 2026-10-14), or
    None when the page stands: no publisher abstract, a complete page, a PDF,
    or prose beyond the abstract plus PUBLISHER_ABSTRACT_EXTRA_PROSE_WORDS."""
    if landing is None or not landing.abstract:
        return None
    if ((landing.metadata or {}).get("abstract_provenance") or {}).get("basis") != "publisher_landing_page":
        return None
    representation = getattr(result, "representation", None)
    if representation is None or not representation.content:
        return None
    if str(representation.completeness or "").casefold() == "complete":
        return None
    if RepresentationKind.HTML not in (representation.kind, representation.original_kind):
        return None
    prose = body_prose_words(representation.content.decode("utf-8", "replace"))
    return prose if prose <= _words(landing.abstract) + PUBLISHER_ABSTRACT_EXTRA_PROSE_WORDS else None


def _doi_key(value) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(value or "").strip(),
                  flags=re.IGNORECASE).casefold()


def doi_confirmed_abstract(holder: RetrievalResult | None, doi: str | None) -> dict | None:
    """The provider of a merged abstract whose record confirmed the cited DOI.

    The abstract must come from an academic adapter that the canonical-work
    graph accepted at high confidence with the cited DOI.
    """
    if holder is None or not holder.abstract or not _doi_key(doi):
        return None
    graph = (holder.metadata or {}).get("canonical_work") or {}
    provider = graph.get("abstract_provider")
    confirmed = sorted({
        str(item.get("provider")) for item in graph.get("identity_evidence") or []
        if isinstance(item, dict) and item.get("confidence") == "high"
        and _doi_key(item.get("doi")) == _doi_key(doi)
    })
    if not provider or provider not in confirmed:
        return None
    return {"provider": provider, "doi": doi, "doi_confirmed_by": confirmed}


# ---- linked Chinese-language abstract pages (cjk-abstract-page-v1) ----------
#
# Owner decision 2026-10-14 (the Regulation paper's references 8 and 13): a
# page the student linked for a Chinese-language work, matched to the work but
# holding an abstract rather than the work (a CNKI abstract page, a database
# record page), gives its abstract as abstract-only evidence. The abstract is
# found deterministically: text after an abstract label (摘要, 内容摘要,
# 内容提要) up to the next field label, else the one page element whose class or
# id names a summary or abstract. Nothing is kept below MIN_CJK_ABSTRACT_CHARS
# Han characters.

CJK_ABSTRACT_POLICY = "cjk-abstract-page-v1"
LINKED_ABSTRACT_BASIS = "linked_abstract_page"
MIN_CJK_ABSTRACT_CHARS = 40
MAX_CJK_ABSTRACT_CHARS = 3_000
_HAN = re.compile(r"[一-鿿]")
_CJK_LABELLED = re.compile(
    r"(?:内容摘要|内容提要|摘\s*要)\s*[:：]\s*(?P<text>.+?)\s*"
    r"(?=关键词|关键字|关\s*键\s*词|基金|分类号|中图分类|DOI|专辑|专题|文献来源|作者单位|$)", re.DOTALL)
_SUMMARY_ELEMENT = re.compile(
    r"<(?P<tag>div|p|section|span|dd)\b[^>]*\b(?:class|id)=[\"'][^\"']*(?:summary|abstract)[^\"']*[\"'][^>]*>"
    r"(?P<inner>.*?)</(?P=tag)>", re.IGNORECASE | re.DOTALL)


def _visible(html: str) -> str:
    from html import unescape
    text = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html or "")
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())


def cjk_page_abstract(html: str) -> dict | None:
    """The abstract of a Chinese-language abstract page, or None."""
    candidates = [match.group("text") for match in _CJK_LABELLED.finditer(_visible(html))]
    if not any(len(_HAN.findall(value)) >= MIN_CJK_ABSTRACT_CHARS for value in candidates):
        candidates = [_visible(match.group("inner")) for match in _SUMMARY_ELEMENT.finditer(html or "")]
    candidates = [value.strip() for value in candidates if len(_HAN.findall(value)) >= MIN_CJK_ABSTRACT_CHARS]
    if not candidates:
        return None
    abstract = max(candidates, key=lambda value: len(_HAN.findall(value)))[:MAX_CJK_ABSTRACT_CHARS]
    return {"abstract": abstract, "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
            "abstract_characters": len(abstract)}


def is_mainly_abstract(page_text: str, abstract: str) -> bool:
    """The page's readable text is little more than its abstract: at most three
    times its Han characters plus 400 (navigation, title, metadata). Empty page
    text (nothing readable) counts as mainly abstract."""
    page = len(_HAN.findall(page_text or ""))
    return page <= 3 * len(_HAN.findall(abstract or "")) + 400


def cjk_abstract_provenance(observation: dict, *, provider: str, content_sha256: str | None,
                            doi: str | None) -> dict:
    """Bounded, content-bound provenance; no address is kept."""
    return {"policy": CJK_ABSTRACT_POLICY, "basis": LINKED_ABSTRACT_BASIS, "provider": str(provider or "")[:80],
            "doi": doi or None, "page_content_sha256": content_sha256,
            "abstract_sha256": observation["abstract_sha256"],
            "abstract_characters": observation["abstract_characters"]}
