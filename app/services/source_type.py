"""Source-type detection for routing references to the right resolution path.

Different source types need different retrieval strategies:

  - Academic works (articles, books): → academic databases (OpenAlex, CORE,
    S2, Crossref) + S3 cache. These have DOIs/ISBNs or searchable titles.
  - Traditional media (films, TV, albums, artwork): NOT in academic databases
    in a useful form. Routing them to title search produces false-positive
    keyword matches (e.g. "Rain Man" → papers *about* the film). These should
    skip academic DBs entirely.
  - Websites / digital-native: → live web-fetch path (Phase 3.7). These live
    at a URL, not in an academic index.
  - Physical archives: unverifiable automatically. Skip all retrieval.

This module provides the detection used by both the source resolver (for
routing) and count_missing_identifiers (for the penalty policy).
"""

from dataclasses import dataclass
import json
import re
from urllib.parse import urlparse


SOURCE_KINDS = frozenset(
    {
        "journal_article",
        "book_review",
        "monograph",
        "edited_collection",
        "book_section",
        "conference_paper",
        "report",
        "thesis",
        "webpage",
        "news_article",
        "blog_post",
        "social_media_post",
        "video",
        "podcast_episode",
        "dataset",
        "software",
        "traditional_media",
        "archival_source",
        "unknown",
    }
)

SOURCE_KIND_ALIASES = {
    "article": "journal_article",
    "journalarticle": "journal_article",
    "bookreview": "book_review",
    "book": "monograph",
    "editedbook": "edited_collection",
    "editedcollection": "edited_collection",
    "bookchapter": "book_section",
    "booksection": "book_section",
    "chapter": "book_section",
    "conferencepaper": "conference_paper",
    "proceedingsarticle": "conference_paper",
    "dissertation": "thesis",
    "website": "webpage",
    "webpage": "webpage",
    "news": "news_article",
    "newspaperarticle": "news_article",
    "blog": "blog_post",
    "socialmedia": "social_media_post",
    "podcast": "podcast_episode",
    "audio": "podcast_episode",
    "film": "traditional_media",
}

SOURCE_KIND_TO_DOCUMENT_KIND = {
    "journal_article": "article",
    "conference_paper": "article",
    "book_review": "article",
    "monograph": "book",
    "edited_collection": "book",
    "book_section": "chapter",
}

_CONFIDENCE_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class SourceKindAssessment:
    """Inspectably classified bibliographic work type."""

    kind: str = "unknown"
    confidence: str = "unknown"
    evidence: tuple[str, ...] = ()

    @property
    def is_known(self) -> bool:
        return self.kind != "unknown"


@dataclass(frozen=True)
class SourceKindCompatibility:
    """Relationship between the cited work type and an observed candidate."""

    verdict: str  # compatible | incompatible | unknown
    reason: str


def normalize_source_kind(value: str | None, *, strict: bool = False) -> str:
    """Normalize a bounded source-kind name without guessing unknown values."""
    if not value:
        return "unknown"
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    normalized = SOURCE_KIND_ALIASES.get(normalized.replace("_", ""), normalized)
    if normalized in SOURCE_KINDS:
        return normalized
    if strict:
        supported = ", ".join(sorted(SOURCE_KINDS - {"unknown"}))
        raise ValueError(f"Unsupported source_kind {value!r}; expected one of {supported}")
    return "unknown"


def document_kind_for_source_kind(value: str | None) -> str:
    """Map bibliographic identity to the coarser completeness contract."""
    return SOURCE_KIND_TO_DOCUMENT_KIND.get(normalize_source_kind(value), "unknown")


def _assessment(kind: str, confidence: str, *evidence: str) -> SourceKindAssessment:
    return SourceKindAssessment(kind, confidence, tuple(item for item in evidence if item))


_BOOK_REVIEW_RE = re.compile(
    r"(?:\bbook\s+review\b|"
    r"\[\s*review\s+of\s+(?:the\s+)?book\b|"
    r"\breview\s+of\s+.{3,180},\s+by\s+[A-Z])",
    re.IGNORECASE,
)
_JOURNAL_STRUCTURE_RE = re.compile(
    r"(?:\b(?:journal|quarterly|review)\b.{0,100})?"
    r"\b\d{1,4}\s*\(\s*\d{1,4}\s*\)\s*[,.:]?\s*(?:pp?\.\s*)?\d{1,6}",
    re.IGNORECASE,
)
_JOURNAL_VOLUME_ISSUE_RE = re.compile(
    r",\s*\d{1,4}\s*\(\s*\d{1,4}\s*\)\s*\.?\s*(?=(?:https?://|www\.|$))",
    re.IGNORECASE,
)
_JOURNAL_FRONT_MATTER_RE = re.compile(
    r"(?:\bISSN\s*:|\bjournal\s+homepage\b|\bto\s+cite\s+this\s+article\b|"
    r"\bVolume\s+[A-Z0-9IVXLC]+\s*,?\s*(?:Number|No\.)\s*\d+\s*,?\s*pp?\.)",
    re.IGNORECASE,
)
_BOOK_PUBLISHER_RE = re.compile(
    r"\b(?:university\s+press|press|routledge|sage|springer|wiley|palgrave|"
    r"penguin|knopf|norton|bloomsbury|blackwell|elsevier|harpercollins|"
    r"random\s+house|cambridge|oxford|columbia\s+global\s+reports)\b",
    re.IGNORECASE,
)
_BOOK_EDITION_RE = re.compile(r"\(\s*\d+(?:st|nd|rd|th)\s+ed\.?(?:ition)?\s*\)", re.I)
_BOOK_SECTION_RE = re.compile(
    r"(?:\bIn\s+.{1,160}\((?:Ed|Eds)\.\)|\b(?:edited\s+by|chapter\s+\d+)\b)",
    re.IGNORECASE,
)
_REPORT_RE = re.compile(
    r"(?:\[(?:technical\s+)?report\]|\breport\s+(?:no\.?|number)\s*[A-Z0-9-]+|"
    r"\bworking\s+paper\s+(?:no\.?|number)\s*[A-Z0-9-]+)",
    re.IGNORECASE,
)
_OECD_REPORT_SERIES_RE = re.compile(
    r"\bOECD\b.{0,160}\b(?:economic\s+outlook|survey|statistics|indicators)\b",
    re.IGNORECASE,
)
_THESIS_RE = re.compile(r"\b(?:doctoral\s+dissertation|master'?s\s+thesis|phd\s+thesis)\b", re.I)
_DATASET_RE = re.compile(r"(?:\[(?:data\s*set|dataset)\]|\bdata\s*set\s+version\b)", re.I)
_SOFTWARE_RE = re.compile(r"(?:\[(?:computer\s+)?software\]|\bsoftware\s+version\b)", re.I)
_VIDEO_RE = re.compile(r"(?:\[(?:video|webinar)\]|\bYouTube\b|\bVimeo\b)", re.I)
_PODCAST_RE = re.compile(r"(?:\[(?:audio\s+)?podcast(?:\s+episode)?\]|\bpodcast\s+episode\b)", re.I)
_SOCIAL_RE = re.compile(r"(?:\[(?:tweet|social\s+media\s+post)\]|\b(?:twitter|x|mastodon|weibo)\.com/)", re.I)
_BLOG_RE = re.compile(r"(?:\[blog\s+post\]|\bsubstack\.com/|\bmedium\.com/)", re.I)
_NEWS_RE = re.compile(r"\[(?:news|newspaper|magazine)\s+article\]", re.I)


def classify_reference_source_kind(
    raw_ref: str | None,
    *,
    title: str | None = None,
    url: str | None = None,
) -> SourceKindAssessment:
    """Classify a citation's expected work type from bibliographic evidence.

    The classifier is deliberately conservative.  Its result is an expected
    identity constraint, not a claim that every malformed reference can be
    typed.  Explicit labels and distinctive container structures are strong;
    a bare URL is only weak evidence for a generic web resource.
    """
    raw = " ".join((raw_ref or "").split())
    combined = " ".join(part for part in (raw, title or "", url or "") if part)
    normalized_url = (url or "").casefold()

    if is_archive_source(raw):
        return _assessment("archival_source", "high", "archive/manuscript citation marker")
    if _BOOK_REVIEW_RE.search(combined) or re.search(r"\breviewed\s+by\b", raw, re.I):
        return _assessment("book_review", "high", "explicit review-of/book-review marker")
    if _DATASET_RE.search(combined):
        return _assessment("dataset", "high", "explicit dataset marker")
    if _SOFTWARE_RE.search(combined) or "github.com/" in normalized_url:
        return _assessment("software", "high" if _SOFTWARE_RE.search(combined) else "medium", "software marker or repository host")
    if _PODCAST_RE.search(combined):
        return _assessment("podcast_episode", "high", "explicit podcast marker")
    if _VIDEO_RE.search(combined) or any(host in normalized_url for host in ("youtube.com/", "youtu.be/", "vimeo.com/")):
        return _assessment("video", "high", "explicit video marker or dedicated video host")
    if _SOCIAL_RE.search(combined):
        return _assessment("social_media_post", "high", "explicit social-media marker or host")
    if _BLOG_RE.search(combined):
        return _assessment("blog_post", "high" if "[blog" in combined.casefold() else "medium", "blog marker or publishing host")
    if _NEWS_RE.search(combined):
        return _assessment("news_article", "high", "explicit news/newspaper/magazine marker")
    if _THESIS_RE.search(combined):
        return _assessment("thesis", "high", "explicit thesis/dissertation marker")
    if _REPORT_RE.search(combined):
        return _assessment("report", "high", "explicit report or working-paper number")
    if _OECD_REPORT_SERIES_RE.search(combined):
        return _assessment("report", "high", "identified OECD report-series title")
    if _BOOK_SECTION_RE.search(raw):
        return _assessment("book_section", "high", "chapter container/editor structure")
    if is_traditional_media(raw):
        return _assessment("traditional_media", "high", "explicit media role or format marker")
    if _JOURNAL_STRUCTURE_RE.search(raw) or _JOURNAL_VOLUME_ISSUE_RE.search(raw):
        return _assessment("journal_article", "high", "journal volume/issue/page structure")
    if re.search(r"\bISBN(?:-1[03])?\b", raw, re.I) or _BOOK_EDITION_RE.search(raw):
        return _assessment("monograph", "high", "ISBN or edition marker")
    raw_without_url = re.sub(r"https?://\S+|www\.\S+", " ", raw, flags=re.I).strip()
    publisher_matches = list(_BOOK_PUBLISHER_RE.finditer(raw_without_url))
    if publisher_matches and len(raw_without_url) - publisher_matches[-1].end() <= 80:
        return _assessment("monograph", "high", "terminal book-publisher citation structure")
    if re.search(r"https?://|www\.", combined, re.I):
        return _assessment("webpage", "low", "URL without a more specific work-type marker")
    return SourceKindAssessment()


_PROVIDER_TYPE_MAP = {
    "journal-article": "journal_article",
    "article": "journal_article",
    "review": "journal_article",  # literature-review genre is not necessarily a book review
    "editorial": "journal_article",
    "letter": "journal_article",
    "proceedings-article": "conference_paper",
    "proceedings_article": "conference_paper",
    "book-chapter": "book_section",
    "book-section": "book_section",
    "book_chapter": "book_section",
    "book": "monograph",
    "monograph": "monograph",
    "edited-book": "edited_collection",
    "reference-book": "edited_collection",
    "report": "report",
    "report-series": "report",
    "dissertation": "thesis",
    "dataset": "dataset",
    "software": "software",
}


def classify_provider_source_kind(metadata: dict | None) -> SourceKindAssessment:
    """Read a bounded provider work type, ignoring MIME/link ``type`` fields."""
    if not isinstance(metadata, dict):
        return SourceKindAssessment()
    records: list[dict] = [metadata]
    for key in ("message", "provider_metadata"):
        value = metadata.get(key)
        if isinstance(value, dict):
            if key == "provider_metadata":
                records.extend(item for item in value.values() if isinstance(item, dict))
            else:
                records.append(value)
    for record in tuple(records):
        message = record.get("message")
        if isinstance(message, dict):
            records.append(message)
    for record in records[:16]:
        for key in ("work_type", "source_kind"):
            value = record.get(key)
            if isinstance(value, str):
                kind = _PROVIDER_TYPE_MAP.get(value.strip().casefold()) or normalize_source_kind(value)
                if kind != "unknown":
                    return _assessment(kind, "high", f"provider {key}={value}")
        # Top-level scholarly APIs use ``type`` for work type. Nested link and
        # location objects are intentionally not traversed, because there it is
        # normally a MIME/host type.
        value = record.get("type")
        if isinstance(value, str):
            kind = _PROVIDER_TYPE_MAP.get(value.strip().casefold())
            if kind:
                return _assessment(kind, "high", f"provider type={value}")
    return SourceKindAssessment()


def classify_content_source_kind(
    text: str,
    *,
    source_url: str | None = None,
) -> SourceKindAssessment:
    """Classify only strong, bounded signals in acquired source content."""
    sample = " ".join(text[:20000].split())
    if _BOOK_REVIEW_RE.search(sample) or re.search(
        r"\breviewed\s+by\b", sample[:3000], re.I
    ):
        return _assessment("book_review", "high", "acquired content explicitly identifies a review")
    if _DATASET_RE.search(sample):
        return _assessment("dataset", "high", "acquired content explicitly identifies a dataset")
    if _SOFTWARE_RE.search(sample):
        return _assessment("software", "high", "acquired content explicitly identifies software")
    if _THESIS_RE.search(sample[:6000]):
        return _assessment("thesis", "high", "front matter identifies a thesis/dissertation")
    if _JOURNAL_FRONT_MATTER_RE.search(sample[:6000]):
        return _assessment(
            "journal_article", "high", "front matter identifies a journal article"
        )
    if _REPORT_RE.search(sample[:6000]):
        return _assessment("report", "high", "front matter identifies a numbered report")
    if _JOURNAL_STRUCTURE_RE.search(sample[:6000]):
        return _assessment("journal_article", "medium", "journal-like volume/issue/page front matter")
    if re.search(r"\bISBN(?:-1[03])?\s*[:=]?\s*(?:97[89][ -]?)?[0-9Xx -]{9,17}", sample[:12000], re.I):
        return _assessment("monograph", "medium", "ISBN appears in bounded front matter")
    if source_url and re.search(r"https?://", source_url):
        return _assessment("webpage", "low", "network representation without stronger content marker")
    return SourceKindAssessment()


def classify_html_source_kind(html: str, url: str) -> SourceKindAssessment:
    """Classify a web work from trusted structural metadata plus its URL."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    for meta in soup.find_all("meta"):
        key = (meta.get("property") or meta.get("name") or "").casefold()
        if key in {"og:type", "article:section", "twitter:card"}:
            candidates.append(str(meta.get("content") or ""))
    for script in soup.find_all("script", attrs={"type": "application/ld+json"})[:8]:
        try:
            payload = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        nodes = payload if isinstance(payload, list) else [payload]
        for node in nodes:
            if isinstance(node, dict):
                value = node.get("@type")
                candidates.extend(value if isinstance(value, list) else [str(value or "")])
    normalized = " ".join(candidates).casefold()
    if any(marker in normalized for marker in ("newsarticle", "news article")):
        return _assessment("news_article", "high", "HTML structured metadata identifies NewsArticle")
    if any(marker in normalized for marker in ("blogposting", "blog posting")):
        return _assessment("blog_post", "high", "HTML structured metadata identifies BlogPosting")
    if any(marker in normalized for marker in ("report", "governmentservice")):
        return _assessment("report", "medium", "HTML structured metadata identifies report-like content")
    if any(marker in normalized for marker in ("videoobject", "video.other", "video.movie")):
        return _assessment("video", "high", "HTML structured metadata identifies video")
    if "podcastepisode" in normalized:
        return _assessment("podcast_episode", "high", "HTML structured metadata identifies PodcastEpisode")
    if "dataset" in normalized:
        return _assessment("dataset", "high", "HTML structured metadata identifies Dataset")
    if any(marker in normalized for marker in ("scholarlyarticle", "article")):
        return _assessment("journal_article", "medium", "HTML structured metadata identifies article content")
    return _assessment("webpage", "medium", "successfully extracted canonical HTML page")


def compare_source_kinds(
    expected: SourceKindAssessment,
    observed: SourceKindAssessment,
) -> SourceKindCompatibility:
    """Reject only definite identity conflicts; uncertainty remains inspectable."""
    if not expected.is_known or not observed.is_known:
        return SourceKindCompatibility("unknown", "expected or observed work type is unknown")
    if expected.kind == observed.kind:
        return SourceKindCompatibility("compatible", f"matching work type {expected.kind}")

    # Scholarly indexes commonly expose book reviews only through the broader
    # journal-article record type. That broad provider label is compatible with
    # an explicitly cited book review; the reverse is not true when acquired
    # content specifically identifies itself as a book review.
    if expected.kind == "book_review" and observed.kind == "journal_article":
        return SourceKindCompatibility(
            "compatible", "provider journal-article type can contain a book-review item"
        )

    book_family = {"monograph", "edited_collection"}
    if expected.kind in book_family and observed.kind in book_family:
        return SourceKindCompatibility("compatible", "compatible whole-book types")
    if expected.kind == "book_section" and observed.kind in book_family:
        return SourceKindCompatibility(
            "unknown",
            "a parent book may contain the cited section but requires container confirmation",
        )
    web_family = {"webpage", "news_article", "blog_post"}
    if expected.kind == "webpage" and observed.kind in web_family:
        return SourceKindCompatibility("compatible", "generic webpage expectation permits specific web subtype")
    if observed.kind == "webpage" and expected.kind in web_family:
        return SourceKindCompatibility("unknown", "generic observed webpage does not confirm its expected subtype")

    if (
        _CONFIDENCE_RANK.get(expected.confidence, 0) >= _CONFIDENCE_RANK["high"]
        and _CONFIDENCE_RANK.get(observed.confidence, 0) >= _CONFIDENCE_RANK["high"]
    ):
        return SourceKindCompatibility(
            "incompatible",
            f"expected {expected.kind}, observed {observed.kind}",
        )
    return SourceKindCompatibility(
        "unknown",
        f"possible type conflict: expected {expected.kind}, observed {observed.kind}",
    )

# Traditional-media markers. These sources are cited by title/creator/year per
# convention and do not belong in academic-database title search.
TRADITIONAL_MEDIA_RE = re.compile(
    r"\[(?:film|motion picture|tv series|television series|album|"
    r"recording|painting|sculpture|play|performance|photograph|"
    r"dvd|blu-?ray|cd|lp|ep|videorecording|video recording|"
    r"video game|game|opera|ballet|musical|concert|television episode|"
    r"tv episode|podcast episode)\]",
    re.IGNORECASE,
)

# Director/artist/creator credits also signal traditional media.
DIRECTOR_RE = re.compile(
    r"\b(?:director|dir\.|performer|perf\.|artist|creator|choreographer|"
    r"conductor|host|narrator)\b",
    re.IGNORECASE,
)

# Physical-archive markers. Unverifiable automatically.
ARCHIVE_RE = re.compile(
    r"\b(?:archive|archives|manuscript|ms\.|mss\.|"
    r"special collections|box\s+\d|folder\s+\d|"
    r"unpublished manuscript)\b",
    re.IGNORECASE,
)

# Catalog/discovery/login pages identify a source but are not source content.
# They may later feed an authorized library adapter; their page text must not
# be used as evidence for verifying the cited work.
LIBRARY_LOCATOR_HOST_MARKERS = (
    "ebscohost.com",
    "ebscohost-com",
    "ebsco.com",
    "proquest.com",
    "worldcat.org",
    "jstor.org/stable/",
    "books.google.",
)


def is_library_locator_url(url: str) -> bool:
    """Return True for known catalog/discovery URLs, including proxies."""
    normalized = url.strip().lower()
    if normalized.startswith("www."):
        normalized = f"https://{normalized}"
    try:
        parsed = urlparse(normalized)
    except ValueError:
        return False
    target = f"{parsed.netloc}{parsed.path}".lower()
    return any(marker in target for marker in LIBRARY_LOCATOR_HOST_MARKERS)


def is_traditional_media(raw_ref: str) -> bool:
    """Return True if the reference cites traditional media (film, TV, album, etc.).

    Such references should NOT be sent to academic-database title search —
    they produce false-positive keyword matches (papers *about* the work)
    rather than the work itself.
    """
    return bool(TRADITIONAL_MEDIA_RE.search(raw_ref) or DIRECTOR_RE.search(raw_ref))


def is_archive_source(raw_ref: str) -> bool:
    """Return True if the reference cites a physical archive / manuscript.

    These cannot be verified automatically (no API to a physical archive).
    """
    # A URL such as web.archive.org is a locator, not evidence that the cited
    # work is a physical archival holding. Likewise, Internet Archive is a
    # digital host rather than the special-collections source type routed here.
    bibliographic_text = re.sub(r"https?://\S+|www\.\S+", " ", raw_ref, flags=re.I)
    bibliographic_text = re.sub(
        r"\bInternet\s+Archive\b", " ", bibliographic_text, flags=re.I
    )
    return bool(ARCHIVE_RE.search(bibliographic_text))
