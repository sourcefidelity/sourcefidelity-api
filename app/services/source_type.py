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
    # Broadcast and performance kinds collapse to "unknown" unless named here,
    # and "unknown" is now searchable.  An unrecognised media descriptor must
    # not be mistaken for an unclassified article.
    "televisionseries": "traditional_media",
    "tvseries": "traditional_media",
    "televisionprogramme": "traditional_media",
    "televisionprogram": "traditional_media",
    "radioprogramme": "traditional_media",
    "radioprogram": "traditional_media",
    "radiobroadcast": "traditional_media",
    "documentary": "traditional_media",
    "motionpicture": "traditional_media",
    "musicalbum": "traditional_media",
    "album": "traditional_media",
    "song": "traditional_media",
    "artwork": "traditional_media",
    "painting": "traditional_media",
    "photograph": "traditional_media",
    "performance": "traditional_media",
    "play": "traditional_media",
    "exhibition": "traditional_media",
    "manuscript": "archival_source",
    "archivalmaterial": "archival_source",
    "personalcommunication": "archival_source",
    "interview": "archival_source",
}

# Kinds a bibliographic search cannot settle from author, title and year.
# Academic indexes either do not carry them or carry writing *about* them, so
# an absent match is silence rather than evidence of fabrication.  Everything
# not named here is searchable — including an unclassified reference, whose
# missing kind is a parser gap, not a property of the cited work.
BIBLIOGRAPHICALLY_UNSEARCHABLE_KINDS = frozenset(
    {
        "webpage",
        "news_article",
        "blog_post",
        "social_media_post",
        "video",
        "podcast_episode",
        "traditional_media",
        "archival_source",
        "dataset",
        "software",
    }
)

# Kinds whose identity is settled against a book catalogue rather than an
# article index.  Deliberately excludes "unknown": an unclassified reference
# must not inherit a book's evidence requirement on a guess.
BOOK_CATALOGUE_KINDS = frozenset({"monograph", "edited_collection", "book_section"})


def is_bibliographically_searchable(value: str | None) -> bool:
    """Whether author/title/year can settle this kind's identity."""
    return normalize_source_kind(value) not in BIBLIOGRAPHICALLY_UNSEARCHABLE_KINDS


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
# A journal page locator may carry a single-letter article/supplement prefix
# (E276, S45, e1234). The letter must bind directly to the digits; a separated
# word is not a locator.
# Two accepted shapes. Volume(issue) followed by a page, where the issue may be
# a range or carry a letter ("32(5-6)"); or volume followed by a page *range*
# with no issue at all ("System, 93, 1-11"), which is ordinary APA and was
# previously unclassifiable. The second shape requires the range: a bare
# "volume, number" pair is not evidence of an article.
_JOURNAL_STRUCTURE_RE = re.compile(
    r"(?:\b(?:journal|quarterly|review)\b.{0,100})?"
    # A magazine cites discontinuous pages as "pp. 34-36, 91-93". Without these
    # guards the second branch reads "36" as a volume and "91-93" as its pages.
    # Reject a volume that continues a page range or follows a page marker.
    r"(?<![-\u2013\u2014])(?<!pp\. )(?<!p\. )"
    r"\b\d{1,4}\s*(?:"
    r"\(\s*[\dA-Za-z]{1,4}(?:\s*[-\u2013\u2014/]\s*[\dA-Za-z]{1,4})?\s*\)"
    r"\s*[,.:]?\s*(?:pp?\.\s*|article\s+)?[A-Za-z]?\d{1,6}"
    r"|,\s*[A-Za-z]?\d{1,6}\s*[-\u2013\u2014]\s*[A-Za-z]?\d{1,6}"
    # Article-number journals cite an e-locator instead of pages: "Journal of
    # English for Academic Purposes, 50, Article 100957". Ordinary modern APA,
    # and previously unclassifiable because the locator is not a page number.
    r"|,\s*article\s+\d{1,6}"
    r")",
    re.IGNORECASE,
)
_JOURNAL_VOLUME_ISSUE_RE = re.compile(
    r",\s*\d{1,4}\s*\(\s*\d{1,4}\s*\)\s*\.?\s*(?=(?:https?://|www\.|$))",
    re.IGNORECASE,
)
# Literal emphasis around the journal or journal-plus-volume can survive in
# submitted text. Require the complete container/volume/page-range structure;
# an emphasized phrase alone is not evidence of an article.
_MARKED_JOURNAL_STRUCTURE_RE = re.compile(
    r'\*[^*\n]{2,180}(?:\*\s*,\s*\d{1,4}|,\s*\d{1,4}\*)\s*'
    r'(?:\(\s*\d{1,4}\s*\))?\s*,\s*(?:pp?\.\s*)?'
    r'\d{1,6}\s*[-–—]\s*\d{1,6}(?=[.\s]|$)',
)
_JOURNAL_FRONT_MATTER_RE = re.compile(
    r"(?:\bISSN\s*:|\bjournal\s+homepage\b|\bto\s+cite\s+this\s+article\b|"
    r"\bVolume\s+[A-Z0-9IVXLC]+\s*,?\s*(?:Number|No\.)\s*\d+\s*,?\s*pp?\.)",
    re.IGNORECASE,
)
# Imprint names, not places. A textbook publisher that does not put "Press" in
# its name was read as no publisher at all, so a real book parsed as an unknown
# kind and lost its catalog route entirely (Carlton & Perloff, "Pearson.").
# Names that also occur as venue or title words — "harvard", "chicago" — are
# deliberately excluded: "Harvard Law Review" must stay a journal.
_BOOK_PUBLISHER_RE = re.compile(
    r"\b(?:[A-Za-z]{2,60}UniversityPress|university\s+press|press|routledge|sage|springer|wiley|palgrave|"
    r"penguin|knopf|norton|bloomsbury|blackwell|elsevier|harpercollins|"
    r"random\s+house|cambridge|oxford|columbia\s+global\s+reports|"
    r"pearson|mcgraw[\s-]?hill|prentice[\s-]?hall|cengage|wadsworth|macmillan|"
    r"pergamon|ashgate|edward\s+elgar|de\s+gruyter|rowman|"
    # Submitted text loses spaces often enough that "New York UniversityPress"
    # reached the classifier as one token and matched nothing.
    r"university\s*press|universitet|"
    # Institutional imprints. A body that publishes its own work is a publisher
    # even when its name says nothing about publishing.
    r"british\s+council|british\s+film\s+institute|kamera\s+books|"
    # Major academic imprints missing above (paper 7's Taylor & Francis, 2026-10-04).
    r"taylor\s*(?:&|and)\s*francis|peter\s+lang)\b",
    re.IGNORECASE,
)
_BOOK_EDITION_RE = re.compile(r"\(\s*\d+(?:st|nd|rd|th)\s+ed\.?(?:ition)?\s*\)", re.I)
_BOOK_SECTION_RE = re.compile(
    r"\bIn\s+.{1,160}\((?:Ed|Eds)\.?\)"
    # Editors omitted: "Chapter. In Book title (pp. 75-116)." (2026-09-29).
    r"|\.\s+In\s+[^()]{3,300}?\(\s*pp?\.\s*\d{1,6}\s*[-–—]\s*\d{1,6}\s*\)"
    # Editors after the book title: "Chapter. In Book title (A. Name & B. Name,
    # Eds.)." A common student variant (paper 5, Bordwell, 2026-10-01).
    r"|\.\s+In\s+[^()]{3,300}?\([^()]{2,160}?,\s*Eds?\.\s*\)",
    re.IGNORECASE,
)
_COLLECTION_LEAD_RE = re.compile(
    r"^(?:[^()\n]{2,160}\(Eds?\.\)\s*\.?\s*\((?:18|19|20)\d{2}[a-z]?\)"
    r"|[^.\n]{2,160},\s*editors?\.\s+\S)", re.I,
)


def _reference_component_kind(raw: str, title: str | None) -> SourceKindAssessment | None:
    """Distinguish cited contribution from editor credit, not attribution guilt."""
    # Ignore content-like cues inside a separately bound title and all URLs.
    structural = re.sub(r'https?://\S+|www\.\S+', '', raw, flags=re.I)
    if title and structural.count(title) == 1:
        structural = structural.replace(title, ' ' * len(title))
    apa_section = _BOOK_SECTION_RE.search(structural)
    # MLA requires a distinct quoted contribution and a separate container,
    # contributor credit and page range. "Edited by" alone can describe an
    # edition of an entire authored work (e.g. Beowulf).
    mla_section = re.search(
        r'[“"][^”"]{3,300}[”"]\.?\s+[^“”"]{3,200}?\bEdited\s+by\s+'
        r'.{2,180}?\b(?:pp?\.|pages)\s*\d+\s*[-–—]\s*\d+', raw, re.I)
    collection = _COLLECTION_LEAD_RE.search(structural)
    if collection and (apa_section or mla_section):
        return _assessment('unknown', 'unknown', 'conflicting collection/contribution structure')
    if apa_section or mla_section:
        return _assessment('book_section', 'high', 'distinct contribution/container/editor structure v2')
    if collection:
        return _assessment('edited_collection', 'high', 'explicit leading editor role for whole work v1')
    return None
_REPORT_RE = re.compile(
    r"(?:\[(?:technical\s+)?report\]|\breport\s+(?:no\.?|number)\s*[A-Z0-9-]+|"
    r"\bworking\s+paper\s+(?:no\.?|number)\s*[A-Z0-9-]+)",
    re.IGNORECASE,
)
_OECD_REPORT_SERIES_RE = re.compile(
    r"\bOECD\b.{0,160}\b(?:economic\s+outlook|survey|statistics|indicators)\b",
    re.IGNORECASE,
)
_THESIS_RE = re.compile(r"\b(?:doctoral\s+dissertation|(?:master|bachelor|honou?rs)['’]?s?\s+thesis|phd\s+thesis)\b", re.I)
_DATASET_RE = re.compile(r"(?:\[(?:data\s*set|dataset)\]|\bdata\s*set\s+version\b)", re.I)
_SOFTWARE_RE = re.compile(r"(?:\[(?:computer\s+)?software\]|\bsoftware\s+version\b)", re.I)
_VIDEO_RE = re.compile(r"(?:\[(?:video|webinar)\]|\bYouTube\b|\bVimeo\b)", re.I)
_PODCAST_RE = re.compile(r"(?:\[(?:audio\s+)?podcast(?:\s+episode)?\]|\bpodcast\s+episode\b)", re.I)
_SOCIAL_RE = re.compile(r"(?:\[(?:tweet|social\s+media\s+post)\]|\b(?:twitter|x|mastodon|weibo)\.com/)", re.I)
_BLOG_RE = re.compile(r"(?:\[blog\s+post\]|\bsubstack\.com/|\bmedium\.com/)", re.I)
_NEWS_RE = re.compile(r"\[(?:news|newspaper|magazine)\s+article\]", re.I)


# A body publishing a document about itself - a university language policy, a
# ministry circular - is not deposited in a bibliographic index, so an absent
# match is silence rather than evidence of fabrication. This holds whether or
# not the document is still reachable online; going offline does not make it
# newly searchable. Requires an organizational author AND no venue, publisher,
# identifier or edition, all of which are tested earlier in the classifier.
_ORGANIZATION_AUTHOR_RE = re.compile(
    r"\b(?:universit(?:y|ies|e|ät|à)|college|institute|institution|ministry"
    r"|department|association|council|commission|foundation|organi[sz]ation"
    r"|agency|bureau|society|authority|board|centre|center|trust|office"
    r"|school|academy|federation|committee|consortium|secretariat"
    r"|directorate|parliament|government)\b",
    re.I,
)
# APA personal authors carry initials: "Berg, S. V., & Forsyth, P." An
# organization never does, and that is the cheapest reliable separator.
_PERSONAL_AUTHOR_INITIALS_RE = re.compile(r"[A-Z][\w'\u2019-]+,\s*(?:[A-Z]\.\s*)")
_AUTHOR_SEGMENT_RE = re.compile(r"^(.*?)\(\s*(?:n\.d\.|\d{4})", re.I)
_ANY_DOI_RE = re.compile(r"\b10\.\d{4,}/", re.I)
# An actual DOI statement, not any "10.NNNN/" that happens to occur in a URL
# path. Only this may keep a web address from classifying as a webpage.
_DOI_STATEMENT_RE = re.compile(r"\bdoi\.org/10\.\d{4,}/|\bdoi:\s*10\.\d{4,}/", re.I)
# What follows the year: the title, and then any venue or publisher statement.
_AFTER_YEAR_RE = re.compile(r"\(\s*(?:n\.d\.|\d{4}[a-z]?)\s*\)\.?\s*(.*)$", re.I | re.S)
_STATEMENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z\u00C0-\u024F])")


def _has_statement_after_title(raw: str) -> bool:
    """Whether anything - a publisher, a venue - follows the title."""
    match = _AFTER_YEAR_RE.search(raw)
    if match is None:
        return False
    segments = [part for part in _STATEMENT_SPLIT_RE.split(match.group(1).strip()) if part.strip()]
    return len(segments) > 1


def _organisation_report(raw: str) -> bool:
    """An organizational author, a four-digit year and a title of three words or more."""
    segment = _AUTHOR_SEGMENT_RE.match(raw)
    after = _AFTER_YEAR_RE.search(raw)
    if segment is None or after is None or not re.search(r"\(\s*\d{4}", raw[:segment.end() + 2]):
        return False
    author = segment.group(1).strip()
    if not author or _PERSONAL_AUTHOR_INITIALS_RE.search(author) or not _ORGANIZATION_AUTHOR_RE.search(author):
        return False
    title = _STATEMENT_SPLIT_RE.split(after.group(1).strip())[0] if after.group(1).strip() else ""
    return len(re.findall(r"[^\W\d_]{2,}", title)) >= 3 and not re.match(r"(?i)retrieved|available|https?://", title)


def _institutional_self_published(raw: str) -> bool:
    """Whether an organization is citing a document it published about itself.

    Requires an organizational author and *nothing after the title*: no
    identifier, and no venue or publisher statement of any kind. "World Health
    Organization (2021). Title. World Health Organization." names a publisher,
    is held by the book catalogues, and must stay reviewable; the rule was
    first written to exclude it, which removed org-authored grey literature -
    including fabricated grey literature - from review altogether.
    """
    if _ANY_DOI_RE.search(raw) or _has_statement_after_title(raw):
        return False
    segment = _AUTHOR_SEGMENT_RE.match(raw)
    if segment is None:
        return False
    author = segment.group(1).strip()
    if not author or _PERSONAL_AUTHOR_INITIALS_RE.search(author):
        return False
    return bool(_ORGANIZATION_AUTHOR_RE.search(author))


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
    component_kind = _reference_component_kind(raw, title)
    if component_kind is not None:
        return component_kind
    if is_traditional_media(raw):
        return _assessment("traditional_media", "high", "explicit media role or format marker")
    if (_JOURNAL_STRUCTURE_RE.search(raw) or _JOURNAL_VOLUME_ISSUE_RE.search(raw)
            or _MARKED_JOURNAL_STRUCTURE_RE.search(raw)):
        return _assessment("journal_article", "high", "journal volume/issue/page structure")
    if re.search(r"\bISBN(?:-1[03])?\b", raw, re.I) or _BOOK_EDITION_RE.search(raw):
        return _assessment("monograph", "high", "ISBN or edition marker")
    raw_without_url = re.sub(r"https?://\S+|www\.\S+", " ", raw, flags=re.I).strip()
    publisher_matches = list(_BOOK_PUBLISHER_RE.finditer(raw_without_url))
    if publisher_matches and len(raw_without_url) - publisher_matches[-1].end() <= 80:
        return _assessment("monograph", "high", "terminal book-publisher citation structure")
    # A DOI is a more specific work-type marker, not a bare URL: it says the
    # work is registered and resolvable. Treating a doi.org link as a webpage
    # classified the reference into the unsearchable set, so a fabricated
    # reference carrying an invented DOI was never reviewed at all.
    # An organisation's titled, dated document is a report, searched and
    # assessed like one (owner decision 2026-10-02, reversing 2026-09-23:
    # almost every report a student cites is online).
    if _organisation_report(raw):
        return _assessment("report", "medium", "organizational author with a dated, titled document")
    if re.search(r"https?://|www\.", combined, re.I) and not _DOI_STATEMENT_RE.search(combined):
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


def visible_journal_masthead(soup) -> bool:
    """A compact volume/issue/article label immediately before the work title."""
    heading = soup.find('h1')
    if heading is None or (heading.find_parent('article') is None and heading.find_next('article') is None):
        return False
    return any(re.fullmatch(r'Vol(?:ume)?\.?\s*\d+\s+No\.?\s*\d+\s+Article',
                           node.get_text(' ', strip=True), re.I)
               for node in heading.find_all_previous(['div', 'p', 'span'], limit=24))


def classify_html_source_kind(html: str, url: str) -> SourceKindAssessment:
    """Classify a web work from trusted structural metadata plus its URL."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    if visible_journal_masthead(soup):
        return _assessment('journal_article', 'high', 'visible article masthead identifies journal volume and issue')
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
    r"[\[(]\s*(?:director|dir\.|performer|perf\.|artist|creator|choreographer|"
    r"conductor|host|narrator)s?\s*[\])]",
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
    if not ARCHIVE_RE.search(bibliographic_text):
        return False
    # An organisation named "... Archive" can publish an ordinary web page
    # (UCLA Film & Television Archive, paper 5, 2026-10-01). When the archive
    # wording is only in the author segment and the reference is marked as
    # online, it cites a web page, not a physical holding.
    author, rest = _author_segment(bibliographic_text)
    online = bool(_ONLINE_MARKER_RE.search(raw_ref))
    return not (online and ARCHIVE_RE.search(author) and not ARCHIVE_RE.search(rest))


# Online markers, including MLA's scheme-less addresses ("hammer.ucla.edu/...").
_ONLINE_MARKER_RE = re.compile(
    r"https?://|www\.|\[online\]|\bretrieved\b[^.]{0,80}\bfrom\b"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:edu|org|com|net|gov|ac\.[a-z]{2}|co\.[a-z]{2})/\S", re.IGNORECASE)


def _author_segment(text: str) -> tuple[str, str]:
    """(author, remainder): before an APA date in parentheses, else before the first full stop."""
    date = re.search(r"\(\s*(?:(?:18|19|20)\d{2}[a-z]?|n\.\s*d\.)[^)]{0,40}\)", text, re.IGNORECASE)
    if date and date.start() <= 200:
        return text[:date.start()], text[date.end():]
    stop = re.search(r"\.\s", text)
    if stop and stop.start() <= 160:
        return text[:stop.start()], text[stop.end():]
    return "", text
