"""Deterministic unquoted-wording and close-paraphrase comparison (patchwriting-v4).

Compares student wording with the sentences of authorized source texts and
reports exact matched wording on both sides, the measures that produced each
finding and the coverage of the comparison.

Boundaries (PLAN "Attribution review", ARCHITECTURE §6):
- only the supplied authorized sources are compared; no match is not evidence
  that wording is original, and a match never establishes copying direction
  or intent;
- wording inside quotation marks, citation markers and citation parentheticals
  is excluded, because quotation accuracy is the quotation check's job;
- every threshold below is provisional; it selects passages for human
  inspection and is not a similarity standard.

v2 (owner review 2026-09-29): the unit of comparison is the best local region
(a clause or any window) found by local alignment of content words, not the
whole sentence; coverage is measured on the source side too; adjacent
transpositions and one-for-one substitutions are tolerated and counted;
possessives and simple inflections are normalized and function words are free.
v3 (owner calibration review 2026-09-29, 30 blind items): a close paraphrase
needs 5 matched content words, and a one-for-one substitution in the same slot
counts as kept structure: matched plus substituted words must make up 0.6 of
the region's content words on each side. Owner agreement rose from 25/30 to
28/30 with no new flags on unrelated sources, controls or real statements.
v1 and v2 results remain readable by the result model.

Model-free; no network or database access. The public entry points never
raise: unexpected input degrades to ``not_assessed`` or a recorded limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import unicodedata
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

POLICY_VERSION = "patchwriting-v4"

# --- Provisional thresholds (for owner calibration) ------------------------
# unquoted_verbatim: a contiguous shared word run (exact word forms).
VERBATIM_MIN_WORDS = 8
VERBATIM_MIN_CONTENT_WORDS = 4
# A qualifying close-paraphrase region is reported as verbatim instead when
# its verbatim run makes up at least this share of the region's words.
VERBATIM_REGION_SHARE = 0.8
# close_paraphrase: the best local alignment region of content words.
PARAPHRASE_MIN_MATCHED = 5          # matched content words in the region
# (matched + same-slot substitutions) / content words of the region, on the
# student and the source side alike (the structural_retention measure).
PARAPHRASE_MIN_RETAINED = 0.6
PARAPHRASE_MIN_ORDER_AGREEMENT = 0.5  # Kendall-tau style, 1 = same order
# Shared wording (owner request 2026-10-02): similar words in the same order are
# not patchwriting without copied runs. A close paraphrase also needs one shared
# run of 4+ identical words, or two runs of 2+, each holding a content word.
PARAPHRASE_MIN_LONG_RUN = 4
PARAPHRASE_MIN_SHORT_RUN = 2
PARAPHRASE_MIN_SHORT_RUNS = 2
# Local alignment scores over content words (function words are free).
ALIGN_MATCH = 2.0
ALIGN_SUBSTITUTION = -1.0   # one-for-one replacement in the same slot
ALIGN_GAP = -1.5            # a content word present on one side only
ALIGN_TRANSPOSITION = -1.0  # adjacent swap, added to two matches
# Candidate retrieval.
CANDIDATE_TOP_K = 5
CANDIDATE_MIN_SHARED_STEMS = 3
MAX_POSTING_SHARE = 0.05     # stems in more than this share of sentences...
MAX_POSTING_FLOOR = 200      # ...and more than this many are not used to retrieve
MAX_FINDINGS_PER_STUDENT_SENTENCE = 3
# Matches against these source roles are measured but never flagged: a
# bibliography entry or note reproduces titles, not the source's prose.
NOT_FLAGGED_SOURCE_ROLES = frozenset({
    "reference_list", "citation_notes", "publication_metadata", "document_metadata",
    "page_furniture", "author_biography",
})
# A student region whose content words are mostly capitalized (not sentence-
# initial) is a title or name sequence, not prose; measured but not flagged.
TITLE_CASE_SHARE = 0.75

# --- Input bounds (degrade and record; never fail) --------------------------
MAX_BODY_CHARACTERS = 400_000
MAX_STATEMENT_CHARACTERS = 20_000
MAX_ALIGN_STUDENT_CONTENT = 80
MAX_ALIGN_SOURCE_CONTENT = 160
SOURCE_ALIGN_STEP = 100
MAX_SOURCE_SENTENCES = 250_000
MAX_SOURCE_SENTENCE_TEXT = 1_500

THRESHOLDS = {
    "verbatim_min_words": VERBATIM_MIN_WORDS,
    "verbatim_min_content_words": VERBATIM_MIN_CONTENT_WORDS,
    "verbatim_region_share": VERBATIM_REGION_SHARE,
    "paraphrase_min_matched": PARAPHRASE_MIN_MATCHED,
    "paraphrase_min_retained": PARAPHRASE_MIN_RETAINED,
    "paraphrase_min_order_agreement": PARAPHRASE_MIN_ORDER_AGREEMENT,
    "paraphrase_min_long_run": PARAPHRASE_MIN_LONG_RUN,
    "paraphrase_min_short_run": PARAPHRASE_MIN_SHORT_RUN,
    "paraphrase_min_short_runs": PARAPHRASE_MIN_SHORT_RUNS,
    "align_match": ALIGN_MATCH,
    "align_substitution": ALIGN_SUBSTITUTION,
    "align_gap": ALIGN_GAP,
    "align_transposition": ALIGN_TRANSPOSITION,
    "candidate_top_k": CANDIDATE_TOP_K,
    "candidate_min_shared_stems": CANDIDATE_MIN_SHARED_STEMS,
    "title_case_share": TITLE_CASE_SHARE,
}

STOPWORDS = frozenset(
    """a about above after again against all also although am among an and another any are
    as at be because been before being below between both but by can could did do does
    doing down during each either else even ever every few for from further had has have
    having he her here hers herself him himself his how however i if in into is it its
    itself just many may me might more most much must my myself neither no nor not now of
    off on once one only or other others our ours ourselves out over own per rather same
    shall she should since so some such than that the their theirs them themselves then
    there these they this those though through thus to too under until up upon us very
    was we were what when where whether which while who whom whose why will with within
    without would yet you your yours yourself yourselves""".split()
)

# Conventional phrases: their words count toward a verbatim run's length but
# never as content words (PLAN risk "Attribution false positives").
CONVENTIONAL_PHRASES = tuple(tuple(phrase.split()) for phrase in (
    "in the united states", "the united states", "united states of america",
    "the united kingdom", "the results show", "the results suggest", "the results indicate",
    "the findings show", "the findings suggest", "research shows", "studies show",
    "as a result", "in the context of", "it is important to note", "it is important to",
    "on the other hand", "in other words", "for example", "for instance", "in addition",
    "in terms of", "with respect to", "in order to", "the fact that", "a number of",
    "at the same time", "in the case of", "as well as", "the role of", "the impact of",
    "the effect of", "the effects of", "the importance of", "in recent years",
    "over the past", "in the past", "the present study", "this study", "the study",
    "the authors", "the author", "the researchers", "take into account", "play a role",
    "plays a role", "an important role", "a significant role", "in particular",
    "on the basis of", "per cent", "percent of", "high school", "long term", "short term",
))
_CONVENTIONAL_BY_FIRST: dict[str, list[tuple[str, ...]]] = {}
for _phrase in sorted(CONVENTIONAL_PHRASES, key=len, reverse=True):
    _CONVENTIONAL_BY_FIRST.setdefault(_phrase[0], []).append(_phrase)

# Same sentence boundaries as facet_evidence_judgment._passage_sentences, so a
# reported sentence has the identity the report's evidence lists use.
_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=[A-Z0-9\"'‘“(])|\n{2,}")
_CLAUSE_BOUNDARY = re.compile(r"[,;:()\[\]—–]|\s-\s")
_WORD_CHAR = re.compile(r"\w")
_TOKEN = re.compile(
    r"(?P<hb>[^\W\d_]+-[ \t]*\n[ \t]*[a-z]+)|(?P<w>[^\W_]+(?:['’][^\W_]+)*)"
)
_REFERENCE_HEADING = re.compile(r"(?im)^\s*(?:references|bibliography|works\s+cited)\s*$")
_QUOTE_PATTERNS = (
    ("double_quotation", re.compile(r'"[^"]*"')),
    ("double_quotation", re.compile(r"“[^”]*”")),
    ("single_quotation", re.compile(r"(?<!\w)‘[^’\n]{2,}?’(?!\w)")),
    ("single_quotation", re.compile(r"(?<![\w'])'[^'\n]{2,}?'(?![\w])")),
)
# Source-side quotation spans may cross PDF line breaks.
_SOURCE_QUOTE_PATTERNS = (
    re.compile(r'"[^"]{2,1500}"'),
    re.compile(r"“[^”]{2,1500}”"),
    re.compile(r"(?<!\w)‘[^’]{2,1500}?’(?!\w)"),
    re.compile(r"(?<![\w'])'[^']{2,1500}?'(?![\w])"),
)
# A parenthetical holding a digit is a citation or locator, never prose.
_CITATION_PARENTHETICAL = re.compile(r"\([^()\n]{0,160}\d[^()\n]{0,160}\)")

Kind = Literal["unquoted_verbatim", "close_paraphrase"]
Label = Literal["citation_statement_vs_cited_source", "body_sentence"]
ExclusionKind = Literal[
    "double_quotation", "single_quotation", "unclosed_quotation", "block_quotation",
    "citation_marker", "citation_parenthetical",
]


# --- Inputs -----------------------------------------------------------------

@dataclass(frozen=True)
class StudentStatement:
    """Exact student wording; offsets are local to `text`.

    `segments` maps local ranges to paper offsets: (local_start, local_end,
    paper_start). A citation unit is one segment starting at passage_start.
    """

    text: str
    segments: tuple[tuple[int, int, int], ...] = ()
    marker_spans: tuple[tuple[int, int], ...] = ()
    claim_type: str = "paraphrase"
    claim_id: str | None = None


@dataclass(frozen=True)
class CitationStatement:
    """A citation claim's paper range and the representations it cites."""

    claim_id: str
    paper_start: int
    paper_end: int
    cited_representation_ids: tuple[str, ...] = ()
    marker_spans: tuple[tuple[int, int], ...] = ()     # paper offsets
    block_quotation: bool = False


@dataclass(frozen=True)
class SourceSentence:
    text: str
    page_index: int | None
    page_label: str | None
    absolute_start: int
    absolute_end: int
    role: str = "body"


# --- Output -----------------------------------------------------------------

class StudentSpan(BaseModel):
    local_start: int
    local_end: int
    paper_start: int | None = None
    paper_end: int | None = None
    text: str


class SourceSpan(BaseModel):
    absolute_start: int
    absolute_end: int


class MatchedSourceSentence(BaseModel):
    sentence_key: str
    page_index: int | None
    page_label: str | None
    absolute_start: int
    absolute_end: int
    role: str
    text: str
    text_truncated: bool = False
    matched_spans: list[SourceSpan] = Field(default_factory=list)


class PairMeasures(BaseModel):
    candidate_score: float
    student_content_words: int
    region_student_content_words: int = 0
    region_source_content_words: int = 0
    matched_content_words: int = 0
    aligned_substitutions: int = 0
    transpositions: int = 0
    student_gaps: int = 0
    source_gaps: int = 0
    student_density: float = 0.0
    source_coverage: float = 0.0
    student_clause_coverage: float = 0.0
    source_clause_coverage: float = 0.0
    order_agreement: float = 0.0
    structural_retention: float = 0.0
    alignment_score: float = 0.0
    longest_run_words: int = 0
    longest_run_content_words: int = 0
    qualifying_run_words: int = 0
    # Share of matched source words that sit inside quotation marks in the
    # source itself (the source quoting someone else: common-source wording).
    source_quoted_share: float = 0.0
    suppressed_reason: str | None = None


class PatchwritingFinding(BaseModel):
    kind: Kind
    label: Label = "body_sentence"
    claim_ids: list[str] = Field(default_factory=list)
    student_sentence_index: int
    student_sentence: StudentSpan
    student_region: StudentSpan
    student_matched_spans: list[StudentSpan]
    source_representation_id: str | None = None
    source: MatchedSourceSentence
    additional_source_sentences: list[MatchedSourceSentence] = Field(default_factory=list)
    measures: PairMeasures


class SentenceComparison(BaseModel):
    """Best candidate's measures for one student sentence (calibration data)."""

    student_sentence_index: int
    compared_content_words: int
    candidates_evaluated: int
    best_sentence_key: str | None = None
    best_measures: PairMeasures | None = None


class ExcludedStudentSpan(BaseModel):
    kind: ExclusionKind
    local_start: int
    local_end: int


class SourceCoverage(BaseModel):
    representation_id: str | None = None
    content_sha256: str | None = None
    extracted_text_sha256: str | None = None
    comparison_scope: Literal["full_text", "passages"] = "full_text"
    source_sentences_indexed: int = 0
    source_sentences_compared: int = 0


class PatchwritingCoverage(BaseModel):
    comparison_scope: Literal["full_text", "passages"] = "full_text"
    representation_id: str | None = None
    content_sha256: str | None = None
    extracted_text_sha256: str | None = None
    source_sentences_indexed: int = 0
    source_sentences_compared: int = 0
    sources: list[SourceCoverage] = Field(default_factory=list)
    student_sentences: int = 0
    student_sentences_compared: int = 0
    student_words_total: int = 0
    student_words_excluded_quotation: int = 0
    student_words_excluded_marker: int = 0
    student_words_compared: int = 0


class PatchwritingResult(BaseModel):
    policy_version: Literal["patchwriting-v1", "patchwriting-v2", "patchwriting-v3", "patchwriting-v4"] = POLICY_VERSION
    status: Literal["compared", "not_assessed"]
    reason: str | None = None
    claim_id: str | None = None
    findings: list[PatchwritingFinding] = Field(default_factory=list)
    sentence_comparisons: list[SentenceComparison] = Field(default_factory=list)
    excluded_student_spans: list[ExcludedStudentSpan] = Field(default_factory=list)
    coverage: PatchwritingCoverage
    limitations: list[str] = Field(default_factory=list)
    thresholds: dict[str, float | int] = Field(default_factory=lambda: dict(THRESHOLDS))
    # Evidence for inspection only; never a determination of copying or intent.
    decision_applied: Literal[False] = False


# --- Tokenization -------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Token:
    start: int
    end: int
    norm: str          # exact word form (casefolded, possessive removed)
    stem: str          # light inflectional stem
    content: bool
    clause_break_after: bool = False   # clause punctuation follows (student side)
    capitalized: bool = False


def _shared_content_runs(student: list, source: list) -> list[int]:
    """Lengths of identical word runs shared by the two regions that hold a
    content word, longest first."""
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, [t.norm for t in student], [t.norm for t in source], autojunk=False)
    return sorted((block.size for block in matcher.get_matching_blocks()
                   if block.size and any(student[block.a + k].content for k in range(block.size))), reverse=True)


def _normalize_word(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("’", "'")
    if value.endswith("'s"):
        value = value[:-2]
    return value.replace("'", "")


def stem(word: str) -> str:
    """Conservative suffix stripper (-s/-es/-ed/-ing/-ly/-ies), both sides alike."""
    if len(word) <= 3 or not word.isalpha():
        return word
    base = word
    if base.endswith("ies") and len(base) > 4:
        base = base[:-3] + "y"
    elif base.endswith("sses"):
        base = base[:-2]
    elif base.endswith("ing") and len(base) > 5:
        base = base[:-3]
    elif base.endswith("ed") and len(base) > 4:
        base = base[:-2]
    elif base.endswith("ly") and len(base) > 4:
        base = base[:-2]
    elif base.endswith("es") and len(base) > 4 and re.search(r"(?:[sxz]|ch|sh)es$", base):
        base = base[:-2]
    elif base.endswith("s") and not re.search(r"(?:ss|us|is)$", base):
        base = base[:-1]
    if base != word and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsaeiou":
        base = base[:-1]                     # stopped -> stop
    if len(base) > 4 and base.endswith("e"):
        base = base[:-1]                     # relate / related -> relat
    if len(base) > 3 and base.endswith("y"):
        base = base[:-1] + "i"               # signify / signified -> signifi
    return base


def _is_content(norm: str) -> bool:
    return norm not in STOPWORDS and (len(norm) > 1 or norm.isdigit())


def _hyphen_break_forms(span_text: str) -> list[str]:
    """Join a line-break hyphen as text_quality.readable_text does."""
    try:
        from app.services.text_quality import readable_text

        joined = readable_text(span_text)
    except Exception:  # word lists or import unavailable: join mechanically
        joined = re.sub(r"-[ \t]*\n[ \t]*", "", span_text)
    return [part for part in joined.split("-") if part]


def _make(start: int, end: int, word: str) -> _Token | None:
    norm = _normalize_word(word)
    if not norm:
        return None
    return _Token(start, end, norm, stem(norm), _is_content(norm), False, word[:1].isupper())


def tokenize(text: str, offset: int = 0) -> list[_Token]:
    tokens: list[_Token] = []
    for match in _TOKEN.finditer(text):
        if match.group("hb"):
            forms = _hyphen_break_forms(match.group("hb"))
            if len(forms) == 1:
                token = _make(match.start() + offset, match.end() + offset, forms[0])
                if token:
                    tokens.append(token)
                continue
            left, right = re.split(r"-[ \t]*\n[ \t]*", match.group("hb"), maxsplit=1)
            for s, e, word in ((match.start(), match.start() + len(left), left),
                               (match.end() - len(right), match.end(), right)):
                token = _make(s + offset, e + offset, word)
                if token:
                    tokens.append(token)
            continue
        token = _make(match.start() + offset, match.end() + offset, match.group("w"))
        if token:
            tokens.append(token)
    return _discount_conventional(tokens)


def _discount_conventional(tokens: list[_Token]) -> list[_Token]:
    conventional: set[int] = set()
    for i, token in enumerate(tokens):
        for phrase in _CONVENTIONAL_BY_FIRST.get(token.norm, ()):
            if tuple(t.norm for t in tokens[i:i + len(phrase)]) == phrase:
                conventional.update(range(i, i + len(phrase)))
                break
    if not conventional:
        return tokens
    return [
        _Token(t.start, t.end, t.norm, t.stem, False, t.clause_break_after, t.capitalized)
        if i in conventional and t.content else t
        for i, t in enumerate(tokens)
    ]


# --- Sentences and clauses -------------------------------------------------------

def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _ends_with_initial(text: str, position: int) -> bool:
    """Equivalent of re.search(r"\\b[A-Z]\\.\\s*$", text[:position]) in O(gap)."""
    j = position
    while j > 0 and text[j - 1].isspace():
        j -= 1
    if j < 2 or text[j - 1] != "." or not ("A" <= text[j - 2] <= "Z"):
        return False
    return j < 3 or not _WORD_CHAR.match(text[j - 3])


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _ends_with_initial(text, match.start()):
            continue
        if trimmed := _trim(text, start, match.start()):
            spans.append(trimmed)
        start = match.end()
    if trimmed := _trim(text, start, len(text)):
        spans.append(trimmed)
    return spans


def clause_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Punctuation-delimited clauses (commas, semicolons, colons, brackets, dashes)."""
    end = len(text) if end is None else end
    spans, cursor = [], start
    for match in _CLAUSE_BOUNDARY.finditer(text, start, end):
        if trimmed := _trim(text, cursor, match.start()):
            spans.append(trimmed)
        cursor = match.end()
    if trimmed := _trim(text, cursor, end):
        spans.append(trimmed)
    return spans


def source_sentences_from_pages(pages: Iterable[Any]) -> list[SourceSentence]:
    """Split extracted source pages (index, label, text, structural_spans).

    Offsets are page-absolute, so `sentence_key` matches
    evidence_report.evidence_sentence_key for a whole-page passage.
    """
    sentences: list[SourceSentence] = []
    in_references = False
    for page in pages:
        text = getattr(page, "text", "") or ""
        structural = [
            (span.start, span.end, span.role)
            for span in (getattr(page, "structural_spans", ()) or ())
        ]
        heading = _REFERENCE_HEADING.search(text)
        for start, end in sentence_spans(text):
            if heading is not None and not in_references and start >= heading.start():
                in_references = True
            role = "reference_list" if in_references else "body"
            if role == "body" and _looks_bibliographic(text[start:end]):
                role = "citation_notes"
            for s_start, s_end, s_role in structural:
                # A structural role labels a sentence only when it covers most
                # of it; a header touching a long sentence does not.
                if min(end, s_end) - max(start, s_start) >= (end - start) / 2:
                    role = s_role
                    break
            sentences.append(SourceSentence(
                text=text[start:end],
                page_index=getattr(page, "index", None),
                page_label=getattr(page, "label", None),
                absolute_start=start,
                absolute_end=end,
                role=role,
            ))
    return sentences


_BIBLIOGRAPHIC = re.compile(
    r"University Press|\bPress\b|\(\s*[A-Z][\w .]+:\s*[^()]{2,80}\)|\bdoi\b|https?://|\bpp?\.\s*\d|\bvol\.\s*\d",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:1[6-9]|20)\d{2}\b")


def _looks_bibliographic(text: str) -> bool:
    """A note or bibliography entry: publication details plus a year."""
    return bool(_BIBLIOGRAPHIC.search(text) and _YEAR.search(text))


def sentence_key(page_index, absolute_start: int, absolute_end: int) -> str:
    # Same form as evidence_report.evidence_sentence_key (not imported: that
    # module is the report renderer).
    return f"{page_index}:{absolute_start}:{absolute_end}"


# --- Source index ---------------------------------------------------------------

@dataclass
class _IndexedSentence:
    sentence: SourceSentence
    tokens: list[_Token]
    clause_of_token: list[int]        # clause number per token (sentence-local)
    clause_content: list[int]         # content-word count per clause


@dataclass
class SourceIndex:
    sentences: list[_IndexedSentence]
    stem_postings: dict[str, list[int]]
    comparison_scope: Literal["full_text", "passages"] = "full_text"
    representation_id: str | None = None
    content_sha256: str | None = None
    extracted_text_sha256: str | None = None
    limitations: list[str] = field(default_factory=list)

    @property
    def max_posting(self) -> int:
        return max(MAX_POSTING_FLOOR, int(MAX_POSTING_SHARE * len(self.sentences)))

    def coverage(self, compared: int = 0) -> SourceCoverage:
        return SourceCoverage(
            representation_id=self.representation_id,
            content_sha256=self.content_sha256,
            extracted_text_sha256=self.extracted_text_sha256,
            comparison_scope=self.comparison_scope,
            source_sentences_indexed=len(self.sentences),
            source_sentences_compared=compared,
        )


def _clauses_for(text: str, offset: int, tokens: list[_Token]) -> tuple[list[int], list[int]]:
    clauses = clause_spans(text)
    clause_of, counts = [], [0] * max(1, len(clauses))
    index = 0
    for token in tokens:
        local = token.start - offset
        while index + 1 < len(clauses) and local >= clauses[index][1]:
            index += 1
        clause_of.append(index)
        if token.content:
            counts[index] += 1
    return clause_of, counts


def build_source_index(
    sentences: Iterable[SourceSentence],
    *,
    comparison_scope: Literal["full_text", "passages"] = "full_text",
    representation_id: str | None = None,
    content_sha256: str | None = None,
    extracted_text_sha256: str | None = None,
) -> SourceIndex:
    limitations: list[str] = []
    indexed: list[_IndexedSentence] = []
    postings: dict[str, list[int]] = {}
    for sentence in sentences:
        if len(indexed) >= MAX_SOURCE_SENTENCES:
            limitations.append("source_sentence_limit_reached")
            break
        tokens = tokenize(sentence.text, sentence.absolute_start)
        if not tokens:
            continue
        clause_of, counts = _clauses_for(sentence.text, sentence.absolute_start, tokens)
        position = len(indexed)
        indexed.append(_IndexedSentence(sentence, tokens, clause_of, counts))
        for word in {t.stem for t in tokens if t.content}:
            postings.setdefault(word, []).append(position)
    return SourceIndex(indexed, postings, comparison_scope, representation_id,
                       content_sha256, extracted_text_sha256, limitations)


# --- Student side -------------------------------------------------------------------

def statement_from_claim(claim: Any) -> StudentStatement:
    """Build the input from a stored claim (dict or ClaimEvidence-like object)."""
    get = claim.get if isinstance(claim, dict) else (lambda key, default=None: getattr(claim, key, default))
    text = get("text") or ""
    segments: list[tuple[int, int, int]] = []
    for segment in get("source_segments") or []:
        seg = segment if isinstance(segment, dict) else vars(segment)
        segments.append((int(seg["local_start"]), int(seg["local_end"]), int(seg["paper_start"])))
    passage_start = get("passage_start")
    if not segments and isinstance(passage_start, int) and passage_start >= 0:
        segments.append((0, len(text), passage_start))
    markers: list[tuple[int, int]] = []
    for marker in get("citation_markers") or []:
        item = marker if isinstance(marker, dict) else vars(marker)
        start, end = item.get("local_start"), item.get("local_end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            markers.append((start, end))
    legacy = get("citation_marker") or ""
    if not markers and legacy and text.count(legacy) == 1:
        start = text.index(legacy)
        markers.append((start, start + len(legacy)))
    return StudentStatement(
        text=text,
        segments=tuple(segments),
        marker_spans=tuple(markers),
        claim_type=get("claim_type") or "paraphrase",
        claim_id=get("claim_id"),
    )


def citation_statement_from_claim(claim: Any, cited_representation_ids: Sequence[str] = ()) -> CitationStatement | None:
    """Paper-offset view of a stored claim for body-wide comparison."""
    statement = statement_from_claim(claim)
    if not statement.segments or not statement.text:
        return None
    start = statement.segments[0][2]
    quoted = bool(re.search(r'["“”]', statement.text))
    return CitationStatement(
        claim_id=statement.claim_id or "",
        paper_start=start,
        paper_end=start + len(statement.text),
        cited_representation_ids=tuple(cited_representation_ids),
        marker_spans=tuple((start + s, start + e) for s, e in statement.marker_spans),
        block_quotation=statement.claim_type == "quotation" and not quoted,
    )


def _excluded_spans(text: str, *, marker_spans=(), block_ranges=(), claim_type: str = "paraphrase",
                    citation_parentheticals: bool = True) -> list[ExcludedStudentSpan]:
    spans: list[ExcludedStudentSpan] = []
    occupied: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return not any(start < o_end and o_start < end for o_start, o_end in occupied)

    for kind, pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(text):
            if free(match.start(), match.end()):
                occupied.append((match.start(), match.end()))
                spans.append(ExcludedStudentSpan(kind=kind, local_start=match.start(),
                                                 local_end=match.end()))
    # An opening mark with no closing mark: exclude to the end of its paragraph.
    for opener in ("“", '"'):
        for match in re.finditer(re.escape(opener), text):
            if free(match.start(), match.end()):
                paragraph_end = text.find("\n\n", match.start())
                end = len(text) if paragraph_end < 0 else paragraph_end
                occupied.append((match.start(), end))
                spans.append(ExcludedStudentSpan(kind="unclosed_quotation",
                                                 local_start=match.start(), local_end=end))
    if claim_type == "quotation" and not spans and text:
        # A quotation unit with no marks is a block quotation.
        spans.append(ExcludedStudentSpan(kind="block_quotation", local_start=0, local_end=len(text)))
    for start, end in block_ranges:
        spans.append(ExcludedStudentSpan(kind="block_quotation", local_start=start, local_end=end))
    for start, end in marker_spans:
        spans.append(ExcludedStudentSpan(kind="citation_marker", local_start=start, local_end=end))
    if citation_parentheticals:
        for match in _CITATION_PARENTHETICAL.finditer(text):
            spans.append(ExcludedStudentSpan(kind="citation_parenthetical",
                                             local_start=match.start(), local_end=match.end()))
    return sorted(spans, key=lambda span: (span.local_start, span.local_end))


# --- Alignment ------------------------------------------------------------------

# Lowercase words that may join a title's capitalized words ("Harry Potter
# and the Deathly Hallows").
_TITLE_JOINERS = frozenset({"and", "the", "of", "a", "an", "in", "on", "for", "to", "with", "at", "from", "by"})


def _title_run_positions(stream: list) -> frozenset[int]:
    """Stream indexes inside a title or name: two or more capitalized content
    words joined only by title joiners or numbers (Franchise 1, 2026-09-30).

    A title or name repeated from the source is not the source's wording, so
    it neither counts toward a close paraphrase nor makes a verbatim run.
    """
    positions: set[int] = set()
    run: list[int] = []

    def close() -> None:
        while run and not (stream[run[-1]].capitalized or stream[run[-1]].norm.isdigit()):
            run.pop()
        if sum(1 for k in run if stream[k].capitalized and stream[k].content) >= 2:
            positions.update(run)
        run.clear()

    for index, token in enumerate(stream):
        if token is None:
            close()
            continue
        if token.capitalized or token.norm.isdigit():
            run.append(index)
        elif run and token.norm in _TITLE_JOINERS:
            run.append(index)
        else:
            close()
    close()
    return frozenset(positions)


def _longest_runs(student: list[_Token | None], source: list[_Token],
                  titles: frozenset[int] = frozenset()) -> tuple[int, int, tuple | None, int]:
    """Longest contiguous shared runs of exact word forms.

    Returns (longest words, its content words, qualifying run (i0, i1, j0, j1)
    or None, qualifying run words).
    """
    positions: dict[str, list[int]] = {}
    for j, token in enumerate(source):
        positions.setdefault(token.norm, []).append(j)
    previous: dict[int, int] = {}
    runs: list[tuple[int, int, int]] = []
    for i, token in enumerate(student):
        current: dict[int, int] = {}
        if token is not None:
            for j in positions.get(token.norm, ()):
                current[j] = previous.get(j - 1, 0) + 1
        for j, length in previous.items():
            if current.get(j + 1) is None:
                runs.append((length, i - 1, j))
        previous = current
    for j, length in previous.items():
        runs.append((length, len(student) - 1, j))
    longest = (0, 0)
    qualifying, q_key = None, (0, 0)
    for length, end_i, end_j in runs:
        start_j = end_j - length + 1
        content = sum(1 for offset, k in enumerate(range(end_i - length + 1, end_i + 1))
                      if student[k].content
                      and not (k in titles and source[start_j + offset].capitalized))
        longest = max(longest, (length, content))
        if length >= VERBATIM_MIN_WORDS and content >= VERBATIM_MIN_CONTENT_WORDS and (length, content) > q_key:
            qualifying = (end_i - length + 1, end_i, end_j - length + 1, end_j)
            q_key = (length, content)
    return longest[0], longest[1], qualifying, q_key[0]


@dataclass
class _Alignment:
    score: float
    pairs: list[tuple[int, int, str]]   # (student idx, source idx, op) in student order
    i0: int
    i1: int
    j0: int
    j1: int
    gaps_s: int
    gaps_t: int


def _local_alignment(a: list[str], b: list[str]) -> _Alignment | None:
    """Smith-Waterman over content stems with adjacent transpositions."""
    rows, cols = len(a), len(b)
    if not rows or not cols:
        return None
    H = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    P = [[0] * (cols + 1) for _ in range(rows + 1)]   # 1 diag, 2 up, 3 left, 4 transpose
    best, best_i, best_j = 0.0, 0, 0
    for i in range(1, rows + 1):
        ai = a[i - 1]
        row, prev = H[i], H[i - 1]
        prow = P[i]
        for j in range(1, cols + 1):
            bj = b[j - 1]
            diag = prev[j - 1] + (ALIGN_MATCH if ai == bj else ALIGN_SUBSTITUTION)
            up = prev[j] + ALIGN_GAP
            left = row[j - 1] + ALIGN_GAP
            score, pointer = 0.0, 0
            if diag > score:
                score, pointer = diag, 1
            if up > score:
                score, pointer = up, 2
            if left > score:
                score, pointer = left, 3
            if i > 1 and j > 1 and ai != bj and ai == b[j - 2] and a[i - 2] == bj:
                swap = H[i - 2][j - 2] + 2 * ALIGN_MATCH + ALIGN_TRANSPOSITION
                if swap > score:
                    score, pointer = swap, 4
            row[j], prow[j] = score, pointer
            if score > best:
                best, best_i, best_j = score, i, j
    if best <= 0:
        return None
    pairs: list[tuple[int, int, str]] = []
    gaps_s = gaps_t = 0
    i, j = best_i, best_j
    while i > 0 and j > 0 and H[i][j] > 0:
        pointer = P[i][j]
        if pointer == 1:
            pairs.append((i - 1, j - 1, "match" if a[i - 1] == b[j - 1] else "substitution"))
            i, j = i - 1, j - 1
        elif pointer == 4:
            pairs.append((i - 1, j - 2, "transposition"))
            pairs.append((i - 2, j - 1, "transposition"))
            i, j = i - 2, j - 2
        elif pointer == 2:
            gaps_s += 1
            i -= 1
        elif pointer == 3:
            gaps_t += 1
            j -= 1
        else:
            break
    pairs.sort()
    # Trim leading/trailing substitutions (a region starts and ends on a match).
    while pairs and pairs[0][2] == "substitution":
        pairs.pop(0)
    while pairs and pairs[-1][2] == "substitution":
        pairs.pop()
    if not pairs:
        return None
    return _Alignment(best, pairs, pairs[0][0], pairs[-1][0],
                      min(p[1] for p in pairs), max(p[1] for p in pairs), gaps_s, gaps_t)


def _order_agreement(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for x in range(n):
        for y in range(x + 1, n):
            ds = pairs[y][0] - pairs[x][0]
            dt = pairs[y][1] - pairs[x][1]
            if ds * dt > 0:
                concordant += 1
            elif ds * dt < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


# --- Student units --------------------------------------------------------------

@dataclass
class _StudentSentence:
    index: int
    start: int
    end: int
    stream: list[_Token | None]     # compared tokens, None = excluded barrier
    claim_ids: tuple[str, ...] = ()
    cited: frozenset[str] = frozenset()


@dataclass
class _Candidate:
    source: SourceIndex
    position: int
    score: float


@dataclass
class _Evaluated:
    kind: Kind | None
    measures: PairMeasures
    student_region: tuple[int, int]
    student_spans: list[tuple[int, int]]
    source_spans: list[tuple[int, int, int]]      # (sentence position, abs start, abs end)
    source_positions: list[int]
    candidate: _Candidate


def _sentence_units(text: str, excluded: list[ExcludedStudentSpan], *,
                    statements: Sequence[CitationStatement] = ()) -> tuple[list[_StudentSentence], dict]:
    tokens = tokenize(text)
    quoted = [(s.local_start, s.local_end) for s in excluded
              if s.kind not in {"citation_marker", "citation_parenthetical"}]
    markers = [(s.local_start, s.local_end) for s in excluded
               if s.kind in {"citation_marker", "citation_parenthetical"}]
    counts = {"total": len(tokens), "quotation": 0, "marker": 0, "compared": 0}
    quoted.sort()
    markers.sort()

    def inside(token: _Token, spans: list[tuple[int, int]]) -> bool:
        return any(token.start < end and start < token.end for start, end in spans)

    flags: list[str] = []
    for token in tokens:
        if inside(token, quoted):
            flags.append("quotation")
        elif inside(token, markers):
            flags.append("marker")
        else:
            flags.append("compared")
        counts[flags[-1]] += 1
    units: list[_StudentSentence] = []
    cursor = 0
    for index, (start, end) in enumerate(sentence_spans(text)):
        stream: list[_Token | None] = []
        while cursor < len(tokens) and tokens[cursor].end <= start:
            cursor += 1
        k = cursor
        while k < len(tokens) and tokens[k].start < end:
            if flags[k] == "compared":
                stream.append(tokens[k])
            elif stream and stream[-1] is not None:
                stream.append(None)
            k += 1
        while stream and stream[-1] is None:
            stream.pop()
        claim_ids, cited = [], set()
        for statement in statements:
            if statement.paper_start < end and start < statement.paper_end:
                claim_ids.append(statement.claim_id)
                cited.update(statement.cited_representation_ids)
        units.append(_StudentSentence(index, start, end, stream, tuple(claim_ids), frozenset(cited)))
    return units, counts


def _candidates(unit: _StudentSentence, sources: Sequence[SourceIndex]) -> list[_Candidate]:
    stems = {t.stem for t in unit.stream if t is not None and t.content}
    if len(stems) < CANDIDATE_MIN_SHARED_STEMS:
        return []
    found: list[_Candidate] = []
    for source in sources:
        total = len(source.sentences) or 1
        limit = source.max_posting
        shared: dict[int, int] = {}
        weight: dict[int, float] = {}
        for word in stems:
            posting = source.stem_postings.get(word)
            if not posting or len(posting) > limit:
                continue
            idf = math.log(1 + total / len(posting))
            for position in posting:
                shared[position] = shared.get(position, 0) + 1
                weight[position] = weight.get(position, 0.0) + idf
        for position, count in shared.items():
            if count >= CANDIDATE_MIN_SHARED_STEMS:
                found.append(_Candidate(source, position, weight[position]))
    found.sort(key=lambda c: (-c.score, c.source.representation_id or "", c.position))
    return found[:CANDIDATE_TOP_K]


def _offsets(length: int, width: int, step: int) -> list[int]:
    """Window starts covering [0, length) with windows of `width`."""
    if length <= width:
        return [0]
    starts = list(range(0, length - width + 1, step))
    if starts[-1] != length - width:
        starts.append(length - width)
    return starts


def _evaluate(unit: _StudentSentence, candidate: _Candidate) -> _Evaluated | None:
    source = candidate.source
    positions = [candidate.position]
    if candidate.position + 1 < len(source.sentences):
        positions.append(candidate.position + 1)   # a statement may span two source sentences
    window: list[tuple[int, int]] = []             # (sentence position, token index)
    for position in positions:
        window.extend((position, k) for k in range(len(source.sentences[position].tokens)))
    source_tokens = [source.sentences[p].tokens[k] for p, k in window]
    source_content = [n for n, t in enumerate(source_tokens) if t.content]

    # Student content words in barrier-separated segments.
    segments: list[list[int]] = [[]]
    for i, token in enumerate(unit.stream):
        if token is None:
            segments.append([])
        elif token.content:
            segments[-1].append(i)
    best: tuple[_Alignment, list[int], int] | None = None
    for seg in segments:
        if not seg:
            continue
        for s_off in _offsets(len(seg), MAX_ALIGN_STUDENT_CONTENT, MAX_ALIGN_STUDENT_CONTENT // 2):
            s_part = seg[s_off:s_off + MAX_ALIGN_STUDENT_CONTENT]
            a = [unit.stream[i].stem for i in s_part]
            for t_off in _offsets(len(source_content), MAX_ALIGN_SOURCE_CONTENT, SOURCE_ALIGN_STEP):
                t_part = source_content[t_off:t_off + MAX_ALIGN_SOURCE_CONTENT]
                alignment = _local_alignment(a, [source_tokens[n].stem for n in t_part])
                if alignment and (best is None or alignment.score > best[0].score):
                    best = (alignment, s_part, t_off)
    titles = _title_run_positions(unit.stream)
    longest, longest_content, run, run_words = _longest_runs(unit.stream, source_tokens, titles)
    n_content = sum(1 for t in unit.stream if t is not None and t.content)
    measures = PairMeasures(candidate_score=round(candidate.score, 3), student_content_words=n_content,
                            longest_run_words=longest, longest_run_content_words=longest_content,
                            qualifying_run_words=run_words)
    region = None
    kind: Kind | None = None
    student_spans: list[tuple[int, int]] = []
    source_spans: list[tuple[int, int, int]] = []
    if best is not None:
        alignment, s_part, t_off = best
        t_part = source_content[t_off:t_off + MAX_ALIGN_SOURCE_CONTENT]
        matched = [(s_part[i], t_part[j]) for i, j, op in alignment.pairs if op != "substitution"]
        subs = sum(1 for *_x, op in alignment.pairs if op == "substitution")
        transpositions = sum(1 for *_x, op in alignment.pairs if op == "transposition") // 2
        s_first, s_last = s_part[alignment.i0], s_part[alignment.i1]
        t_first, t_last = t_part[alignment.j0], t_part[alignment.j1]
        region_s = sum(1 for i in range(s_first, s_last + 1)
                       if unit.stream[i] is not None and unit.stream[i].content)
        region_t = sum(1 for n in range(t_first, t_last + 1) if source_tokens[n].content)
        m = len(matched)
        # Clause coverage: matched words over the content words of the clauses touched.
        student_clause_total = _student_clause_content(unit, s_first, s_last)
        touched: set[tuple[int, int]] = set()
        for n in range(t_first, t_last + 1):
            p, k = window[n]
            touched.add((p, source.sentences[p].clause_of_token[k]))
        source_clause_total = sum(source.sentences[p].clause_content[c] for p, c in touched)
        measures = measures.model_copy(update={
            "region_student_content_words": region_s,
            "region_source_content_words": region_t,
            "matched_content_words": m,
            "aligned_substitutions": subs,
            "transpositions": transpositions,
            "student_gaps": max(0, region_s - m - subs),
            "source_gaps": max(0, region_t - m - subs),
            "student_density": round(m / region_s, 4) if region_s else 0.0,
            "source_coverage": round(m / region_t, 4) if region_t else 0.0,
            "student_clause_coverage": round(m / student_clause_total, 4) if student_clause_total else 0.0,
            "source_clause_coverage": round(m / source_clause_total, 4) if source_clause_total else 0.0,
            "order_agreement": round(_order_agreement(matched), 4),
            "structural_retention": round((m + subs) / max(region_s, region_t), 4)
            if max(region_s, region_t) else 0.0,
            "alignment_score": round(alignment.score, 2),
        })
        # Matched words of a title or name capitalized on both sides do not
        # count toward the minimum (2026-09-30).
        own_words = sum(1 for i, n in matched if not (i in titles and source_tokens[n].capitalized))
        runs = _shared_content_runs([t for t in unit.stream[s_first:s_last + 1] if t is not None],
                                    source_tokens[t_first:t_last + 1])
        copied = bool(runs) and (runs[0] >= PARAPHRASE_MIN_LONG_RUN
                                 or sum(1 for r in runs if r >= PARAPHRASE_MIN_SHORT_RUN) >= PARAPHRASE_MIN_SHORT_RUNS)
        if (own_words >= PARAPHRASE_MIN_MATCHED
                and measures.structural_retention >= PARAPHRASE_MIN_RETAINED
                and measures.order_agreement >= PARAPHRASE_MIN_ORDER_AGREEMENT
                and copied):
            kind = "close_paraphrase"
            region = (_extend_left(unit.stream, s_first), s_last)
            student_spans = _merge_adjacent(unit.stream, sorted(i for i, _ in matched))
            source_spans = _source_spans(window, source_tokens, sorted(n for _, n in matched))
    if run is not None:
        i0, i1, j0, j1 = run
        region_words = (sum(1 for t in unit.stream[region[0]:region[1] + 1] if t is not None)
                        if region else 0)
        if kind is None or run_words >= VERBATIM_REGION_SHARE * region_words:
            kind = "unquoted_verbatim"
            region = (i0, i1)
            student_spans = [(unit.stream[i0].start, unit.stream[i1].end)]
            source_spans = _source_spans(window, source_tokens, list(range(j0, j1 + 1)))
    if kind is not None and region is not None:
        measures = measures.model_copy(update={
            "source_quoted_share": _quoted_share(source, source_spans)})
        roles = {source.sentences[p].sentence.role for p, _s, _e in source_spans}
        blocked = sorted(roles & NOT_FLAGGED_SOURCE_ROLES)
        if blocked:
            measures = measures.model_copy(update={"suppressed_reason": f"source_role:{blocked[0]}"})
            kind = None
        elif _title_like(unit.stream[region[0]:region[1] + 1], unit.start, unit.stream):
            measures = measures.model_copy(update={"suppressed_reason": "title_or_name"})
            kind = None
    if kind is None or region is None:
        return _Evaluated(None, measures, (0, 0), [], [], positions, candidate)
    return _Evaluated(kind, measures,
                      (unit.stream[region[0]].start, unit.stream[region[1]].end),
                      student_spans, source_spans, positions, candidate)


def _quoted_share(source: SourceIndex, spans: list[tuple[int, int, int]]) -> float:
    if not spans:
        return 0.0
    inside = total = 0
    quotes_by_position: dict[int, list[tuple[int, int]]] = {}
    for position, start, end in spans:
        if position not in quotes_by_position:
            sentence = source.sentences[position].sentence
            quotes_by_position[position] = [
                (sentence.absolute_start + m.start(), sentence.absolute_start + m.end())
                for pattern in _SOURCE_QUOTE_PATTERNS for m in pattern.finditer(sentence.text)
            ]
        length = end - start
        total += length
        if any(q0 <= start and end <= q1 for q0, q1 in quotes_by_position[position]):
            inside += length
    return round(inside / total, 4) if total else 0.0


def _title_like(region: list, sentence_start: int, stream: list) -> bool:
    first = next((t for t in stream if t is not None), None)
    words = [t for t in region if t is not None and t.content and t is not first and not t.norm.isdigit()]
    if len(words) < 3:
        return False
    return sum(1 for t in words if t.capitalized) / len(words) >= TITLE_CASE_SHARE


def _student_clause_content(unit: _StudentSentence, first: int, last: int) -> int:
    """Content words in the student clauses (punctuation-delimited) touched by the region."""
    tokens = [t for t in unit.stream if t is not None]
    if not tokens:
        return 0
    lo, hi = unit.stream[first].start, unit.stream[last].end
    # Clause edges are punctuation between consecutive compared tokens.
    clause_ids: list[int] = []
    clause = 0
    previous = None
    for token in unit.stream:
        if token is None:
            clause += 1
            previous = None
            continue
        if previous is not None and previous.clause_break_after:
            clause += 1
        clause_ids.append(clause)
        previous = token
    touched = {c for t, c in zip(tokens, clause_ids) if t.start < hi and lo < t.end}
    return sum(1 for t, c in zip(tokens, clause_ids) if c in touched and t.content)


def _extend_left(stream: list[_Token | None], first: int) -> int:
    """Include function words directly before the region (e.g. "The")."""
    i = first
    while i > 0 and stream[i - 1] is not None and not stream[i - 1].content \
            and not stream[i - 1].clause_break_after:
        i -= 1
    return i


def _merge_adjacent(tokens: list, indexes: list[int]) -> list[tuple[int, int]]:
    """One span per run of consecutive token indexes."""
    spans: list[tuple[int, int]] = []
    previous = None
    for index in indexes:
        token = tokens[index]
        if token is None:
            continue
        if previous is not None and index == previous + 1 and spans:
            spans[-1] = (spans[-1][0], token.end)
        else:
            spans.append((token.start, token.end))
        previous = index
    return spans


def _source_spans(window, source_tokens, indexes: list[int]) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    previous = None
    for n in indexes:
        position = window[n][0]
        token = source_tokens[n]
        if previous is not None and n == previous + 1 and spans and spans[-1][0] == position:
            spans[-1] = (position, spans[-1][1], token.end)
        else:
            spans.append((position, token.start, token.end))
        previous = n
    return spans


def _mark_clause_breaks(text: str, units: list[_StudentSentence]) -> None:
    """Annotate compared tokens followed by clause punctuation (dataclass is frozen)."""
    for unit in units:
        stream = unit.stream
        for i, token in enumerate(stream):
            if token is None:
                continue
            nxt = next((t for t in stream[i + 1:] if t is not None), None)
            gap = text[token.end:nxt.start] if nxt is not None else ""
            object.__setattr__(token, "clause_break_after", bool(_CLAUSE_BOUNDARY.search(gap)))


# --- Detection -----------------------------------------------------------------

def _span(text: str, start: int, end: int, paper_offset) -> StudentSpan:
    paper = paper_offset(start, end)
    return StudentSpan(local_start=start, local_end=end,
                       paper_start=paper[0] if paper else None,
                       paper_end=paper[1] if paper else None,
                       text=text[start:end])


def _matched_sentence(source: SourceIndex, position: int, spans) -> MatchedSourceSentence:
    sentence = source.sentences[position].sentence
    return MatchedSourceSentence(
        sentence_key=sentence_key(sentence.page_index, sentence.absolute_start, sentence.absolute_end),
        page_index=sentence.page_index,
        page_label=sentence.page_label,
        absolute_start=sentence.absolute_start,
        absolute_end=sentence.absolute_end,
        role=sentence.role,
        text=sentence.text[:MAX_SOURCE_SENTENCE_TEXT],
        text_truncated=len(sentence.text) > MAX_SOURCE_SENTENCE_TEXT,
        matched_spans=[SourceSpan(absolute_start=s, absolute_end=e) for p, s, e in spans if p == position],
    )


def _compare(text: str, units: list[_StudentSentence], sources: Sequence[SourceIndex], *,
             paper_offset, include_sentence_measures: bool) -> tuple[list[PatchwritingFinding], list[SentenceComparison], dict[int, set[int]], int]:
    findings: list[PatchwritingFinding] = []
    comparisons: list[SentenceComparison] = []
    compared_positions: dict[int, set[int]] = {id(s): set() for s in sources}
    compared_units = 0
    for unit in units:
        n_content = sum(1 for t in unit.stream if t is not None and t.content)
        if n_content == 0:
            continue
        compared_units += 1
        candidates = _candidates(unit, sources)
        evaluated = []
        for candidate in candidates:
            compared_positions[id(candidate.source)].add(candidate.position)
            result = _evaluate(unit, candidate)
            if result is not None:
                evaluated.append(result)
        evaluated.sort(key=_rank, reverse=True)
        if include_sentence_measures:
            best = evaluated[0] if evaluated else None
            comparisons.append(SentenceComparison(
                student_sentence_index=unit.index,
                compared_content_words=n_content,
                candidates_evaluated=len(evaluated),
                best_sentence_key=_matched_sentence(best.candidate.source, best.candidate.position, []).sentence_key
                if best else None,
                best_measures=best.measures if best else None,
            ))
        kept = 0
        seen_regions: set[tuple] = set()
        for item in evaluated:
            if item.kind is None or kept >= MAX_FINDINGS_PER_STUDENT_SENTENCE:
                continue
            # One finding per student sentence and source: the best-aligned
            # source sentence (repeats of the same wording elsewhere in the
            # source are not separate findings).
            key = (id(item.candidate.source),)
            if key in seen_regions:
                continue
            seen_regions.add(key)
            kept += 1
            source = item.candidate.source
            touched = sorted({p for p, _s, _e in item.source_spans}) or [item.candidate.position]
            label: Label = ("citation_statement_vs_cited_source"
                            if source.representation_id and source.representation_id in unit.cited
                            else "body_sentence")
            findings.append(PatchwritingFinding(
                kind=item.kind,
                label=label,
                claim_ids=list(unit.claim_ids),
                student_sentence_index=unit.index,
                student_sentence=_span(text, unit.start, unit.end, paper_offset),
                student_region=_span(text, *item.student_region, paper_offset),
                student_matched_spans=[_span(text, s, e, paper_offset) for s, e in item.student_spans],
                source_representation_id=source.representation_id,
                source=_matched_sentence(source, touched[0], item.source_spans),
                additional_source_sentences=[_matched_sentence(source, p, item.source_spans)
                                             for p in touched[1:]],
                measures=item.measures,
            ))
    return findings, comparisons, compared_positions, compared_units


def _rank(item: _Evaluated) -> tuple:
    m = item.measures
    return ({"unquoted_verbatim": 2, "close_paraphrase": 1, None: 0}[item.kind],
            m.alignment_score, m.matched_content_words, m.qualifying_run_words, m.candidate_score)


def _not_assessed(reason: str, limitations: list[str], sources: Sequence[SourceIndex] = (),
                  excluded=None, claim_id=None) -> PatchwritingResult:
    coverage = PatchwritingCoverage(sources=[s.coverage() for s in sources if isinstance(s, SourceIndex)])
    if len(coverage.sources) == 1:
        only = coverage.sources[0]
        coverage.comparison_scope = only.comparison_scope
        coverage.representation_id = only.representation_id
        coverage.content_sha256 = only.content_sha256
        coverage.extracted_text_sha256 = only.extracted_text_sha256
        coverage.source_sentences_indexed = only.source_sentences_indexed
    return PatchwritingResult(status="not_assessed", reason=reason, claim_id=claim_id,
                              excluded_student_spans=excluded or [], coverage=coverage,
                              limitations=limitations)


def _run(text: str, excluded: list[ExcludedStudentSpan], sources: Sequence[SourceIndex], *,
         paper_offset, statements: Sequence[CitationStatement] = (), limitations: list[str],
         include_sentence_measures: bool, claim_id: str | None = None) -> PatchwritingResult:
    for source in sources:
        limitations.extend(f"{source.representation_id}:{item}" for item in source.limitations)
    if not any(source.sentences for source in sources):
        return _not_assessed("no_source_text", limitations, sources, excluded, claim_id)
    units, counts = _sentence_units(text, excluded, statements=statements)
    _mark_clause_breaks(text, units)
    findings, comparisons, compared, compared_units = _compare(
        text, units, sources, paper_offset=paper_offset,
        include_sentence_measures=include_sentence_measures)
    source_coverage = [s.coverage(len(compared.get(id(s), ()))) for s in sources]
    coverage = PatchwritingCoverage(
        sources=source_coverage,
        student_sentences=len(units),
        student_sentences_compared=compared_units,
        student_words_total=counts["total"],
        student_words_excluded_quotation=counts["quotation"],
        student_words_excluded_marker=counts["marker"],
        student_words_compared=counts["compared"],
    )
    if len(source_coverage) == 1:
        only = source_coverage[0]
        coverage.comparison_scope = only.comparison_scope
        coverage.representation_id = only.representation_id
        coverage.content_sha256 = only.content_sha256
        coverage.extracted_text_sha256 = only.extracted_text_sha256
        coverage.source_sentences_indexed = only.source_sentences_indexed
        coverage.source_sentences_compared = only.source_sentences_compared
    if counts["compared"] == 0:
        result = _not_assessed("no_unquoted_student_wording", limitations, sources, excluded, claim_id)
        result.coverage = coverage
        return result
    return PatchwritingResult(status="compared", claim_id=claim_id, findings=findings,
                              sentence_comparisons=comparisons, excluded_student_spans=excluded,
                              coverage=coverage, limitations=limitations)


def detect_patchwriting(statement: StudentStatement, source: SourceIndex | Sequence[SourceIndex], *,
                        include_sentence_measures: bool = True) -> PatchwritingResult:
    """Compare one citation statement with one or more indexed sources; never raises."""
    sources = [source] if isinstance(source, SourceIndex) else list(source or [])
    try:
        limitations: list[str] = []
        text = statement.text or ""
        markers = statement.marker_spans
        if len(text) > MAX_STATEMENT_CHARACTERS:
            limitations.append("statement_truncated")
            text = text[:MAX_STATEMENT_CHARACTERS]
            markers = tuple(s for s in markers if s[1] <= MAX_STATEMENT_CHARACTERS)
        if not statement.segments:
            limitations.append("paper_offsets_unavailable")
        excluded = _excluded_spans(text, marker_spans=markers, claim_type=statement.claim_type)

        def paper_offset(start: int, end: int):
            for seg_start, seg_end, paper_start in statement.segments:
                if seg_start <= start and end <= seg_end:
                    return paper_start + start - seg_start, paper_start + end - seg_start
            return None

        return _run(text, excluded, sources, paper_offset=paper_offset, limitations=limitations,
                    include_sentence_measures=include_sentence_measures, claim_id=statement.claim_id)
    except Exception as exc:  # degrade the one record, never the paper run
        return _not_assessed("internal_error", [f"internal_error:{type(exc).__name__}"],
                             [s for s in sources if isinstance(s, SourceIndex)],
                             claim_id=getattr(statement, "claim_id", None))


def detect_in_body(body_text: str, sources: Sequence[SourceIndex], *,
                   statements: Sequence[CitationStatement] = (),
                   include_sentence_measures: bool = False) -> PatchwritingResult:
    """Compare every body sentence (paper offsets = body offsets) with the sources; never raises."""
    try:
        limitations: list[str] = []
        text = body_text or ""
        if len(text) > MAX_BODY_CHARACTERS:
            limitations.append("body_truncated")
            text = text[:MAX_BODY_CHARACTERS]
        markers = [span for statement in statements for span in statement.marker_spans
                   if span[1] <= len(text)]
        blocks = [(s.paper_start, min(s.paper_end, len(text))) for s in statements
                  if s.block_quotation and s.paper_start < len(text)]
        excluded = _excluded_spans(text, marker_spans=markers, block_ranges=blocks)
        return _run(text, excluded, list(sources), paper_offset=lambda s, e: (s, e),
                    statements=statements, limitations=limitations,
                    include_sentence_measures=include_sentence_measures)
    except Exception as exc:
        return _not_assessed("internal_error", [f"internal_error:{type(exc).__name__}"],
                             [s for s in sources if isinstance(s, SourceIndex)])
