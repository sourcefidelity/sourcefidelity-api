"""High-precision local reporting-frame and source-voice cues.

The pattern families follow the recurring reporting structures described by
Graff and Birkenstein's *They Say / I Say*: ``according to X``, ``X argues
that``, ``as X puts it``, ``in X's view``, possessive position labels, and
passive ``... by X`` frames.  They are structural cues only.  Downstream code
must still establish that the attributed material bears on the fixed facet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


SOURCE_ATTRIBUTION_STRUCTURE_VERSION = "source-cue-content-relations-v6"

_INITIALS = r"(?:[A-Z]\.){1,3}"
_NAME = rf"(?:{_INITIALS}|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)"
_NAME_PARTICLE = r"(?:da|de|del|della|der|di|du|la|le|van|von)"
_PERSON = rf"{_NAME}(?:\s+(?:{_NAME_PARTICLE}\s+)?{_NAME}){{0,3}}"
ACTOR_PATTERN = (
    rf"{_PERSON}(?:\s+(?:&|and)\s+{_PERSON})?"
    r"(?:\s+et\s+al\.)?"
)

_REPORTING_VERBS = (
    r"acknowledge(?:s|d)?|add(?:s|ed)?|adopt(?:s|ed)?|advocate(?:s|d)?|agree(?:s|d)?|"
    r"argue(?:s|d)?|assert(?:s|ed)?|assume(?:s|d)?|believe(?:s|d)?|celebrate(?:s|d)?|"
    r"claim(?:s|ed)?|complain(?:s|ed)?|complicate(?:s|d)?|concede(?:s|d)?|"
    r"confirm(?:s|ed)?|contend(?:s|ed)?|corroborate(?:s|d)?|conclude(?:s|d)?|"
    r"consider(?:s|ed)?|criticiz(?:e|es|ed)|criticis(?:e|es|ed)|demonstrate(?:s|d)?|"
    r"den(?:y|ies|ied)|deplore(?:s|d)?|describe(?:s|d)?|disagree(?:s|d)?|"
    r"distinguish(?:es|ed)?|emphasiz(?:e|es|ed)|emphasis(?:e|es|ed)|endorse(?:s|d)?|"
    r"establish(?:es|ed)?|explain(?:s|ed)?|impl(?:y|ies|ied)|indicate(?:s|d)?|"
    r"insist(?:s|ed)?|maintain(?:s|ed)?|note(?:s|d)?|observe(?:s|d)?|"
    r"point(?:s|ed)?\s+out|propose(?:s|d)?|question(?:s|ed)?|refute(?:s|d)?|"
    r"reject(?:s|ed)?|remind(?:s|ed)?|report(?:s|ed)?|prove(?:s|d)?|"
    r"respond(?:s|ed)?|state(?:s|d)?|suggest(?:s|ed)?|show(?:s|ed)?|"
    r"think(?:s)?|thought|verif(?:y|ies|ied)|write(?:s)?|wrote|warn(?:s|ed)?|urge(?:s|d)?"
)

# ``find`` is unusually ambiguous in narrative and humanities prose (for
# example, "Rick finds Redemption").  Treat it as attribution only when its
# grammar explicitly introduces a proposition.  Document-voice detection may
# still recognize ``we found`` and ``the study found`` below.
_FINDING_REPORT_VERBS = r"find(?:s)?|found"


@dataclass(frozen=True)
class AttributionCue:
    actor_text: str
    cue_text: str
    family: str
    start: int
    end: int
    actor_start: int
    actor_end: int


@dataclass(frozen=True)
class AttributionRelation:
    """Exact local source–cue–content proposal over one source sentence."""

    actor_text: str
    cue_text: str
    content_text: str
    family: str
    resolution_status: str
    reason_code: str
    actor_start: int
    actor_end: int
    cue_start: int
    cue_end: int
    content_start: int | None
    content_end: int | None


@dataclass(frozen=True)
class EpistemicCommitmentCue:
    """Exact reporting expression without an application-imposed strength order."""

    cue_text: str
    family: str
    holder_role: str
    actor_text: str
    resolution_status: str
    start: int
    end: int
    content_start: int | None
    content_end: int | None


_EXTERNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "according_to",
        re.compile(
            rf"(?i:\baccording\s+to\s+(?:both\s+)?)"
            rf"(?P<actor>{ACTOR_PATTERN})"
        ),
    ),
    (
        "named_actor_reporting_verb",
        re.compile(
            rf"\b(?!(?:(?:The\s+)?United\s+States)\b)(?P<actor>{ACTOR_PATTERN})"
            r"(?<!['’]s)"
            rf"(?:\s*\((?:19|20)\d{{2}}(?::[^)]*)?\))?\s+"
            rf"(?:{_REPORTING_VERBS})\b"
        ),
    ),
    (
        "named_actor_finding_that",
        re.compile(
            rf"\b(?P<actor>{ACTOR_PATTERN})"
            r"(?<!['’]s)"
            rf"(?:\s*\((?:19|20)\d{{2}}(?::[^)]*)?\))?\s+"
            rf"(?:{_FINDING_REPORT_VERBS})\b\s+(?=(?:that|whether)\b)"
        ),
    ),
    (
        "as_actor_puts_it",
        re.compile(
            rf"(?i:\bas\s+(?:(?:the\s+)?(?:prominent|leading|noted|"
            rf"influential)\s+(?:author|critic|economist|historian|"
            rf"philosopher|researcher|scholar)\s+)?)"
            rf"(?P<actor>{ACTOR_PATTERN})\s+"
            r"(?i:puts?\s+it|writes?|notes?|observes?|explains?|states?)\b"
        ),
    ),
    (
        "actor_himself_reports",
        re.compile(
            rf"\b(?P<actor>{ACTOR_PATTERN})\s+"
            r"(?i:(?:him|her|them)self\s+)(?:writes?|states?|notes?|"
            r"argues?|maintains?)\b"
        ),
    ),
    (
        "publication_context_actor_reports",
        re.compile(
            r"(?i:\b(?:writing|reporting)\s+in\s+[^,;]{1,120},\s*)"
            rf"(?P<actor>{ACTOR_PATTERN})\s+(?i:{_REPORTING_VERBS})\b"
        ),
    ),
    (
        "actor_publication_reports",
        re.compile(
            r"(?i:\bin\s+(?:his|her|their)\s+(?:article|book|essay|report|"
            r"study|work)[^,;]{0,120},\s*)"
            rf"(?P<actor>{ACTOR_PATTERN})\s+(?i:{_REPORTING_VERBS})\b"
        ),
    ),
    (
        "named_actor_opinion_frame",
        re.compile(
            rf"\b(?P<actor>{ACTOR_PATTERN})\s+"
            r"(?i:(?:is|was)\s+of\s+the\s+(?:opinion|view)\s+that|"
            r"came\s+to\s+the\s+(?:conclusion|view)\s+that)"
        ),
    ),
    (
        "work_author_reports",
        re.compile(
            r"(?P<actor>(?:The|the)\s+author\s+of\s+"
            r"[A-Z][^.!?]{3,180}?)\s+"
            rf"(?i:{_REPORTING_VERBS}|(?:is|was)\s+of\s+the\s+(?:opinion|view))\b"
        ),
    ),
    (
        "to_named_actor_view",
        re.compile(
            rf"(?:^|[.!?]\s+)(?i:to\s+)(?P<actor>{ACTOR_PATTERN})"
            r"(?=\s*,|\s+(?:a|an|the)\b)"
        ),
    ),
    (
        "to_work_author_view",
        re.compile(
            r"(?:^|[.!?]\s+)(?i:to\s+)"
            r"(?P<actor>(?:the\s+)?author\s+of\s+[A-Z][^.!?]{3,180}?)"
            r"(?=\s*[—–-]\s*)"
        ),
    ),
    (
        "actor_is_saying",
        re.compile(
            rf"(?i:\b(?:basically|essentially|in\s+other\s+words)\s*,?\s*)"
            rf"(?P<actor>{ACTOR_PATTERN})\s+(?i:is\s+saying)\b"
        ),
    ),
    (
        "in_actor_view",
        re.compile(
            rf"(?i:\bin\s+)(?P<actor>{ACTOR_PATTERN})"
            r"(?:['’]s)?\s+(?i:view|words?|account|analysis)\b"
        ),
    ),
    (
        "actor_possessive_position",
        re.compile(
            rf"\b(?P<actor>{ACTOR_PATTERN})['’]s\s+"
            r"(?i:argument|claim|conclusion|concept|contention|finding|idea|point|"
            r"interpretation|observation|position|reflection|theory|view|work)s?\b"
        ),
    ),
    (
        "position_passive_by_actor",
        re.compile(
            r"(?i:\b(?:argument|claim|conclusion|contention|finding|idea|"
            r"interpretation|observation|position|sentiment|theory|view)s?\s+"
            r"(?:is|are|was|were)\s+(?:independently\s+)?(?:adopted|advanced|asserted|challenged|developed|echoed|"
            r"expressed|formulated|made|proposed|questioned|rejected|reported|shared|taken\s+over|"
            r"stated|supported)(?:\s+independently)?\s+by\s+)"
            rf"(?P<actor>{ACTOR_PATTERN})"
        ),
    ),
    (
        "views_or_work_of_actor",
        re.compile(
            r"(?i:\b(?:argument|claim|concept|conclusion|finding|idea|"
            r"position|view|work)s?\s+of\s+)"
            rf"(?P<actor>{ACTOR_PATTERN})"
        ),
    ),
)

_NON_ACTORS = {
    "a",
    "an",
    "article",
    "author",
    "analysis",
    "appendix",
    "chapter",
    "conclusion",
    "data",
    "discussion",
    "evidence",
    "experiment",
    "figure",
    "findings",
    "he",
    "her",
    "hers",
    "his",
    "i",
    "it",
    "its",
    "our",
    "ours",
    "paper",
    "report",
    "research",
    "researchers",
    "results",
    "sample",
    "section",
    "she",
    "study",
    "table",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "to",
    "we",
    "you",
    "your",
}

_DOCUMENT_COMPONENT_SUBJECT = (
    r"(?:this|the)\s+(?:analysis|article|chapter|discussion|experiment|paper|report|study)|"
    r"(?:the\s+)?(?:analysis|data|evidence|findings|results)|"
    r"(?:appendix|figure|section|table)\s+[A-Za-z0-9.-]+"
)

_DOCUMENT_VOICE_PATTERN = re.compile(
    rf"(?i:\b(?:I|we)\s+(?:{_REPORTING_VERBS}|{_FINDING_REPORT_VERBS})\b)|"
    rf"(?i:\b(?:{_DOCUMENT_COMPONENT_SUBJECT})\s+"
    rf"(?:{_REPORTING_VERBS}|{_FINDING_REPORT_VERBS})\b)|"
    r"(?i:\bour\s+(?:analysis|argument|conclusion|finding|position|study|thesis)\b)|"
    r"(?i:\b(?:the\s+)?purpose\s+of\s+this\b)"
)

_COMMITMENT_EXPRESSIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "conclusive_evidence",
        re.compile(r"\b(?:prov(?:e|es|ed)|establish(?:es|ed)?|confirm(?:s|ed)?|verif(?:y|ies|ied))\b", re.IGNORECASE),
    ),
    (
        "evidence_claim",
        re.compile(r"\b(?:demonstrat(?:e|es|ed)|show(?:s|ed)?|finds?|found|conclud(?:e|es|ed))\b", re.IGNORECASE),
    ),
    (
        "tentative_inference",
        re.compile(r"\b(?:suggest(?:s|ed)?|indicat(?:e|es|ed)|impl(?:y|ies|ied)|appear(?:s|ed)?|seem(?:s|ed)?)\b", re.IGNORECASE),
    ),
    (
        "position_assertion",
        re.compile(r"\b(?:argu(?:e|es|ed)|assert(?:s|ed)?|claim(?:s|ed)?|contend(?:s|ed)?|maintain(?:s|ed)?|insist(?:s|ed)?|believ(?:e|es|ed)|propos(?:e|es|ed))\b", re.IGNORECASE),
    ),
    (
        "neutral_report",
        re.compile(r"\b(?:stat(?:e|es|ed)|report(?:s|ed)?|note(?:s|d)?|observ(?:e|es|ed)|writ(?:e|es|ten)|wrote|describ(?:e|es|ed)|explain(?:s|ed)?|discuss(?:es|ed)?|points?\s+out|says?|said)\b", re.IGNORECASE),
    ),
)


def find_epistemic_commitment_cues(text: str) -> list[EpistemicCommitmentCue]:
    """Return exact reporting cues while deliberately withholding comparison.

    The families are descriptive, not a universal ordinal scale. A later
    calibrated procedure may compare a student's cue with source posture, but
    this local foundation never turns lexical choice into overstatement by
    itself.
    """
    relations = find_source_attribution_relations(text)
    relation_by_key = {
        (relation.actor_start, relation.family): relation for relation in relations
    }
    cues: list[EpistemicCommitmentCue] = []
    occupied: set[tuple[int, int, str]] = set()
    for attribution in find_external_attribution_cues(text):
        relation = relation_by_key.get((attribution.actor_start, attribution.family))
        expression = _commitment_expression(text, attribution.start, attribution.end)
        if expression is None:
            family = (
                "attributed_position"
                if attribution.family
                in {
                    "in_actor_view",
                    "named_actor_opinion_frame",
                    "to_named_actor_view",
                    "to_work_author_view",
                    "actor_possessive_position",
                }
                else "neutral_attribution"
            )
            start, end = attribution.start, attribution.end
        else:
            family, start, end = expression
        key = (start, end, attribution.actor_text.casefold())
        if key in occupied:
            continue
        occupied.add(key)
        cues.append(
            EpistemicCommitmentCue(
                cue_text=text[start:end],
                family=family,
                holder_role="external_actor",
                actor_text=attribution.actor_text,
                resolution_status=(
                    relation.resolution_status if relation is not None else "unresolved"
                ),
                start=start,
                end=end,
                content_start=(relation.content_start if relation is not None else None),
                content_end=(relation.content_end if relation is not None else None),
            )
        )
    for match in _DOCUMENT_VOICE_PATTERN.finditer(text):
        expression = _commitment_expression(text, match.start(), match.end())
        if expression is None:
            continue
        family, start, end = expression
        key = (start, end, "document_author")
        if key in occupied:
            continue
        occupied.add(key)
        content = _trim_optional_span(text, match.end(), len(text))
        cues.append(
            EpistemicCommitmentCue(
                cue_text=text[start:end],
                family=family,
                holder_role="document_author",
                actor_text="document_author",
                resolution_status="resolved" if content is not None else "unresolved",
                start=start,
                end=end,
                content_start=content[0] if content is not None else None,
                content_end=content[1] if content is not None else None,
            )
        )
    return sorted(cues, key=lambda item: (item.start, item.end, item.holder_role))


def _commitment_expression(
    text: str, start: int, end: int
) -> tuple[str, int, int] | None:
    matches = []
    for family, pattern in _COMMITMENT_EXPRESSIONS:
        for match in pattern.finditer(text, start, end):
            matches.append((match.start(), match.end(), family))
    if not matches:
        return None
    match_start, match_end, family = min(matches)
    return family, match_start, match_end


def find_external_attribution_cues(text: str) -> list[AttributionCue]:
    """Return exact reporting-frame cues with plausible named actors."""
    cues: list[AttributionCue] = []
    seen: set[tuple[int, int, str]] = set()
    for family, pattern in _EXTERNAL_PATTERNS:
        for match in pattern.finditer(text):
            actor = _normalize_actor(match.group("actor"))
            if not _informative_actor(actor):
                continue
            key = (match.start(), match.end(), actor.casefold())
            if key in seen:
                continue
            seen.add(key)
            cues.append(
                AttributionCue(
                    actor_text=actor[:200],
                    cue_text=re.sub(r"\s+", " ", match.group(0)).strip()[:300],
                    family=family,
                    start=match.start(),
                    end=match.end(),
                    actor_start=match.start("actor"),
                    actor_end=match.end("actor"),
                )
            )
    work_alias_spans = [
        (item.start, item.end)
        for item in cues
        if item.family in {"work_author_reports", "to_work_author_view"}
    ]
    cues = [
        item
        for item in cues
        if not (
            item.family in {"named_actor_reporting_verb", "named_actor_opinion_frame"}
            and any(
                outer_start <= item.start < outer_end
                for outer_start, outer_end in work_alias_spans
            )
        )
    ]
    return sorted(cues, key=lambda item: (item.start, item.end, item.family))


_POST_CUE_CONTENT_FAMILIES = {
    "according_to",
    "named_actor_reporting_verb",
    "as_actor_puts_it",
    "actor_himself_reports",
    "publication_context_actor_reports",
    "actor_publication_reports",
    "named_actor_opinion_frame",
    "work_author_reports",
    "to_named_actor_view",
    "to_work_author_view",
    "actor_is_saying",
    "in_actor_view",
    "named_actor_finding_that",
}


def find_source_attribution_relations(text: str) -> list[AttributionRelation]:
    """Resolve only explicit actor/cue/content relations inside one sentence.

    A named actor mention is not enough.  Generic position labels, possessive
    topic mentions, passive idea genealogy, multiple competing frames, and
    mixed document/external voice remain inspectable but cannot establish a
    proposition holder automatically.
    """
    cues = find_external_attribution_cues(text)
    own_voice = bool(document_voice_cues(text))
    multiple_frames = len(cues) > 1
    relations = []
    for cue in cues:
        content_span = _relation_content_span(text, cue)
        if content_span is None:
            status = "unresolved"
            reason = "attributed_content_span_unresolved"
            content_start = content_end = None
            content_text = ""
            cue_end = cue.end
        else:
            content_start, content_end = content_span
            content_text = text[content_start:content_end]
            cue_end = content_start
            if own_voice:
                status = "ambiguous"
                reason = "mixed_document_and_external_voice"
            elif multiple_frames:
                status = "ambiguous"
                reason = "multiple_attribution_frames"
            else:
                status = "resolved"
                reason = "explicit_actor_cue_content"
        cue_start, cue_end = _trim_span(text, cue.start, cue_end)
        relations.append(
            AttributionRelation(
                actor_text=text[cue.actor_start:cue.actor_end],
                cue_text=text[cue_start:cue_end],
                content_text=content_text,
                family=cue.family,
                resolution_status=status,
                reason_code=reason,
                actor_start=cue.actor_start,
                actor_end=cue.actor_end,
                cue_start=cue_start,
                cue_end=cue_end,
                content_start=content_start,
                content_end=content_end,
            )
        )
    return relations


def _relation_content_span(text: str, cue: AttributionCue) -> tuple[int, int] | None:
    if cue.family == "actor_possessive_position":
        # "Ricardo's point is that ..." attributes the following proposition;
        # "Ricardo's argument concerns ..." merely mentions/describes it.
        tail = re.match(r"\s+(?:is|was)\s+that\s+", text[cue.end:], re.IGNORECASE)
        if tail is None:
            return None
        return _trim_optional_span(text, cue.end + tail.end(), len(text))
    if cue.family not in _POST_CUE_CONTENT_FAMILIES:
        return None
    tail = re.match(
        r"\s*(?:[,;:]|[—–-])?\s*(?:(?:that|whether)\s+)?",
        text[cue.end:],
        re.IGNORECASE,
    )
    start = cue.end + (tail.end() if tail is not None else 0)
    return _trim_optional_span(text, start, len(text))


def _trim_optional_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    start, end = _trim_span(text, start, end)
    if start >= end or not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", text[start:end]):
        return None
    return start, end


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def document_voice_cues(text: str) -> list[str]:
    """Return exact first-person/document-author reporting cues."""
    return list(
        dict.fromkeys(
            re.sub(r"\s+", " ", match.group(0)).strip()[:300]
            for match in _DOCUMENT_VOICE_PATTERN.finditer(text)
        )
    )


def source_voice_fields(text: str) -> dict:
    """Classify one exact source span without deciding proposition relevance."""
    external_cues = find_external_attribution_cues(text)
    relations = find_source_attribution_relations(text)
    resolved_relations = [
        relation for relation in relations if relation.resolution_status == "resolved"
    ]
    actors = list(dict.fromkeys(item.actor_text for item in external_cues))
    cues = list(dict.fromkeys(item.cue_text for item in external_cues))
    own_voice = document_voice_cues(text)
    for cue in own_voice:
        if cue not in cues:
            cues.append(cue)
    if own_voice and external_cues:
        role = "mixed_or_uncertain"
    elif resolved_relations and len(resolved_relations) == len(relations):
        role = "explicit_external_attribution"
    elif external_cues:
        role = "mixed_or_uncertain"
    else:
        role = "unmarked_document_voice"
    return {
        "voice_role": role,
        "attributed_actor_texts": actors[:8],
        "voice_cues": cues[:8],
        "attribution_relations": [asdict(relation) for relation in relations[:8]],
        "epistemic_commitment_cues": [
            asdict(cue) for cue in find_epistemic_commitment_cues(text)[:8]
        ],
    }


def _informative_actor(actor: str) -> bool:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", actor.casefold())
    meaningful = [token for token in tokens if token not in {"and", "et", "al"}]
    return (
        bool(meaningful)
        and any(len(token) > 1 for token in meaningful)
        and not all(token in _NON_ACTORS for token in meaningful)
    )


def _normalize_actor(actor: str) -> str:
    value = re.sub(r"\s+", " ", actor).strip()
    value = re.sub(r"(?i)^(?:As|According|In|The|To)\s+", "", value)
    value = re.sub(r"[’']s$", "", value)
    return value.strip()
