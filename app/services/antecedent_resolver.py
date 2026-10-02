"""Local-only, exact-span antecedent rescue for citation verification.

The resolver searches only the extracted student paper.  It never uses an
assessment brief and never calls an LLM.  It records bounded exact candidate
phrases and fails closed when the referent is not uniquely evidenced.
"""

from __future__ import annotations

import hashlib
import re

from app.services.verification_evidence import (
    ClaimAntecedentCandidateEvidence,
    ClaimAntecedentDependency,
    ClaimEvidence,
)


ANTECEDENT_RESOLVER_VERSION = "local-document-antecedent-rescue-v3"
MAX_DOCUMENT_SEARCH_CHARACTERS = 100_000

_TYPED_HEADS = (
    "act", "acts", "measure", "measures", "policy", "policies", "law",
    "laws", "regulation", "regulations", "rule", "rules", "provision",
    "provisions", "reform", "reforms", "restriction", "restrictions",
    "action", "actions", "decision", "decisions", "proposal", "proposals",
)
_TYPE_PATTERN = "|".join(sorted(_TYPED_HEADS, key=len, reverse=True))
_MENTION_BOUNDARY_WORDS = {
    "is", "are", "was", "were", "has", "have", "had", "can", "could",
    "may", "might", "must", "should", "will", "would", "leads", "lead",
    "creates", "create", "causes", "cause", "results", "result", "shows",
    "show", "supports", "support", "limits", "limit", "derails", "derail",
    "affects", "affect", "encourages", "encourage",
    # Reporting verbs: "This suggests that …" refers back like "This shows"
    # (paper 4 citation 23 went unjudged; 2026-10-02).
    "suggests", "suggest", "indicates", "indicate", "implies", "imply", "means", "mean",
    "demonstrates", "demonstrate", "highlights", "highlight", "reflects", "reflect",
    "illustrates", "illustrate", "reveals", "reveal",
}
_DEMONSTRATIVE_WORD = (
    rf"(?!(?:{'|'.join(sorted(_MENTION_BOUNDARY_WORDS))})\b)"
    rf"[A-Za-z][\w'’\-]*"
)
_LEADING_DEPENDENCY = re.compile(
    rf"^\s*(?P<mention>"
    rf"(?:this|that|these|those)\s+"
    rf"(?P<demonstrative>(?:{_DEMONSTRATIVE_WORD}\s+){{0,3}}{_DEMONSTRATIVE_WORD})"
    rf"|the\s+(?P<typed>{_TYPE_PATTERN})"
    rf"|(?:this|that|these|those|it|its|they|their|them|he|his|she|her)\b)",
    re.IGNORECASE,
)
_NAMED_TYPED_PHRASE = re.compile(
    rf"(?<!\w)(?:the\s+)?"
    rf"(?:(?:[A-Z][\w'’\-]*|(?:19|20)\d{{2}}|of|and|the|for|on)\s+){{1,10}}"
    rf"(?:{_TYPE_PATTERN})(?!\w)",
    re.IGNORECASE,
)
_DOMAIN_TYPED_PHRASE = re.compile(
    rf"\b(?:media|broadcasting|telecommunications?|communications?|press|"
    rf"information|radio|television|regulatory|competition|copyright|privacy)"
    rf"(?:\s+[A-Za-z][\w'’\-]*){{0,3}}\s+(?:{_TYPE_PATTERN})\b",
    re.IGNORECASE,
)
_STRUCTURAL_TYPED_PHRASE = re.compile(
    rf"(?<!\w)(?:the\s+)?(?:[A-Za-z][\w'’\-]*\s+){{0,3}}"
    rf"(?:media|broadcasting|telecommunications?|communications?|press|"
    rf"information|radio|television|regulatory|competition|copyright|privacy)"
    rf"(?:\s+[A-Za-z][\w'’\-]*){{0,3}}\s+(?:{_TYPE_PATTERN})"
    rf"(?:\s+of\s+(?:19|20)\d{{2}})?(?!\w)",
    re.IGNORECASE,
)
_GENERIC_GROUP_HEADS = {
    "individual", "individuals", "person", "persons", "people",
}
_INFORMATIVE_GROUP_HEADS = (
    "people", "individuals", "persons", "students", "participants",
    "respondents", "children", "adults", "men", "women", "workers",
    "patients", "teachers", "researchers", "viewers", "audiences",
)
_GROUP_HEAD_PATTERN = "|".join(
    sorted(_INFORMATIVE_GROUP_HEADS, key=len, reverse=True)
)
_GROUP_POSTMODIFIER_WORD = (
    r"(?!(?:is|are|was|were|has|have|had|can|could|may|might|must|should|"
    r"will|would|seem|seems|appear|appears|represent|represents|represented|"
    r"view|views|viewed)\b)[A-Za-z][\w'’\-]*"
)
_COMPATIBLE_GROUP_PHRASE = re.compile(
    rf"(?<!\w)(?:[A-Za-z][\w'’\-]*\s+)?(?:{_GROUP_HEAD_PATTERN})"
    rf"(?:\s+(?:with|without|living\s+with|diagnosed\s+with)"
    rf"\s+(?:{_GROUP_POSTMODIFIER_WORD}\s*){{1,4}}"
    rf"|\s+who\s+(?:have|live\s+with|are\s+diagnosed\s+with)"
    rf"\s+(?:{_GROUP_POSTMODIFIER_WORD}\s*){{1,4}})?(?!\w)",
    re.IGNORECASE,
)
_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)


def resolve_claim_antecedents(
    body_text: str,
    claim: ClaimEvidence,
) -> ClaimEvidence:
    """Resolve a leading dependency through ordered local-paper search tiers."""
    mention = _dependency_mention(claim)
    if mention is None:
        return claim.model_copy(
            update={
                "antecedent_dependencies": [],
                "context_dependency_status": "not_required",
            }
        )

    local_start, local_end, mention_text, head, plural, compatible_group = mention
    tiers = _search_tiers(body_text, claim)
    for tier, ranges in tiers:
        candidates = _candidates_for_ranges(
            body_text,
            ranges,
            head=head,
            typed=head in _TYPED_HEADS,
            compatible_group=compatible_group,
            tier=tier,
        )
        if not candidates:
            continue
        status, selected = _resolve_candidates(candidates, plural=plural)
        dependency = _dependency(
            claim,
            local_start,
            local_end,
            mention_text,
            tier=tier,
            candidates=candidates,
            status=status,
            selected=selected,
        )
        return claim.model_copy(
            update={
                "antecedent_dependencies": [dependency],
                "context_dependency_status": status,
            }
        )

    dependency = _dependency(
        claim,
        local_start,
        local_end,
        mention_text,
        tier=None,
        candidates=[],
        status="unresolved",
        selected=[],
    )
    return claim.model_copy(
        update={
            "antecedent_dependencies": [dependency],
            "context_dependency_status": "unresolved",
        }
    )


def _dependency_mention(claim):
    marker_start = claim.text.find(claim.citation_marker) if claim.citation_marker else -1
    searchable = claim.text if marker_start < 0 else claim.text[:marker_start]
    match = _LEADING_DEPENDENCY.match(searchable)
    if not match:
        return None
    start, end = match.span("mention")
    typed = match.group("typed")
    demonstrative = match.group("demonstrative")
    if typed:
        head = typed.casefold()
    elif demonstrative:
        words = list(re.finditer(r"[A-Za-z][\w'’\-]*", demonstrative))
        kept = []
        singular = match.group("mention").split(None, 1)[0].casefold() in {"this", "that"}
        for word in words:
            value = word.group().casefold()
            if value in _MENTION_BOUNDARY_WORDS:
                break
            # After a singular demonstrative and its noun, a word in -s is the
            # verb: "This portrayal reflects" ends at "portrayal" (v3).
            if singular and kept and re.fullmatch(r"[a-z]+[^su]s", value):
                break
            kept.append(word)
        if kept:
            end = match.start("demonstrative") + kept[-1].end()
            head = kept[-1].group().casefold()
        else:
            head = ""
    else:
        head = ""
    text = claim.text[start:end]
    first = text.split()[0].casefold()
    plural = first in {"these", "those", "they", "their", "them"} or head.endswith("s")
    compatible_group = (
        first in {"they", "their", "them"}
        or head in _GENERIC_GROUP_HEADS
    )
    return start, end, text, head, plural, compatible_group


def _search_tiers(body_text, claim):
    immediate = [
        (segment.paper_start, min(segment.paper_end, claim.passage_start))
        for segment in claim.antecedent_context
        if 0 <= segment.paper_start < min(segment.paper_end, claim.passage_start)
    ]
    paragraph_start = claim.passage_start
    for paragraph in _PARAGRAPH.finditer(body_text):
        if paragraph.start() <= claim.passage_start <= paragraph.end():
            paragraph_start = paragraph.start()
            break
    paragraph = (
        [(paragraph_start, claim.passage_start)]
        if paragraph_start < claim.passage_start
        else []
    )
    document_start = max(0, paragraph_start - MAX_DOCUMENT_SEARCH_CHARACTERS)
    anchors = _document_anchor_ranges(body_text, claim.passage_start)
    nearby = (
        [(document_start, paragraph_start)]
        if document_start < paragraph_start
        else []
    )
    return [
        ("immediate_context", immediate),
        ("full_paragraph", paragraph),
        ("document_anchor", anchors),
        ("nearby_section_document", nearby),
    ]


def _document_anchor_ranges(body_text, claim_start):
    """Return exact title/opening/heading ranges before the citation."""
    before = body_text[:claim_start]
    ranges = []
    # Title/opening block and first prose sentence are stable document anchors.
    first_nonspace = re.search(r"\S", before)
    if first_nonspace:
        block_end = re.search(r"\n\s*\n", before[first_nonspace.start():])
        end = (
            first_nonspace.start() + block_end.start()
            if block_end
            else min(len(before), first_nonspace.start() + 1_000)
        )
        if end > first_nonspace.start():
            ranges.append((first_nonspace.start(), end))
    sentence = re.search(r"\S(?:.*?)(?:[.!?](?=\s|$)|$)", before, re.DOTALL)
    if sentence:
        ranges.append(sentence.span())
    # Short standalone lines act as section-heading anchors.
    search_start = max(0, claim_start - 20_000)
    for match in re.finditer(r"(?m)^\s*(\S[^\n]{1,159}?)\s*$", body_text[search_start:claim_start]):
        text = match.group(1).strip()
        words = re.findall(r"[A-Za-z0-9]+", text)
        if not 2 <= len(words) <= 18 or text.endswith((".", "?", "!", ";")):
            continue
        start = search_start + match.start(1)
        ranges.append((start, start + len(match.group(1))))
    deduplicated = []
    for item in ranges:
        if item[1] > item[0] and item not in deduplicated:
            deduplicated.append(item)
    return deduplicated[:8]


def _candidates_for_ranges(
    body_text, ranges, *, head, typed, compatible_group, tier
):
    if not head and not compatible_group:
        return []
    candidates = []
    for start, end in ranges:
        if start < 0 or end <= start or end > len(body_text):
            continue
        text = body_text[start:end]
        if compatible_group:
            matches = list(_COMPATIBLE_GROUP_PHRASE.finditer(text))
        elif typed:
            matches = [*_NAMED_TYPED_PHRASE.finditer(text), *_DOMAIN_TYPED_PHRASE.finditer(text)]
            if tier == "document_anchor":
                matches.extend(_STRUCTURAL_TYPED_PHRASE.finditer(text))
        else:
            pattern = re.compile(
                rf"\b(?:[A-Za-z][\w'’\-]*\s+){{1,4}}{re.escape(head)}\b",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(text))
            derived = _derived_verb_pattern(head) if not matches and tier == "immediate_context" else None
            if derived is not None:
                # "This portrayal" after "…, portraying Alice as a naive child":
                # the verb phrase, to the end of its clause, is the antecedent (v3).
                matches = list(derived.finditer(text))
        for match in matches:
            candidate_start = start + match.start()
            candidate_end = start + match.end()
            exact = body_text[candidate_start:candidate_end]
            exact = exact.strip()
            candidate_start += len(body_text[candidate_start:candidate_end]) - len(
                body_text[candidate_start:candidate_end].lstrip()
            )
            candidate_end = candidate_start + len(exact)
            if not exact or (
                _uninformative_group(exact)
                if compatible_group
                else _uninformative(exact, head)
            ):
                continue
            match_kind = (
                "derived_verb_phrase"
                if derived_head(head) and not typed and not compatible_group
                and not re.search(rf"\b{re.escape(head)}\b", exact, re.IGNORECASE)
                else "compatible_group_phrase"
                if compatible_group
                else "coordinated_named_group"
                if _typed_head_count(exact) >= 2
                else "named_type" if typed else "exact_head_phrase"
            )
            candidates.append(
                ClaimAntecedentCandidateEvidence(
                    candidate_id=_stable_id(
                        ANTECEDENT_RESOLVER_VERSION,
                        str(candidate_start),
                        str(candidate_end),
                        exact,
                    ),
                    text=exact,
                    paper_start=candidate_start,
                    paper_end=candidate_end,
                    search_tier=tier,
                    match_kind=match_kind,
                )
            )
    # Bound adversarial/repetitive papers before coordinated-group pairing.
    base_by_name = {}
    for candidate in candidates:
        key = _normalized(candidate.text)
        current = base_by_name.get(key)
        if current is None or candidate.paper_end > current.paper_end:
            base_by_name[key] = candidate
    candidates = sorted(
        base_by_name.values(), key=lambda item: item.paper_end, reverse=True
    )[:16]
    if typed:
        ordered = sorted(candidates, key=lambda item: item.paper_start)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                if right.paper_end - left.paper_start > 300:
                    break
                # A coordinated phrase must belong to one searched context.
                # Joining separate context sentences can create an anchor for
                # which no exact antecedent_context_index exists.
                if not any(start <= left.paper_start and right.paper_end <= end
                           for start, end in ranges):
                    continue
                exact = body_text[left.paper_start:right.paper_end]
                if _typed_head_count(exact) < 2 or not re.search(
                    r"\b(?:and|or)\b|,", exact, re.IGNORECASE
                ):
                    continue
                candidates.append(
                    ClaimAntecedentCandidateEvidence(
                        candidate_id=_stable_id(
                            ANTECEDENT_RESOLVER_VERSION,
                            str(left.paper_start),
                            str(right.paper_end),
                            exact,
                        ),
                        text=exact,
                        paper_start=left.paper_start,
                        paper_end=right.paper_end,
                        search_tier=tier,
                        match_kind="coordinated_named_group",
                    )
                )
    # Repeated exact names are one semantic alternative; keep the nearest span.
    by_name = {}
    for candidate in candidates:
        key = _normalized(candidate.text)
        current = by_name.get(key)
        if current is None or candidate.paper_end > current.paper_end:
            by_name[key] = candidate
    return sorted(by_name.values(), key=lambda item: item.paper_end, reverse=True)[:8]


def _resolve_candidates(candidates, *, plural):
    if len(candidates) == 1:
        selected = candidates
        if plural and not _plural_phrase(candidates[0].text):
            return "unresolved", []
        return "resolved", selected
    if not plural:
        widest = max(
            candidates,
            key=lambda item: (item.paper_end - item.paper_start, item.paper_end),
        )
        if all(
            widest.paper_start <= candidate.paper_start
            and candidate.paper_end <= widest.paper_end
            for candidate in candidates
        ):
            # Regex variants of one title (for example, "Act", the full named
            # Act, and the same title plus its year) are one alternative, not
            # three ambiguous antecedents. Preserve the widest exact anchor.
            return "resolved", [widest]
    coordinated = [candidate for candidate in candidates if _plural_phrase(candidate.text)]
    if plural and len(coordinated) == 1:
        return "resolved", coordinated
    if plural and coordinated:
        widest = max(
            coordinated,
            key=lambda item: (item.paper_end - item.paper_start, item.paper_end),
        )
        if all(
            widest.paper_start <= candidate.paper_start
            and candidate.paper_end <= widest.paper_end
            for candidate in coordinated
        ):
            return "resolved", [widest]
    return "ambiguous", []


def _dependency(
    claim,
    local_start,
    local_end,
    mention_text,
    *,
    tier,
    candidates,
    status,
    selected,
):
    selected_item = selected[0] if len(selected) == 1 else None
    context_index = None
    if selected_item is not None and tier == "immediate_context":
        context_index = next(
            (
                context.context_index
                for context in claim.antecedent_context
                if context.paper_start <= selected_item.paper_start
                and selected_item.paper_end <= context.paper_end
            ),
            None,
        )
    return ClaimAntecedentDependency(
        mention_text=mention_text,
        mention_local_start=local_start,
        mention_local_end=local_end,
        mention_paper_start=claim.passage_start + local_start,
        mention_paper_end=claim.passage_start + local_end,
        resolution_status=status,
        confidence=(
            "high"
            if status == "resolved" and tier in {"immediate_context", "full_paragraph"}
            else "medium" if status == "resolved" else "low" if status == "ambiguous" else "none"
        ),
        antecedent_context_index=context_index,
        antecedent_text=None if selected_item is None else selected_item.text,
        antecedent_paper_start=None if selected_item is None else selected_item.paper_start,
        antecedent_paper_end=None if selected_item is None else selected_item.paper_end,
        search_tier=tier,
        candidates=candidates,
        selected_candidate_ids=[candidate.candidate_id for candidate in selected],
        method=(
            f"{ANTECEDENT_RESOLVER_VERSION}:{tier}"
            if tier is not None
            else f"{ANTECEDENT_RESOLVER_VERSION}:no_local_candidate"
        ),
    )


def _plural_phrase(text):
    return _typed_head_count(text) >= 2 or bool(
        re.search(
            rf"\b(?:acts|laws|regulations|policies|measures|rules|provisions|"
            rf"reforms|restrictions|actions|decisions|proposals|{_GROUP_HEAD_PATTERN})\b",
            text,
            re.IGNORECASE,
        )
    )


def _typed_head_count(text):
    return len(re.findall(rf"\b(?:{_TYPE_PATTERN})\b", text, re.IGNORECASE))


_NOMINAL_SUFFIXES = ("ation", "ition", "ment", "ance", "ence", "al", "ion")


def derived_head(head: str) -> str | None:
    """The verb stem of a noun formed from a verb (portrayal -> portray), or None."""
    for suffix in _NOMINAL_SUFFIXES:
        if head.endswith(suffix) and len(head) - len(suffix) >= 4:
            return head[: -len(suffix)]
    return None


def _derived_verb_pattern(head: str):
    stem = derived_head(head or "")
    if stem is None:
        return None
    return re.compile(rf"\b{re.escape(stem)}(?:s|ed|ing|es)?\b[^.;:!?,]{{3,160}}", re.IGNORECASE)


def _uninformative(text, head):
    words = re.findall(r"[A-Za-z][\w'’\-]*", text.casefold())
    return not words or words in [[head], ["the", head], ["this", head], ["these", head]]


def _uninformative_group(text):
    words = re.findall(r"[A-Za-z][\w'’\-]*", text.casefold())
    if not words:
        return True
    generic = {"person", "persons", "people", "individual", "individuals"}
    content = [word for word in words if word not in {"a", "an", "the", "this", "these", "those"}]
    return len(content) == 1 and content[0] in generic


def _normalized(text):
    words = re.findall(r"[a-z0-9]+", text.casefold())
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)


def _stable_id(*parts):
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
