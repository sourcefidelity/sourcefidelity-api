"""Fail-closed presentation of retained, source-bound findings. No assessment."""
import hashlib
import re


def partial_relevance(member: dict) -> bool:
    """Ordinary reports do not project advisory retrieval labels as judgments."""
    return False


def citation_partial_relevance(citation: dict) -> bool:
    return any(partial_relevance(member) for member in citation.get('members', []))


def _scope_coverage_agrees(member: dict, scope: dict) -> bool:
    """A judgment made on one kind of evidence cannot be read as another.

    Records written before the retrieved-document contract carry no coverage
    and were always abstracts, so their absence is read as `abstract_only`.
    """
    coverage = member.get('coverage_level')
    stated = scope.get('scope_coverage') or 'abstract_only'
    if stated != coverage:
        return False
    if coverage == 'abstract_only':
        return scope.get('scope_policy_version') in {
            'abstract-topic-v4', 'abstract-topic-v5', 'abstract-topic-v6', 'abstract-topic-v7'}
    return scope.get('scope_policy_version') == 'fulltext-topic-v1'


# Calibrated below the observed floor of ordinary use. Measured over the
# full-text members on 2026-09-22, claim vocabulary present in the source
# ran 42% to 100% for judgments of general relevance, and 83% for the one
# mark the model called a mismatch -- which was a false positive. A work
# that genuinely discusses something else shares almost none of the
# citation's vocabulary, so the ground is reserved for that case.
MAX_CLAIM_TERM_PRESENCE_FOR_DIFFERENT_SUBJECT = 0.25
MIN_CLAIM_TERMS_FOR_ABSENCE = 4


def _claim_absent_from_source(scope: dict) -> bool:
    """Does the complete document lack what the citation attributed to it?

    Counted locally over the whole extracted text, not the excerpt the
    judgment saw. Too few distinctive terms to measure is an abstention:
    silence about a claim nobody can characterise is not evidence.
    """
    total = scope.get("claim_terms_total")
    present = scope.get("claim_terms_present")
    if not isinstance(total, int) or not isinstance(present, int):
        return False
    if total < MIN_CLAIM_TERMS_FOR_ABSENCE or present < 0 or present > total:
        return False
    return present / total <= MAX_CLAIM_TERM_PRESENCE_FOR_DIFFERENT_SUBJECT


def _qualifying_scope_ground(scope: dict) -> bool:
    """One of the two affirmative grounds, at the contract version that has it.

    `abstract-topic-v4` knew only the different-subject ground, so its stored
    records are checked exactly as they were written. `v5` adds the stated-scope
    ground: a source whose own declared jurisdiction, period or population
    excludes what the citation attributes to it, which the broad-subject test
    calls compatible and therefore always discarded. `fulltext-topic-v1` is the
    same gate applied to a retrieved document's opening rather than an
    abstract, so it carries both grounds; the coverage check keeps the two
    contracts from being read in place of each other.
    """
    version = scope.get('scope_policy_version')
    different_subject = (scope.get('topic_relation') == 'disjoint'
                         and scope.get('broad_subject_relation') == 'incompatible'
                         and scope.get('plausible_connection') == 'absent')
    if version == 'abstract-topic-v4':
        return different_subject
    if version not in {'abstract-topic-v5', 'abstract-topic-v6', 'abstract-topic-v7', 'fulltext-topic-v1'}:
        return False
    if scope.get('abstract_truncated'):
        # The assessment saw a prefix, not the abstract the reader is shown.
        return False
    stated_scope = (scope.get('stated_scope_conflict') == 'present'
                    and scope.get('discrepancy') == 'incompatible_stated_scope'
                    and str(scope.get('scope_dimension') or '').strip()
                    and str(scope.get('source_scope') or '').strip()
                    and str(scope.get('claim_scope') or '').strip()
                    and scope.get('broad_subject_relation') != 'uncertain')
    if version == 'fulltext-topic-v1':
        # A retrieved document is judged from a bounded opening -- 1,200
        # characters of a work that may run to 126 pages -- so absence of a
        # topic there is not evidence the work is about something else.
        # Both measured marks were that mistake: Pallant's opening omitted
        # narrative and Khan's omitted the Telecommunications Act, and both
        # works discuss them at length. The different-subject ground is
        # therefore allowed only when the COMPLETE document also lacks the
        # vocabulary the citation attributed to it. A stated scope needs no
        # such support: it is an affirmative claim the text makes about
        # itself, and the opening is where a work makes it.
        return bool(stated_scope or (different_subject and _claim_absent_from_source(scope)))
    return bool(different_subject or stated_scope)


# Only a reference whose work the app actually identified can carry a subject
# comparison. Belton's chapter was never identified — 28 candidates, every one
# recorded `plausible_identity_match: False` — and the abstract displayed for
# it belonged to a review of Geoffrey Block's "A Fine Romance". Comparing that
# abstract to the citation produced a mismatch mark, which reads as a finding
# against the student for a source the app fetched by mistake. Every mark in
# that report sat on an unidentified reference; every identified one was clean.
_IDENTIFIED_FOR_SUBJECT_COMPARISON = frozenset({
    'confirmed', 'confirmed_with_minor_differences',
})


def reference_identified(member: dict) -> bool:
    """Do we know which work this is, well enough to compare its subject?

    The Belton failure was a reference that was never located: 28 candidates,
    none a plausible match, and an abstract belonging to a different book. The
    guard against repeating it is not that every bibliographic field agrees,
    but that the located record is recognisably the cited work. A record that
    agrees on both title and author qualifies even when another field
    conflicts -- a wrong year is a citation error, not a failure to identify --
    while an unlocated or merely possible match does not.
    """
    identity = member.get('reference_identity') or {}
    status = identity.get('status')
    if status in _IDENTIFIED_FOR_SUBJECT_COMPARISON:
        return True
    if member.get('text_is_located_book') and member.get('located_book'):
        # The text is the identified single-authored book a chapter was cited
        # from (Synthetic Embrace citation 3, 2026-10-18).
        return True
    # A possible match agreeing on title and author is recognisably the work
    # too (Hettich: Crossref's online-first year left it a possible match;
    # owner decision 2026-10-18; +3 matches, no mismatch over stored reports).
    return bool(status in {'bibliographic_conflict', 'possible_match'}
                and identity.get('title_and_author_agree'))


# A fully retrieved text is judged, never marked a topical mismatch; the
# mark is for abstracts and limited text (owner decision 2026-10-14).
_SCOPE_COMPARABLE_COVERAGE = {'abstract_only', 'partial_text'}


def _scope_source_text(member: dict) -> tuple[str, bool]:
    """The text the scope judgment was made against, and whether it is bound.

    For an abstract that is the displayed summary itself. For a retrieved
    document it is the bounded leading excerpt the assessment was given, which
    is recorded separately because the displayed passage answers a different
    question -- what was cited, not what this work is about.
    """
    if member.get('coverage_level') == 'abstract_only':
        evidence = member.get('best_evidence') or {}
        return str(evidence.get('text') or ''), not evidence.get('excerpt_truncated')
    scope_source = member.get('scope_source') or {}
    # The assessment was bound to the excerpt with its whitespace collapsed.
    from app.services.passage_relevance import scope_bound_text
    text = scope_bound_text(scope_source.get('text'))
    return text, bool(text)


BROAD_CITATION_SOURCES = 3


def assessed_claim_text(citation: dict) -> str:
    """The statement the subject comparison was made for: the whole claim,
    also for a citation the report shows as one clause of it (the clause
    projection of 2026-10-14 had left every clause citation without its topic
    label, mark or note; Synthetic Embrace citation 7, 2026-10-18)."""
    return str((citation.get('clause_of') or citation).get('student_text') or '')


def _bound_scope(member: dict, citation: dict) -> dict | None:
    """The stored subject comparison, when it may speak about this citation.

    Shared by the mismatch mark and the match label, so the two always answer
    under the same conditions: an abstract or limited text, a statement naming
    fewer than three sources, an identified work, and a complete assessment
    bound by hash to the exact text assessed and the statement as written.
    """
    if member.get('coverage_level') not in _SCOPE_COMPARABLE_COVERAGE:
        return None
    # A statement about a body of research cites its sources as examples; one
    # study's own subject cannot mismatch it (Academic Article, 2026-09-30).
    # Removed 2026-10-17 on a measurement the clause-citation fault had
    # distorted; restored 2026-10-18 (owner decision) when that fault's fix
    # revealed three such false marks ("Most research on ... (A; B; C)").
    if len(citation.get('members') or []) >= BROAD_CITATION_SOURCES:
        return None
    if not reference_identified(member):
        # The subject shown may not be the cited work's subject.
        return None
    return _hash_bound_scope(member, citation)


def _hash_bound_scope(member: dict, citation: dict) -> dict | None:
    """A complete comparison bound by hash to the text shown and the statement
    assessed, on the kind of evidence it was made on: the one test every
    topic label, mark and note reads (rules audit 2026-10-18)."""
    if member.get('coverage_level') not in _SCOPE_COMPARABLE_COVERAGE:
        return None
    text, bound = _scope_source_text(member)
    claim = assessed_claim_text(citation)
    assessment = member.get('abstract_relevance') or {}
    scope = assessment.get('scope_assessment') or {}
    hashes = {'abstract_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'claim_sha256': hashlib.sha256(claim.encode()).hexdigest()}
    if (text and claim and bound
            and assessment.get('status') == scope.get('status') == 'complete'
            and all(assessment.get(k) == value == scope.get(k) for k, value in hashes.items())
            and _scope_coverage_agrees(member, scope)):
        return scope
    return None


def topical_mismatch(member: dict, citation: dict) -> bool:
    """Only a complete, hash-bound scope contract qualifies.

    Abstract and retrieved-document scope share one gate: the comparison is
    bound to the exact text assessed, both grounds are affirmative, and a
    reference whose work was not identified never carries the mark. Low
    relevance scores and missing passages never become topical mismatch.
    """
    scope = _bound_scope(member, citation)
    if scope is None:
        return False
    text, _bound = _scope_source_text(member)
    claim = assessed_claim_text(citation)
    return bool(scope.get('relevance') == 'apparent_mismatch' and scope.get('attention') is True
        and _qualifying_scope_ground(scope)
        and str(scope.get('subject_comparison') or '').strip()
        and scope.get('confidence') == 'high'
        and scope.get('discrepancy') in {'different_subject', 'incompatible_stated_scope'}
        and scope.get('abstract_span') and scope['abstract_span'] in text
        and scope.get('claim_span') and scope['claim_span'] in claim
        and str(scope.get('rationale') or '').strip())


UNCERTAIN_NOTE_CLOSING = 'Read the source to judge whether it supports the statement.'


def topical_uncertain(member: dict, citation: dict) -> bool:
    """A comparison that ran but settled neither way: uncertain, or an apparent
    mismatch short of the mark (owner decision 2026-10-18)."""
    scope = _bound_scope(member, citation)
    return bool(scope is not None and not topical_mismatch(member, citation)
                and not topical_match(member, citation))


def topical_match(member: dict, citation: dict) -> bool:
    """The comparison found the abstract or limited text generally relevant to
    the statement (owner decision 2026-10-17): a statement about the source's
    topic, not support for its claim. Uncertain and unmarked comparisons are
    neither a match nor a mismatch."""
    scope = _bound_scope(member, citation)
    return bool(scope is not None and scope.get('relevance') == 'generally_relevant'
                and not topical_mismatch(member, citation))




def abstract_scope_note(member: dict, citation: dict) -> dict | None:
    """The stored subject comparison for an abstract-backed source (owner
    decision 2026-10-12) or a limited-text source (2026-10-13), shown even
    where the comparison found nothing to disagree about.

    Separate from `scope_disagreement` and from the topical-mismatch mark,
    which keep their own presentation. Only a complete assessment bound by
    hash to the abstract shown and the statement as written, for a reference
    whose work was identified, qualifies; no model is called.
    """
    if not reference_identified(member) or topical_mismatch(member, citation):
        return None
    scope = _hash_bound_scope(member, citation) or {}
    comparison = str(scope.get('subject_comparison') or '').strip()
    if not comparison:
        return None
    # No closing advice: reading the source is implied (owner decision 2026-10-18).
    return {'note': comparison.rstrip('. ') + '.'}


def check_shown(member: dict, kind: str) -> bool:
    """A quotation or locator check is shown only for retrieved text: one rule
    for the window, the PDF, the summary and the paper marks (rules audit
    2026-10-18: they had four)."""
    return bool(member.get(f'show_{kind}_check')
                and member.get('coverage_level') in {'full_text', 'abstract_only', 'partial_text'})


def check_attention(member: dict, kind: str) -> bool:
    """A shown check that found something, from a finished check."""
    check = member.get(f'{kind}_check') or {}
    return bool(check_shown(member, kind) and check.get('attention')
                and check.get('status') in {'complete', 'incomplete'})


def locator_attention(member: dict) -> bool:
    """A quotation located outside the supplied locator: an Academic Practice
    issue, marked yellow on the citation (owner request 2026-10-03), not orange."""
    return check_attention(member, 'locator')


def locator_rectangles(citation: dict, words_by_page: dict) -> list[dict]:
    """The locator's own words in the citation ("p. 27"), never the year
    (owner review 2026-10-14); none when they cannot be placed exactly."""
    from app.services.judgment_layer import claim_span_rectangles
    text = str(citation.get('student_text') or '')
    marker = str(citation.get('citation_marker') or '')
    locator = str(citation.get('page_locator') or '').strip()
    start = citation.get('paper_character_start')
    at = text.rfind(marker) if marker else -1
    if not locator or at < 0 or not isinstance(start, int):
        return []
    found = list(re.finditer(r'(?:\b(?:pp?|para|paras|ch|sec)\.\s*)?' + re.escape(locator) + r'(?![\w])',
                             marker))
    if len(found) != 1:
        return []
    begin = start + at + found[0].start()
    rects, placement = claim_span_rectangles(citation, [(begin, begin + len(found[0].group(0)))], words_by_page)
    return rects if placement == 'exact' else []


def member_layers(member: dict, citation: dict) -> set[str]:
    layers = set()
    if topical_mismatch(member, citation):
        layers.add('relevance')
    if check_attention(member, 'quotation'):
        layers.add('practice')
    if any(f.get('finding_type') == 'duplicate_citation_key' for f in member.get('reference_findings') or []):
        layers.add('reference')
    return layers


def member_marks(member: dict, citation: dict, target: dict, words=()) -> str:
    layers = member_layers(member, citation)
    # The quoted words themselves are highlighted when quotation geometry was
    # bound to the paper. A separate marker shape beside the citation then
    # repeats that signal without adding a fact, so drop it. Keep the marker
    # when no geometry was located: there the shape is the only visible notice.
    if citation.get('quotation_difference_rectangles'):
        layers.discard('practice')
    titles = {'relevance': 'Possible topical mismatch',
              'reference': 'Citation formatting issue', 'practice': 'Quotation issue: compare source wording'}
    marks = []
    for offset, layer in enumerate(sorted(layers)):
        if layer == 'relevance':
            marks.append(f'<rect class="layer-mark mark-relevance" x="{target["x0"]-1.5:.3f}" y="{target["y0"]-1.5:.3f}" '
                         f'width="{target["x1"]-target["x0"]+3:.3f}" height="{target["y1"]-target["y0"]+3:.3f}"><title>{titles[layer]}</title></rect>')
            continue
        if layer == 'reference':
            marks.append(f'<rect class="layer-indicator indicator-reference" x="{target["x0"]:.3f}" y="{target["y0"]:.3f}" '
                         f'width="{target["x1"]-target["x0"]:.3f}" height="{target["y1"]-target["y0"]:.3f}"><title>{titles[layer]}</title></rect>')
            continue
        radius = 2
        marks.append(f'<circle class="layer-indicator indicator-{layer}" cx="{target["x1"]+4+offset*7:.3f}" cy="{target["y0"]:.3f}" r="{radius}"><title>{titles[layer]}</title></circle>')
    return ''.join(marks)


# Why the judgment stopped short of a mark. Retained on the record for
# diagnosis and kept OUT of the reader's note: a reader is served by the
# comparison between the source and the citation, not by the rules the
# assessment applied to it.
_SCOPE_DISSENT_REASONS = {
    'connection present': 'plausible connection found',
    'subjects compatible': 'broad subjects judged compatible',
    'confidence not high': 'confidence below the threshold',
    'ground unmet': 'stated grounds not met',
    'scopes differ without conflict': 'scopes named but not treated as incompatible',
}


def scope_disagreement(member: dict, citation: dict) -> dict | None:
    """The comparison to show when the judgment stopped short of a mark.

    One flag, not a gradient. A mark means the signals agreed; anything less
    is shown to the reader as the comparison itself plus the reason it is not
    a mark, in the evidence window rather than beside the citation. An
    abstract is a partial view of a work by construction, so a disagreement
    over one is information for a reader, not a finding against a student.

    Never shown for a reference whose work was not identified: the comparison
    would then describe whichever record the search merged, which is the
    Belton failure in a quieter voice.
    """
    if member.get('coverage_level') not in _SCOPE_COMPARABLE_COVERAGE:
        return None
    if not reference_identified(member):
        return None
    # The same hash-bound comparison as the label and the note: a stale one
    # (made for other wording) is not shown (rules audit 2026-10-18).
    scope = _hash_bound_scope(member, citation)
    if scope is None:
        return None
    comparison = str(scope.get('subject_comparison') or scope.get('rationale') or '').strip()
    if not comparison:
        return None
    if topical_mismatch(member, citation):
        return None  # It is a mark; the mark speaks for itself.

    reasons = []
    if scope.get('relevance') == 'apparent_mismatch':
        if scope.get('plausible_connection') == 'present':
            reasons.append('connection present')
        if scope.get('broad_subject_relation') == 'compatible':
            reasons.append('subjects compatible')
        if scope.get('confidence') != 'high':
            reasons.append('confidence not high')
        if not _qualifying_scope_ground(scope):
            reasons.append('ground unmet')
    elif (str(scope.get('scope_dimension') or '').strip()
            and str(scope.get('source_scope') or '').strip()
            and str(scope.get('claim_scope') or '').strip()
            and scope.get('stated_scope_conflict') != 'present'):
        reasons.append('scopes differ without conflict')
    if not reasons:
        return None
    return {
        'comparison': comparison,
        'reasons': reasons,
        # The reader gets the comparison and what to do with it. The
        # dissent stays in `reasons` for diagnosis, not in the note.
        'note': comparison.rstrip('. ') + '.',
        'source_scope': str(scope.get('source_scope') or '').strip(),
        'claim_scope': str(scope.get('claim_scope') or '').strip(),
        'scope_dimension': str(scope.get('scope_dimension') or '').strip(),
    }
