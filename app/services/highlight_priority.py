"""Geometric paint precedence; findings and their evidence are unchanged."""
from copy import deepcopy
import xml.etree.ElementTree as ET

ACADEMIC_FINDINGS = frozenset({'duplicate_reference_entry', 'reference_author_conflict'})
# A DOI that registers a different work is a submitted-link issue, whichever
# check noticed it (owner decision 2026-09-24).
SUBMITTED_LINK_FINDINGS = frozenset({'submitted_link_issue', 'reference_identifier_conflict',
                                     'doi_registers_a_different_title'})
# Every link issue is drawn as the purple diamond (owner decisions 2026-09-30),
# including a DOI or link the reference should include but does not, and an
# unfinished identifier; those stay Citation and reference formatting findings
# in windows and summaries.
LINK_MARKER_FINDINGS = SUBMITTED_LINK_FINDINGS | {'required_doi_missing', 'assessment_link_missing',
                                                  'reference_identifier_placeholder'}
# Reference findings about the evidence for the cited work: whether it could be
# located, and how the located record's details differ from the reference.
# The former review flag is a legacy name for the same presentation.
UNVERIFIED_FINDINGS = frozenset({'unverified_reference', 'potentially_fabricated_reference'})
REFERENCE_DIFFERENCE_FINDINGS = frozenset({'bibliographic_conflict', 'bibliographic_field_conflict',
                                           'publication_year_discrepancy', 'doi_registered_title_differs'})
EVIDENCE_REFERENCE_FINDINGS = UNVERIFIED_FINDINGS | REFERENCE_DIFFERENCE_FINDINGS


def finding_category(kind: str | None) -> str:
    """The report category a reference finding belongs to.

    ``evidence``, ``formatting`` or ``academic``; submitted-link issues are
    Academic Practice and topical mismatch is Evidence.
    """
    if kind in EVIDENCE_REFERENCE_FINDINGS or kind == 'source_topical_mismatch':
        return 'evidence'
    if kind in ACADEMIC_FINDINGS | SUBMITTED_LINK_FINDINGS | {
            'missing_reference', 'missing_reference_entry', 'quotation_difference'}:
        return 'academic'
    return 'formatting'


def subtract_rectangles(rect, covers):
    parts = [tuple(rect)]
    for cover in covers:
        remaining = []
        for x0, y0, x1, y1 in parts:
            a, b, c, d = max(x0, cover[0]), max(y0, cover[1]), min(x1, cover[2]), min(y1, cover[3])
            if a >= c or b >= d:
                remaining.append((x0,y0,x1,y1))
                continue
            remaining.extend(r for r in ((x0,y0,x1,b),(x0,d,x1,y1),(x0,b,a,d),(c,b,x1,d))
                             if r[0] < r[2] and r[1] < r[3])
        parts = remaining
    return parts


def prioritize_svg_highlights(markup):
    root = ET.fromstring('<g>' + markup + '</g>')
    rows = []
    for parent in root.iter():
        for element in list(parent):
            if element.tag != 'rect':
                continue
            classes = set(element.get('class', '').split())
            priority = (3 if classes & {'academic-highlight', 'unverified-highlight', 'quote-difference-mark', 'mark-practice'} else
                        2 if classes & {'reference-formatting-hit', 'indicator-reference'} else
                        1.5 if 'mark-relevance' in classes else
                        1 if classes & {'source-highlight', 'selection-bg'} else 0)
            if priority:
                x, y, w, h = (float(element.get(k, '0')) for k in ('x','y','width','height'))
                rows.append((priority, (x,y,x+w,y+h), parent, element))
    for priority, rect, parent, element in rows:
        covers = [r for p,r,_,_ in rows if p > priority]
        parts = subtract_rectangles(rect, covers)
        if parts == [rect]:
            continue
        index = list(parent).index(element)
        parent.remove(element)
        for offset, (x0,y0,x1,y1) in enumerate(parts):
            new = deepcopy(element)
            for key, value in zip(('x','y','width','height'), (x0,y0,x1-x0,y1-y0)):
                new.set(key, f'{value:.3f}')
            parent.insert(index+offset, new)
    return ''.join(ET.tostring(child, encoding='unicode') for child in root)
