from app.services.verification_evidence import (
    _SourcePage, _SourceStructuralSpan, _candidate_union_candidates,
    _is_explanatory_note,
    _linked_explanatory_notes, _PassageCandidate,
)
from app.services.evidence_report import _passage_view, _remaining_source_context
import pytest
from types import SimpleNamespace


@pytest.mark.parametrize('mutation,accepted', [
    (None, True), ('channel', False), ('method', False), ('bibliographic', False),
])
def test_explanatory_note_persistence_contract(monkeypatch, mutation, accepted):
    from app.services import verification_report as report
    from app.services.verification_evidence import _passage_boundary_status
    text = '1 Earlier animated series were broadcast in France during the preceding decade. This history was documented independently.'
    if mutation == 'bibliographic':
        text = 'References\nSmith, A. (2001). Television was created for the younger generation. London: Press.'
    passage = SimpleNamespace(passage_id='p1', text=text, passage_role='citation_notes',
        boundary_status=_passage_boundary_status(text),
        retrieval_method='lexical_overlap' if mutation == 'method' else 'explanatory_note_context')
    item = SimpleNamespace(passage_id='p1', rank=1,
        channels=['candidate_lexical'] if mutation == 'channel' else ['candidate_explanatory_note_context'])
    artifact = SimpleNamespace(passages=[passage], candidate_passage_retrieval=SimpleNamespace(
        status='complete', selections=[SimpleNamespace(candidate_id='c1',passages=[item])]))
    monkeypatch.setattr(report, '_routed_eligible_ids', lambda _: {'c1'})
    if accepted:
        report._validate_candidate_passage_retrieval(artifact, {'p1'})
    else:
        with pytest.raises(report.ReportAuthorizationError):
            report._validate_candidate_passage_retrieval(artifact, {'p1'})


def test_substantive_note_coexists_with_body():
    body = 'The archive documents the history of television broadcasting in France.'
    note = '1 Earlier animated series were broadcast in France during the preceding decade, before this transaction.'
    text = body + '\n' + note
    page = _SourcePage(index=0, label='1', text=text, structural_spans=(
        _SourceStructuralSpan(start=len(body)+1, end=len(text), role='citation_notes'),))
    rows, _, _ = _candidate_union_candidates([page], query_text=body,
        page_locator='', broad_passages=[], top_k=3)
    assert any('candidate_explanatory_note_context' in channels for _, channels in rows)
    assert any(c.passage_role != 'citation_notes' for c, _ in rows)
    assert len(rows) <= 3


def test_bibliographic_and_cross_reference_notes_abstain():
    for note in ['1 Smith, A. (2001). How television was created for the younger generation. London: Press.',
                 '2 See the long discussion of how television was created in Smith (2001).',
                 '3 Jones (1984), pp. 23–25.', '4 The archive and international television distribution.',
                 '5 On the history of television and why it had a wide impact, cf. Smith 2001; Jones 2002.']:
        assert not _is_explanatory_note(note)
    assert not _is_explanatory_note('1 See Smith for further reading.\n2 The actor was in a revival and the part was originally written for another performer.')


def test_explanatory_note_is_labelled_in_report():
    view = _passage_view({'excerpt':'Earlier series were broadcast in France.',
        'retrieval_method':'explanatory_note_context','page_label':'8'})
    assert view['evidence_note'].startswith('Explanatory source note.')
    remaining = _remaining_source_context([{'passage_id':'note-1',
        'excerpt':'Earlier series were broadcast in France.',
        'retrieval_method':'explanatory_note_context','page_label':'8'}], [])
    assert remaining[0]['evidence_note'].startswith('Explanatory source note.')


def _linked_fixture(marker='3', note=None, page_index=0):
    body = 'The excavation established the settlement chronology.' + marker
    note = note or '3 Earlier layers were dated by pottery comparisons rather than the surviving inscription.'
    text = body + '\n' + note
    page = _SourcePage(index=page_index, label='1', text=text, structural_spans=(
        _SourceStructuralSpan(start=len(body)+1, end=len(text), role='citation_notes'),))
    candidate = _PassageCandidate(page_index=0, page_label='1', start=0, end=len(body),
        text=body, method='lexical_overlap', score=0.5, passage_role='body_prose')
    return page, candidate


def test_explicit_link_recovers_note_without_shared_query_words():
    page, body = _linked_fixture()
    notes = _linked_explanatory_notes([page], [body])
    assert len(notes) == 1
    assert notes[0].text.startswith('3 Earlier layers')
    assert page.text[notes[0].start:notes[0].end].strip() == notes[0].text
    assert notes[0].passage_role == 'citation_notes'


def test_link_requires_same_page_unique_number_and_substantive_note():
    for kwargs in [dict(marker='4'), dict(marker=''), dict(page_index=1),
                   dict(note='3 Smith, A. (2001). The layers were dated by pottery comparisons.'),
                   dict(note='3 Earlier layers were dated by pottery comparisons.\n4 The inscription was dated separately.')]:
        page, body = _linked_fixture(**kwargs)
        assert not _linked_explanatory_notes([page], [body])


def test_link_never_uses_reference_list_as_note():
    page, body = _linked_fixture()
    span=page.structural_spans[0]
    page = _SourcePage(index=0, label='1', text=page.text, structural_spans=(
        _SourceStructuralSpan(start=span.start, end=span.end, role='reference_list'),))
    assert not _linked_explanatory_notes([page], [body])
