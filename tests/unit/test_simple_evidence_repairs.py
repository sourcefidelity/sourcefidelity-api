"""Optional advice cannot invalidate evidence; display never cuts a quotation."""
import json
import pytest
from app.services import passage_relevance as gate
from app.services.evidence_report import (
    _responsive_display_excerpt, _bounded_display_context, _closed_display_delimiters,
)
from tests.unit.test_passage_relevance import _artifact


@pytest.mark.parametrize('hint', ['too_many', 'wrong_container', 'unknown_id'])
def test_invalid_optional_advice_preserves_required_assessments(monkeypatch, hint):
    artifact = _artifact()
    def response(system, prompt, **kwargs):
        ids = [p['passage_id'] for p in json.loads(prompt)['passages']]
        hints = {ids[0]: dict(basis='direct_attribution', claim_token_ranges=[[0, 2]],
                             source_sentence_ids=['s000', 's001', 's002'])}
        if hint == 'wrong_container': hints = []
        if hint == 'unknown_id': hints = {'unknown': hints[ids[0]]}
        return dict(assessments=[dict(passage_id=pid, relevance='relevant', confidence='high',
            evidence_role='source_own_claim_or_finding') for pid in ids], display_observations=hints)
    monkeypatch.setattr(gate, 'chat_completion_json', response)
    result = gate.apply_passage_relevance_gate(artifact)
    assert result.passage_relevance.status == 'complete'
    assert result.passage_relevance.assessments
    assert all(a.display_observation is None for a in result.passage_relevance.assessments)
    assert any('Optional display advice omitted' in s for s in result.passage_relevance.limitations)
    assert result.passages == artifact.passages


@pytest.mark.parametrize('invalid', ['empty', 'duplicate', 'unknown', 'bad_label'])
def test_required_assessment_failures_remain_failures(monkeypatch, invalid):
    def response(system, prompt, **kwargs):
        ids = [p['passage_id'] for p in json.loads(prompt)['passages']]
        rows = [dict(passage_id=pid, relevance='relevant', confidence='high',
                     evidence_role='source_own_claim_or_finding') for pid in ids]
        if invalid == 'empty': rows = []
        if invalid == 'duplicate': rows[-1] = rows[0]
        if invalid == 'unknown': rows[0]['passage_id'] = 'unknown'
        if invalid == 'bad_label': rows[0]['relevance'] = 'invented'
        return dict(assessments=rows, display_observations=[])
    monkeypatch.setattr(gate, 'chat_completion_json', response)
    assert gate.apply_passage_relevance_gate(_artifact()).passage_relevance.status == 'not_assessed'


@pytest.mark.parametrize('opening,closing', [('“','”'), ('"','"')])
def test_compact_excerpt_keeps_multisentence_quotation(opening, closing):
    text = f'She explained: {opening}I left because I died so often. Dying was my best role.{closing} Later she returned home.'
    excerpt = _responsive_display_excerpt(text, 'She said "I died so often".')
    assert 'I died so often.' in excerpt and 'Dying was my best role.' in excerpt
    assert _closed_display_delimiters(excerpt) and excerpt in text


def test_context_is_local_complete_and_bounded():
    before = 'Unrelated background has a long history. ' * 30
    target = 'Wong explained why she left America.'
    text = before + 'The interview happened that year. ' + target + ' She wanted different roles. ' + before
    context = _bounded_display_context(text, target)
    assert target in context and context in text and len(context) <= 1400
    assert context == 'The interview happened that year. ' + target + ' She wanted different roles.'


def test_context_does_not_cut_long_quote_or_expand_unbound_excerpt():
    quote = 'She said: “' + 'This is a long sentence. ' * 80 + '”'
    assert _bounded_display_context(quote, 'This is a long sentence.') == ''
    assert _bounded_display_context('A complete sentence.', 'Invented') == ''


def test_unclosed_quote_is_not_a_compact_excerpt():
    assert _responsive_display_excerpt('She said: “I died so often. More words', 'I died so often') == ''


def test_empty_context_never_restores_full_raw_parent():
    from app.services.evidence_report import _member_evidence_contexts
    primary = dict(text='Primary.', display_text='Primary.', context_text='')
    other = dict(text='Unbounded raw parent. ' * 300, display_text='Extra evidence.', context_text='')
    contexts = _member_evidence_contexts(dict(best_evidence=primary, additional_evidence=[other]))
    assert [c['context_text'] for c in contexts] == ['Extra evidence.']


def test_joint_selection_default_is_off():
    from app.config import Settings
    assert Settings.model_fields['PAPER_EXPERIMENTAL_JOINT_SELECTION_ENABLED'].default is False


def test_expanded_context_does_not_restore_leading_fragment():
    text = 'from the older discussion. Images are produced by industries. Publicity contributes to these images.'
    context = _bounded_display_context(text, 'Images are produced by industries.')
    assert context.startswith('Images are produced')
    assert 'older discussion' not in context


@pytest.mark.parametrize('heading', ['Popular Culture', 'visibility and representation',
                                    'Changing National\nIdentity—\nThe Cultural Image'])
def test_raw_headings_are_not_joined_to_prose(heading):
    from app.services.evidence_report import _readable_display_units
    text = 'Earlier history ended.\n' + heading + '\nThe actor faced limited employment.\nHer opportunities were constrained.'
    units = _readable_display_units(text)
    assert 'The actor faced limited employment.' in units
    context = _bounded_display_context(text, 'The actor faced limited employment.')
    assert context.startswith('The actor') and 'Earlier history' not in context
    assert heading.splitlines()[0] not in context


def test_wrapped_prose_is_not_removed_as_heading():
    from app.services.evidence_report import _readable_display_units
    text = 'The actor in the\nAmerican film industry faced limited work.'
    assert _readable_display_units(text) == ['The actor in the American film industry faced limited work.']


def test_synthesis_note_is_internal_not_report_text():
    from app.services.evidence_report import _passage_view, _display_evidence_note
    p = _passage_view({'excerpt':'The author discussed public culture.'},
                      {'evidence_role':'source_synthesis_or_conclusion'}, claim_text='Public culture')
    assert not p['evidence_note']
    assert not _display_evidence_note({}, {'evidence_note':"This passage states the source author's synthesis or conclusion."})
