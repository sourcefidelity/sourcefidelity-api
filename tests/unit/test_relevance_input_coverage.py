"""Fixed-batch input-coverage regressions; no provider/network calls."""

from app.config import settings
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from app.services import passage_relevance as gate
from app.services import llm_service
from app.services.llm_input_boundary import LLMInputBudgetExceeded
from app.services.verification_evidence import CoverageLevel, ObligationPassageRelevanceEvidence


def _text(size):
    return 'Evidence ' + 'x' * (size - 10) + '.'


def _plain(row):
    return re.sub(r'\[s\d+\] ', '', row['text'])


@pytest.fixture
def runner(monkeypatch):
    calls = []
    monkeypatch.setattr(gate.settings, 'VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS', 4000)
    monkeypatch.setattr(gate.settings, 'VERIFICATION_JUDGMENT_MAX_OUTPUT_TOKENS', 1600)
    monkeypatch.setattr(gate, 'get_provider_config', lambda _: SimpleNamespace(input_batch_tokens=1500))

    def reply(system, prompt, **kwargs):
        payload = json.loads(prompt)
        calls.append((system, payload, kwargs, prompt))
        return {'assessments': [dict(passage_id=p['passage_id'], relevance='relevant',
                                    confidence='high', evidence_role='source_own_claim_or_finding',
                                    rationale='Addresses the claim.') for p in payload['passages']]}

    monkeypatch.setattr(gate, 'chat_completion_json', reply)

    def run(texts, context='', target='Evidence addresses a claim.', source_title=''):
        passages = [SimpleNamespace(passage_id=f'p{i}', text=text, page_label=str(i + 1),
                                    passage_role='body') for i, text in enumerate(texts)]
        artifact = SimpleNamespace(
            claim=SimpleNamespace(text=target, claim_id='claim', antecedent_context=(
                [SimpleNamespace(text=context, context_index=0)] if context else [])),
            coverage=SimpleNamespace(level=CoverageLevel.FULL_TEXT))
        result = gate._assess_relevance_target(artifact, passages, target_text=target, obligation=None,
                                                source_title=source_title)
        return result, passages

    return run, calls, reply


def test_source_title_is_orientation_not_target(runner):
    run, calls, _ = runner
    result, _ = run(['The ending returns the protagonists to society.'],
                   target='The ending restores social order.',
                   source_title='A reading of Dracula', context='The film is Dracula.')
    assert result.status == 'complete'
    system, payload, _, _ = calls[0]
    assert payload['submitted_source_title'] == 'A reading of Dracula'
    assert payload['student_context'][0]['text'] == 'The film is Dracula.'
    assert 'Dracula' not in payload['source_attributed_text']
    assert 'orientation only' in system


def test_compact_ordinary_instructions_retain_more_whole_text(monkeypatch, runner):
    run, calls, _ = runner
    texts = [_text(2700)] * 3
    current, _ = run(texts)
    assert current.status == 'complete'
    assert all(not a.assessment_input_truncated for a in current.assessments)
    current_ids = [p['passage_id'] for p in calls[0][1]['passages']]
    calls.clear()
    monkeypatch.setattr(gate, '_system_prompt', lambda _: gate._SYSTEM_PROMPT)
    previous, _ = run(texts)
    assert previous.status == 'complete'
    assert any(a.assessment_input_truncated for a in previous.assessments)
    assert [p['passage_id'] for p in calls[0][1]['passages']] == current_ids
    assert current.batch_count == previous.batch_count == 1


def test_ordinary_inspection_keeps_adverse_evidence_and_topic_boundary(runner):
    run, calls, _ = runner
    result, _ = run(['Investment requires approval.'])
    assert result.status == 'complete'
    prompt = ' '.join(calls[0][0].split())
    assert 'Explicit constraints, exceptions or counterexamples' in prompt
    assert 'Do not require the passage to establish' in prompt
    assert 'Sharing actors or a broad topic alone is not relevant' in prompt
    assert 'Unclear wording remains unresolved' in prompt
    assert 'do not output a support verdict' in prompt
    assert 'Do not substitute a neighboring phenomenon' in prompt
    assert 'do not reject the audience evidence for failing to prove the whole compound claim' in prompt


def test_compact_sentence_inventory_keeps_inline_ids_and_exact_offsets(runner):
    result, _ = runner[0](['Evidence explains a claim. Another sentence adds context.'])
    row = runner[1][0][1]['passages'][0]
    assert row['source_sentences'] == ['s000', 's001']
    assert row['text'] == '[s000] Evidence explains a claim. [s001] Another sentence adds context.'
    assert result.assessments[0].assessed_text_offset_start == 0
    assert result.assessments[0].assessed_text_offset_end == len(_plain(row))


def test_abstract_keeps_separate_existing_protocol(monkeypatch):
    from tests.unit.test_passage_relevance import _artifact
    calls = []
    def reply(system, prompt, **kwargs):
        calls.append(system)
        data = json.loads(prompt)
        return {'assessments': [dict(passage_id=p['passage_id'], relevance='not_relevant',
            confidence='high', evidence_role='unclear') for p in data['passages']]}
    monkeypatch.setattr(gate, 'chat_completion_json', reply)
    gate.assess_abstract_relevance(_artifact().claim, 'An abstract about licensing.')
    assert calls == [gate._SYSTEM_PROMPT
                     + gate._scope_prompt(settings.ABSTRACT_SCOPE_POLICY_VERSION)]


def _source_capacity(monkeypatch, capacity):
    """Deterministic fit boundary, in addition to the real complete-prompt check."""
    original = gate.enforce_complete_prompt_budget

    def check(system, prompt, **kwargs):
        assert prompt.endswith(llm_service.JSON_REPAIR_SUFFIX * 2)
        data = json.loads(prompt[:-len(llm_service.JSON_REPAIR_SUFFIX * 2)])
        if sum(len(_plain(p)) for p in data['passages']) > capacity:
            raise LLMInputBudgetExceeded('Synthetic source capacity')
        return original(system, prompt, **kwargs)

    monkeypatch.setattr(gate, 'enforce_complete_prompt_budget', check)


@pytest.mark.parametrize('capacity,whole', [(5100, [True, True, True]),
                                           (4800, [True, True, False]),
                                           (4200, [False, False, False])])
def test_fixed_batch_maximizes_whole_count_then_original_order(monkeypatch, runner, capacity, whole):
    _source_capacity(monkeypatch, capacity)
    run, calls, _ = runner
    result, passages = run([_text(1700)] * 3)
    assert result.status == 'complete' and result.batch_count == len(calls) == 1
    assert [not a.assessment_input_truncated for a in result.assessments] == whole
    assert [p['passage_id'] for p in calls[0][1]['passages']] == [p.passage_id for p in passages]
    for a, row, passage in zip(result.assessments, calls[0][1]['passages'], passages, strict=True):
        excerpt = passage.text[a.assessed_text_offset_start:a.assessed_text_offset_end]
        assert excerpt == _plain(row)
        assert a.assessed_text_sha256 == hashlib.sha256(excerpt.encode()).hexdigest()
        assert a.model_input_text_sha256 == hashlib.sha256(row['text'].encode()).hexdigest()
        assert calls[0][2]['max_tokens'] == 1600 and calls[0][2]['max_retries'] == 2
        assert calls[0][2]['disable_thinking'] is True


def test_character_coverage_breaks_whole_count_tie(monkeypatch, runner):
    _source_capacity(monkeypatch, 4600)
    run, calls, _ = runner
    result, _ = run([_text(1600), _text(1700), _text(1800)])
    assert result.status == 'complete'
    assert [not a.assessment_input_truncated for a in result.assessments] == [False, False, True]
    assert sum(len(_plain(p)) for p in calls[0][1]['passages']) == 4600


def test_oversized_whole_is_not_redacted_or_labelled(monkeypatch, runner):
    run, calls, _ = runner
    original = gate.redact_direct_identifiers
    lengths = []

    def redact(text):
        lengths.append(len(text))
        assert len(text) <= 16000
        return original(text)

    monkeypatch.setattr(gate, 'redact_direct_identifiers', redact)
    result, _ = run([_text(17000)])
    assert result.status == 'complete' and len(calls) == 1
    assert result.assessments[0].assessment_input_truncated
    assert max(lengths) == 1400


def test_variant_probes_do_not_multiply_redaction_counts(monkeypatch, runner):
    _source_capacity(monkeypatch, 4800)
    monkeypatch.setattr(gate, 'redact_direct_identifiers',
                        lambda text: SimpleNamespace(text=text, redaction_counts={'probe': 1}))
    result, _ = runner[0]([_text(1700)] * 3)
    assert result.status == 'complete'
    assert result.direct_identifier_redactions == {'probe': 5}  # claim, target, three selected inputs


def test_exact_budget_boundary_includes_retry_reserve(monkeypatch, runner):
    from app.services.llm_input_boundary import estimate_prompt_tokens

    run, calls, _ = runner
    result, _ = run(['Short evidence.'])
    assert result.status == 'complete'
    system, _, _, prompt = calls.pop()
    exact = estimate_prompt_tokens(system, prompt + llm_service.JSON_REPAIR_SUFFIX * 2)
    monkeypatch.setattr(gate.settings, 'VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS', exact)
    assert run(['Short evidence.'])[0].status == 'complete'
    calls.clear()
    monkeypatch.setattr(gate.settings, 'VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS', exact - 1)
    assert run(['Short evidence.'])[0].status == 'not_assessed'
    assert not calls


@pytest.mark.parametrize('ids', [['s999'], ['s001', 's000'], ['s000', 's000']])
def test_invalid_selected_sentence_ids_omit_only_observation(monkeypatch, runner, ids):
    run, _, reply = runner

    def response(*args, **kwargs):
        result = reply(*args, **kwargs)
        result['display_observations'] = {'p0': dict(basis='direct_attribution',
            claim_token_ranges=[[0, 3]], source_sentence_ids=ids)}
        return result

    monkeypatch.setattr(gate, 'chat_completion_json', response)
    result, _ = run(['Evidence addresses a claim. More evidence follows.'])
    assert result.status == 'complete' and result.assessments[0].display_observation is None


def test_fallback_rebuilds_sentence_map_and_binds_late_source_offsets(monkeypatch, runner):
    _source_capacity(monkeypatch, 1400)
    run, calls, reply = runner

    def response(*args, **kwargs):
        result = reply(*args, **kwargs)
        result['display_observations'] = {'p0': dict(basis='direct_attribution',
            claim_token_ranges=[[0, 3]], source_sentence_ids=['s000'])}
        return result

    monkeypatch.setattr(gate, 'chat_completion_json', response)
    text = 'Unrelated history. ' * 150 + 'Evidence addresses a claim. Additional details follow.'
    result, _ = run([text])
    assessment = result.assessments[0]
    assert result.status == 'complete' and assessment.assessed_text_offset_start > 0
    sent = _plain(calls[0][1]['passages'][0])
    assert sent == text[assessment.assessed_text_offset_start:assessment.assessed_text_offset_end]
    assert assessment.display_observation.source_span == gate.split_sentences(sent)[0]


def test_masked_and_original_input_hashes_remain_separate(runner):
    result, _ = runner[0](['Evidence addresses a claim. Contact reviewer@example.edu.'])
    assessment = result.assessments[0]
    assert result.status == 'complete'
    assert assessment.assessed_text_sha256 != assessment.model_input_text_sha256
    assert 'reviewer@example.edu' not in runner[1][0][3]


@pytest.mark.parametrize('context', ['', 'Long context ' * 2000])
def test_impossible_plan_fails_before_any_calls(monkeypatch, runner, context):
    _source_capacity(monkeypatch, 4100)
    run, calls, _ = runner
    result, _ = run(['Short evidence.'] * 3 + [_text(1700)] * 3, context=context)
    assert result.status == 'not_assessed' and result.batch_count == 0
    assert result.method == 'passage_relevance_prompt_budget_exceeded'
    assert not result.assessments and not calls


@pytest.mark.parametrize('provider_boundary,batch_sizes', [(1500, [3] * 6), (4000, [6] * 3)])
def test_eighteen_candidates_keep_existing_batches_and_schema(monkeypatch, runner, provider_boundary, batch_sizes):
    monkeypatch.setattr(gate, 'get_provider_config', lambda _: SimpleNamespace(input_batch_tokens=provider_boundary))
    run, calls, _ = runner
    result, _ = run(['Short evidence.'] * 18)
    assert result.status == 'complete'
    assert [len(c[1]['passages']) for c in calls] == batch_sizes
    assert [a.passage_id for a in result.assessments] == [f'p{i}' for i in range(18)]
    assert result.batch_count <= 6
    assert ObligationPassageRelevanceEvidence.model_validate_json(result.model_dump_json()) == result


def test_long_optional_display_span_is_omitted_without_losing_relevance(monkeypatch, runner):
    run, calls, reply = runner

    def response(*args, **kwargs):
        result = reply(*args, **kwargs)
        result['display_observations'] = {'p0': dict(basis='direct_attribution',
            claim_token_ranges=[[0, 3]], source_sentence_ids=['s000'])}
        return result

    monkeypatch.setattr(gate, 'chat_completion_json', response)
    result, _ = run([_text(1700)])
    assert result.status == 'complete' and len(calls) == 1
    assert not result.assessments[0].assessment_input_truncated
    assert result.assessments[0].display_observation is None


def test_later_response_failure_keeps_progress_but_no_partial_assessments(monkeypatch, runner):
    run, calls, reply = runner

    def response(*args, **kwargs):
        if calls:
            raise llm_service.LLMCallFailure('empty_response')
        return reply(*args, **kwargs)

    monkeypatch.setattr(gate, 'chat_completion_json', response)
    result, _ = run(['Short evidence.'] * 6)
    assert result.status == 'not_assessed' and result.batch_count == 1
    assert not result.assessments and not result.relevant_passage_ids


def test_shared_suffix_preserves_wrapper_retry_prompts(monkeypatch):
    prompts = []
    responses = iter(['not JSON', 'still not JSON', '{"ok":true}'])

    def completion(**kwargs):
        prompts.append(kwargs['user_prompt'])
        return next(responses)

    monkeypatch.setattr(llm_service, 'chat_completion', completion)
    assert llm_service.chat_completion_json('system', '{}', max_retries=2) == {'ok': True}
    assert prompts == ['{}' + llm_service.JSON_REPAIR_SUFFIX * n for n in range(3)]
    assert len(llm_service.JSON_REPAIR_SUFFIX) == 65
