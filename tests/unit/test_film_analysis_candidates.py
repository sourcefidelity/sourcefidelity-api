import hashlib

from app.services.schemas import SubjectIdentification
from app.services.subject_identifier import bind_film_analysis_candidates, identify_subject


def test_exact_candidate_binding_does_not_establish_semantic_quality_or_absence():
    body = 'Opening. Example Film uses fragmented editing to show uncertainty. Closing.'
    passage = 'Example Film uses fragmented editing to show uncertainty.'
    result = bind_film_analysis_candidates([dict(title='Example Film', passage=passage,
        proposed_role='substantive_analysis')], body, [])
    assert result.status == 'bound_candidates'
    c = result.candidates[0]
    assert body[c.title_start:c.title_end] == c.title
    assert body[c.passage_start:c.passage_end] == passage
    assert c.passage_sha256 == hashlib.sha256(passage.encode()).hexdigest()
    assert result.automatic_findings_enabled is False


def test_invalid_ambiguous_and_injected_candidates_fail_closed():
    base = dict(title='Film', passage='Film is mentioned.', proposed_role='incidental_mention')
    for raw, body in [([base], 'Film is mentioned. Film is mentioned.'),
                      ([base], 'Different text'),
                      ([dict(base, proposed_role='missing_reference')], base['passage']),
                      ([dict(base, reference_absent=True)], base['passage']),
                      ([dict(base, title='')], base['passage']),
                      ([base]*13, base['passage']), (None, base['passage'])]:
        r = bind_film_analysis_candidates(raw, body, [])
        assert r.status == 'invalid'
        assert not r.candidates
        assert not r.automatic_findings_enabled


def test_uncertainty_and_empty_results_are_not_omissions():
    r = bind_film_analysis_candidates([dict(title='Film', passage='Film is mentioned.',
        proposed_role='uncertain')], 'Film is mentioned.', [])
    assert r.candidates[0].proposed_role == 'uncertain'
    assert bind_film_analysis_candidates([], 'body', []).status == 'no_candidates'
    assert SubjectIdentification().film_analysis_preflight.status == 'not_requested'
    assert identify_subject('', [], collect_film_candidates=True).film_analysis_preflight.status == 'unavailable'


def test_opt_in_uses_existing_single_call_and_preserves_default(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    body='Example Film uses fragmented editing to show uncertainty.'
    def reply(**kwargs):
        calls.append(kwargs)
        return {'primary_subject': 'film: Example Film', 'film_analysis_candidates': [
            dict(title='Example Film', passage=body, proposed_role='substantive_analysis')]}
    monkeypatch.setattr('app.services.subject_identifier.chat_completion_json',reply)
    monkeypatch.setattr('app.services.providers.get_provider_config',
                        lambda: SimpleNamespace(input_batch_tokens=10000))
    assert identify_subject(body,[]).film_analysis_preflight.status == 'not_requested'
    assert 'Additional development-only output' not in calls[0]['system_prompt']
    result=identify_subject(body,[],collect_film_candidates=True)
    assert len(calls)==2
    assert result.film_analysis_preflight.status == 'bound_candidates'
    assert 'Additional development-only output' in calls[1]['system_prompt']
