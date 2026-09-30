from copy import deepcopy
from app.services.evidence_report import _prioritize_display_passages

def test_truncated_parent_does_not_receive_complete_excerpt_preference():
    rows=[{'passage_id':'cut','excerpt':'Markets develop through invest',
           'boundary_status':'sentence_complete','retrieval_score':1.0},
          {'passage_id':'whole','excerpt':'Markets develop.',
           'boundary_status':'sentence_complete','retrieval_score':0.1}]
    before=deepcopy(rows)
    ranked=_prioritize_display_passages(rows,'Markets develop')
    assert ranked[0]['passage_id']=='whole'
    assert rows==before

def test_conservative_parent_status_and_explicit_preference_preserved():
    rows=[{'passage_id':'fragment','excerpt':'Markets develop through investment.',
           'boundary_status':'bounded_fragment_or_nonprose','retrieval_score':1.0},
          {'passage_id':'whole','excerpt':'Markets develop.',
           'boundary_status':'sentence_complete','retrieval_score':0.1}]
    assert _prioritize_display_passages(rows,'Markets develop')[0]['passage_id']=='whole'
    assert _prioritize_display_passages(rows,'Markets develop through investment',preferred_passage_ids=['fragment'])[0]['passage_id']=='fragment'

def test_empty_excerpt_cannot_inherit_complete_parent_status():
    rows=[{'passage_id':'empty','excerpt':'','boundary_status':'sentence_complete'},
          {'passage_id':'complete','excerpt':'Markets develop.','boundary_status':'sentence_complete'}]
    assert _prioritize_display_passages(rows,'')[0]['passage_id']=='complete'
