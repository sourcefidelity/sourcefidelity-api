from test_source_informed_interpretation import fixture
from app.services.source_informed_interpretation import (
    build_source_informed_retrieval, project_source_informed_explanation,
)
from app.services.evidence_package import build_evidence_package


def test_competing_readings_preserve_contrary_evidence_and_original():
    source, artifact, proposal = fixture()
    before = artifact.model_dump_json()
    package = build_evidence_package(artifact).model_dump_json()
    assert any('Other shows retained' in p.text for p in artifact.passages)
    other = proposal.model_copy(update={
        'interpreted_statement': 'Other shows retained their original cultural identity.',
        'unresolved_scope': 'Some shows retained identity; no general conclusion follows.'})
    record = build_source_informed_retrieval(source, artifact, [proposal, other], enabled=True)
    view = project_source_informed_explanation(record, source, artifact)
    assert record.selected_interpretation_id is None
    assert len(view['possible_readings']) == 2
    assert artifact.model_dump_json() == before
    assert build_evidence_package(artifact).model_dump_json() == package
    assert view['original_passage_ids'] == [p.passage_id for p in artifact.passages]


def test_overgeneralized_reading_is_not_promoted_or_hidden():
    source, artifact, proposal = fixture()
    proposal = proposal.model_copy(update={
        'interpreted_statement': 'All shows lost every trace of cultural identity.',
        'wording_difference': 'Deliberately overgeneralized calibration control, not accepted wording.',
        'unresolved_scope': 'All and every are unsupported scope expansions; retain original and contrary evidence.'})
    record = build_source_informed_retrieval(source, artifact, [proposal], enabled=True)
    view = project_source_informed_explanation(record, source, artifact)
    assert 'unsupported scope' in view['possible_readings'][0]['unresolved_scope']
    assert view['original_statement'] == artifact.claim.text
    assert view['accuracy_judgment'] is None
    assert not record.searches[0].accuracy_judgment_allowed
    assert not record.searches[0].source_blind_repair


def test_same_query_control_does_not_manufacture_gain_or_judgment():
    source, artifact, proposal = fixture()
    proposal = proposal.model_copy(update={'interpreted_statement': artifact.claim.text})
    record = build_source_informed_retrieval(source, artifact, [proposal], enabled=True)
    original = {(p.page_index,p.character_start,p.character_end,p.text) for p in artifact.passages}
    additions = {(p.page_index,p.character_start,p.character_end,p.text)
                 for p in record.searches[0].passages} - original
    assert not additions
    assert record.original_accuracy_judgment is None
