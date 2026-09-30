from types import SimpleNamespace
from app.services.verification_evidence import ClaimEvidence
from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.evidence_obligations import claim_source_attributed_text

class Input(SimpleNamespace):
    def model_copy(self,*,update):return Input(**(vars(self)|update))

def generate(text,marker='(Smith, 2020)',kind='parenthetical'):
    claim=ClaimEvidence(claim_id='scope-case',paper_version_id='scope-paper',text=text,
        granularity='citation_unit',reference_ids=['r'],citation_marker=marker,
        citation_marker_type=kind,passage_start=0,passage_end=len(text))
    return claim,attach_verification_candidates(Input(claim=claim)).verification_candidates

def test_following_independent_claim_is_retained_but_not_source_attributed():
    text='Investment became a government priority (Smith, 2020), and cooperation was effective in stimulating domestic demand.'
    claim,candidates=generate(text)
    assert claim.text==text
    source=[c for c in candidates.candidates if c.relationship_eligible]
    assert source and all('domestic demand' not in c.text for c in source)
    trailing=[c for c in candidates.candidates if c.generation_method=='post_marker_outside_attributed_scope']
    assert trailing and all(c.verification_scope=='not_source_verification' for c in trailing)
    assert all(not c.relationship_eligible for c in trailing)
    assert 'domestic demand' not in claim_source_attributed_text(claim)

def test_final_parenthetical_keeps_both_preceding_clauses():
    _,candidates=generate('Investment became a priority and cooperation increased demand (Smith, 2020).')
    assert any('demand' in c.text for c in candidates.candidates if c.relationship_eligible)
    assert not any(c.generation_method=='post_marker_outside_attributed_scope' for c in candidates.candidates)

def test_narrative_marker_does_not_strip_following_content():
    _,candidates=generate('Smith (2020) reports that investment increased domestic demand.',marker='Smith (2020)',kind='narrative')
    assert any('demand' in c.text for c in candidates.candidates if c.relationship_eligible)
