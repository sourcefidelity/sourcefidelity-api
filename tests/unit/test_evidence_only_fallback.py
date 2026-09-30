from datetime import datetime,timezone
import hashlib
from dataclasses import replace
from app.services.verification_evidence import AuthorizedRepresentation,ClaimEvidence,build_passage_evidence,attach_candidate_passage_retrieval
from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.evidence_package import build_evidence_package,validate_evidence_package

def fixture():
    content=b'The program is dangerous and cannot reflect on itself. Its behavior has no explanation.'
    now=datetime.now(timezone.utc)
    source=AuthorizedRepresentation(representation_id='synthetic-r',canonical_work_id='synthetic-w',
        content_object_id='synthetic-o',content_sha256=hashlib.sha256(content).hexdigest(),content=content,
        representation_kind='plain_text',media_type='text/plain',provenance='authorized_upload',
        scope_type='personal_owner',scope_id='synthetic-owner',identity_verdict='verified',identity_confidence=.99,
        completeness_verdict='complete',text_quality='digital',edition_or_version=None,created_at=now,admitted_at=now)
    text='The program is dangerous and cannot reflect on itself (Smith, 2020).'
    claim=ClaimEvidence(claim_id='synthetic-c',paper_version_id='synthetic-p',text=text,granularity='citation_unit',
        reference_ids=['ref-1'],citation_marker='(Smith, 2020)',citation_marker_type='parenthetical',passage_start=0,passage_end=len(text))
    artifact=attach_verification_candidates(build_passage_evidence(source,claim=claim))
    candidates=[c.model_copy(update={'relationship_eligible':False,'limitations':['candidate_integrity:materially_redundant_candidate']})
                for c in artifact.verification_candidates.candidates]
    artifact=artifact.model_copy(update={'verification_candidates':artifact.verification_candidates.model_copy(update={'candidates':candidates})})
    return source,artifact

def test_fallback_keeps_rejected_candidates_and_judgments_unchanged():
    source,artifact=fixture();result=attach_candidate_passage_retrieval(source,artifact)
    r=result.candidate_passage_retrieval
    assert r.method=='exact_citation_evidence_only_fallback_v1'
    assert r.evidence_only_passage_ids and not r.selections and r.status=='not_assessed'
    assert r.evidence_only_query_sha256==hashlib.sha256(artifact.claim.text.encode()).hexdigest()
    assert result.verification_candidates==artifact.verification_candidates
    assert result.relationship==artifact.relationship and result.candidate_relationships==artifact.candidate_relationships
    assert {p.passage_id for p in artifact.passages}<={p.passage_id for p in result.passages}
    package=build_evidence_package(result);validate_evidence_package(package)
    assert package.retrieval.evidence_only_passage_ids==r.evidence_only_passage_ids

def test_mismatched_scope_cannot_trigger_fallback():
    source,artifact=fixture()
    result=attach_candidate_passage_retrieval(replace(source,scope_id='other-owner'),artifact)
    assert not result.candidate_passage_retrieval.evidence_only_passage_ids
    assert result.candidate_passage_retrieval.method=='authorized_source_mismatch'

def test_missing_exact_binding_cannot_trigger_fallback():
    source,artifact=fixture()
    result=attach_candidate_passage_retrieval(source,artifact.model_copy(update={'source_binding':None}))
    assert not result.candidate_passage_retrieval.evidence_only_passage_ids
