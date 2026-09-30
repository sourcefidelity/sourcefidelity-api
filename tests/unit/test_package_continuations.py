import hashlib,json
from app.services.evidence_package import build_evidence_package,validate_evidence_package,EvidencePackageV1,_continuation_ranges,MAX_PASSAGE_CONTINUATIONS
from test_evidence_only_fallback import fixture

def test_continuations_preserve_exact_offsets_hashes_and_parent():
    source,artifact=fixture();p=artifact.passages[0]
    text=('Background detail. '*95)+'The final result is fully explained here.'+(' More context.'*100)
    p=p.model_copy(update={'text':text,'character_start':20,'character_end':20+len(text)})
    artifact=artifact.model_copy(update={'passages':[p]})
    package=build_evidence_package(artifact);validate_evidence_package(package)
    children=[c for c in package.passages if c.parent_passage_id]
    assert children and len(children)<=MAX_PASSAGE_CONTINUATIONS
    covered=set(range(min(1800,len(text))))
    for c in children:
        assert c.parent_passage_id==p.passage_id
        assert text[c.character_start-20:c.character_end-20]==c.excerpt
        assert hashlib.sha256(c.excerpt.encode()).hexdigest()==c.passage_text_sha256
        assert c.passage_id in package.retrieval.displayed_passage_ids
        covered.update(range(c.character_start-20,c.character_end-20))
    assert covered==set(range(len(text)))
    assert any('The final result is fully explained here.' in c.excerpt for c in children)

def test_continuation_work_is_bounded():
    assert not _continuation_ranges('short')
    assert len(_continuation_ranges('x'*100000))==MAX_PASSAGE_CONTINUATIONS

def test_historical_package_hash_survives_missing_optional_fields():
    _,artifact=fixture();p=build_evidence_package(artifact).model_dump(mode='json')
    # This field did not exist in the historical package being simulated.
    p.pop('alternate_edition',None)
    for row in p['passages']:row.pop('parent_passage_id',None)
    p['retrieval'].pop('evidence_only_passage_ids',None);p['retrieval'].pop('evidence_only_query_sha256',None)
    payload={k:v for k,v in p.items() if k!='package_sha256'}
    p['package_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    validate_evidence_package(EvidencePackageV1.model_validate(p))
