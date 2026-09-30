import copy
import json

import pytest

from app.services import evidence_context_experiment as c
from app.services.llm_input_boundary import LLMInputBudgetExceeded
from app.services.verification_evidence import ClaimEvidence, _SourcePage


def inputs():
    text=''.join(f'The numbered observation {i} describes the development of a public policy. ' for i in range(230))
    return dict(claim=ClaimEvidence(claim_id='c',paper_version_id='p',text='Public policy developed.',claim_type='paraphrase'),
        source_title='Policy',regions=[c.Region(0,0,len(text),text,'body_prose',('original',))],
        pages=[_SourcePage(0,'1',text)],source_binding={'content':'bound'},mode='test')


def bind(request,raw,kw,**overrides):
    args={k:kw[k] for k in ('claim','pages','source_binding','source_title')}
    return c.bind_extended_comparison(request,raw,**dict(args,**overrides))


def test_explicit_larger_input_and_unchanged_default():
    kw=inputs()
    with pytest.raises(LLMInputBudgetExceeded):c.prepare_comparison(**kw)
    with pytest.raises(ValueError):c.prepare_comparison(**kw,max_input_tokens=7000)
    request=c.prepare_extended_comparison(**kw)
    assert 4000<request['estimated_input_tokens']<=7000
    assert request['input_token_ceiling']==7000
    result=bind(json.loads(json.dumps(request)),{'selected':[]},kw)
    assert result['request_sha256']==request['request_sha256']
    assert result['version']==c.EXTENDED_CONTEXT_VERSION
    assert not result['source_support_assessed']
    with pytest.raises(LLMInputBudgetExceeded):bind(request,{'selected':[]},kw,max_input_tokens=4000)
    with pytest.raises(ValueError):c.bind_comparison(request,{'selected':[]},**{k:kw[k] for k in ('claim','pages','source_binding')})


@pytest.mark.parametrize('limit',[0,7001,True,7000.0])
def test_bad_limit(limit):
    with pytest.raises(ValueError):c.prepare_extended_comparison(**inputs(),max_input_tokens=limit)


@pytest.mark.parametrize('field',['prompt','system','source_title_sha256'])
def test_recomputed_hash_does_not_authorize_changed_request(field):
    kw=inputs();request=c.prepare_extended_comparison(**kw)
    request[field]='changed';request['request_sha256']=c.digest({k:v for k,v in request.items() if k!='request_sha256'})
    with pytest.raises(ValueError):bind(request,{'selected':[]},kw)


def test_output_limits_and_source_title_remain_bound():
    kw=inputs();request=c.prepare_extended_comparison(**kw)
    with pytest.raises(ValueError):bind(request,{'selected':[]},kw,source_title='Changed')
    choice=dict(region_id='r000',sentence_ids=['s000'],purpose='primary',context_for=None,why_useful='Policy development.')
    assert bind(request,{'selected':[choice]},kw)['selected'][0]['text'].startswith('The numbered observation 0 ')
    bad=copy.deepcopy(choice);bad['sentence_ids']=[f's{i:03d}' for i in range(30)]
    with pytest.raises(ValueError):bind(request,{'selected':[bad]},kw)
    with pytest.raises(ValueError):bind(request,{'selected':[choice]*4},kw)
