"""Complete catalog dispositions, not shared-author existence heuristics."""
import pytest
from app.services.reference_credibility import assess_reference_credibility
from app.services.reference_discovery import build_reference_discovery_candidate, ExpectedBibliographicFields
from app.services.retrieval.base import RetrievalResult
from app.services.reference_review_scope import text_key
from test_bounded_reference_review import review_fixture, kinds
import hashlib


def catalog_fixture():
    ref,t=review_fixture();ref.source_kind='monograph';t['expected']['source_kind']='monograph'
    result=RetrievalResult(source_name='google_books',success=True,title='Volcanic sediment dynamics',
        authors=['A. River'],year='2002',metadata={'book_edition_metadata':dict(volume_id='catalog-one',
            record_sha256='a'*64,published_date='2002')})
    c=build_reference_discovery_candidate(attempt_id='openalex',provider='google_books',
        expected=ExpectedBibliographicFields(**t['expected']),result=result,
        acquisition_outcome='identity_rejected',disposition_reason_code='edition_metadata_only')
    t['candidates']=[c.model_dump(mode='json')]
    t['attempts'][1].update(provider='google_books',outcome='candidate_found')
    query=text_key(f'intitle:{ref.title} inauthor:{ref.author.split(",",1)[0]}')
    t['queries'][1].update(provider='google_books',execution_provider='google_books',
        execution_outcome='results',result_count=1,normalized_query=query,
        query_sha256=hashlib.sha256(query.encode()).hexdigest())
    return ref,t


def test_all_outside_scope_books_complete_catalog_review_coverage():
    ref,t=catalog_fixture()
    assert kinds(assess_reference_credibility(ref,None,t))==['potentially_fabricated_reference']


@pytest.mark.parametrize('mutation',['missing_count','missing_candidate','extra_count','missing_catalog_receipt',
                                  'same_title','failed_query','old_policy'])
def test_incomplete_catalog_and_plausible_books_remain_protected(mutation):
    ref,t=catalog_fixture()
    if mutation=='missing_count':t['queries'][1]['result_count']=None
    if mutation=='missing_candidate':t['candidates']=[]
    if mutation=='extra_count':t['queries'][1]['result_count']=2
    if mutation=='missing_catalog_receipt':t['candidates'][0]['edition_metadata']=None
    if mutation=='same_title':t['candidates'][0]['observed']['title']=ref.title
    if mutation=='failed_query':t['queries'][1]['execution_outcome']='timeout'
    if mutation=='old_policy':t['bounded_review_policy_version']='bounded-reference-review-v4'
    assert not assess_reference_credibility(ref,None,t)['findings']
