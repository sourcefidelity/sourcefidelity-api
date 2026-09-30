"""Prospective abbreviation handling never rewrites authors or old comparisons."""
import pytest
from app.services.reference_discovery import ExpectedBibliographicFields, build_reference_discovery_candidate, _value_hash
from app.services.retrieval.base import RetrievalResult


def candidate(author, observed, *, policy=True, title='Deep learning'):
    expected=ExpectedBibliographicFields(title='Deep learning',authors=[author],year='2015',
        **({'author_normalization_policy_version':'author-etal-comparison-v1'} if policy else {}))
    result=RetrievalResult(source_name='fixture',success=True,title=title,authors=observed,year='2015')
    return expected,build_reference_discovery_candidate(attempt_id='fixture',provider='fixture',expected=expected,result=result)


@pytest.mark.parametrize('author',['Yann LeCun et al.','LeCun, Y., et al.','Yann LeCun et. al.'])
def test_explicit_suffix_is_not_a_surname(author):
    expected,c=candidate(author,['Yann LeCun','Yoshua Bengio','Geoffrey Hinton'])
    comparison=next(x for x in c.comparisons if x.field_name=='author')
    assert comparison.outcome=='minor_difference'
    assert expected.authors==[author]
    assert comparison.expected_sha256==_value_hash(author)


def test_legacy_comparison_and_serialization_unchanged():
    expected,c=candidate('Yann LeCun et al.',['Yann LeCun'],policy=False)
    assert 'author_normalization_policy_version' not in expected.model_dump()
    assert next(x for x in c.comparisons if x.field_name=='author').outcome=='material_conflict'


def test_abbreviation_does_not_invent_author_or_rescue_different_title():
    _,c=candidate('Yann LeCun et al.',['David Silver'])
    assert next(x for x in c.comparisons if x.field_name=='author').outcome=='material_conflict'
    _,c=candidate('Yann LeCun et al.',['Yann LeCun'],title='Botanical classification of tropical trees')
    assert next(x for x in c.comparisons if x.field_name=='title').outcome=='material_conflict'
