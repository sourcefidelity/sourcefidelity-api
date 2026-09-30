import hashlib
import pytest
from app.services.reference_formatting import ContributionReferenceExpectation, assess_contribution_reference
from app.services.schemas import ParsedReference


@pytest.mark.parametrize('variant,expected', [('same_editor','difference'),('whole','not_assessed'),
    ('unknown','not_assessed'),('changed_source','not_assessed'),('changed_review','not_assessed'),
    ('changed_body','not_assessed'),('changed_reference','not_assessed'),('chapter','not_assessed'),
    ('review','not_assessed'),('mla','not_assessed'),('details_present','not_assessed')])
def test_specific_contribution_does_not_require_different_author(variant,expected):
    h=lambda s:hashlib.sha256(s.encode()).hexdigest()
    body='A claim (Morgan, 2004).'
    ref=ParsedReference(reference_id='one',raw_ref='Morgan, A. (Ed.). (2004). Collection. Press.',
        author='Morgan, A.',source_kind='edited_collection',source_kind_confidence='high')
    evidence=ContributionReferenceExpectation(paper_body_sha256=h(body),reference_id='one',
        reference_text_sha256=h(ref.raw_ref),source_sha256=h('source'),review_record_sha256=h('review'),
        use_scope='specific_contribution',contribution_title='Introduction',contribution_author='Morgan, A.',
        contribution_pages='xiii–xx',basis='owner_review')
    source=h('source');review=h('review');style='apa'
    if variant in {'whole','unknown'}:evidence=evidence.model_copy(update={'use_scope':'whole_collection' if variant=='whole' else 'unknown'})
    if variant=='changed_source':source=h('changed')
    if variant=='changed_review':review=h('changed')
    if variant=='changed_body':body+=' changed'
    if variant=='changed_reference':ref.raw_ref+=' changed'
    if variant=='chapter':ref.source_kind='book_section'
    if variant=='review':ref.needs_review=True
    if variant=='mla':style='mla'
    if variant=='details_present':
        ref.raw_ref+=' Introduction.'
        evidence=evidence.model_copy(update={'reference_text_sha256':h(ref.raw_ref)})
    result=assess_contribution_reference(citation_format=style,body=body,reference=ref,
        expectation=evidence,source_sha256=source,review_record_sha256=review)
    assert result['status']==expected
    assert not result['wrong_author_assessed'] and not result['automatic_findings_enabled']
