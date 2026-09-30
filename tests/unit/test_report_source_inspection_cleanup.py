from app.services.evidence_report import summary_text
from app.services.evidence_report import _render_panel_template


def test_homepage_summary_does_not_invent_a_network_response():
    from app.services.evidence_report import _build_report_summary
    finding=dict(reference_id='r',finding_type='submitted_link_issue',
        link_outcome='website_level_address',rectangles=[dict(page_index=0,x0=1,y0=1,x1=2,y1=2)])
    result=_build_report_summary(citations=[],overview={},pervasive_hanging_indent=False,
        reference_practice=[finding])
    message=' '.join(map(summary_text, result['academic_practice']))
    assert 'website address rather than a page' in message
    assert 'returned missing' not in message


def test_redundant_notices_and_abstract_action_are_omitted():
    member = dict(reference_id='r', coverage_level='abstract_only',
        source=dict(raw_reference='Writer (2020). A source.', title='A source', author='Writer', year='2020'),
        availability='Source completeness is uncertain. Check the source manually.',
        reference_identity=dict(edition_year_unresolved=True),
        source_action=dict(enabled=True, href='/report/test/source/id', label='Open abstract'),
        best_evidence=dict(text='The complete available abstract.'))
    html = _render_panel_template(dict(members=[member]), 1)
    assert 'The complete available abstract.' in html
    assert 'Source completeness is uncertain' not in html
    assert 'year differs' not in html
    assert 'Open abstract' not in html
    member['coverage_level'] = 'partial_text'
    member['source_action']['label'] = 'Open available text'
    assert 'Open available text' in _render_panel_template(dict(members=[member]), 1)


def test_standalone_bibliographic_note_not_primary_but_protected_evidence_survives():
    from app.services.evidence_report import _eligible_display_passages
    note = dict(passage_id='note', excerpt='330. See A. Writer, Article Title (2019), https://example.org/article.')
    prose = dict(passage_id='prose', excerpt='Platforms control entry and competition through their terms.')
    assert _eligible_display_passages([note, prose], None) == [prose]
    assert _eligible_display_passages([note], None, preferred_passage_ids=['note']) == [note]
    explanatory = dict(passage_id='explanatory', excerpt='330. These rules were applied differently because local conditions varied.')
    assert _eligible_display_passages([explanatory], None) == [explanatory]


def test_bibliographic_note_titles_and_quoted_descriptions_are_not_own_prose():
    from app.services.evidence_report import _eligible_display_passages
    note=dict(passage_id='note',excerpt='330. See A. Writer, The Opinion, 7 Journal 117 (2019) '
        '(“The decision suggests another approach”); B. Writer, America Has a Problem (2018), https://example.org/article.')
    assert _eligible_display_passages([note],None)==[]
    assert _eligible_display_passages([note],None,preferred_passage_ids=['note'])==[note]
    explanatory={**note,'excerpt':note['excerpt']+' This shows why different rules were applied.'}
    assert _eligible_display_passages([explanatory],None)==[explanatory]


def test_transient_cleanup_does_not_disable_verified_public_source_access():
    from app.services.evidence_report import enable_authenticated_paper_actions
    m=dict(coverage_level='partial_text',source={},verification_report_id='evidence',
        source_navigation=dict(status='ready',representation_id='verification-run:run'),
        source_action=dict(enabled=True,status='verified_public_source_available',
                           href='https://example.org/article.pdf',label='Open available text'))
    view=dict(citations=[dict(members=[m])],paper_surface={})
    out=enable_authenticated_paper_actions(view,report_id='report')
    assert out['citations'][0]['members'][0]['source_action']['href']=='https://example.org/article.pdf'
    m.pop('source_action')
    assert not enable_authenticated_paper_actions(view,report_id='report')['citations'][0]['members'][0]['source_action']['enabled']


def test_uncertain_identity_does_not_disable_independently_retained_candidate_access():
    from app.services.evidence_report import enable_authenticated_paper_actions
    m=dict(coverage_level='partial_text',source={},verification_report_id='evidence',
        source_navigation=dict(status='ready',representation_id='verification-run:run'),
        source_action=dict(enabled=True,status='public_candidate_available',
                           href='https://example.org/preview.pdf',label='Open retrieved candidate'))
    view=dict(citations=[dict(members=[m])],paper_surface={})
    action=enable_authenticated_paper_actions(view,report_id='report')['citations'][0]['members'][0]['source_action']
    assert action['enabled'] and action['label']=='Open retrieved candidate'
    assert action['status']=='public_candidate_available'
