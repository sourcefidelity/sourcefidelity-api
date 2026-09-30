from copy import deepcopy
from app.services.evidence_report import _remaining_source_context, _render_more_source_context, _render_member

def rows():
    return [{'passage_id':str(i),'excerpt':f'Context number {i} discusses a different condition.',
             'page_label':str(i+1),'excerpt_truncated':i==4} for i in range(5)]

def test_remaining_context_preserves_package_and_excludes_primary_duplicates():
    source=rows();before=deepcopy(source)
    views=_remaining_source_context(source+[source[3]],source[:3])
    assert [v['passage_id'] for v in views]==['3','4']
    assert source==before
    assert all('does not establish' in v['evidence_note'] for v in views)
    assert views[-1]['locator']=='Page 5'

def test_context_control_is_collapsed_escaped_and_omitted_when_empty():
    source=rows();source[4]['excerpt']='<script>unsafe()</script>'
    html=_render_more_source_context(_remaining_source_context(source,source[:3]))
    assert html.startswith('<details class="more-source-context">')
    assert 'More source context (2)' in html
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert '(p. 5)' in html and 'bounded' in html
    assert _render_more_source_context([])==''

def test_member_renders_separate_context_without_checks_or_models():
    m={'source':{'author':'Example','year':'2024','title':'A study','raw_reference':'Example (2024). A study.'},
       'reference_id':'ref-a','availability':'','best_evidence':None,'additional_evidence':[],
       'more_source_context':_remaining_source_context(rows(),rows()[:3])}
    html=_render_member(m)
    assert 'More source context' not in html and 'data-reference-id="ref-a"' in html
    assert 'Quotation:' not in html
