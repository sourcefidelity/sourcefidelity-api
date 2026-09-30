import json
import fitz
from app.services.evidence_report import _responsive_display_excerpt, _prioritize_display_passages, _normalize_source_display_text, _render_member
from app.services.web_completeness import assess_web_completeness, extract_complete_article_body
from app.services.verification_evidence import _pdf_reading_order_text_and_spans


def test_bracketed_printed_pagination_overrides_conflicting_embedded_labels():
    from app.services.verification_evidence import _SourcePage,_SourceStructuralSpan,_resolved_pdf_page_labels
    pages=[_SourcePage(index=i,label=str(i-3),text=f'[{i+1}]') for i in [11,12]]
    spans={p.index:(_SourceStructuralSpan(start=0,end=len(p.text),role='page_furniture'),) for p in pages}
    assert _resolved_pdf_page_labels(pages,spans)[12]=='13'


def test_distinctive_claim_beats_repeated_name_and_title():
    claim='The performer combined exoticism on screen with modernity off screen.'
    passage='On screen she was exotically presented; off screen she was a modern performer. The performer appeared in many films and was a film fan.'
    assert _responsive_display_excerpt(passage,claim).startswith('On screen')


def test_responsive_sentence_not_window_word_accumulation():
    claim='The artist only recently received recognition after years of neglect.'
    rows=[{'passage_id':'direct','excerpt':'The artist recently received recognition after years of neglect. Other career details follow.', 'boundary_status':'sentence_complete'},
          {'passage_id':'diffuse','excerpt':'The artist had a long career. Recognition helped other artists. Several years brought challenges.', 'boundary_status':'sentence_complete'}]
    assert _prioritize_display_passages(rows,claim)[0]['passage_id']=='direct'


def test_source_spelling_corroboration_keeps_ambiguous_compounds():
    assert _normalize_source_display_text('mys-\ntery off-screen','mystery appeared again')=='mystery off-screen'
    assert _normalize_source_display_text('English-\nspeaking','speaking English')=='English-speaking'


def test_duplicate_link_removed_and_candidate_reserve_not_rendered():
    member={'source':{'author':'A','year':'2024','title':'Title','raw_reference':'A (2024). Title.', 'url':'https://example.org/article'},'reference_id':'r',
            'source_action':{'enabled':True,'href':'https://example.org/article','label':'Open cited source link'},
            'more_source_context':[{'text':'private reserve'}]}
    html=_render_member(member)
    assert 'Open cited source link' not in html and 'More source context' not in html
    assert 'https://example.org/article' in html


def page(body, extra=''):
    return '<html><head>'+extra+'</head><body><div itemprop="articleBody">'+body+'</div></body></html>'


def test_the_window_no_longer_shows_relevance_gate_passages():
    member={'source':{'author':'A','year':'2024','title':'Title','raw_reference':'Reference'},
            'best_evidence':{'text':'primary','context_text':'Primary longer context'},
            'additional_evidence':[
                {'text':'first raw','display_text':'first shortened','context_text':'First longer context'},
                {'text':'Second full passage','display_text':'second shortened'},
                {'text':'Third omitted passage'}]}
    html=_render_member(member)
    # Owner decision 2026-09-28: GLM-selected sentences are the window's one evidence source.
    assert 'Additional evidence and context' not in html and '<blockquote' not in html
    assert not any(text in html for text in ['primary','Primary longer context','First longer context'])


def test_retrieval_headings_and_media_notice_order_for_saved_reports():
    from app.services.evidence_report import _render_panel_template
    # Owner-approved label (2026-09-28): "<judgment> - <evidence kind>".
    labels={'full_text':'Full Text Retrieved','partial_text':'Limited Text Retrieved',
            'abstract_only':'Abstract Retrieved','unavailable':'No Text Retrieved'}
    for coverage,label in labels.items():
        member={'source':{'author':'A','year':'2024','title':'Title','raw_reference':'Reference'},
                'coverage_level':coverage,
                'availability':'This is a media reference. Automated source-text checks are not available.',
                'limitations':['A retained limitation.']}
        html=_render_panel_template({'members':[member],'student_text':'Citation'},1)
        assert f'kind-{coverage}">{label}</span></span></h3>' in html
        assert 'Automated source retrieval and checks are not available.' in html
        assert 'source-text checks' not in html
        assert html.index('This is a media reference.') < html.index('class="full-reference"')
        assert 'Citation Information' not in html


def test_web_complete_requires_every_body_paragraph_and_recovers_missing_one():
    one='The study describes the setting and its participants. '*12
    two='The final paragraph discusses the conclusions and limitations.'
    html=page('<p>'+one+'</p><p>'+two+'</p>')
    assert assess_web_completeness(html,one)['verdict']=='not_assessed'
    recovered=extract_complete_article_body(html,one)
    assert two in recovered and assess_web_completeness(html,recovered)['verdict']=='complete'
    assert assess_web_completeness(page('<p>'+one+'</p>','<link rel="next" href="/2">'),one)['verdict']=='not_assessed'


def test_web_drop_cap_and_access_prompt():
    text='Widely known in the community, the artist developed a long career. '*12
    html=page('<p><span>W</span>'+text[1:]+'</p>')
    assert assess_web_completeness(html,text)['verdict']=='complete'
    blocked=page('<p>'+text+'</p><p>Subscribe to continue reading.</p>')
    assert assess_web_completeness(blocked,text+'Subscribe to continue reading.')['verdict']=='not_assessed'


def test_four_columns_are_not_interleaved_by_vertical_position():
    with fitz.open() as doc:
        p=doc.new_page(width=800,height=1200)
        for col,x in enumerate([80,245,410,575]):
            for row in range(4):
                p.insert_textbox(fitz.Rect(x,200+row*120,x+140,300+row*120),f'Column {col} paragraph {row}. '+('An independent account explains the event. '*4),fontsize=9)
        text,_,rebuilt=_pdf_reading_order_text_and_spans(p,0)
        assert rebuilt
        assert text.index('Column 0 paragraph 3')<text.index('Column 1 paragraph 0')
