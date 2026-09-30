from app.services.evidence_report import _reportable_bibliographic_difference, _render_continuous_paper
from app.services.web_completeness import extract_complete_article_body, assess_web_completeness
from bs4 import BeautifulSoup


def test_missing_metadata_and_joined_words_are_not_affirmative_conflicts():
    check=lambda a,b,field='title': _reportable_bibliographic_difference(
        dict(field_name=field,submitted_value=a,located_value=b))
    assert not check('A Journal','', 'container_title')
    assert not check('The construction of player identity','Theconstruction ofplayer identity')
    assert not check('Creativity and Technology','CreativityandTechnology.JOURNAL')
    assert not check('Digital histories in games. Journal of History','Digital histories in games')
    assert not check('The growth of digital narratives','The growth of digital narratives. Creative processes')
    assert check('Digital histories in games','Digital histories in schools')
    assert check('2014','2013','year')
    assert check('Journal One','Journal Two','container_title')


def test_gold_area_is_clickable_but_exact_formatting_target_stays_on_top():
    rect=dict(page_index=0,x0=50,y0=90,x1=300,y1=105)
    surface=dict(page_dimensions=[dict(page_index=0,width=600,height=800)],page_href_template='p-{page_index}')
    html,_=_render_continuous_paper(surface,[],[
        dict(finding_type='reference_title_style',rectangles=[rect]),
        dict(finding_type='source_topical_mismatch',rectangles=[rect])])
    soup=BeautifulSoup(html,'html.parser')
    hit=soup.select_one('.topical-reference-hit')
    assert hit and float(hit['width'])==258
    targets=soup.select('.reference-practice-overlay')
    assert targets[0]['data-panel-template']=='reference-panel-2'
    assert targets[1]['data-panel-template']=='reference-panel-1'


def test_saved_journal_reader_body_excludes_collateral_and_checks_complete_coverage():
    paragraphs=[' '.join(['A complete review paragraph about literary characters.']*12),
                ' '.join(['Another paragraph explaining the argument.']*12)]
    body='\n'.join(paragraphs)
    html='<article><header>Book review</header><section id="bodymatter"><div class="core-container">'+''.join(
        '<div>'+p+'</div>' for p in paragraphs)+'</div></section><div class="core-collateral">Metrics and author biography</div></article>'
    assert extract_complete_article_body(html,'')==body
    assert assess_web_completeness(html,body)['verdict']=='complete'
    assert assess_web_completeness(html,paragraphs[0])['verdict']=='not_assessed'
    assert extract_complete_article_body(html+'<link rel="next" href="/next">','fallback')=='fallback'
    assert extract_complete_article_body(html.replace('class="core-container"','class="core-container" hidden'),'fallback')=='fallback'


def test_workspace_entry_snap_holds_one_gesture_then_releases():
    import os, shutil, subprocess
    from pathlib import Path
    import pytest
    node=os.environ.get('SOURCEFIDELITY_TEST_NODE') or shutil.which('node')
    if not node:
        pytest.skip('Node runtime needed for wheel-state regression')
    source=Path('app/services/report_interactions.js').read_text()
    block=source[source.index('  let entryCaught='):source.index("  document.querySelectorAll('[data-report-step]')")]
    harness=r'''
const assert=require('node:assert/strict');
let handler,timer;
const window={scrollY:0,innerHeight:800,scrollTo({top}){this.scrollY=top;}};
const document={getElementById(){return {getBoundingClientRect(){return {top:500-window.scrollY};}};},
 addEventListener(name,fn,options){assert.equal(name,'wheel');assert.equal(options.passive,false);handler=fn;}};
const setTimeout=(fn)=>{timer=fn;return 1;},clearTimeout=()=>{};
function wheel(delta,extra={}){let prevented=false;handler({deltaX:0,deltaY:delta,deltaMode:0,cancelable:true,
 target:{closest(){return null;}},preventDefault(){prevented=true;},...extra});return prevented;}
'''+block+r'''
assert.equal(wheel(700,{ctrlKey:true}),false);
assert.equal(wheel(700),true);assert.equal(window.scrollY,500);
assert.equal(wheel(100),true);assert.equal(window.scrollY,500);
timer();assert.equal(wheel(100),false);
window.scrollY=0;assert.equal(wheel(100),false);
assert.equal(wheel(700),true);assert.equal(window.scrollY,500);
assert.equal(wheel(-100),false);
'''
    subprocess.run([node,'-e',harness],check=True,capture_output=True,text=True)
