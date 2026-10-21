from copy import deepcopy
import pytest
from app.services.report_member_navigation import member_targets, availability_tone


def fixture(text, marker, sources):
    citation={'citation_marker':marker,'members':[
        {'source':{'author':author,'year':year},'coverage_level':coverage}
        for author,year,coverage in sources], 'paper_location':{'rectangles':[
            {'page_index':0,'x0':0,'y0':0,'x1':1000,'y1':30}]}}
    words={0:[(i*40,0,i*40+38,20,word) for i,word in enumerate(text.split())]}
    return citation,words


@pytest.mark.parametrize('text,marker,sources,count',[
    ('A claim (Smith, 2020; Jones, 2021).','(Smith, 2020; Jones, 2021)', [('Smith','2020','full_text'),('Jones','2021','unavailable')],2),
    ('Smith (2020) describes a result.','Smith (2020)', [('Smith','2020','full_text')],1),
    ('Claim (Alhaisoni & Alhaysoy, 2017).','(Alhaisoni & Alhaysoy, 2017)', [('Alhaisoni & Alhaysoy','2017','full_text')],1),
    ('Smith (2020, 2021) describes results.','Smith (2020, 2021)', [('Smith','2020','full_text'),('Smith','2021','abstract_only')],2),
    ('Claim (Modern Screen, 1946; Screenland, 1946).','(Modern Screen, 1946; Screenland, 1946)', [('Modern Screen','1946','unavailable'),('Screenland','1946','unavailable')],2),
    ('Claim (Smith, 2020).','(Smith, 2020)', [('Smith','2020','full_text'),('Smith','2020','unavailable')],0),
    ('In 2020 a claim was made.','(Smith, 2020)', [('Smith','2020','full_text')],0),
])
def test_exact_member_targets(text,marker,sources,count):
    citation,words=fixture(text,marker,sources)
    before=deepcopy(citation)
    result=member_targets(citation,words)
    assert len(result)==count and citation==before
    assert all(r['tone']==availability_tone(citation['members'][r['member_index']]) for r in result)


def test_line_wrapped_marker_and_repeated_occurrence_are_not_guessed():
    citation,words=fixture('Claim (Smith, 2020).','(Smith, 2020)',[('Smith','2020','full_text')])
    words[0][2]=(0,21,38,29,'2020).')
    assert len(member_targets(citation,words))==2
    words[0]+= [(400,0,430,20,'(Smith,'),(440,0,470,20,'2020).')]
    assert member_targets(citation,words)==[]


def test_availability_is_not_relevance_or_correctness():
    assert availability_tone({'coverage_level':'full_text','relevance_status':'not_relevant'})=='evidence_available'


def test_fullwidth_joined_narrative_marker_matches_split_glyph_words():
    citation, words = fixture('In A Film （1931） a claim.', 'A Film（1931）', [('Director','1931','unavailable')])
    citation['members'][0]['source'].update(source_kind='traditional_media',title='A Film [Film]')
    targets=member_targets(citation,words)
    assert len(targets)==1 and targets[0]['x0']==120


@pytest.mark.parametrize('text,marker,start,end',[
    ('Claim (Smith, 2020, p. 25).','(Smith, 2020, p. 25)',1,4),
    ('Smith (2020, p. 25) describes it.','Smith (2020, p. 25)',1,1),
])
def test_parenthetical_includes_name_and_locator_but_narrative_keeps_year(text,marker,start,end):
    citation,words=fixture(text,marker,[('Smith','2020','full_text')])
    targets=member_targets(citation,words)
    assert len(targets)==1
    assert targets[0]['x0']==words[0][start][0]
    assert targets[0]['x1']==words[0][end][2]


def test_parenthetical_sources_with_locators_keep_separate_targets():
    citation,words=fixture('Claim (Smith, 2020, p. 5; Jones, 2021, pp. 6–8).',
        '(Smith, 2020, p. 5; Jones, 2021, pp. 6–8)',
        [('Smith','2020','full_text'),('Jones','2021','abstract_only')])
    targets=member_targets(citation,words)
    assert [(t['member_index'],t['x0'],t['x1']) for t in targets]==[
        (0,words[0][1][0],words[0][4][2]),(1,words[0][5][0],words[0][8][2])]


def test_shared_author_multiple_year_parenthetical_does_not_overlap_sources():
    citation,words=fixture('Claim (Smith, 2020, 2021).','(Smith, 2020, 2021)',
        [('Smith','2020','full_text'),('Smith','2021','unavailable')])
    targets=member_targets(citation,words)
    assert len(targets)==2 and targets[0]['x1']<targets[1]['x0']


def test_same_year_narrative_sources_keep_year_targets():
    citation,words=fixture('Smith (2020) agrees with Jones (2020).','Smith (2020); Jones (2020)',
        [('Smith','2020','full_text'),('Jones','2020','abstract_only')])
    targets=member_targets(citation,words)
    assert [(t['x0'],t['x1']) for t in targets]==[
        (words[0][1][0],words[0][1][2]),(words[0][5][0],words[0][5][2])]


def test_a_narrative_mention_does_not_hide_the_bracketed_marker():
    # Paper 6, 2026-10-04: "Onay (2024) argues … (Onay, 2024)." placed no target.
    citation,words=fixture('Onay (2024) argues a claim (Onay, 2024).','(Onay, 2024)',[('Onay','2024','unavailable')])
    result=member_targets(citation,words)
    assert len(result)==1 and result[0]['x0']==5*40 and result[0]['x1']==6*40+38
