from types import SimpleNamespace
from app.services.evidence_package import _display_consolidations

def passage(pid,start,end,page=0):
    return SimpleNamespace(passage_id=pid,page_index=page,character_start=start,character_end=end)

def consolidate(*passages):
    return _display_consolidations([p.passage_id for p in passages],SimpleNamespace(passages=passages))

def test_long_page_cannot_hide_later_retrieved_window():
    ids,groups=consolidate(passage('page',0,4011),passage('ending',635,3179))
    assert ids==['page','ending']
    assert groups=={'page':['page'],'ending':['ending']}

def test_other_source_offset_and_partial_overlap_preserve_visible_text():
    assert consolidate(passage('a',200,3200),passage('b',1300,2500))[0]==['a','b']
    assert consolidate(passage('a',0,900),passage('b',500,1300))[0]==['a','b']

def test_fully_visible_nested_excerpt_still_deduplicates():
    assert consolidate(passage('a',0,1200),passage('b',200,800))==(['a'],{'a':['a','b']})

def test_other_pages_and_larger_reverse_window_stay_separate():
    assert consolidate(passage('a',0,1200),passage('b',200,800,1))[0]==['a','b']
    assert consolidate(passage('a',200,800),passage('b',0,1200))[0]==['a','b']
