import pytest
from app.services.source_type import classify_reference_source_kind, document_kind_for_source_kind, compare_source_kinds, SourceKindAssessment


@pytest.mark.parametrize('reference', [
    'Smith, J. (Ed.). (2023). Collected studies. Example Press.',
    'Smith, J., & Jones, A. (Eds.). (2023). Collected studies. Example Press.',
    'Smith, John, editor. Collected Studies. Example Press, 2023.',
])
def test_explicit_lead_editor_is_collection_not_chapter(reference):
    kind = classify_reference_source_kind(reference)
    assert kind.kind == 'edited_collection' and kind.confidence == 'high'
    assert document_kind_for_source_kind(kind.kind) == 'book'
    assert compare_source_kinds(kind, SourceKindAssessment('book_section','high')).verdict != 'compatible'


@pytest.mark.parametrize('reference', [
    'Smith, J. (2023). A contribution. In A. Jones (Ed.), Collected studies (pp. 10–25). Example Press.',
    'Smith, John. “A Contribution.” Collected Studies, edited by Alice Jones, Example Press, 2023, pp. 10–25.',
    'Smith, John. “A Contribution.” From Collected Studies. Edited by Alice Jones. Example Press: 2023. Pages 10-25.',
])
def test_distinct_contribution_remains_chapter(reference):
    kind = classify_reference_source_kind(reference)
    assert kind.kind == 'book_section' and kind.confidence == 'high'
    assert document_kind_for_source_kind(kind.kind) == 'chapter'


@pytest.mark.parametrize('reference', [
    'Beowulf. Translated by Alan Sullivan and Timothy Murphy, edited by Sarah Anderson, Pearson, 2004.',
    'Smith, J. (2023). Chapter 4: A history of measurement. Example Press.',
    'Smith, John. A Complete Work. Edited by Alice Jones. Example Press, 2023.',
    'Smith, J. (2023). A complete work (A. Jones (Ed.)). Example Press.',
])
def test_editor_credit_or_chapter_word_alone_does_not_establish_component(reference):
    assert classify_reference_source_kind(reference).kind not in {'book_section','edited_collection'}


def test_role_in_url_or_title_does_not_make_collection_or_chapter():
    r = 'Smith, J. (2023). The chapter 4 debate. Example Press. https://example.org/edited-by/chapter-2'
    assert classify_reference_source_kind(r, title='The chapter 4 debate').kind == 'monograph'


def test_component_cue_inside_bound_title_is_not_a_container():
    title = 'Reading In A. Jones (Ed.) as a publishing convention'
    r = 'Smith, J. (2023). '+title+'. Example Press.'
    assert classify_reference_source_kind(r, title=title).kind == 'monograph'


def test_conflicting_editor_lead_and_chapter_container_abstains():
    r = 'Smith, J. (Ed.). (2023). A contribution. In A. Jones (Ed.), A collection (pp. 10–25). Example Press.'
    assert classify_reference_source_kind(r).kind == 'unknown'
