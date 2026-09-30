import fitz
import pytest
from app.services.source_validator import _late_book_title_identity

TITLE = 'Understanding Narrative Form'

def book(title_page=4, author='Alice Morgan', year='2004', extra=''):
    with fitz.open() as doc:
        for i in range(7):
            p=doc.new_page()
            if i==title_page:
                p.insert_text((70,100),TITLE,fontsize=24)
                p.insert_text((70,170),author,fontsize=12)
                if extra:p.insert_textbox(fitz.Rect(70,220,500,700),extra,fontsize=10)
            if i==title_page+1:
                p.insert_text((70,100),'Copyright '+year,fontsize=12)
        return doc.tobytes()

def check(data):
    return _late_book_title_identity(data,TITLE,'Morgan, A.','2004')

def test_late_title_with_author_and_adjacent_date():
    assert check(book())


def test_edited_collection_reuses_guarded_late_title_identity():
    from app.services.source_validator import validate_retrieved_pdf
    result=validate_retrieved_pdf(book(),expected_title=TITLE,
        expected_author='Morgan, A.',expected_year='2004',is_article=False,
        expected_source_kind='edited_collection',skip_completeness=True,
        skip_text_quality=True)  # Deliberately blank synthetic opening pages.
    assert result.identity_confidence=='high'

@pytest.mark.parametrize('kwargs',[
    {'title_page':6}, {'author':'Bob Other'}, {'year':'2005'},
    {'extra':'This is a book review.'},
    {'extra':'Discussion and bibliography. '*100},
])
def test_wrong_or_out_of_bounds_context_is_not_promoted(kwargs):
    assert not check(book(**kwargs))

def test_reissue_is_not_silently_promoted():
    assert not check(book(year='2004; reprinted 2012'))

@pytest.mark.parametrize('kind',['journal_article','book_section','unknown'])
def test_nonbooks_do_not_invoke_fallback(monkeypatch,kind):
    from app.services import source_validator as validator
    def forbidden(*args):
        raise AssertionError('book fallback invoked for nonbook')
    monkeypatch.setattr(validator,'_late_book_title_identity',forbidden)
    validator.validate_retrieved_pdf(book(),expected_title=TITLE,
        expected_author='Morgan, A.',expected_year='2004',expected_source_kind=kind)
