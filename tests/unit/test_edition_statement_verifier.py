import hashlib
import pytest
from app.services.edition_statement_verifier import PublicationPage,verify_reprint_statement

def check(text):
    return verify_reprint_statement([PublicationPage(pdf_page_index=4,text=text)],
        representation_sha256='a'*64,submitted_reference_sha256='b'*64,
        cited_year='2004',retrieved_year='2012')

def test_explicit_statement_exact_bound_without_equivalence():
    text='Publisher information\nReprint 2012 of the 2004 edition.\nAll rights reserved.'
    result=check(text)
    assert result.status=='documented_reprint'
    assert text[result.character_start:result.character_end]==result.exact_statement
    assert result.statement_sha256==hashlib.sha256(result.exact_statement.encode()).hexdigest()
    assert not result.whole_work_equivalence and not result.task_usability_granted

@pytest.mark.parametrize('text',[
    'Copyright 2004. Published 2012.',
    'First published 2004. Reprinted 2012.',
    'Reprint 2012 of the 2004 edition.\nRevised text.',
    'Reprint 2012 of the 2004 edition.\nTranslation.',
    'Catalogue\nReprint 2012 of the 2004 edition.',
    'Reprint 2012 of the 2004 edition.\nReprint 2015.',
    'Reprint 2012 of the 2004 edition.\nReprint 2012 of the 2004 edition.',
    'Reprint 2012 of the 2005 edition.',
    'This is not a reprint 2012 of the 2004 edition.',
])
def test_uncertain_or_conflicting_material_abstains(text):
    assert check(text).status=='unresolved'

def test_duplicate_pages_and_invalid_hash_rejected():
    p=PublicationPage(pdf_page_index=0,text='Test')
    with pytest.raises(ValueError):
        verify_reprint_statement([p,p],representation_sha256='a'*64,
            submitted_reference_sha256='b'*64,cited_year='2004',retrieved_year='2012')

def test_normal_library_cataloguing_notice_is_not_a_catalogue_page():
    result=check('A catalogue record for this book is available from the British Library.\nReprint 2012 of the 2004 edition.')
    assert result.status=='documented_reprint'
