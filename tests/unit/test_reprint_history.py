import hashlib
import pytest
from app.services.edition_statement_verifier import PublicationPage, verify_reprint_history


def check(text, index=2):
    return verify_reprint_history([PublicationPage(pdf_page_index=index, text=text)],
        representation_sha256='a'*64, submitted_reference_sha256='b'*64,
        cited_year='2000', retrieved_year='2002')


PAGE = '© Publisher 2000\nFirst published 2000\nReprinted 2002\nISBN 1 234 56789 0'


def test_adjacent_history_is_bound_but_not_equivalence():
    r = check(PAGE)
    assert r.status == 'documented_reprint'
    assert r.version == 'publication-history-v2'
    assert PAGE[r.character_start:r.character_end] == r.exact_statement
    assert hashlib.sha256(r.exact_statement.encode()).hexdigest() == r.statement_sha256
    assert not r.whole_work_equivalence and not r.task_usability_granted
    assert r.indexed_input_sha256 != check(PAGE, index=3).indexed_input_sha256


@pytest.mark.parametrize('text', [
    PAGE + '\nSecond edition 2005',
    PAGE + '\nPaperback edition 2003',
    PAGE + '\nFirst published 1998',
    PAGE + '\nReprinted 2004',
    PAGE + '\nTransferred to digital printing 2006',
    PAGE + '\nRevised edition',
    PAGE + '\nThis translation © 2000',
    PAGE + '\nTranslated by Someone',
    PAGE + '\nBibliography',
    PAGE + '\n' + PAGE,
    PAGE.replace('Reprinted 2002', 'Reprinted 2002, 2003'),
    PAGE.replace('Reprinted 2002', 'Reprinted with corrections 2002'),
    PAGE.replace('First published 2000', 'First published 1999'),
    PAGE.replace('First published 2000', 'Not first published 2000'),
    PAGE.replace('© Publisher 2000\n', '').replace('ISBN', 'Identifier'),
    PAGE.replace('First published 2000\n', 'First published 2000. '),
])
def test_ambiguous_or_insufficient_history_abstains(text):
    assert check(text).status == 'unresolved'


def test_subject_matter_is_not_an_edition_change():
    r = check(PAGE + '\nTranslation Studies discusses revised models of language.')
    assert r.status == 'documented_reprint'


def test_history_cannot_be_stitched_across_pages():
    r = verify_reprint_history([
        PublicationPage(pdf_page_index=2, text='© 2000 ISBN 123\nFirst published 2000'),
        PublicationPage(pdf_page_index=3, text='Reprinted 2002')],
        representation_sha256='a'*64, submitted_reference_sha256='b'*64,
        cited_year='2000', retrieved_year='2002')
    assert r.status == 'unresolved'
