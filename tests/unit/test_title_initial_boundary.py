from app.services.ref_field_extractor import _apa_title


def test_personal_initials_do_not_truncate_book_title():
    for title in ('The life of Pearl S. Buck and her work', 'Memo from David O. Selznick'):
        assert _apa_title(title+'. University Press.')==title


def test_ordinary_title_publisher_boundary_is_preserved():
    assert _apa_title('A complete book title. University Press.')=='A complete book title'
