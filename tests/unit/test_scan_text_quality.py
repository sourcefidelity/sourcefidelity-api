"""Boilerplate text over a scan must not stand in for readable article text."""
from types import SimpleNamespace
import pytest
from app.services import page_layout

NOTICE = ('Reproduced with permission of the copyright owner.  '
          'Further reproduction prohibited without permission.')

@pytest.mark.parametrize(('text','image','expected'), [
    (NOTICE, True, 'pure_scan'),
    (NOTICE.replace('  ', '\n'), True, 'pure_scan'),
    (NOTICE + '\n125', True, 'pure_scan'),
    (NOTICE + '\n' + 'Substantive article text about project organizations. '*3, True, 'scan_ocr'),
    (NOTICE, False, 'digital'),
    ('The author discusses copyright permissions and reproduction practices. '*3, True, 'scan_ocr'),
])
def test_scan_rights_notice_does_not_count_as_article(monkeypatch,text,image,expected):
    class Page:
        rect=SimpleNamespace(width=100,height=100)
        def get_text(self): return text
        def get_images(self): return [(1,)] if image else []
        def get_image_rects(self,unused): return [self.rect]
    class Document:
        def __len__(self): return 3
        def __getitem__(self,index): return Page()
        def close(self): pass
    monkeypatch.setattr(page_layout.fitz,'open',lambda **kwargs: Document())
    result=page_layout.classify_text_quality(b'fixture')
    assert result.verdict==expected
    assert Page().get_text()==text
