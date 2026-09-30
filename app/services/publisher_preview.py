"""Bounded publisher-preview exception; never complete-work admission."""
import hashlib
import re
from urllib.parse import urlsplit

POLICY = 'publisher-preview-limited-v1'


def publisher_preview_url(url):
    try:
        value = urlsplit(url or '')
        return bool(value.scheme == 'https' and value.hostname == 'hkupress.hku.hk'
                    and not value.username and not value.password and value.port in (None,443)
                    and not value.query and not value.fragment
                    and re.fullmatch(r'/image/catalog/pdf-preview/97[89]\d{10}\.pdf', value.path))
    except ValueError:
        return False


def preview_receipt(content, url, *, identity, completeness, source_kind, text_quality, cleanliness):
    if not (publisher_preview_url(url) and identity == 'high' and completeness == 'incomplete'
            and source_kind in {'book', 'monograph'} and text_quality in {'digital', 'scan_ocr'}
            and cleanliness == 'clean'):
        return None
    import fitz
    try:
        with fitz.open(stream=content, filetype='pdf') as doc:
            if doc.is_encrypted or len(doc) < 3:
                return None
            # Exclude a bare cover/catalog leaf. This is readability, not a
            # claim that every supplied page is relevant or complete.
            words = sum(len(p.get_text().split()) for p in doc)
            if words < 500:
                return None
            pages = len(doc)
    except Exception:
        return None
    return dict(policy=POLICY, content_sha256=hashlib.sha256(content).hexdigest(),
                source_url=url, coverage='partial_text', completeness='incomplete',
                identity='high', cleanliness='clean', text_quality=text_quality,
                source_kind=source_kind, page_count=pages)


def valid_preview_receipt(receipt, digest, url):
    return bool(isinstance(receipt, dict) and receipt.get('policy') == POLICY
                and receipt.get('content_sha256') == digest and receipt.get('source_url') == url
                and publisher_preview_url(url) and receipt.get('coverage') == 'partial_text'
                and receipt.get('completeness') == 'incomplete' and receipt.get('identity') == 'high'
                and receipt.get('cleanliness') == 'clean' and receipt.get('source_kind') in {'book','monograph'}
                and receipt.get('text_quality') in {'digital','scan_ocr'})
