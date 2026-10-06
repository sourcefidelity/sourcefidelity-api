"""Bounded public item-file discovery; metadata is never source evidence."""
import json
import re
from urllib.parse import quote, urlsplit

from app.services.candidate_budget import require_source_candidate
from app.services.safe_fetch import safe_fetch_bytes
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind

VERSION = 'archive-public-files-v1'


def item_identifier(url: str) -> str | None:
    p = urlsplit(url)
    if p.scheme != 'https' or p.netloc not in {'archive.org', 'www.archive.org'}:
        return None
    match = re.match(r'^/details/([A-Za-z0-9][A-Za-z0-9_.-]{0,99})(?:/|$)', p.path)
    return match[1] if match else None


def public_pdf_locations(url: str, *, timeout: float, max_bytes: int) -> list[AcquisitionLocation]:
    identifier = item_identifier(url)
    if not identifier:
        return []
    endpoint = f'https://archive.org/metadata/{identifier}'
    require_source_candidate(endpoint)
    raw = safe_fetch_bytes(endpoint, usage_label="adapter:internet_archive", max_bytes=2*1024*1024,
                           accept_content_types=('application/json',), timeout=timeout)
    return locations_from_metadata(json.loads(raw), url, max_bytes=max_bytes)


def locations_from_metadata(data, url: str, *, max_bytes: int) -> list[AcquisitionLocation]:
    identifier = item_identifier(url)
    if not identifier or not isinstance(data, dict) or data.get('error'):
        raise ValueError('archive_metadata_unavailable')
    meta = data.get('metadata')
    if not isinstance(meta, dict) or not meta:
        raise ValueError('archive_metadata_unavailable')
    if meta.get('identifier') != identifier:
        raise ValueError('archive_item_identity_mismatch')
    # No lending/login/print-disabled branch, even if a derivative is listed.
    if any(str(meta.get(k, '')).lower() not in {'', 'false', '0', 'none'}
           for k in ('access-restricted-item', 'is_dark', 'noindex')) or data.get('is_dark'):
        raise ValueError('archive_item_restricted')
    files = data.get('files')
    if not isinstance(files, list) or len(files) > 2000:
        raise ValueError('archive_file_inventory_invalid')
    candidates = []
    for f in files:
        if not isinstance(f, dict):
            continue
        name = f.get('name')
        if (not isinstance(name, str) or len(name)>240 or '/' in name or '\\' in name
                or not name.lower().endswith('.pdf') or any(ord(c)<32 for c in name)
                or f.get('private') or f.get('rights') or f.get('format') not in {'Text PDF','Image Container PDF','PDF'}):
            continue
        try:
            size = int(f.get('size', 0))
        except (ValueError, TypeError):
            continue
        if not 0 < size <= max_bytes:
            continue
        candidates.append((f.get('format') != 'Text PDF', name, size))
    return [AcquisitionLocation(
        url=f'https://archive.org/download/{identifier}/{quote(name, safe="")}',
        provider='internet_archive', representation_kind=RepresentationKind.PDF,
        media_type='application/pdf', landing_page_url=url,
        metadata={'discovery_signal': VERSION, 'archive_item_id': identifier,
                  'listed_size': size})
        for _, name, size in sorted(set(candidates))[:2]]


TEXT_IDENTITY_VERSION = 'archive-item-text-v1'
_TEXT_MAX_BYTES = 3 * 1024 * 1024


def _words(value) -> str:
    return ' '.join(re.findall(r'[0-9a-z]+', str(value or '').casefold()))


def item_text_identity(meta: dict, text: str, *, title: str, author: str) -> dict | None:
    """The cited work as an archived item's own text and record name it.

    Owner decision 2026-10-06 (Wagner): a link to one page of a whole archived
    magazine issue names the article only inside the issue, so the item's
    title never matches the cited one. When the cited title appears in the
    item's text, the item's publication date, title and volume are compared
    with the reference like any other record. Identity only: the text is not
    admitted as source evidence.
    """
    cited = _words(title)
    if len(cited) < 12 or cited not in _words(text):
        return None
    from app.services.relevance import extract_surnames
    body = set(_words(text).split())
    surnames = [s for s in extract_surnames(author or '') if s.casefold() in body]
    item_title = str(meta.get('title') or '')
    creator = meta.get('creator')
    creator = creator[0] if isinstance(creator, list) and creator else creator
    issue = meta.get('issue') or (re.search(r'\bNo\.?\s*(\d+)\b', item_title) or [None, None])[1]
    date = str(meta.get('date') or '')
    return {
        'title': title, 'authors': [author] if surnames else [],
        'year': date[:4] if re.match(r'(?:1[5-9]|20)\d{2}', date) else None, 'doi': None,
        'date_method': 'catalog_publication_date',
        'container_title': str(creator or '') if creator and _words(item_title).startswith(_words(creator)) else '',
        'volume': str(meta.get('volume') or ''), 'issue': str(issue or ''),
        'item_title': item_title, 'work_identity_basis': TEXT_IDENTITY_VERSION,
    }


def item_text_observation(url: str, *, title: str, author: str, timeout: float) -> dict | None:
    """Read a public item's record and OCR text and apply item_text_identity."""
    identifier = item_identifier(url)
    if not identifier:
        return None
    endpoint = f'https://archive.org/metadata/{identifier}'
    require_source_candidate(endpoint)
    data = json.loads(safe_fetch_bytes(endpoint, usage_label="adapter:internet_archive", max_bytes=2*1024*1024,
                                       accept_content_types=('application/json',), timeout=timeout))
    meta = data.get('metadata') if isinstance(data, dict) else None
    if (not isinstance(meta, dict) or meta.get('identifier') != identifier or data.get('is_dark')
            or any(str(meta.get(k, '')).lower() not in {'', 'false', '0', 'none'}
                   for k in ('access-restricted-item', 'is_dark', 'noindex'))):
        return None
    names = [f.get('name') for f in data.get('files') or [] if isinstance(f, dict)
             and f.get('format') == 'DjVuTXT' and not f.get('private') and isinstance(f.get('name'), str)
             and '/' not in f['name'] and 0 < int(f.get('size') or 0) <= _TEXT_MAX_BYTES]
    if not names:
        return None
    target = f'https://archive.org/download/{identifier}/{quote(names[0], safe="")}'
    require_source_candidate(target)
    raw = safe_fetch_bytes(target, usage_label="adapter:internet_archive", max_bytes=_TEXT_MAX_BYTES,
                           accept_content_types=('text/plain',), timeout=timeout)
    observed = item_text_identity(meta, raw.decode('utf-8', 'replace'), title=title, author=author)
    if observed:
        import hashlib
        observed['item_text_sha256'] = hashlib.sha256(raw).hexdigest()
    return observed
