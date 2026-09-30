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
