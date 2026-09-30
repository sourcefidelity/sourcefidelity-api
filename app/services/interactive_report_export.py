"""Bounded, read-only offline derivative of an authorized report bundle."""
import base64
import hashlib
import json
import ipaddress
import secrets
from copy import deepcopy
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qsl

import fitz
from bs4 import BeautifulSoup

from app.services.evidence_report import render_evidence_report_html, project_reference_flags
from app.services.report_member_navigation import marker_words


def _public_report_href(href):
    """Portable links must not carry local capabilities or credential queries."""
    try:
        parsed=urlsplit(str(href or ''))
        host=parsed.hostname or ''
        public=bool(host) and host!='localhost' and not host.endswith(('.localhost','.local','.internal'))
        try: public=ipaddress.ip_address(host).is_global
        except ValueError: pass
        sensitive=any(k.casefold() in {'token','access_token','api_key','key','password','session'}
                      for k,_ in parse_qsl(parsed.query))
        return parsed.scheme in {'http','https'} and public and not parsed.username and not sensitive
    except ValueError:
        return False


def build_interactive_report_html(view: dict, paper_content: bytes) -> bytes:
    """Caller must use the scoped bundle loader."""
    result = deepcopy(view)
    digest = hashlib.sha256(paper_content).hexdigest()
    if result.get('paper_surface', {}).get('presentation_sha256') != digest:
        raise ValueError('Portable paper hash mismatch')
    with fitz.open(stream=paper_content,filetype='pdf') as document:
        if len(document) > 100 or len(paper_content) > 30*1024*1024:
            raise ValueError('Portable report exceeds bounded export size')
        result = project_reference_flags(result, document, digest)
        surface = result['paper_surface']
        surface['selectable_words']={p.number:p.get_text('words',sort=True) for p in document}
        surface['marker_words']=marker_words(document)
        surface['page_dimensions']=[{'page_index':p.number,'width':p.rect.width,'height':p.rect.height} for p in document]
        surface['page_href_template']='offline-page-{page_index}'
        images={};image_bytes=0
        for p in document:
            scale=min(1.3,2000/max(p.rect.width,p.rect.height))
            encoded=base64.b64encode(p.get_pixmap(matrix=fitz.Matrix(scale,scale)).tobytes('png')).decode()
            image_bytes+=len(encoded)
            if image_bytes>60*1024*1024:raise ValueError('Portable page images exceed export size')
            images[f'offline-page-{p.number}']='data:image/png;base64,'+encoded
    # No Judgment layout offline: a forwarded file can reach someone who never
    # accepted its notice (ARCHITECTURE §7).
    result['portable_export']=True
    # Remove server operations recursively before serialization, not just visually.
    def clean(value):
        if isinstance(value,dict):
            result={k:clean(v) for k,v in value.items() if not str(k).endswith('_action') and k not in {'storage_key','token','session_token'}}
            action=value.get('source_action') or {}
            if (action.get('status') in {'verified_public_source_available','public_candidate_available'} and action.get('enabled')
                    and _public_report_href(action.get('href'))):
                result['source_action']={k:action[k] for k in ('status','enabled','href','label') if k in action}
            return result
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    result=clean(result)
    result['export_action']={}
    nonce=secrets.token_urlsafe(24)
    soup=BeautifulSoup(render_evidence_report_html(result,csp_nonce=nonce),'html.parser')
    if not soup.find(id='select-text'):
        select=soup.new_tag('button',id='select-text');select['type']='button';select['aria-pressed']='false';select.string='Select text'
        soup.select_one('.paper-toolbar').append(select)
    for node in soup.select('image'):
        href=node.get('href','')
        if href not in images:raise ValueError('Unbound portable image')
        node['href']=images[href]
    for node in soup.select('form, .technical-export'):
        node.decompose()
    judgment=soup.find(id='judgment-layout')
    if judgment:judgment.decompose()
    for node in soup.find_all(True):
        for key in list(node.attrs):
            if key.startswith('data-') and any(s in key for s in ('href','url')):
                del node.attrs[key]
        if node.name=='a':
            href=node.get('href','')
            if href.startswith('#'):continue
            if _public_report_href(href):
                node['rel']='noopener noreferrer';node['target']='_blank'
                node['title']='Requires an internet connection'
            else:
                node.unwrap()
    notice=soup.new_tag('p')
    notice.string=('Read-only report snapshot • '+datetime.now(timezone.utc).strftime('%Y-%m-%d UTC')+
        ' • Report '+str(view.get('report_id') or '')+' • Version '+str(view.get('report_version') or 'retained snapshot')+
        '. This file does not update. Anyone receiving it can read and forward its contents; downloaded copies cannot be revoked.')
    soup.body.insert(0,notice)
    meta=soup.new_tag('meta');meta['http-equiv']='Content-Security-Policy'
    meta['content']=f"default-src 'none'; img-src data:; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"
    soup.head.insert(0,meta)
    for key,value in {'paper-sha256':digest,'report-snapshot-sha256':hashlib.sha256(json.dumps(view,sort_keys=True,default=str).encode()).hexdigest()}.items():
        binding=soup.new_tag('meta');binding['name']=key;binding['content']=value;soup.head.append(binding)
    content=str(soup).encode()
    if len(content)>80*1024*1024:raise ValueError('Portable report exceeds bounded export size')
    return content
