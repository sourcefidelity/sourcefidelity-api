"""Curated official style help, separate from findings and source evidence.

Only recognized findings receive a link, placed at the bottom of the window
that explains them (owner decision 2026-09-25). Unknown styles, evidence
findings and retrieval limitations do not acquire a formatting recommendation.
"""
from html import escape

APA = 'https://apastyle.apa.org/style-grammar-guidelines/'
MLA = 'https://style.mla.org/'
GUIDANCE = {
    'duplicate': ('Distinguishing works by the same author', APA + 'citations/basic-principles/same-year-author', MLA + 'works-with-the-same-title/'),
    'reference': ('Reference-list entries', APA + 'references/basic-principles', MLA + 'works-cited/citations-by-format/'),
    'quotation': ('Quoting and indicating changes', APA + 'citations/quotations', MLA + 'avoid-bracketed-changes/'),
    'indirect': ('Citing an indirect source', APA + 'citations/secondary-sources', MLA + 'paraphrasing-indirect-sources/'),
    'citation': ('Connecting citations to references', APA + 'citations/basic-principles', MLA + 'in-text-citations-overview/'),
    # Titles of standalone works (films, books) are italicized in the text
    # (2026-10-03). No MLA page was verified, so MLA papers get no link.
    'italics': ('Italics', APA + 'italics-quotations/italics', None),
}


def render_guidance(text: str, citation_format: str) -> str:
    style = citation_format.strip().upper()
    if not (style.startswith('APA') or style.startswith('MLA')):
        return ''
    wording = text.casefold()
    kind = None
    if 'same author and year' in wording or 'distinguishing labels' in wording:
        kind = 'duplicate'
    elif 'quotation' in wording and ('difference' in wording or 'correct' in wording):
        kind = 'quotation'
    elif 'another study' in wording or 'indirect citation' in wording:
        kind = 'indirect'
    elif 'hanging indent' in wording or 'reference details' in wording or 'title(s)' in wording:
        kind = 'reference'
    elif 'missing reference' in wording or 'no matching reference' in wording:
        kind = 'citation'
    if kind is None:
        return ''
    label, apa, mla = GUIDANCE[kind]
    name, url = ('APA', apa) if style.startswith('APA') else ('MLA', mla)
    return f' <a class="style-guidance" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{name}: {escape(label)}</a>'


# The style page that explains each finding type. Evidence findings (whether a
# work was located, what its record says) are not style questions and have no
# entry; neither does a finding without a verified matching page.
FINDING_GUIDANCE = {
    'duplicate_citation_key': 'duplicate',
    'duplicate_reference_entry': 'reference',
    'formatting': 'reference',
    'reference_title_style': 'reference',
    'reference_order': 'reference',
    'required_doi_missing': 'reference',
    'required_author_missing': 'reference',
    'reference_identifier_placeholder': 'reference',
    'contribution_author_is_volume_editor': 'reference',
    'chapter_editors_missing': 'reference',
    'reference_publisher_repeated': 'reference',
    'reference_title_missing': 'reference',
    'chapter_pages_missing': 'reference',
    'required_quotation_locator_missing': 'quotation',
    'body_title_style': 'italics',
    'quotation_difference': 'quotation',
    'indirect_source': 'indirect',
    'missing_reference_entry': 'citation',
    'citation_reference_mismatch': 'citation',
}


def guidance_link(kind: str | None, citation_format: str) -> str:
    """One style-guide link for a guidance kind, or '' for an unsupported style."""
    style = str(citation_format or '').strip().upper()
    if kind not in GUIDANCE or not (style.startswith('APA') or style.startswith('MLA')):
        return ''
    label, apa, mla = GUIDANCE[kind]
    name, url = ('APA', apa) if style.startswith('APA') else ('MLA', mla)
    if not url:
        return ''
    return (f'<a class="style-guidance" href="{escape(url, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{name}: {escape(label)}</a>')


def guidance_links(finding_types, citation_format: str) -> str:
    """The de-duplicated style-guide paragraph for a window section."""
    kinds = list(dict.fromkeys(FINDING_GUIDANCE[t] for t in finding_types if t in FINDING_GUIDANCE))
    links = [link for link in (guidance_link(kind, citation_format) for kind in kinds) if link]
    return f'<p class="style-guidance-links">Style guide: {" · ".join(links)}</p>' if links else ''
