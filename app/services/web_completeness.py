"""Bounded article-body coverage, independent of source identity or relevance."""
import hashlib
import json
import re
from bs4 import BeautifulSoup


def _article_bodies(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    observations = []
    restricted = False
    body_selectors = []
    def visit(value):
        nonlocal restricted
        if isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, dict):
            kind = value.get('@type', [])
            if isinstance(kind, str): kind = [kind]
            if any(k in {'Article','NewsArticle','BlogPosting','ReportageNewsArticle'} for k in kind):
                restricted |= value.get('isAccessibleForFree') in (False, 'false')
                if isinstance(value.get('articleBody'), str): observations.append(('schema_article_body',value['articleBody']))
                parts=value.get('hasPart',[])
                if isinstance(parts,dict):parts=[parts]
                for part in parts if isinstance(parts,list) else []:
                    selector=part.get('cssSelector') if isinstance(part,dict) else None
                    if isinstance(selector,str) and re.fullmatch(r'[.#][\w-]{1,80}',selector):body_selectors.append(selector)
            if '@graph' in value: visit(value['@graph'])
    for script in soup.find_all('script',type='application/ld+json'):
        try: visit(json.loads(script.string or script.get_text()))
        except (ValueError,TypeError): pass
    selector=', '.join(['[itemprop="articleBody"]', '[data-gu-name="body"]', '.article-body', '.article__body', '.entry-content',*body_selectors])
    for node in soup.select(selector):
        paragraphs = [p.get_text().strip() for p in node.find_all('p') if p.get_text().strip()]
        if paragraphs: observations.append(('explicit_article_body', '\n'.join(paragraphs)))
    # Saved journal reader pages expose body paragraphs as direct divs rather
    # than p elements; collateral author/metrics panels are outside bodymatter.
    reader_bodies = soup.select('article > section#bodymatter > .core-container')
    def explicitly_hidden(node):
        return (node.has_attr('hidden') or node.get('aria-hidden') == 'true'
                or bool(re.search(r'(?:display\s*:\s*none|visibility\s*:\s*hidden)',node.get('style',''),re.I)))
    if len(reader_bodies) == 1 and not any(explicitly_hidden(n) for n in
            [reader_bodies[0], *reader_bodies[0].parents, *reader_bodies[0].find_all(True)]):
        blocks = reader_bodies[0].find_all(['p', 'div'], recursive=False)
        paragraphs = [p.get_text(' ', strip=True) for p in blocks if p.get_text(strip=True)]
        if paragraphs:
            observations.append(('explicit_article_body', '\n'.join(paragraphs)))
    # Some journal templates keep the complete visible text in this sibling
    # container and expose footnotes through widgets. Preserve the notes as a
    # separate section instead of interleaving bibliographic widgets in prose.
    from app.services.source_type import visible_journal_masthead
    if visible_journal_masthead(soup):
        bodies=soup.select('.content-article')
        if len(bodies)==1:
            body=BeautifulSoup(str(bodies[0]),'html.parser')
            notes=[]
            for widget in body.select('cite.footnote'):
                note=widget.select_one('.footnote-text')
                count=widget.select_one('.aside-footnote-count')
                if note is None or count is None:continue
                number=count.get_text(strip=True)
                if not number.isdigit():continue
                notes.append(note.get_text(' ',strip=True))
                widget.replace_with(number)
            text=body.get_text('\n',strip=True)
            if notes:text+='\n\nNotes\n'+'\n'.join(notes)
            observations.append(('explicit_article_body',text))
    return soup, observations, restricted


def extract_complete_article_body(html: str, fallback: str) -> str:
    """Recover omitted paragraphs only from one explicit visible article body.

    Existing fetch, identity and authorization rules remain the caller's job.
    Do not expose hidden metadata-only text or bypass an access prompt.
    """
    _, observations, _ = _article_bodies(html)
    bodies=list(dict.fromkeys(body for method,body in observations if method=='explicit_article_body'))
    if len(bodies)!=1 or len(bodies[0])>500_000:return fallback
    body=bodies[0]
    if assess_web_completeness(html,body)['verdict']!='complete':return fallback
    # Prefer explicit body order over a heuristic extractor's omissions.
    return body if len(body.split()) >= len(fallback.split()) * .8 else fallback


# A link line a site inserts into its own article body ("Related: 10 Things
# Harry Potter Fans Get Wrong…", ScreenRant, 2026-09-30). Extractors rightly
# drop it; it is not the article's text, so it cannot make a page incomplete.
_PROMOTION = re.compile(
    r'^\s*(?:related(?:\s+(?:article|story|reading))?|read\s+(?:more|next|also)|also\s+read|see\s+also|'
    r'more|recommended|you\s+(?:may|might)\s+also\s+like)\s*:', re.IGNORECASE)


def assess_web_completeness(html: str, extracted: str) -> dict:
    soup, observations, restricted = _article_bodies(html)
    normalize=lambda s:' '.join(re.findall(r'\w+',s.casefold()))
    actual=normalize(extracted)
    result={'version':'web-article-coverage-v1','verdict':'not_assessed','reason':'article_body_boundary_unavailable'}
    if soup.select('link[rel="next"]'):
        return {**result,'reason':'restricted_or_paginated_article'}
    for method,body in observations:
        expected=normalize(body)
        if len(expected.split()) < 80: continue
        # Exact normalized full-body containment, not a word-count heuristic.
        if restricted and method != 'explicit_article_body':continue
        if re.search(r'\b(?:subscribe to (?:continue|read)|sign in to (?:continue|read)|remaining article|unlock this article)\b',body,re.I):continue
        if all(normalize(p) in actual for p in body.split('\n') if normalize(p) and not _PROMOTION.match(p)):
            return {**result,'verdict':'complete','reason':'explicit_article_body_fully_extracted','method':method,
                    'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'body_words':len(expected.split()),
                    'html_sha256':hashlib.sha256(html.encode()).hexdigest(),
                    'extracted_sha256':hashlib.sha256(extracted.encode()).hexdigest()}
    return {**result,'reason':'article_body_not_fully_extracted' if observations else result['reason']}


# Plausible lengths of a whole work, in words, by kind (owner decision
# 2026-09-30): a page far longer than an article is not the article alone,
# and a page far shorter than a book cannot be the whole book.
KIND_WORD_RANGES = {
    "journal_article": (1_000, 25_000), "conference_paper": (1_000, 25_000), "book_review": (1_000, 25_000),
    "book_section": (1_500, 30_000),
    "monograph": (25_000, None), "edited_collection": (25_000, None),
    "thesis": (10_000, None), "report": (1_000, 150_000),
    "webpage": (150, 20_000), "news_article": (150, 20_000), "blog_post": (150, 20_000),
}
STATED_PAGE_MIN_WORDS = 1_000
_CUT_OFF = re.compile(
    r"\b(?:continue reading|read the full (?:article|story|essay)|to read (?:more|the rest)|"
    r"(?:subscribe|sign in|log in|register) to (?:continue|read|view|access)|already a subscriber|"
    r"members only|remaining (?:article|content)|unlock this (?:article|story)|preview only|"
    r"pages? \d+(?:\s*[-–]\s*\d+)? (?:is|are) not (?:shown|available))\b", re.IGNORECASE)


def length_fits_kind(words: int, kind: str | None) -> bool | None:
    """Whether a text of this many words can be a whole work of this kind; None when unknown."""
    bounds = KIND_WORD_RANGES.get(str(kind or ""))
    if bounds is None:
        return None
    low, high = bounds
    return words >= low and (high is None or words <= high)


def stated_page_completeness(html: str, text: str, kind: str | None) -> dict:
    """Completeness of a page whose own title, author and year name the cited work.

    Complete only when the page shows no sign of being cut off, has at least
    STATED_PAGE_MIN_WORDS words and its length fits the cited kind of work
    (owner decision 2026-09-30, rule A). The caller establishes the identity.
    """
    words = len(re.findall(r"\w+", text or ""))
    result = {"version": "stated-page-coverage-v1", "verdict": "not_assessed", "words": words}
    soup = BeautifulSoup(html or "", "html.parser")
    if soup.select('link[rel="next"], a[rel="next"]'):
        return {**result, "reason": "paginated_page"}
    # The page outside its menus, sidebars and related-article lists: a
    # "Continue reading" link beside other articles does not cut this one off.
    for chrome in soup.select('nav, aside, header, footer, [role="navigation"], [role="complementary"], '
                              '[class*="sidebar"], [class*="related"], [class*="recommend"], [id*="sidebar"]'):
        chrome.decompose()
    if _CUT_OFF.search(text or "") or _CUT_OFF.search(soup.get_text(" ", strip=True)):
        return {**result, "reason": "cut_off_signal"}
    if words < STATED_PAGE_MIN_WORDS:
        return {**result, "reason": "too_short_for_a_whole_work"}
    fits = length_fits_kind(words, kind)
    if fits is None:
        return {**result, "reason": "kind_unknown"}
    if not fits:
        return {**result, "reason": "length_does_not_fit_kind"}
    return {**result, "verdict": "complete", "reason": "stated_work_whole_page",
            "html_sha256": hashlib.sha256((html or "").encode()).hexdigest()}
