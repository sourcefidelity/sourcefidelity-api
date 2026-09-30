from app.services.source_type import classify_html_source_kind
from app.services.web_source_metadata import extract_web_source_metadata


HTML = '''<title>An article</title><meta name="author" content="Example Review">
<meta property="og:site_name" content="Example Review">
<script type="application/ld+json">{"@type":"BlogPosting"}</script>
<div>Vol. 12 No. 4 Article</div><h1>An article</h1>
<article><h3 class="single-content-author">A. Writer*</h3><p>Body prose.</p></article>'''


def test_visible_journal_masthead_and_byline_override_generic_site_metadata():
    assert classify_html_source_kind(HTML, 'https://example.org/article').kind == 'journal_article'
    assert extract_web_source_metadata(HTML, 'https://example.org/article')['authors'] == ['A. Writer']


def test_blog_and_explicit_citation_author_are_not_overridden():
    blog = HTML.replace('Vol. 12 No. 4 Article', 'Blog')
    assert classify_html_source_kind(blog, 'https://example.org/article').kind == 'blog_post'
    specific = HTML + '<meta name="citation_author" content="B. Contributor">'
    assert extract_web_source_metadata(specific, 'https://example.org/article')['authors'] == ['B. Contributor']


def test_journal_site_suffix_is_not_part_of_work_title():
    html=HTML.replace('<title>An article</title>', '<meta property="og:title" content="An article - Example Review">')
    assert extract_web_source_metadata(html,'https://example.org/article')['title']=='An article'


def test_explicit_visible_journal_body_keeps_notes_separate():
    from app.services.web_completeness import extract_complete_article_body, assess_web_completeness
    body=('These regulations affect market entry and competition across several sectors. '*12)
    html=HTML+'<div class="content-article"><h2>Introduction</h2><p>'+body+'''<cite class="footnote">
    <span class="footnote-text"><span class="aside-footnote-count">1</span>Writer (2020). A source.</span>
    <button>Close</button></cite></p><h2>Conclusion</h2><p>Concluding discussion.</p></div>'''
    extracted=extract_complete_article_body(html,'short fallback')
    assert 'Notes\n1 Writer (2020)' in extracted and 'Close' not in extracted
    assert assess_web_completeness(html,extracted)['verdict']=='complete'
