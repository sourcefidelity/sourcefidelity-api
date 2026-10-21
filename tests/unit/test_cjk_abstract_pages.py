"""Abstracts from linked Chinese-language abstract pages (cjk-abstract-page-v1, owner decision 2026-10-14)."""
from types import SimpleNamespace

from app.services.abstract_only_pages import LINKED_ABSTRACT_BASIS, cjk_page_abstract
from app.services.retrieval.base import RetrievalResult
from app.services.source_resolver import _attach_linked_abstract, _landing_abstract_holder

ABSTRACT = "《网络出版服务管理规定》将网络出版服务平台经营者纳入许可准入的范围，基本解决了其前身中网络出版主体准入制度因容易被规避而形同虚设的问题。"
LABELLED = f"<html><body><h1>网络出版准入制度研究</h1><div>摘要：{ABSTRACT}</div><p>关键词：网络出版；准入</p></body></html>"
BLOCK = f'<html><body><nav>摘要 全文 标题</nav><div class="summaryCon summaryCon_m" style="x">{ABSTRACT}</div></body></html>'


def test_a_labelled_abstract_and_a_summary_block_are_found():
    assert cjk_page_abstract(LABELLED)["abstract"] == ABSTRACT
    assert cjk_page_abstract(BLOCK)["abstract"] == ABSTRACT
    # A navigation word alone, an empty pop-up page or a short label is not an abstract.
    assert cjk_page_abstract('<script>alert("资源不存在，请联系工作人员");window.close();</script>') is None
    assert cjk_page_abstract("<div>摘要：很短。</div>") is None


def test_an_attached_abstract_becomes_the_abstract_only_fallback():
    named = RetrievalResult(source_name="web_fetch", success=False, title="网络出版准入制度研究")
    _attach_linked_abstract(named, SimpleNamespace(text=LABELLED, content=LABELLED.encode()), None)
    assert named.abstract == ABSTRACT
    assert named.metadata["abstract_provenance"]["basis"] == LINKED_ABSTRACT_BASIS
    holder = _landing_abstract_holder(named)
    assert holder is not None and holder.success and holder.abstract == ABSTRACT
    # An abstract already held is never replaced.
    held = RetrievalResult(source_name="openalex", success=True, abstract="An adapter abstract.",
                           metadata={"abstract_provenance": {"basis": "doi_confirmed_adapter_record"}})
    _attach_linked_abstract(held, SimpleNamespace(text=LABELLED, content=b""), None)
    assert held.abstract == "An adapter abstract."


def test_page_text_is_not_an_abstract_and_a_full_article_keeps_its_text():
    from app.services.abstract_only_pages import is_mainly_abstract
    page = RetrievalResult(source_name="web_fetch", success=False, full_text=("导航 " + ABSTRACT).encode(),
                           abstract="导航 " + ABSTRACT)
    _attach_linked_abstract(page, SimpleNamespace(text=LABELLED, content=LABELLED.encode()), None)
    assert page.abstract == ABSTRACT and page.full_text is None and _landing_abstract_holder(page) is not None
    assert not is_mainly_abstract(ABSTRACT * 20, ABSTRACT)
