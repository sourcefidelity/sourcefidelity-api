"""One sentence-boundary rule for Han, kana and Hangul text, and a dropped space after a closing quote (2026-10-14)."""
from types import SimpleNamespace

from app.services.evidence_translation import sentence_units
from app.services.facet_evidence_judgment import _passage_sentences
from app.services.sentence_splitter import split_sentences

TEXT = "未经批准，任何单位不得开展活动。\n互联网出版机构依法从事活动。 第七条　应当具备条件：“有确定的出版范围。”下一句。"
EXPECTED = ["未经批准，任何单位不得开展活动。", "互联网出版机构依法从事活动。", "第七条　应当具备条件：“有确定的出版范围。”", "下一句。"]


def test_every_splitter_ends_chinese_sentences_whatever_spacing_follows():
    assert split_sentences(TEXT) == EXPECTED
    passage = SimpleNamespace(text=TEXT, passage_id="p1", page_index=0, character_start=0, passage_role="body")
    assert [sentence.text for sentence in _passage_sentences(passage)] == EXPECTED
    units = sentence_units(TEXT)
    assert "".join(units) == TEXT and [unit.strip() for unit in units] == EXPECTED


def test_a_closing_quote_followed_directly_by_a_capital_ends_the_sentence():
    assert split_sentences("for the individual.”Different from before, it changed (Smith, 2016).") == [
        "for the individual.”", "Different from before, it changed (Smith, 2016)."]
    # Ordinary spacing, decimals and abbreviations are unchanged.
    assert split_sentences('He said "Yes." Then he left. Values like 3.5 stay. U.S. policy holds.') == [
        'He said "Yes."', "Then he left.", "Values like 3.5 stay.", "U.S. policy holds."]
