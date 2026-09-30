import copy
import json
from dataclasses import replace

import pytest

from app.services.evidence_context_experiment import (
    Region, prepare_multipage_extract, bind_multipage_extract, prepare_multipage_transition,
    _closed_extract_delimiters,
)
from app.services.verification_evidence import _SourcePage, _SourceStructuralSpan, _PdfLayoutSpan


def fixture():
    texts = ["The account describes a journey into the ", "2\nworld and its consequences."]
    pages = [_SourcePage(0, "1", texts[0]), _SourcePage(1, "2", texts[1],
             (_SourceStructuralSpan(0, 2, "page_furniture"),))]
    parts = [Region(0, 0, len(texts[0]), texts[0], "body_prose", ("a",)),
             Region(1, 2, len(texts[1]), texts[1][2:], "body_prose", ("b",))]
    def box(page, start, end, y):
        return _PdfLayoutSpan(page, start, end, texts[page][start:end], 20, y, 400, y+12, 600, 800)
    layout = {0: [box(0, 0, len(texts[0]), 700)],
              1: [box(1, 0, 2, 760), box(1, 2, len(texts[1]), 80)]}
    return parts, dict(inspected_regions=copy.deepcopy(parts), pages=pages,
                       layout_by_page=layout, source_binding={"content": "hash", "scope": "personal"})


def test_exact_parts_and_omissions_survive_json_transport():
    parts, kw = fixture()
    before = copy.deepcopy((parts, kw))
    record = prepare_multipage_extract(parts, **kw)
    assert record["text"] == parts[0].text + "\n" + parts[1].text
    assert "page_index" not in record and "start" not in record
    assert [p["page_index"] for p in record["parts"]] == [0, 1]
    for p in record["parts"]:
        assert record["text"][p["display_start"]:p["display_end"]] == p["text"]
    assert record["omitted"][0]["start"] == 0 and record["omitted"][0]["end"] == 2
    assert not record["semantic_acceptance"] and not record["source_support_assessed"]
    assert bind_multipage_extract(json.loads(json.dumps(record)), **kw) == record
    assert (parts, kw) == before


@pytest.mark.parametrize("change", ["scope", "content", "page", "geometry", "role", "reservoir"])
def test_stale_context_rejected(change):
    parts, kw = fixture()
    record = prepare_multipage_extract(parts, **kw)
    if change in ("scope", "content"):
        kw["source_binding"][change] = "changed"
    elif change == "page":
        kw["pages"][0] = replace(kw["pages"][0], text=kw["pages"][0].text + "extra")
    elif change == "geometry":
        kw["layout_by_page"][0][0] = replace(kw["layout_by_page"][0][0], y0=699)
    elif change == "role":
        kw["pages"][1] = replace(kw["pages"][1], structural_spans=())
    else:
        kw["inspected_regions"] = []
    with pytest.raises(ValueError):
        bind_multipage_extract(record, **kw)


@pytest.mark.parametrize("change", ["text", "omission", "offset", "separator", "fingerprint"])
def test_record_tampering_rejected(change):
    parts, kw = fixture()
    record = prepare_multipage_extract(parts, **kw)
    if change == "text": record["text"] += "Invented."
    elif change == "omission": record["omitted"] = []
    elif change == "offset": record["parts"][1]["display_start"] -= 1
    elif change == "separator": record["separators"][0]["text"] = " "
    else: record["fingerprint"] = "wrong"
    with pytest.raises(ValueError):
        bind_multipage_extract(record, **kw)


@pytest.mark.parametrize("change", ["reversed", "duplicate", "notes", "uninspected", "missing_layout",
    "unverified_header", "central_header", "wrong_text", "columns", "nonfinite", "fragment", "hyphen"])
def test_unsafe_join_rejected(change):
    parts, kw = fixture()
    if change == "reversed": parts.reverse()
    elif change == "duplicate": parts[1] = parts[0]
    elif change == "notes": parts[0] = replace(parts[0], role="citation_notes")
    elif change == "uninspected": kw["inspected_regions"] = [parts[1]]
    elif change == "missing_layout": kw["layout_by_page"] = {}
    elif change == "unverified_header": kw["pages"][1] = replace(kw["pages"][1], structural_spans=())
    elif change == "central_header": kw["layout_by_page"][1][0] = replace(kw["layout_by_page"][1][0], y0=300, y1=312)
    elif change == "wrong_text": kw["layout_by_page"][0][0] = replace(kw["layout_by_page"][0][0], text="Wrong")
    elif change == "columns":
        s = kw["layout_by_page"][0][0]
        mid = 20
        kw["layout_by_page"][0] = [replace(s, end=mid, text=s.text[:mid], x1=200),
            replace(s, start=mid, text=s.text[mid:], x0=300, y0=80, y1=92)]
    elif change == "nonfinite": kw["layout_by_page"][0][0] = replace(kw["layout_by_page"][0][0], x0=float("nan"))
    else:
        old = parts[0]
        text = old.text[:-1] + ("-" if change == "hyphen" else ".")
        parts[0] = replace(old, text=text)
        kw["pages"][0] = replace(kw["pages"][0], text=text)
        kw["inspected_regions"][0] = parts[0]
        kw["layout_by_page"][0][0] = replace(kw["layout_by_page"][0][0], text=text)
    with pytest.raises(ValueError):
        prepare_multipage_extract(parts, **kw)


@pytest.mark.parametrize("text,expected", [
    ('The hero comes back from this [.', False),
    ('A hero “ventures forth and returns.', False),
    ('The hero comes back from this [. . .] adventure.', True),
    ('A hero “returns” (1993: 30).', True),
    ('The source says "unfinished.', False),
    ('A stray closing bracket].', False),
])
def test_incomplete_ellipsis_and_quotation_not_complete_extracts(text, expected):
    assert _closed_extract_delimiters(text) is expected


def test_transition_reuses_exact_spans_without_extending_inspected_text():
    parts, kw = fixture()
    record = prepare_multipage_transition(*parts, **kw)
    for p in record['parts']:
        page = next(page for page in kw['pages'] if page.index == p['page_index'])
        assert p['text'] == page.text[p['start']:p['end']]
    assert bind_multipage_extract(record, **kw) == record
    assert record['omitted'][0]['reason'] == 'whitespace_or_verified_page_furniture'


def test_transition_crosses_spaced_ellipsis_without_truncating_quotation():
    parts, kw = fixture()
    text = '2\nworld with [. . .] altered conditions (2001: 9). Another statement.'
    kw['pages'][1] = replace(kw['pages'][1], text=text)
    parts[1] = replace(parts[1], end=len(text), text=text[2:])
    kw['inspected_regions'][1] = parts[1]
    kw['layout_by_page'][1][1] = replace(kw['layout_by_page'][1][1], end=len(text), text=text[2:])
    record = prepare_multipage_transition(*parts, **kw)
    assert record['text'].endswith('altered conditions (2001: 9).')
    assert 'Another statement' not in record['text']
    assert bind_multipage_extract(record, **kw) == record


def test_transition_does_not_extend_beyond_inspected_parent():
    parts, kw = fixture()
    text = '2\nworld with [. . .] altered conditions (2001: 9).'
    kw['pages'][1] = replace(kw['pages'][1], text=text)
    end = text.index('.') + 1
    parts[1] = replace(parts[1], end=end, text=text[2:end])
    kw['inspected_regions'][1] = parts[1]
    kw['layout_by_page'][1][1] = replace(kw['layout_by_page'][1][1], end=len(text), text=text[2:])
    with pytest.raises(ValueError, match='multipage_extract_boundary'):
        prepare_multipage_transition(*parts, **kw)


def test_overlong_exact_continuation_remains_rejected():
    parts, kw = fixture()
    text = '2\nworld ' + 'continues ' * 150 + 'onward.'
    kw['pages'][1] = replace(kw['pages'][1], text=text)
    parts[1] = replace(parts[1], end=len(text), text=text[2:])
    kw['inspected_regions'][1] = parts[1]
    kw['layout_by_page'][1][1] = replace(kw['layout_by_page'][1][1], end=len(text), text=text[2:])
    with pytest.raises(ValueError, match='multipage_extract_boundary'):
        prepare_multipage_transition(*parts, **kw)
