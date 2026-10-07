"""The rendered-report checks (owner request 2026-10-07) on small invented reports."""
from app.services.report_check import check_expectations, check_report


def _report(entry_html: str, extra: str = "") -> str:
    return (f'<html><body>{extra}<template id="reference-entry-panel-1"><h2>Reference 1</h2>'
            f'<p class="full-reference reference-window-entry">{entry_html}</p></template></body></html>')


def _checks(html, view=None):
    return [v["check"] for v in check_report(html, view or {})]


def test_a_whole_link_passes_and_a_split_one_fails():
    whole = 'Ames, R. (2020). Title. <a href="https://example.test/a?b=1&amp;c=2">https://example.test/a?b=1&amp;c=2</a>'
    assert _checks(_report(whole)) == []
    split = 'Ames, R. (2020). Title. <a href="https://example.test/a?b=1&amp;c_so">https://example.test/a?b=1&amp;c_so</a> urce=app'
    assert _checks(_report(split)) == ["split_link"]
    glued = 'Ames. <a href="https://example.test/a">https://example.test/a</a>_rest_of_slug'
    assert _checks(_report(glued)) == ["split_link"]
    # Ordinary words after a link are not part of it.
    words = 'Ames. <a href="https://example.test/a">https://example.test/a</a> Retrieved May 5'
    assert _checks(_report(words)) == []


def test_an_address_left_as_plain_text_in_the_entry_fails():
    assert _checks(_report("Ames, R. (2020). Title. https://example.test/a")) == ["unlinked_address"]


def test_a_window_named_but_missing_fails():
    html = _report("Ames.", extra='<button data-go-to="citation-panel-4">Citation 4</button>')
    assert _checks(html) == ["missing_window"]


def test_a_retrieval_phrase_title_and_a_hidden_finding_fail():
    view = {"bibliography": [{"source": {"title": "Retrieved from:", "raw_reference": "Show. (n.d.) Retrieved from: https://x"}}],
            "reference_practice": [{"reference_id": "r1", "finding_type": "chapter_editors_missing", "finding": "x",
                                    "rectangles": []},
                                   {"reference_id": "r2", "finding_type": "reference_title_style", "finding": "y",
                                    "rectangles": [{"page_index": 0}]}]}
    assert sorted(_checks(_report("Ames."), view)) == ["hidden_finding", "retrieval_phrase_title"]


def test_owner_confirmed_expectations_are_checked_per_window():
    html = _report("Ames, R. (2020). Title.")
    assert check_expectations(html, [{"window": "reference-entry-panel-1", "contains": "Ames, R."}]) == []
    [missing] = check_expectations(html, [{"window": "reference-entry-panel-1", "contains": "names no editors",
                                           "note": "chapter form"}])
    assert "chapter form" in missing["detail"]
    [shown] = check_expectations(html, [{"window": "", "absent": "Title."}])
    assert shown["check"] == "expectation"
    [gone] = check_expectations(html, [{"window": "citation-panel-9", "contains": "x"}])
    assert "window missing" in gone["detail"]


def test_the_automatic_check_records_without_text_and_never_raises(monkeypatch):
    from types import SimpleNamespace
    from app.services import report_check
    session = SimpleNamespace(commit=lambda: None, rollback=lambda: None)
    report, job = SimpleNamespace(id="rep-1"), SimpleNamespace(upload_evidence={})
    html = _report('Ames. <a href="https://example.test/a?b=1">https://example.test/a?b=1</a> &amp;c=2')
    monkeypatch.setattr(report_check, "render_for_check", lambda *a: (html, {}))
    outcome = report_check.record_report_check(session, None, report, job)
    assert outcome["passed"] is False and outcome["counts"] == {"split_link": 1}
    assert job.upload_evidence["report_check"]["windows"] == ["reference-entry-panel-1"]
    assert "example.test" not in str(job.upload_evidence["report_check"])

    def broken(*_args):
        raise RuntimeError("render failed")
    monkeypatch.setattr(report_check, "render_for_check", broken)
    assert report_check.record_report_check(session, None, report, job)["passed"] is None
