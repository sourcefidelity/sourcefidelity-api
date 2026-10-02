"""A film or Act named with its year and no reference-list entry is a citation
without an entry (owner decision 2026-10-02)."""
from types import SimpleNamespace

from app.services.evidence_report import _missing_reference_members


def marker(sentence, year, origin=100):
    start = origin + sentence.index(f"({year})")
    return SimpleNamespace(text=f"({year})", link_status="missing_reference", reference_ids=[],
                           candidate_reference_ids=[], marker_type="parenthetical", member_count=1,
                           passage_start=start, passage_end=start + 6, report_text=sentence,
                           report_passage_start=origin)


def members(sentence, year):
    found = marker(sentence, year)
    extraction = SimpleNamespace(citations=[], citation_marker_census=[found])
    return _missing_reference_members(extraction, found.passage_start, found.passage_end)


def test_quoted_film_titles_and_named_acts_become_missing_reference_members():
    assert members('Films such as "Picnic Over the Rock" (1975) were praised.', 1975) == ['Picnic Over the Rock, 1975']
    assert members('The first, “Princess Iron Fan” (1941), was animated.', 1941) == ['Princess Iron Fan, 1941']
    assert members('In conclusion, the Australian Film Development Corporation Act (1970) mattered.', 1970) == [
        'Australian Film Development Corporation Act, 1970']


def test_a_bare_year_after_ordinary_words_stays_unnamed():
    assert members('Attendance rose sharply after the war (1946) in most cities.', 1946) == []


def test_a_title_used_as_author_links_the_one_reference_it_ends_with():
    from app.services.citation_extractor import _title_author_reference
    from app.services.schemas import ParsedReference
    site = ParsedReference(reference_id="r1", author="Dramas_Reference.com", year="2023", title="Fx361.Cc",
                           raw_ref="Dramas_Reference.com. (2023). Fx361.Cc. https://example.test/a")
    other = ParsedReference(reference_id="r2", author="Smith, J.", year="2023", title="Another", raw_ref="Smith, J. (2023). Another.")
    content = "Transmedia Narrative: A Creative Discussion on Adapting Web Novels to Web\nDramas_Reference.com, 2023"
    reference, _author, year = _title_author_reference(content, [site, other])
    assert reference.reference_id == "r1" and year == "2023"
    assert _title_author_reference(content.replace("2023", "2022"), [site, other]) is None


def test_an_organisation_named_with_one_word_different_links_its_reference():
    from app.services.citation_extractor import _organisation_match
    from app.services.schemas import ParsedReference
    commission = ParsedReference(reference_id="r1", author="Australian Film Commission", year="2006", title="A report")
    person = ParsedReference(reference_id="r2", author="Smith, J.", year="2006", title="Other")
    index = {"commission": [commission], "smith": [person]}
    assert _organisation_match("growth. Australian Film ", "Council", "2006", index) is commission
    assert _organisation_match("growth. Australian Film ", "Council", "2007", index) is None
    assert _organisation_match("growth. Canadian Media ", "Council", "2006", index) is None


def test_a_file_carrying_a_borrowed_doi_must_show_the_cited_title():
    import fitz
    from app.services.source_resolver import _cited_title_on_front_pages
    document = fitz.open(); page = document.new_page()
    page.insert_text((72, 100), "Creative sustainability of screen business in the regions", fontsize=11)
    data = document.tobytes(); document.close()
    assert _cited_title_on_front_pages(data, "Creative Sustainability of Screen Business in the Regions")
    assert not _cited_title_on_front_pages(data, "Dependency and Sustainability in the Australian Film Industry")
