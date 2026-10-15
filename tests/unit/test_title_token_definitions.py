"""Two title-token helpers shared one name, so the opening-pages check used the
short-title check's word list (found 2026-10-15)."""
import fitz

from app.services.source_resolver import _opening_pages_present_title


FILLER = ("Interviews covered weather, tides, lighting, radio, charts, tugs, ropes, fuel, crews, shifts, "
          "wages, training, insurance, accidents, cargo, timber, grain, coal, steel, containers, schedules, "
          "inspections, harbours, locks, bridges, currents, fog, ice, storms and maintenance records.")


def _pdf(text):
    document = fitz.open()
    document.new_page().insert_textbox(fitz.Rect(54, 54, 560, 780), f"{text} {FILLER}", fontsize=9)
    return document.tobytes()


def test_generic_title_words_do_not_make_a_stranger_the_cited_work():
    # "Case" and "study" name no work: a different study's opening must not
    # pass for "Cranes: A Case Study" on those words alone.
    other = _pdf("A case study of river pilots. This study reports interviews with pilots "
                 "about night navigation on inland waterways.")
    assert not _opening_pages_present_title(other, "Cranes: A Case Study")
    cited = _pdf("Cranes: A Case Study. Ames, A. This study examines harbour cranes.")
    assert _opening_pages_present_title(cited, "Cranes: A Case Study")


def test_a_missing_title_is_not_present_and_does_not_fail():
    assert not _opening_pages_present_title(_pdf("Any opening text. " * 20), None)
