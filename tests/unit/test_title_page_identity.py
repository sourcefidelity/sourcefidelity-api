"""A copy whose own title page names the cited title and first author is the
cited work (owner decision 2026-10-14, after the review of 8 possible matches)."""
from app.services.source_resolver import title_page_names_work


def test_the_title_and_first_author_on_the_title_page_confirm_the_work():
    page = "Harbour Studies Volume 3 Cranes and Tides: A Case Study Ana Ames University of Ports Abstract Cranes ..."
    assert title_page_names_work(page, "Cranes and Tides: A Case Study", "Ames, A.")
    # Spacing lost when the paper was read ("CranesandTides: ACaseStudy").
    assert title_page_names_work(page, "CranesandTides: ACaseStudy", "Ames, A.")
    # The same words, worded slightly differently by the student.
    report = "Port Council Cranes as a tool of trade - a growing global phenomenon Ana Ames"
    assert title_page_names_work(report, "Cranes tool of trade: A growing global phenomenon", "Ames, A.")


def test_a_shared_phrase_or_the_same_author_alone_is_not_the_work():
    # Another work containing the cited title's words, by other people.
    other = "Working Below the Line in the Crane System: Labour on the Docks Bo Bell and Cy Cole Working Paper"
    assert not title_page_names_work(other, "The Crane System", "Ames, A.")
    # The cited author, but a different work whose pages merely mention its words.
    chapter = ("United Kingdom ANA AMES Ana Ames is a research associate. There is a growing recognition of the "
               "interface between trade and cranes, and of tides, in policy; globalization of ports continues.")
    assert not title_page_names_work(chapter, "Cranes and tides in an age of globalization", "Ames, A.")
    assert not title_page_names_work("", "Cranes and Tides", "Ames, A.")
    assert not title_page_names_work(chapter, "", "Ames, A.")
