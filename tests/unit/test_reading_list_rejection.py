"""A course reading list is not the works it lists.

Found 2026-09-23. A 32-page Talis Aspire list, "HA2224 American Film and
Visual Culture | readinglists@leicester", was acquired as the candidate source
for the monograph *American cinema/American culture (4th ed.)*. Its page one
is a bibliography, so a reader taking the first DOI on the page got
10.1353/flm.0.0105 -- the list's first entry, an unrelated Film & History
article. `_detect_nonwork_listing` existed precisely to refuse documents that
only list works, and did not catch this one.
"""
from app.services.source_validator import _detect_nonwork_listing

_ENTRIES = "\n".join([
    "Agee, James. 1963. 'Comedy's Greatest Era (1949) in Agee on Film'. London: P. Owen.",
    "Akass, Kim and McCabe, Janet. 2005. Reading Six Feet Under: TV to Die For. London: I.B. Tauris.",
    "Allen, Michael. 2003. Contemporary US Cinema. Vol. Inside film. Harlow: Longman.",
    "Allen, Robert Clyde. 1992. Channels of Discourse, Reassembled. 2nd ed. London: Routledge.",
    "Altman, Rick. 1987. The American Film Musical. Bloomington: Indiana University Press.",
    "Balio, Tino. 1985. The American Film Industry. Rev.ed. London: University of Wisconsin Press.",
    "Belton, John. 2013. American Cinema/American Culture. 4th ed. New York: McGraw-Hill.",
    "Bordwell, David. 1985. The Classical Hollywood Cinema. London: Routledge.",
    "Cook, Pam. 2007. The Cinema Book. 3rd ed. London: BFI.",
])

READING_LIST = (
    "09/19/26\n"
    "HA2224 American Film and Visual Culture | readinglists@leicester\n"
    "HA2224 American Film and Visual Culture\n"
    "View Online\n"
    "'A World That Works': Fascism and Media Globalization in Starship Troopers. 2009.\n"
    "Film & History 39(2):17-25. doi:10.1353/flm.0.0105.\n"
    + _ENTRIES
)


def test_a_reading_list_is_refused_even_when_it_lists_the_cited_work() -> None:
    """The list names the book on one of its lines. That is not the book."""
    verdict = _detect_nonwork_listing(
        READING_LIST, "American cinema/American culture (4th ed.)")

    assert verdict == "a course reading list"


def test_a_resource_list_heading_is_recognised_without_the_talis_banner() -> None:
    page = "Module Resource List\nSpring 2026\n" + _ENTRIES

    assert _detect_nonwork_listing(page, "The Cinema Book") == "a course reading list"


def test_the_banner_alone_does_not_refuse_an_ordinary_work() -> None:
    """A work may mention a reading list in passing; that is not its role."""
    article = (
        "Teaching Film History in the Digital Seminar\n"
        "Jane Doe, University of Leicester\n\n"
        "Abstract. This article reports on a redesigned seminar. Materials were "
        "distributed through readinglists@leicester, which students accessed weekly. "
        "We discuss attendance, preparation and assessment outcomes across two cohorts."
    )

    assert _detect_nonwork_listing(article, "Teaching Film History") is None


def test_a_bibliography_alone_does_not_refuse_an_ordinary_work() -> None:
    """Every scholarly work carries references; that is not a listing role."""
    article = (
        "The Classical Hollywood Cinema\nDavid Bordwell\n\n"
        "This study examines the emergence of a mode of production.\n\nReferences\n"
        + _ENTRIES
    )

    assert _detect_nonwork_listing(article, "The Classical Hollywood Cinema") is None
