"""Owner decision 2026-10-04: one reference the list splits in two is joined, checked
and searched as one, and flagged (paper 7's Santillana & Davis)."""
from app.services.reference_parser import merge_split_entries

HEAD = ("Autonomy and National Sovereignty, 2018– Present. In Media and Politics in Post-Authoritarian "
        "Mexico (pp. 179-203). Springer.")
TAIL = "Santillana, M., & Davis, S. (2023). Freedom of the Press: The Struggle Between Journalistic"
OTHERS = ["Curran, J. (2018). Power without responsibility. Routledge.",
          "Smith, J. (2018). The Fourth Estate: Origins of Journalism Ethics in America",
          "Ward, S. J. (2018). Ethical journalism in a populist age. Rowman & Littlefield."]


def test_the_cut_entry_and_its_headless_remainder_are_joined():
    entries = [HEAD, *OTHERS, TAIL]
    merged, splits = merge_split_entries(entries, "apa")
    assert merged == [*OTHERS, f"{TAIL} {HEAD}"]
    assert splits == {" ".join(f"{TAIL} {HEAD}".split()): [TAIL, HEAD]}


def test_nothing_is_joined_without_exactly_one_of_each():
    assert merge_split_entries([*OTHERS, TAIL], "apa") == ([*OTHERS, TAIL], {})          # no remainder
    entries = [HEAD, "Another headless fragment of words here.", *OTHERS, TAIL]
    assert merge_split_entries(entries, "apa")[1] == {}                                   # two remainders
    assert merge_split_entries([HEAD, *OTHERS, TAIL], "mla")[1] == {}                     # APA only
