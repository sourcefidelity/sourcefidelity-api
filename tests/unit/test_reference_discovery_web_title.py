"""A web page's site-name suffix and a citation's bracketed medium are not title words."""
from app.services.reference_discovery import _web_title_furniture_only as same


def test_site_suffix_and_descriptor_are_removed():
    assert same("The Day the Earth Stood Still / The Next Voice You Hear… [Online]",
                "The Day the Earth Stood Still / The Next Voice You Hear… | Hammer Museum", author_match=False)


def test_a_dash_suffix_needs_matching_authors_and_short_titles_never_match():
    assert not same("Long-term effects of exercise", "Long-term effects of exercise - a review", author_match=False)
    assert same("Long-term effects of exercise", "Long-term effects of exercise - a review", author_match=True)
    assert not same("Star Wars", "Star Wars | Lucasfilm", author_match=True)
    assert not same("Long-term effects of exercise - a review", "Long-term effects of exercise", author_match=True)


def test_an_organisation_author_named_in_the_page_text_counts():
    from types import SimpleNamespace as C
    from app.services.source_resolver import _organisation_author_in_page as named
    agree = [C(field_name="title", outcome="minor_difference"), C(field_name="author", outcome="unknown"),
             C(field_name="year", outcome="agreement")]
    page = "This is a past program\nPresented by the UCLA Film & Television Archive.\n"
    assert named("UCLA Film & Television Archive", agree, page)
    assert not named("Smith, J.", agree, "Presented by Smith, J.")
    assert not named("UCLA Film & Television Archive", agree, "Presented by the Hammer Museum.")
    assert not named("UCLA Film & Television Archive", agree[:2], page)
