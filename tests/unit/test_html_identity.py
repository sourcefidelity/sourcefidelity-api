from app.services.source_resolver import _extract_html_titles, _html_title_matches


def test_html_title_extraction_decodes_metadata() -> None:
    page = """
    <html><head>
      <meta name="citation_title" content="Research &amp; Academic Integrity">
      <meta property="og:title" content="A shorter social title">
      <title>Site title</title>
    </head></html>
    """

    assert _extract_html_titles(page) == [
        "Research & Academic Integrity",
        "A shorter social title",
        "Site title",
    ]


def test_related_but_wrong_page_title_is_rejected() -> None:
    assert _html_title_matches(
        "Platform governance: The antitrust option",
        ["Platform regulation and digital advertising markets"],
    ) is False


def test_matching_page_title_is_accepted() -> None:
    assert _html_title_matches(
        "Platform governance: The antitrust option",
        ["Platform Governance — The Antitrust Option | Journal Site"],
    ) is True
