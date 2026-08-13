import pytest

from app.services.relevance import score_relevance, score_title_relevance, verify_authors


def test_short_title_requires_every_significant_token() -> None:
    result = score_title_relevance(
        "Rain Man",
        "Homeostasis model assessment of insulin resistance in man",
    )

    assert result.is_relevant is False
    assert result.score == pytest.approx(0.5)


def test_matching_title_and_author_are_relevant() -> None:
    result = score_relevance(
        "New Media Giants",
        "New Media Giants and the Changing Media Landscape",
        "Croteau, D.",
        ["David Croteau", "William Hoynes"],
    )

    assert result.is_relevant is True


def test_matching_title_with_wrong_author_is_rejected() -> None:
    result = score_relevance(
        "Platform Governance",
        "Platform Governance",
        "Smith, J.",
        ["Martin Moore"],
    )

    assert result.is_relevant is False
    assert "AUTHOR REJECTED" in result.detail


def test_author_matching_handles_citation_and_full_name_formats() -> None:
    passes, similarity, _detail = verify_authors(
        "Parc, J., Messerlin, P., & Kim, K.",
        ["Jimmyn Parc", "Patrick Messerlin", "Keun Kim"],
    )

    assert passes is True
    assert similarity == pytest.approx(1.0)
