import pytest

from app.services.source_type import is_archive_source, is_traditional_media


@pytest.mark.parametrize(
    "reference",
    [
        "Miyazaki, H. (Director). (2001). Spirited Away [Film].",
        "Bowie, D. (1977). Low [Album].",
        "Smith, A. (Host). (2025). Episode title [Podcast episode].",
    ],
)
def test_traditional_media_is_routed_away_from_academic_search(reference: str) -> None:
    assert is_traditional_media(reference) is True


def test_scholarly_work_is_not_misclassified_as_media() -> None:
    reference = "Smith, J. (2024). Film audiences and platform governance. Media Studies, 8(2)."

    assert is_traditional_media(reference) is False


@pytest.mark.parametrize(
    "reference",
    [
        "University Archive, Special Collections, Box 4, Folder 2.",
        "Author papers, unpublished manuscript, 1982.",
    ],
)
def test_physical_archives_are_detected(reference: str) -> None:
    assert is_archive_source(reference) is True
