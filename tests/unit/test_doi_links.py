"""DOI links follow the user's resolver address when one is set (owner decision 2026-10-19)."""
import pytest

from app.config import settings
from app.services.doi_links import doi_href, doi_link_base


@pytest.mark.parametrize("configured,base", [
    (None, "https://doi.org/"),
    ("", "https://doi.org/"),
    ("https://gateway.example/go?url=https://doi.org/", "https://gateway.example/go?url=https://doi.org/"),
    ("javascript:alert(1)//", "https://doi.org/"),
    ("not a url", "https://doi.org/"),
])
def test_the_link_base(monkeypatch, configured, base):
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", configured)
    assert doi_link_base() == base


def test_a_doi_is_quoted_into_the_link(monkeypatch):
    monkeypatch.setattr(settings, "DOI_RESOLVER_URL", "https://resolver.example/")
    assert doi_href("10.1234/a b") == "https://resolver.example/10.1234/a%20b"


def test_the_application_itself_still_resolves_through_doi_org():
    import inspect
    from app.services.source_resolver import SourceResolver
    source = inspect.getsource(SourceResolver._try_doi_resolver)
    assert 'base = "https://doi.org/"' in source and "DOI_RESOLVER_URL" not in source.split('"""')[-1]
