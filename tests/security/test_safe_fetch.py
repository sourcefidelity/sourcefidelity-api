import socket

import httpx
import pytest

from app.services.safe_fetch import (
    UnsafeUrlError,
    _extract_meta_refresh_url,
    safe_request,
    _validate_host,
    _validate_url,
)


@pytest.mark.security
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://10.0.0.7/source.pdf",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/source.pdf",
    ],
)
def test_non_public_targets_and_schemes_are_rejected(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        _validate_url(url)


@pytest.mark.security
def test_hostname_with_any_private_dns_answer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host: str, port: object) -> list[tuple[object, ...]]:
        assert host == "mixed.example"
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(UnsafeUrlError, match="non-public IP"):
        _validate_host("mixed.example")


@pytest.mark.security
def test_public_hostname_is_accepted_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, _port: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        ],
    )

    _validate_url("https://public.example/source.pdf")


@pytest.mark.security
def test_immediate_same_url_meta_refresh_is_bounded_candidate() -> None:
    body = b'<html><head><meta http-equiv="refresh" content="1"></head></html>'

    assert _extract_meta_refresh_url(
        body, "text/html; charset=utf-8", "https://repository.example/view/1"
    ) == "https://repository.example/view/1"


@pytest.mark.security
def test_relative_meta_refresh_target_is_resolved_but_not_pretrusted() -> None:
    body = b'<meta http-equiv="Refresh" content="0; URL=../files/source.pdf">'

    assert _extract_meta_refresh_url(
        body, "text/html", "https://repository.example/view/1"
    ) == "https://repository.example/files/source.pdf"


@pytest.mark.security
@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b'<meta http-equiv="refresh" content="5;url=/slow">', "text/html"),
        (b'<meta http-equiv="refresh" content="now;url=/bad">', "text/html"),
        (b'<meta http-equiv="refresh" content="0;url=/not-html">', "application/pdf"),
    ],
)
def test_delayed_malformed_or_non_html_refresh_is_ignored(
    body: bytes, content_type: str
) -> None:
    assert _extract_meta_refresh_url(
        body, content_type, "https://repository.example/view/1"
    ) is None


@pytest.mark.security
def test_opt_in_meta_refresh_reuses_cookie_session(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                content=b'<meta http-equiv="refresh" content="1">',
                headers={"content-type": "text/html", "set-cookie": "viewer=ready"},
                request=request,
            )
        assert request.headers.get("cookie") == "viewer=ready"
        return httpx.Response(
            200,
            content=b"%PDF-repository",
            headers={"content-type": "application/pdf"},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.services.safe_fetch.httpx.Client", lambda **_kwargs: client)
    monkeypatch.setattr("app.services.safe_fetch._validate_url", lambda _url: None)

    response = safe_request(
        "https://repository.example/view/1", max_meta_refreshes=1
    )

    assert response.content == b"%PDF-repository"
    assert len(requests) == 2
