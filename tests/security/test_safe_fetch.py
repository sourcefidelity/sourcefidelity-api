import socket

import pytest

from app.services.safe_fetch import UnsafeUrlError, _validate_host, _validate_url


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
