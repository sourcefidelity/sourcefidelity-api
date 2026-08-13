"""Safe HTTP fetching: SSRF defense + download-size caps.

Every external fetch in the retrieval/verification chain should go through
`safe_request` (or the `safe_fetch_bytes` helper). These enforce three things
that plain ``httpx.get(url, follow_redirects=True)`` does not:

1. SSRF defense — each URL (including every redirect hop) is resolved and its
   IP addresses are checked against loopback / private / link-local /
   reserved / multicast / unspecified ranges. A student URL or search result
   that points at ``169.254.169.254`` (cloud metadata), ``localhost``, or an
   internal host is rejected *before* any connection is made. See REVIEW §2.1.

2. Download cap — responses are streamed and aborted as soon as they exceed
   ``max_bytes``, so a server cannot exhaust memory by serving a huge body or
   by lying about ``content-length``. See REVIEW §2.2 / §2.7.

3. Redirect control — redirects are followed one hop at a time, re-running the
   IP check on each ``Location``. This blocks the common "external URL that
   302s to an internal host" attack and prevents redirect loops.

Residual risk: a narrow DNS-rebinding TOCTOU window remains between resolve
and connect; fully closing it needs connect-time IP pinning, which httpx does
not expose over TLS. This is consistent with the threat model (untrusted
student/search URLs, not a network-level adversary on the wire).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

# 50 MB default — comfortably above any legitimate academic PDF, well below the
# point where buffering becomes a memory risk.
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_MAX_REDIRECTS = 5
_REDIRECT_CODES = {301, 302, 303, 307, 308}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


class UnsafeUrlError(ValueError):
    """Raised when a URL (or a redirect target) is non-public or otherwise blocked."""


class ResponseTooLargeError(Exception):
    """Raised when a response body exceeds the configured byte cap."""


# ── SSRF guard ───────────────────────────────────────────────────────────

def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True for any non-publicly-routable address.

    Covers loopback (127/8, ::1), private (RFC1918 + fc00::/7), link-local
    (169.254/16 — includes cloud metadata endpoints, fe80::/10), reserved,
    multicast, and unspecified. IPv4-mapped IPv6 addresses (e.g.
    ``::ffff:127.0.0.1``) are normalized by ``ipaddress`` and caught here too.
    """
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_host(host: str) -> None:
    """Resolve ``host`` and reject if it is or resolves to a non-public address."""
    if not host:
        raise UnsafeUrlError("URL has no host")

    host = host.lower()
    if host in ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"):
        raise UnsafeUrlError(f"blocked host: {host}")

    # IP literal (e.g. http://10.0.0.1/ or http://[::1]/)
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(f"blocked IP literal: {ip}")
        return

    # Hostname — resolve and check EVERY returned address. If any is
    # non-public, reject (a host that resolves to both public and private IPs
    # is treated as blocked).
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"cannot resolve host {host}: {e}") from e

    checked = 0
    for _family, _type, _proto, _canon, sockaddr in infos:
        ipstr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ipstr)
        except ValueError:
            continue
        checked += 1
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(f"host {host} resolves to non-public IP {ip}")
    if checked == 0:
        raise UnsafeUrlError(f"host {host} did not resolve to any usable address")


def _validate_url(url: str) -> None:
    """Validate scheme + host of ``url`` before any request is sent."""
    parsed = httpx.URL(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeUrlError(f"blocked scheme: {scheme!r}")
    _validate_host(parsed.host)


# ── Public API ───────────────────────────────────────────────────────────

def safe_request(
    url: str,
    *,
    method: str = "GET",
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    raise_on_status: bool = True,
    max_redirects: int = _MAX_REDIRECTS,
    trust_prefix: str | None = None,
) -> httpx.Response:
    """SSRF-safe, size-capped HTTP request with manual redirect following.

    Follows redirects one hop at a time, re-validating the host on every hop.
    The final response body is fully read (capped at ``max_bytes``) and stored
    on the returned ``httpx.Response`` so callers can use ``.content``,
    ``.text``, ``.status_code``, ``.headers`` and ``.url`` as normal.

    Args:
        url: The URL to fetch.
        method: HTTP method (default ``GET``).
        max_bytes: Abort once the body exceeds this many bytes.
        timeout: Request timeout in seconds.
        headers: Extra headers (merged over the default browser User-Agent).
        raise_on_status: If True (default), raise ``httpx.HTTPStatusError`` on
            4xx/5xx. Set False for callers that need to inspect status codes
            themselves (e.g. the link validator, which categorizes 403/404).
        max_redirects: Maximum redirect hops before giving up.
        trust_prefix: If set, the *initial* URL is exempted from the SSRF
            host check when it starts with this prefix. Use ONLY for
            operator-configured, trusted hosts (e.g. an institution's campus
            EZproxy base in ``DOI_RESOLVER_URL``) that may themselves be on a
            private network. Redirect targets are validated normally, so the
            trust does not extend to wherever the proxy sends us. The
            student-controlled part of such URLs (e.g. the DOI) must still be
            validated/encoded by the caller.

    Raises:
        UnsafeUrlError: URL or any redirect target is non-public / bad scheme.
        ResponseTooLargeError: Body exceeded ``max_bytes``.
        httpx.HTTPStatusError: Non-2xx final status (when ``raise_on_status``).
        httpx.RequestError / httpx.TooManyRedirects: network / redirect errors.
    """
    merged_headers = {**_DEFAULT_HEADERS, **(headers or {})}
    current = url

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for hop in range(max_redirects + 1):
            # Only the very first hop may be a trusted operator host; every
            # redirect target is re-validated regardless of trust_prefix.
            if hop == 0 and trust_prefix and current.startswith(trust_prefix):
                logger.debug("safe_request: trusting operator host for first hop (%s)", current[:60])
            else:
                _validate_url(current)
            with client.stream(method, current, headers=merged_headers) as resp:
                if resp.status_code in _REDIRECT_CODES:
                    location = resp.headers.get("location")
                    if not location:
                        raise UnsafeUrlError(f"redirect with no Location header from {current[:60]}")
                    current = str(httpx.URL(current).join(location))
                    if hop == max_redirects:
                        raise httpx.TooManyRedirects(
                            f"exceeded {max_redirects} redirects (last target: {current[:60]})",
                            request=resp.request,
                        )
                    continue

                # Final response — stream into a capped buffer.
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise ResponseTooLargeError(
                            f"response exceeded {max_bytes} bytes from {current[:60]}"
                        )
                if raise_on_status:
                    resp.raise_for_status()
                # Cache the read body so .content / .text work after close.
                resp._content = bytes(buf)
                return resp

    # Unreachable: the loop either returns or raises on every path.
    raise RuntimeError("safe_request exited its redirect loop without a response")


def safe_fetch_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    accept_content_types: Iterable[str] | None = None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> bytes:
    """SSRF-safe, size-capped fetch returning the response body as bytes.

    If ``accept_content_types`` is given, the response ``Content-Type`` must
    match one of them (or be ``application/octet-stream``, which servers use
    for binary streams of any kind); otherwise ``ValueError`` is raised. Use
    this for PDF downloads where the type should be constrained.
    """
    resp = safe_request(url, max_bytes=max_bytes, timeout=timeout, headers=headers)
    if accept_content_types:
        ct = resp.headers.get("content-type", "").lower()
        allowed = [a.lower() for a in accept_content_types]
        if not (any(a in ct for a in allowed) or "octet-stream" in ct):
            raise ValueError(
                f"unexpected content-type {ct!r} (want one of {allowed}) from {url[:60]}"
            )
    return resp.content
