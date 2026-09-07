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

The connected peer is checked before response bytes are consumed. This closes
the response-disclosure side of DNS rebinding even when the resolver answer
changes between validation and connection. Production deployments should still
enforce the same policy at the egress layer because an HTTP request reaches the
peer before a userspace client can inspect the connected socket.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

# 50 MB default — comfortably above any legitimate academic PDF, well below the
# point where buffering becomes a memory risk.
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_MAX_REDIRECTS = 5
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_MAX_META_REFRESH_BODY_BYTES = 8192

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


class _MetaRefreshParser(HTMLParser):
    """Extract the first bounded HTML refresh directive without executing it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.content: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.content is not None or tag.lower() != "meta":
            return
        values = {key.lower(): value for key, value in attrs if value is not None}
        if values.get("http-equiv", "").strip().lower() == "refresh":
            self.content = values.get("content", "")


def _extract_meta_refresh_url(
    body: bytes,
    content_type: str,
    current_url: str,
) -> str | None:
    """Return a same-session refresh target only for a short, immediate HTML page.

    Repository viewers sometimes set a cookie and return ``<meta refresh>``
    instead of an HTTP redirect before serving a PDF. This parser accepts only
    delay values from zero through one second. The returned URL is still
    validated by ``safe_request`` before the next request, including same-URL
    refreshes, so the directive cannot bypass the SSRF boundary.
    """
    if len(body) > _MAX_META_REFRESH_BODY_BYTES or "html" not in content_type.lower():
        return None
    parser = _MetaRefreshParser()
    try:
        parser.feed(body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if parser.content is None:
        return None
    delay, separator, raw_target = parser.content.partition(";")
    try:
        if float(delay.strip()) > 1:
            return None
    except ValueError:
        return None
    if not separator:
        return current_url
    key, equals, value = raw_target.strip().partition("=")
    if not equals or key.strip().lower() != "url":
        return None
    target = value.strip().strip("'\"")
    return urljoin(current_url, target) if target else current_url


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
        raise UnsafeUrlError("blocked local host")

    # IP literal (e.g. http://10.0.0.1/ or http://[::1]/)
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip):
            raise UnsafeUrlError("blocked non-public IP literal")
        return

    # Hostname — resolve and check EVERY returned address. If any is
    # non-public, reject (a host that resolves to both public and private IPs
    # is treated as blocked).
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError("cannot resolve public host") from exc

    checked = 0
    for _family, _type, _proto, _canon, sockaddr in infos:
        ipstr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ipstr)
        except ValueError:
            continue
        checked += 1
        if _is_blocked_ip(ip):
            raise UnsafeUrlError("hostname resolves to a non-public IP")
    if checked == 0:
        raise UnsafeUrlError("hostname did not resolve to a usable address")


def _validate_url(url: str) -> None:
    """Validate scheme + host of ``url`` before any request is sent."""
    parsed = httpx.URL(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeUrlError(f"blocked scheme: {scheme!r}")
    if parsed.userinfo:
        raise UnsafeUrlError("URL credentials are not permitted")
    _validate_host(parsed.host)


def trusted_url_matches_prefix(url: str, trust_prefix: str) -> bool:
    """Match an operator prefix without allowing userinfo/origin confusion."""
    try:
        candidate = httpx.URL(url)
        trusted = httpx.URL(trust_prefix)
    except Exception:
        return False
    if candidate.userinfo or trusted.userinfo:
        return False
    if (
        candidate.scheme.lower(),
        candidate.host.lower(),
        candidate.port,
    ) != (
        trusted.scheme.lower(),
        trusted.host.lower(),
        trusted.port,
    ):
        return False
    return str(candidate).startswith(str(trusted))


def _validate_connected_peer(response: httpx.Response, *, trusted: bool) -> None:
    """Reject a non-public connected peer before reading its response body."""
    if trusted:
        return
    stream = response.extensions.get("network_stream")
    if stream is None or not hasattr(stream, "get_extra_info"):
        return
    address = stream.get_extra_info("server_addr")
    if not address:
        return
    raw_ip = address[0] if isinstance(address, tuple) else address
    try:
        connected_ip = ipaddress.ip_address(str(raw_ip))
    except ValueError:
        raise UnsafeUrlError("connected peer address is invalid") from None
    if _is_blocked_ip(connected_ip):
        raise UnsafeUrlError("connected peer is a non-public address")


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
    max_meta_refreshes: int = 0,
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
        max_meta_refreshes: Maximum immediate HTML meta-refresh hops. Disabled
            by default; PDF acquisition enables one hop for repository viewers
            that require a cookie-setting interstitial.
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
    redirect_count = 0
    meta_refresh_count = 0

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        while True:
            # Only the very first hop may be a trusted operator host; every
            # redirect target is re-validated regardless of trust_prefix.
            trusted_first_hop = bool(
                redirect_count == 0
                and meta_refresh_count == 0
                and trust_prefix
                and trusted_url_matches_prefix(current, trust_prefix)
            )
            if trusted_first_hop:
                logger.debug("safe_request: using trusted operator origin for first hop")
            else:
                _validate_url(current)
            with client.stream(method, current, headers=merged_headers) as resp:
                _validate_connected_peer(resp, trusted=trusted_first_hop)
                if resp.status_code in _REDIRECT_CODES:
                    location = resp.headers.get("location")
                    if not location:
                        raise UnsafeUrlError("redirect response omitted its target")
                    current = str(httpx.URL(current).join(location))
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        raise httpx.TooManyRedirects(
                            f"exceeded the {max_redirects}-redirect safety limit",
                            request=resp.request,
                        )
                    continue

                # Final response — stream into a capped buffer.
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise ResponseTooLargeError(
                            f"response exceeded the {max_bytes}-byte safety limit"
                        )
                if raise_on_status:
                    resp.raise_for_status()
                # Cache the read body so .content / .text work after close.
                resp._content = bytes(buf)
                if method.upper() == "GET" and meta_refresh_count < max_meta_refreshes:
                    refresh_url = _extract_meta_refresh_url(
                        resp.content,
                        resp.headers.get("content-type", ""),
                        current,
                    )
                    if refresh_url is not None:
                        meta_refresh_count += 1
                        current = refresh_url
                        logger.debug(
                            "safe_request: following bounded same-session meta refresh",
                        )
                        continue
                return resp


def safe_fetch_bytes(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    accept_content_types: Iterable[str] | None = None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    max_meta_refreshes: int = 0,
) -> bytes:
    """SSRF-safe, size-capped fetch returning the response body as bytes.

    If ``accept_content_types`` is given, the response ``Content-Type`` must
    match one of them (or be ``application/octet-stream``, which servers use
    for binary streams of any kind); otherwise ``ValueError`` is raised. Use
    this for PDF downloads where the type should be constrained.
    """
    resp = safe_request(
        url,
        max_bytes=max_bytes,
        timeout=timeout,
        headers=headers,
        max_meta_refreshes=max_meta_refreshes,
    )
    if accept_content_types:
        ct = resp.headers.get("content-type", "").lower()
        allowed = [a.lower() for a in accept_content_types]
        if not (any(a in ct for a in allowed) or "octet-stream" in ct):
            raise ValueError("response content type is not permitted for this route")
    return resp.content
