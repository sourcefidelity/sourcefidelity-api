"""Bounded hostile-file inspection before academic-source parsing."""

from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import re
import socket
import struct

import fitz

from app.config import settings


class SafetyVerdict(str, Enum):
    CLEAN = "clean"
    REJECTED = "rejected"
    NOT_ASSESSED = "not_assessed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FileSafetyReport:
    verdict: SafetyVerdict
    structural_verdict: SafetyVerdict
    malware_verdict: SafetyVerdict
    findings: tuple[str, ...] = field(default_factory=tuple)
    scanner_detail: str | None = None


class FileSafetyUnavailable(RuntimeError):
    """Required malware inspection could not be completed."""


_PDF_NAME_END = rb"(?=[\x00\x09\x0a\x0c\x0d\x20()<>\[\]{}/%])"
_ACTIVE_PDF_MARKERS = (
    (re.compile(rb"/JavaScript" + _PDF_NAME_END), "PDF JavaScript"),
    (re.compile(rb"/JS" + _PDF_NAME_END), "PDF JavaScript action"),
    (re.compile(rb"/AA" + _PDF_NAME_END), "automatic additional action"),
    (re.compile(rb"/Launch" + _PDF_NAME_END), "external launch action"),
    (re.compile(rb"/RichMedia" + _PDF_NAME_END), "embedded rich media"),
    (re.compile(rb"/EmbeddedFile" + _PDF_NAME_END), "embedded file attachment"),
    (re.compile(rb"/XFA" + _PDF_NAME_END), "XFA form content"),
)

_DESTINATION_ARRAY = re.compile(r"^\[\s*\d+\s+\d+\s+R(?=[\s/])")


def _destination_is_valid(
    document: fitz.Document,
    value_type: str,
    value: str,
) -> bool:
    if value_type in {"name", "string"}:
        return True
    if value_type == "array":
        return bool(_DESTINATION_ARRAY.search(value))
    if value_type == "xref":
        try:
            destination_xref = int(value.split()[0])
            raw = document.xref_object(destination_xref, compressed=True).strip()
            return bool(_DESTINATION_ARRAY.search(raw))
        except (IndexError, TypeError, ValueError, RuntimeError):
            return False
    return False


def _open_action_finding(document: fitz.Document) -> str | None:
    """Reject executable/external catalog OpenAction values, not destinations.

    PDF ``OpenAction`` is overloaded: an array, name, or string selects the
    initial page/view and is ordinary document navigation; an action dictionary
    can execute behavior. Inspect the parsed catalog instead of rejecting every
    raw ``/OpenAction`` byte sequence. Only an internal ``GoTo`` action is
    accepted from the action-dictionary form. Unknown or malformed forms fail
    closed.
    """
    catalog = document.pdf_catalog()
    value_type, value = document.xref_get_key(catalog, "OpenAction")
    if value_type == "null":
        return None
    if value_type in {"array", "name", "string"}:
        return (
            None
            if _destination_is_valid(document, value_type, value)
            else "automatic open action (malformed destination)"
        )

    action_type: str | None = None
    destination_valid = False
    chained_action = False
    if value_type == "xref":
        try:
            action_xref = int(value.split()[0])
            raw = document.xref_object(action_xref, compressed=True).strip()
            if raw.startswith("["):
                return (
                    None
                    if _destination_is_valid(document, "array", raw)
                    else "automatic open action (malformed destination)"
                )
            subtype_type, subtype = document.xref_get_key(action_xref, "S")
            if subtype_type == "name":
                action_type = subtype.lstrip("/")
            destination_type, destination = document.xref_get_key(action_xref, "D")
            destination_valid = _destination_is_valid(
                document, destination_type, destination
            )
            next_type, _next_value = document.xref_get_key(action_xref, "Next")
            chained_action = next_type != "null"
        except (IndexError, TypeError, ValueError, RuntimeError):
            action_type = None
    elif value_type == "dict":
        match = re.search(r"/S\s*/([A-Za-z0-9]+)" + _PDF_NAME_END.decode(), value)
        if match:
            action_type = match.group(1)
        chained_action = bool(
            re.search(r"/Next" + _PDF_NAME_END.decode(), value)
        )
        destination_valid = bool(
            re.search(r"/D\s*(?:/[^\s<>{}\[\]()/%]+|\([^)]*\)|"
                      r"\[\s*\d+\s+\d+\s+R(?=[\s/]))", value)
        )

    if chained_action:
        return "automatic open action (chained action)"
    if action_type == "GoTo" and destination_valid:
        return None
    if action_type == "GoTo":
        return "automatic open action (malformed internal destination)"
    if action_type:
        return f"automatic open action ({action_type})"
    return "automatic open action (unknown or malformed action type)"


def _basic_pdf_check(content: bytes) -> tuple[SafetyVerdict, list[str]]:
    findings: list[str] = []
    maximum = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if not content:
        return SafetyVerdict.REJECTED, ["empty file"]
    if len(content) > maximum:
        return SafetyVerdict.REJECTED, ["file exceeds configured upload limit"]
    if not content.lstrip()[:8].startswith(b"%PDF-"):
        return SafetyVerdict.REJECTED, ["content is not a PDF"]
    for marker, label in _ACTIVE_PDF_MARKERS:
        if marker.search(content):
            findings.append(label)
    if findings:
        return SafetyVerdict.REJECTED, findings
    return SafetyVerdict.CLEAN, findings


def _structural_pdf_check(content: bytes) -> tuple[SafetyVerdict, list[str]]:
    basic_verdict, findings = _basic_pdf_check(content)
    if basic_verdict is SafetyVerdict.REJECTED:
        return basic_verdict, findings
    try:
        document = fitz.open(stream=content, filetype="pdf")
        try:
            if document.needs_pass:
                return SafetyVerdict.REJECTED, ["encrypted or password-protected PDF"]
            if document.page_count < 1:
                return SafetyVerdict.REJECTED, ["PDF contains no pages"]
            if document.page_count > settings.MAX_PDF_PAGES:
                return SafetyVerdict.REJECTED, ["PDF exceeds configured page limit"]
            if document.xref_length() > settings.MAX_PDF_OBJECTS:
                return SafetyVerdict.REJECTED, ["PDF exceeds configured object limit"]
            if document.embfile_count() > 0:
                return SafetyVerdict.REJECTED, ["PDF contains embedded file attachments"]
            open_action_finding = _open_action_finding(document)
            if open_action_finding:
                return SafetyVerdict.REJECTED, [open_action_finding]
        finally:
            document.close()
    except (
        fitz.FileDataError,
        fitz.mupdf.FzErrorBase,
        RuntimeError,
        ValueError,
    ) as exc:
        return SafetyVerdict.REJECTED, [f"malformed PDF ({type(exc).__name__})"]
    return SafetyVerdict.CLEAN, []


def _trusted_clamd_addresses(host: str, port: int) -> list[tuple]:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise FileSafetyUnavailable("clamd host did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not (ip.is_private or ip.is_loopback or ip.is_link_local):
            raise FileSafetyUnavailable(
                "clamd TCP endpoint must resolve only to trusted local/private addresses"
            )
    return addresses


def _clamd_socket() -> socket.socket:
    timeout = settings.CLAMD_TIMEOUT_SECONDS
    if settings.CLAMD_UNIX_SOCKET:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect(settings.CLAMD_UNIX_SOCKET)
        return client
    if settings.CLAMD_HOST:
        _trusted_clamd_addresses(settings.CLAMD_HOST, settings.CLAMD_PORT)
        return socket.create_connection(
            (settings.CLAMD_HOST, settings.CLAMD_PORT), timeout=timeout
        )
    raise FileSafetyUnavailable("clamd endpoint is not configured")


def _parse_clamd_reply(reply: bytes) -> tuple[SafetyVerdict, str]:
    detail = reply.rstrip(b"\0\r\n").decode("utf-8", "replace")
    if detail.endswith(": OK"):
        return SafetyVerdict.CLEAN, detail
    if detail.endswith(" FOUND"):
        return SafetyVerdict.REJECTED, detail
    if detail.endswith(" ERROR") or not detail:
        return SafetyVerdict.UNAVAILABLE, detail or "empty clamd response"
    return SafetyVerdict.UNAVAILABLE, detail


def scan_with_clamd(content: bytes) -> tuple[SafetyVerdict, str]:
    """Stream bytes using the official clamd INSTREAM framing."""
    try:
        with _clamd_socket() as client:
            client.sendall(b"zINSTREAM\0")
            for offset in range(0, len(content), 1024 * 1024):
                chunk = content[offset : offset + 1024 * 1024]
                client.sendall(struct.pack(">I", len(chunk)))
                client.sendall(chunk)
            client.sendall(struct.pack(">I", 0))
            reply = bytearray()
            while b"\0" not in reply:
                data = client.recv(4096)
                if not data:
                    break
                reply.extend(data)
                if len(reply) > 16_384:
                    raise FileSafetyUnavailable("clamd response exceeded safety limit")
    except FileSafetyUnavailable:
        raise
    except (OSError, socket.timeout) as exc:
        raise FileSafetyUnavailable(
            f"clamd scan unavailable ({type(exc).__name__})"
        ) from exc
    return _parse_clamd_reply(bytes(reply))


def inspect_uploaded_pdf(content: bytes) -> FileSafetyReport:
    """Inspect a PDF before bibliographic parsing or durable admission."""
    basic_verdict, basic_findings = _basic_pdf_check(content)
    if basic_verdict is SafetyVerdict.REJECTED:
        return FileSafetyReport(
            verdict=SafetyVerdict.REJECTED,
            structural_verdict=basic_verdict,
            malware_verdict=SafetyVerdict.NOT_ASSESSED,
            findings=tuple(basic_findings),
        )

    try:
        malware_verdict, scanner_detail = scan_with_clamd(content)
    except FileSafetyUnavailable as exc:
        if settings.MALWARE_SCAN_REQUIRED:
            raise
        malware_verdict = SafetyVerdict.NOT_ASSESSED
        scanner_detail = str(exc)

    if malware_verdict is SafetyVerdict.REJECTED:
        return FileSafetyReport(
            verdict=SafetyVerdict.REJECTED,
            structural_verdict=SafetyVerdict.NOT_ASSESSED,
            malware_verdict=malware_verdict,
            findings=("malware scanner rejected the upload",),
            scanner_detail=scanner_detail,
        )
    if malware_verdict is SafetyVerdict.UNAVAILABLE and settings.MALWARE_SCAN_REQUIRED:
        raise FileSafetyUnavailable(scanner_detail)

    structural_verdict, structural_findings = _structural_pdf_check(content)
    if structural_verdict is SafetyVerdict.REJECTED:
        return FileSafetyReport(
            verdict=SafetyVerdict.REJECTED,
            structural_verdict=structural_verdict,
            malware_verdict=malware_verdict,
            findings=tuple(structural_findings),
            scanner_detail=scanner_detail,
        )

    final_verdict = (
        SafetyVerdict.CLEAN
        if malware_verdict is SafetyVerdict.CLEAN
        else SafetyVerdict.NOT_ASSESSED
    )
    return FileSafetyReport(
        verdict=final_verdict,
        structural_verdict=SafetyVerdict.CLEAN,
        malware_verdict=malware_verdict,
        scanner_detail=scanner_detail,
    )
