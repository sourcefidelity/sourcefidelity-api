"""Controlled DOCX-to-PDF rendering for a report presentation surface."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile

from app.config import settings


_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_UNSAFE_FIELD = re.compile(r"\b(?:DDE|DDEAUTO|INCLUDEPICTURE|INCLUDETEXT|LINK)\b", re.I)


class DocxPresentationError(ValueError):
    """A DOCX could not be rendered within the controlled boundary."""


@dataclass(frozen=True)
class DocxPresentationRender:
    pdf_bytes: bytes
    provenance: dict


def render_docx_to_pdf(content: bytes) -> DocxPresentationRender:
    """Render sanitized DOCX bytes with an isolated LibreOffice user profile."""
    if not settings.DOCX_PRESENTATION_RENDERING_ENABLED:
        raise DocxPresentationError("DOCX presentation rendering is disabled")
    _reject_renderer_network_dependencies(content)
    executable = _resolve_executable(settings.DOCX_PRESENTATION_RENDERER_EXECUTABLE)
    renderer_version = _renderer_version(executable)
    font_evidence = _font_evidence(content)

    with tempfile.TemporaryDirectory(prefix="sourcefidelity-docx-render-") as root:
        root_path = Path(root)
        input_path = root_path / "input.docx"
        output_path = root_path / "output"
        profile_path = root_path / "profile"
        output_path.mkdir()
        profile_path.mkdir()
        input_path.write_bytes(content)
        command = [
            executable,
            "--headless",
            "--nologo",
            "--nodefault",
            "--nofirststartwizard",
            "--norestore",
            f"-env:UserInstallation={profile_path.as_uri()}",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(output_path),
            str(input_path),
        ]
        environment = os.environ.copy()
        environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "SAL_USE_VCLPLUGIN": "svp"})
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=max(10, settings.DOCX_PRESENTATION_RENDER_TIMEOUT_SECONDS),
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocxPresentationError("DOCX presentation rendering timed out") from exc
        if completed.returncode != 0:
            raise DocxPresentationError("DOCX presentation renderer failed")
        candidates = list(output_path.glob("*.pdf"))
        if len(candidates) != 1:
            raise DocxPresentationError("DOCX presentation renderer produced no unique PDF")
        pdf_bytes = candidates[0].read_bytes()
    maximum = max(1, settings.DOCX_PRESENTATION_RENDER_MAX_MB) * 1024 * 1024
    if not pdf_bytes or len(pdf_bytes) > maximum:
        raise DocxPresentationError("DOCX presentation PDF violates the configured size bound")
    return DocxPresentationRender(
        pdf_bytes=pdf_bytes,
        provenance={
            "renderer": "libreoffice_writer_pdf_export",
            "renderer_version": renderer_version,
            "renderer_options": [
                "headless", "nologo", "nodefault", "nofirststartwizard",
                "norestore", "isolated_user_profile", "writer_pdf_Export",
            ],
            "locale": "C.UTF-8",
            "timezone": "UTC",
            **font_evidence,
        },
    )


def _resolve_executable(value: str) -> str:
    candidate = shutil.which(value)
    if candidate is None:
        raise DocxPresentationError("Configured DOCX presentation renderer is unavailable")
    return candidate


@lru_cache(maxsize=4)
def _renderer_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DocxPresentationError("DOCX presentation renderer version is unavailable") from exc
    version = completed.stdout.decode("utf-8", errors="replace").strip().splitlines()
    if completed.returncode != 0 or not version:
        raise DocxPresentationError("DOCX presentation renderer version is unavailable")
    return version[0][:200]


def _reject_renderer_network_dependencies(content: bytes) -> None:
    """Fail closed on linked content that a renderer might dereference."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise DocxPresentationError("DOCX presentation source is invalid") from exc
    with archive:
        for name in archive.namelist():
            lower = name.casefold()
            if lower.endswith(".rels"):
                try:
                    root = ET.fromstring(archive.read(name))
                except ET.ParseError as exc:
                    raise DocxPresentationError("DOCX relationship data is invalid") from exc
                for relationship in root.findall(f"{{{_RELATIONSHIP_NS}}}Relationship"):
                    if relationship.get("TargetMode", "").casefold() != "external":
                        continue
                    relation_type = relationship.get("Type", "").rsplit("/", 1)[-1].casefold()
                    target_scheme = urlparse(relationship.get("Target", "")).scheme.casefold()
                    if relation_type != "hyperlink" or target_scheme not in {"http", "https", "mailto"}:
                        raise DocxPresentationError(
                            "DOCX contains linked content that cannot be rendered safely"
                        )
            if lower.startswith("word/") and lower.endswith(".xml"):
                try:
                    root = ET.fromstring(archive.read(name))
                except ET.ParseError as exc:
                    raise DocxPresentationError("DOCX word-processing XML is invalid") from exc
                instructions = [node.text or "" for node in
                                root.findall(f".//{{{_WORD_NS}}}instrText")]
                instructions += [node.get(f"{{{_WORD_NS}}}instr", "") for node in
                                 root.findall(f".//{{{_WORD_NS}}}fldSimple")]
                # A complete plain HTTP hyperlink is not an external-content
                # command. Do not interpret URL path/host tokens as commands.
                # Keep all other (including partial or compound) instructions
                # under the conservative existing rejection check.
                field_text = " ".join(
                    instruction for instruction in instructions
                    if not re.fullmatch(r'\s*HYPERLINK\s+"https?://[^"\s]+"\s*',
                                        instruction, re.I)
                )
                if _UNSAFE_FIELD.search(field_text):
                    raise DocxPresentationError(
                        "DOCX contains an external-content field that cannot be rendered safely"
                    )


def _font_evidence(content: bytes) -> dict:
    requested = _requested_fonts(content)
    installed = _installed_fonts()
    missing = sorted(name for name in requested if name.casefold() not in installed)
    manifest = "\n".join(sorted(installed)).encode("utf-8")
    requested_manifest = "\n".join(sorted(name.casefold() for name in requested)).encode("utf-8")
    return {
        "installed_font_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "installed_font_family_count": len(installed),
        "requested_font_manifest_sha256": hashlib.sha256(requested_manifest).hexdigest(),
        "requested_font_family_count": len(requested),
        "missing_requested_font_family_count": len(missing),
        "font_substitution_risk": bool(missing),
    }


def _requested_fonts(content: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if "word/fontTable.xml" not in archive.namelist():
            return set()
        try:
            root = ET.fromstring(archive.read("word/fontTable.xml"))
        except ET.ParseError:
            return set()
    return {
        value.strip()
        for node in root.findall(f".//{{{_WORD_NS}}}font")
        if (value := node.get(f"{{{_WORD_NS}}}name")) and value.strip()
    }


@lru_cache(maxsize=1)
def _installed_fonts() -> frozenset[str]:
    executable = shutil.which("fc-list")
    if executable is None:
        return frozenset()
    try:
        completed = subprocess.run(
            [executable, ":", "family"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if completed.returncode != 0:
        return frozenset()
    families: set[str] = set()
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
        families.update(part.strip().casefold() for part in line.split(",") if part.strip())
    return frozenset(families)
