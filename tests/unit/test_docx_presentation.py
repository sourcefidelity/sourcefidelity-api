"""Security and configuration boundaries for controlled DOCX rendering."""

import io
import zipfile

import pytest
from docx import Document

from app.services.docx_presentation import (
    DocxPresentationError,
    _reject_renderer_network_dependencies,
)


def _docx_bytes() -> bytes:
    document = Document()
    document.add_paragraph("The ordinary word link is safe content.")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _replace_archive_part(content: bytes, name: str, value: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(output, "w") as target:
        for entry in source.infolist():
            target.writestr(entry, value if entry.filename == name else source.read(entry))
    return output.getvalue()


def test_ordinary_text_does_not_trigger_external_field_rejection():
    _reject_renderer_network_dependencies(_docx_bytes())


def test_external_image_relationship_is_rejected():
    content = _docx_bytes()
    name = "word/_rels/document.xml.rels"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        relationships = archive.read(name).replace(
            b"</Relationships>",
            (
                b'<Relationship Id="external-image" '
                b'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                b'Target="https://example.invalid/private.png" TargetMode="External"/>'
                b"</Relationships>"
            ),
        )
    unsafe = _replace_archive_part(content, name, relationships)
    with pytest.raises(DocxPresentationError, match="linked content"):
        _reject_renderer_network_dependencies(unsafe)


def test_external_content_field_is_rejected():
    content = _docx_bytes()
    name = "word/document.xml"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        document = archive.read(name).replace(
            b"</w:body>",
            b'<w:p><w:r><w:instrText>INCLUDEPICTURE "https://example.invalid/x"</w:instrText></w:r></w:p></w:body>',
        )
    unsafe = _replace_archive_part(content, name, document)
    with pytest.raises(DocxPresentationError, match="external-content field"):
        _reject_renderer_network_dependencies(unsafe)


@pytest.mark.parametrize('simple', [False, True])
@pytest.mark.parametrize('instruction,blocked', [
    ('HYPERLINK "https://example.invalid/link/article"', False),
    ('HYPERLINK "https://example.invalid/INCLUDETEXT"', False),
    ('HYPERLINK "https://example.invalid/link" INCLUDETEXT "https://example.invalid/x"', True),
    ('INCLUDETEXT "https://example.invalid/x"', True),
    ('LINK Excel.Sheet "https://example.invalid/x"', True),
    ('DDEAUTO application topic', True),
])
def test_hyperlink_arguments_are_not_external_commands(simple, instruction, blocked):
    from xml.sax.saxutils import escape, quoteattr
    content = _docx_bytes()
    name = 'word/document.xml'
    field = (f'<w:fldSimple w:instr={quoteattr(instruction)}/>' if simple else
             f'<w:r><w:instrText>{escape(instruction)}</w:instrText></w:r>')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        document = archive.read(name).replace(
            b'</w:body>', f'<w:p>{field}</w:p></w:body>'.encode())
    candidate = _replace_archive_part(content, name, document)
    if blocked:
        with pytest.raises(DocxPresentationError, match='external-content field'):
            _reject_renderer_network_dependencies(candidate)
    else:
        _reject_renderer_network_dependencies(candidate)
