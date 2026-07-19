"""StoredDocument model for the source repository.

Represents verified academic documents: journal articles, books, and
book chapters.  Webpages and student papers are NOT stored in this model.
"""

from datetime import datetime, timezone

from sqlalchemy import String, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class StoredDocument(Base):
    """A verified academic document stored in the source repository."""

    __tablename__ = "stored_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Content classification ──────────────────────────
    content_type: Mapped[str] = mapped_column(String(50))
    # "journal_article" | "book" | "book_chapter"

    license_class: Mapped[str] = mapped_column(String(50))
    # "open_access" | "campus_access" | "instructor_upload"

    # ── Bibliographic metadata ──────────────────────────
    title: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(Text)
    year: Mapped[str] = mapped_column(String(10))
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    isbn: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    journal: Mapped[str | None] = mapped_column(Text, nullable=True)
    publisher: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Storage location ────────────────────────────────
    s3_key: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(10))  # "pdf" | "html"
    file_size_bytes: Mapped[int] = mapped_column(Integer)

    # ── Chapter relationships (edited collections) ──────
    parent_isbn: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    parent_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_editors: Mapped[str | None] = mapped_column(Text, nullable=True)
    chapter_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_range_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_range_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Provenance ──────────────────────────────────────
    provenance: Mapped[str] = mapped_column(String(50))
    # "instructor_upload" | "student_link" | "openalex" | "core" | "semantic_scholar"

    uploaded_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    last_accessed: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    access_count: Mapped[int] = mapped_column(Integer, default=0)

    # ── Retention ───────────────────────────────────────
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    # Used for campus_access content. Null = never expires.

    # ── Verification ────────────────────────────────────
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_method: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    # "doi_match" | "title_match" | "crossref_isbn" | "instructor_attestation"

    # ── Completeness (Phase 3.5 hardening) ──────────────
    logical_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # True page count, N-up corrected (e.g. 512 for a 256-page 2-up scan).

    completeness_verdict: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )
    # "COMPLETE" | "INCOMPLETE" | "UNCERTAIN" | None (not checked)

    # Text quality (Phase 3.5 hardening). Determines how much to trust
    # exact-match results from this document's text layer.
    text_quality: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # "digital" (born-digital, authoritative) | "scan_ocr" (OCR layer, may have
    # errors — treat exact-match failures with lower confidence) | "pure_scan"
    # (no text layer, unusable without OCR)

    review_status: Mapped[str] = mapped_column(String(20), default="accepted")
    # "accepted" | "pending_review" | "rejected"
    # In Standard strictness mode, flagged uploads are held as pending_review
    # and excluded from citation-verification lookups until a reviewer approves.
