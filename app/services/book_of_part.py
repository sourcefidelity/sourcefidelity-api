"""The single-authored book a cited chapter belongs to: one rule.

The report's Sources to Upload, the resolver's search for the book's text and
upload matching each chose the book's title their own way, so the report could
ask for a book that an upload under that title was then refused for (rules
audit 2026-10-18). Owner decision 2026-10-18: a chapter of a single-authored
book is treated as the book, as the catalogue records it.
"""
from __future__ import annotations

import re


def cited_book_title(container_title: str) -> str:
    """The book a cited chapter names, without its edition note."""
    from app.services.reference_verification import _EDITION
    container = str(container_title or "").strip()
    return re.sub(r"\s+", " ", _EDITION.sub(" ", container)).strip(" .,") if container else ""


def book_of_cited_part(container: dict | None, *, cited_container_title: str = "",
                       cited_author: str = "") -> dict | None:
    """{"title", "author", "authors", "year"} of the identified single-authored
    book, or None when the containing work is not one.

    Title: the catalogue record's, else the containing work's as identified,
    else the book title as cited (edition note removed). Author: the record's
    single author, else the cited author.
    """
    container = container or {}
    if not container.get("is_monograph"):
        return None
    record = container.get("located_record") or {}
    title = str(record.get("title") or container.get("title")
                or cited_book_title(container.get("searched_title") or cited_container_title) or "").strip()
    authors = [a for a in (record.get("authors") or container.get("authors") or []) if a]
    if not title:
        return None
    return {"title": title, "authors": authors,
            "author": authors[0] if len(authors) == 1 else str(cited_author or ""),
            "year": str(record.get("year") or "")}


def book_identities(container: dict | None, *, cited_container_title: str, cited_author: str) -> list[tuple[str, str]]:
    """(title, author) pairs an uploaded whole book is matched against: the
    identified book first, then the book title as cited."""
    pairs = []
    book = book_of_cited_part(container, cited_container_title=cited_container_title, cited_author=cited_author)
    if book:
        pairs.append((book["title"], book["author"] or cited_author))
    cited = cited_book_title(cited_container_title)
    if cited and (cited, cited_author) not in pairs:
        pairs.append((cited, cited_author))
    return pairs
