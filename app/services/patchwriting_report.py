"""Report projection of patchwriting findings (owner wording 2026-09-29).

The bounded patchwriting block stored under the job's verification summary
(``patchwriting_at_check``) is projected into ``patchwriting_passages`` when the
report view is built. The projection keeps only what the report shows: each
passage's paper range and words (to bind it to the page), which approved line
applies to each matched source, and for each finding the student's words with
the copied words marked and the source clause they follow. Measures,
thresholds, sentence keys and the policy version stay in the stored block and
are never rendered.

Findings whose student regions overlap become one passage for the paper
highlight. A passage has no window of its own (owner decision 2026-09-29): each
matched source is shown in the window of the citation that contains the
passage and cites that source, else in that source's Reference window.
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Callable

from app.services.patchwriting_at_check import SUMMARY_KEY

# Matched source words at least half inside the source's own quotation marks
# are wording the source quotes from another author (owner wording 2026-09-29).
SOURCE_QUOTED_SHARE = 0.5
KINDS = ("close_paraphrase", "unquoted_verbatim")
_RANK = {"close_paraphrase": 1, "unquoted_verbatim": 2}


def stored_block(job, aggregate: dict | None = None) -> dict | None:
    """The job's current patchwriting block, else the report's copy of it."""
    for holder in (getattr(job, "verification_summary", None), aggregate):
        block = (holder or {}).get(SUMMARY_KEY) if isinstance(holder, dict) else None
        if isinstance(block, dict) and isinstance(block.get("sources"), dict):
            return block
    return None


def _only_conventional(region: dict, start: int, spans: list[tuple[int, int]]) -> bool:
    """A stored close paraphrase that rests on terminology the current list
    treats as conventional no longer reaches the matched-word minimum (2026-10-04)."""
    from app.services import patchwriting as pw
    text = str(region.get("text") or "")
    if not text or region.get("text_truncated") or not spans:
        return False
    raw, counted = pw.matched_content_counts(text, start, spans)
    return counted < raw and counted < pw.PARAPHRASE_MIN_MATCHED


def _only_title_words(region: dict, start: int, spans: list[tuple[int, int]], title: str) -> bool:
    """A close paraphrase whose matched words are the matched source's own title
    words ("port crane automation … harbour logistics" against "Port crane
    automation in harbour logistics"), or an acronym the student puts in
    brackets ("(PCA)"), names the topic: at most one matched
    word lies outside them (owner review, 2026-10-07). A title word among
    otherwise borrowed wording does not excuse the rest."""
    from app.services import patchwriting as pw
    text = str(region.get("text") or "")
    if not text or not title or region.get("text_truncated") or not spans:
        return False
    title_stems = {t.stem for t in pw.tokenize(title) if t.content}
    bracketed = {m.start() + 1 for m in re.finditer(r"\(\s*[A-Za-z]*[A-Z][A-Za-z]*[A-Z][A-Za-z]*\s*\)", text)}
    remaining = 0
    for token in pw.tokenize(text, start):
        if not token.content or not any(a <= token.start and token.end <= b for a, b in spans):
            continue
        if token.stem in title_stems or token.start - start in bracketed:
            continue
        remaining += 1
    return remaining <= 1


def _finding_rows(block: dict, known: Callable[[str], bool],
                  title_of: Callable[[str], str] = lambda _r: "") -> list[dict]:
    rows = []
    for reference_id, entry in (block.get("sources") or {}).items():
        if not isinstance(entry, dict) or entry.get("status") != "compared" or not known(reference_id):
            continue
        for finding in entry.get("findings") or []:
            if not isinstance(finding, dict) or finding.get("kind") not in _RANK:
                continue
            region = finding.get("student_region") or {}
            start, end = region.get("paper_start"), region.get("paper_end")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
                continue
            sentences = [s for s in finding.get("source_sentences") or []
                         if isinstance(s, dict) and str(s.get("text") or "").strip()]
            if not sentences:
                continue
            try:
                share = float((finding.get("measures") or {}).get("source_quoted_share") or 0.0)
            except (TypeError, ValueError):
                share = 0.0
            spans = [(span["paper_start"], span["paper_end"])
                     for span in finding.get("student_matched_spans") or []
                     if isinstance(span, dict) and isinstance(span.get("paper_start"), int)
                     and isinstance(span.get("paper_end"), int)]
            from app.services.text_extractor import _LICENCE_LINE
            if _LICENCE_LINE.search(str(region.get("text") or "")) and "://" in str(region.get("text") or ""):
                continue   # a page's licence footer read as paper text (2026-10-07)
            if finding["kind"] == "close_paraphrase" and (
                    _only_conventional(region, start, spans)
                    or _only_title_words(region, start, spans, title_of(reference_id))):
                continue
            sentence = finding.get("student_sentence") or {}
            rows.append({
                "reference_id": reference_id, "kind": finding["kind"],
                "source_quoted": share >= SOURCE_QUOTED_SHARE,
                "start": start, "end": end,
                "text": str(region.get("text") or ""),
                "truncated": bool(region.get("text_truncated")),
                "sentence": (sentence.get("paper_start"), sentence.get("paper_end")),
                "spans": [(span["paper_start"], span["paper_end"])
                          for span in finding.get("student_matched_spans") or []
                          if isinstance(span, dict) and isinstance(span.get("paper_start"), int)
                          and isinstance(span.get("paper_end"), int)],
                "sentences": sentences,
            })
    rows.sort(key=lambda row: (row["start"], row["end"], row["reference_id"]))
    return rows


def _passage_text(rows: list[dict]) -> str | None:
    """The union region's exact words, when every part joins consistently."""
    characters: dict[int, str] = {}
    for row in rows:
        if row["truncated"] or len(row["text"]) != row["end"] - row["start"]:
            return None
        for offset, character in enumerate(row["text"]):
            if characters.setdefault(row["start"] + offset, character) != character:
                return None
    start, end = min(row["start"] for row in rows), max(row["end"] for row in rows)
    if len(characters) != end - start:
        return None
    return "".join(characters[position] for position in range(start, end))


_WORD = re.compile(r"[^\W_](?:[\w'’]*[^\W_])?")
# Words of a copied run shown with the matched content words: a run of at
# least this many identical words that contains a matched content word.
MIN_COPIED_RUN = 2
_CLOSING = ' \t\n.,;:!?)]"”’'


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group().casefold().replace("’", "'"), m.start(), m.end()) for m in _WORD.finditer(text)]


def _inside(token: tuple[str, int, int], spans: list[tuple[int, int]]) -> bool:
    return any(start < token[2] and token[1] < end for start, end in spans)


def _copied_runs(student: list, source: list, student_spans: list, source_spans: list) -> tuple[set, set]:
    """Token indexes of identical runs that include a matched content word."""
    matcher = SequenceMatcher(None, [t[0] for t in student], [t[0] for t in source], autojunk=False)
    marked_student, marked_source = set(), set()
    for a, b, size in matcher.get_matching_blocks():
        if size < MIN_COPIED_RUN:
            continue
        if any(_inside(student[a + k], student_spans) or _inside(source[b + k], source_spans)
               for k in range(size)):
            marked_student.update(range(a, a + size))
            marked_source.update(range(b, b + size))
    return marked_student, marked_source


def _segments(text: str, bold: list[tuple[int, int]]) -> list[dict]:
    """Text in order with the given character ranges marked as copied."""
    segments: list[dict] = []

    def add(value: str, copied: bool) -> None:
        if not value:
            return
        if segments and segments[-1]["copied"] == copied:
            segments[-1]["text"] += value
        else:
            segments.append({"text": value, "copied": copied})

    position = 0
    for low, high in sorted(bold):
        low = max(low, position)
        if high <= low:
            continue
        if segments and segments[-1]["copied"] and not text[position:low].strip():
            add(text[position:high], True)
        else:
            add(text[position:low], False)
            add(text[low:high], True)
        position = high
    add(text[position:], False)
    return segments


def _source_clause(sentence: dict, student: list, student_spans: list) -> tuple[str, set, list]:
    """The part of a source sentence the student follows, with ellipses where cut.

    The clause runs from the first to the last copied word; a sentence whose
    stored text cannot be bound to its offsets is shown whole.
    """
    from app.services.text_quality import readable_text

    text = str(sentence["text"])
    origin, stop = sentence.get("absolute_start"), sentence.get("absolute_end")
    whole = readable_text(text.strip())
    if (sentence.get("text_truncated") or not isinstance(origin, int) or not isinstance(stop, int)
            or len(text) != stop - origin):
        return whole, set(), [{"text": whole, "copied": False}]
    spans = [(span["absolute_start"] - origin, span["absolute_end"] - origin)
             for span in sentence.get("matched_spans") or []
             if isinstance(span, dict) and isinstance(span.get("absolute_start"), int)
             and isinstance(span.get("absolute_end"), int)]
    source = _tokens(text)
    marked_student, marked_source = _copied_runs(student, source, student_spans, spans)
    bounds = [(token[1], token[2]) for index, token in enumerate(source)
              if index in marked_source or _inside(token, spans)]
    if not bounds:
        return whole, marked_student, [{"text": whole, "copied": False}]
    low, high = min(b[0] for b in bounds), max(b[1] for b in bounds)
    clause = readable_text(text[low:high])
    lead = "… " if text[:low].strip() or sentence.get("cut_before") else ""
    tail = " …" if text[high:].strip(_CLOSING) or sentence.get("cut_after") else ""
    # The source's copied words in bold, as on the student side (owner request
    # 2026-10-02). Bold only when the clause is the stored text unchanged.
    if clause == text[low:high]:
        segments = _segments(clause, [(a - low, b - low) for a, b in bounds])
    else:
        segments = [{"text": clause, "copied": False}]
    segments = ([{"text": lead, "copied": False}] if lead else []) + segments + (
        [{"text": tail, "copied": False}] if tail else [])
    return lead + clause + tail, marked_student, segments


def _student_segments(row: dict, marked: set, student: list) -> list[dict]:
    """The student's words in order, copied words marked; ellipses where the sentence continues."""
    text, start = row["text"], row["start"]
    if row["truncated"] or len(text) != row["end"] - start:
        return [{"text": text, "copied": False}] if text else []
    spans = [(s - start, e - start) for s, e in row["spans"]]
    bold = [(t[1], t[2]) for index, t in enumerate(student) if index in marked or _inside(t, spans)]
    segments: list[dict] = []

    def add(value: str, copied: bool) -> None:
        if not value:
            return
        if segments and segments[-1]["copied"] == copied:
            segments[-1]["text"] += value
        else:
            segments.append({"text": value, "copied": copied})

    position = 0
    for low, high in sorted(bold):
        if low < position:
            low = position
        if high <= low:
            continue
        if segments and segments[-1]["copied"] and not text[position:low].strip():
            add(text[position:high], True)   # one run across the space between copied words
        else:
            add(text[position:low], False)
            add(text[low:high], True)
        position = high
    add(text[position:], False)
    sentence_start, sentence_end = row["sentence"]
    if isinstance(sentence_start, int) and start > sentence_start:
        segments.insert(0, {"text": "… ", "copied": False})
    if isinstance(sentence_end, int) and sentence_end - row["end"] > 2:
        segments.append({"text": " …", "copied": False})
    return segments


def _comparison(row: dict) -> dict:
    """One finding: the student's words, then the source clause(s) they follow."""
    spans = [(s - row["start"], e - row["start"]) for s, e in row["spans"]]
    student = _tokens(row["text"])
    marked: set = set()
    excerpts: list[dict] = []
    ordered = sorted(row["sentences"], key=lambda s: (
        s.get("page_index") if isinstance(s.get("page_index"), int) else -1,
        s.get("absolute_start") if isinstance(s.get("absolute_start"), int) else -1))
    for sentence in ordered:
        clause, copied, segments = _source_clause(sentence, student, spans)
        marked |= copied
        page_index = sentence.get("page_index")
        page = sentence.get("page_label") or (page_index + 1 if isinstance(page_index, int) else None)
        page = str(page) if page not in (None, "") else None
        if excerpts and excerpts[-1]["page"] == page:
            excerpts[-1]["text"] += " " + clause
            excerpts[-1]["segments"] += [{"text": " ", "copied": False}] + segments
        else:
            excerpts.append({"page": page, "text": clause, "segments": segments})
    return {"student": _student_segments(row, marked, student), "excerpts": excerpts}


def build_passages(block: dict | None, reference_view: Callable[[str], dict | None]) -> list[dict]:
    """Merge overlapping findings into numbered passages; never raises on bad rows."""
    if not block:
        return []
    views: dict[str, dict | None] = {}

    def known(reference_id: str) -> bool:
        if reference_id not in views:
            views[reference_id] = reference_view(reference_id)
        return views[reference_id] is not None

    groups: list[list[dict]] = []
    end = -1
    def title_of(reference_id: str) -> str:
        view = views.get(reference_id) or {}
        return str(view.get("title") or (view.get("source") or {}).get("title") or "")

    for row in _finding_rows(block, known, title_of):
        if groups and row["start"] < end:
            groups[-1].append(row)
            end = max(end, row["end"])
        else:
            groups.append([row])
            end = row["end"]
    passages = []
    for number, rows in enumerate(groups, 1):
        sources = []
        for reference_id in dict.fromkeys(row["reference_id"] for row in rows):
            own = [row for row in rows if row["reference_id"] == reference_id]
            kind = max((row["kind"] for row in own), key=_RANK.__getitem__)
            wording = "source_quoted" if all(row["source_quoted"] for row in own) else kind
            sources.append({"reference_id": reference_id, "kind": kind, "wording": wording,
                            "comparisons": [_comparison(row) for row in own], "source": views[reference_id]})
        text = _passage_text(rows)
        passages.append({
            "number": number,
            "paper_character_start": min(row["start"] for row in rows),
            "paper_character_end": max(row["end"] for row in rows),
            "kind": max((source["kind"] for source in sources), key=_RANK.__getitem__),
            "student_text": text,
            # Parts bind separately when the union cannot be joined exactly.
            "student_parts": ([] if text is not None else
                              [{"start": row["start"], "end": row["end"], "text": row["text"]}
                               for row in rows if row["text"]]),
            "sources": sources,
            "paper_location": {"localization_level": "semantic_only", "rectangles": []},
        })
    return passages


def _merge_rectangles(rectangles: list[dict]) -> list[dict]:
    """One box per line: parts bound separately may overlap on the same line."""
    merged: list[dict] = []
    for item in sorted(rectangles, key=lambda r: (r["page_index"], r["y0"], r["x0"])):
        last = merged[-1] if merged else None
        if (last and last["page_index"] == item["page_index"]
                and min(last["y1"], item["y1"]) - max(last["y0"], item["y0"])
                > 0.5 * min(last["y1"] - last["y0"], item["y1"] - item["y0"])
                and item["x0"] <= last["x1"] + 1):
            last.update(x0=min(last["x0"], item["x0"]), y0=min(last["y0"], item["y0"]),
                        x1=max(last["x1"], item["x1"]), y1=max(last["y1"], item["y1"]))
        else:
            merged.append(dict(item))
    return merged


def attach_passage_geometry(passages: list[dict], pdf_content: bytes, citations: list[dict]) -> None:
    """Bind each passage's exact words to the page with the citation-anchor binder.

    A passage inside one citation uses that citation's wording as context to
    choose among repeated occurrences. A passage whose words do not bind to one
    place stays unplaced; its window is still reached from the summary.
    """
    from app.services.presentation_anchors import bind_citations_to_pdf
    from app.services.schemas import InTextCitation

    probes, contexts, owners = [], [], []
    for passage in passages:
        passage["paper_location"] = {"localization_level": "semantic_only", "rectangles": []}
        start, end = passage["paper_character_start"], passage["paper_character_end"]
        context = next((c.get("student_text") for c in citations
                        if isinstance(c.get("paper_character_start"), int)
                        and c["paper_character_start"] <= start and end <= c.get("paper_character_end", -1)
                        and c.get("student_text")), None)
        parts = ([{"start": start, "end": end, "text": passage["student_text"]}]
                 if passage.get("student_text") else passage.get("student_parts") or [])
        for part in parts:
            if not str(part.get("text") or "").strip():
                continue
            if context:
                contexts.append(context)
            probes.append(InTextCitation(
                text=part["text"], passage_start=part["start"], passage_end=part["end"],
                paragraph_index=len(contexts) - 1 if context else 10**9))
            owners.append(passage)
    if not probes:
        return
    artifact = bind_citations_to_pdf(pdf_content, citations=probes, paragraphs=contexts)
    anchors = {(a.passage_start, a.passage_end, a.citation_text_sha256): a for a in artifact.anchors}
    bound: dict[int, list[dict] | None] = {}
    for probe, passage in zip(probes, owners):
        key = id(passage)
        anchor = anchors.get((probe.passage_start, probe.passage_end,
                              hashlib.sha256(probe.text.encode("utf-8")).hexdigest()))
        if anchor is None or anchor.localization_level != "exact_rectangle":
            bound[key] = None
        elif bound.get(key, []) is not None:
            bound[key] = (bound.get(key) or []) + [r.model_dump(mode="json") for r in anchor.rectangles]
    for passage in passages:
        rectangles = bound.get(id(passage))
        if rectangles:
            passage["paper_location"] = {
                "localization_level": "exact_rectangle",
                "rectangles": _merge_rectangles(rectangles),
                "geometry_provenance": {"presentation_sha256": artifact.presentation_sha256,
                                        "method": "citation_anchor_binder"},
            }


def passage_windows(passage: dict, citations: list[dict]) -> list[dict]:
    """Where each matched source of a passage is shown.

    The citation whose statement contains the passage (else overlaps it most)
    and cites that source; otherwise that source's Reference window.
    """
    start, end = passage.get("paper_character_start"), passage.get("paper_character_end")
    rows = []
    for item in passage.get("sources") or []:
        reference_id = item.get("reference_id")
        best = None
        for number, citation in enumerate(citations or [], 1):
            low, high = citation.get("paper_character_start"), citation.get("paper_character_end")
            if not all(isinstance(v, int) for v in (low, high, start, end)):
                continue
            overlap = min(high, end) - max(low, start)
            members = [i for i, m in enumerate(citation.get("members") or []) if m.get("reference_id") == reference_id]
            if overlap > 0 and members and (best is None or overlap > best[0]):
                best = (overlap, number, members[0])
        rows.append({"item": item, "reference_id": reference_id,
                     **({"citation": best[1], "member_index": best[2]} if best else {"citation": None})})
    return rows


def summary_counts(passages: list[dict] | None) -> dict[str, list[dict]]:
    """Passages by summary line: a source-quoted passage counts by its kind."""
    found: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    for passage in passages or []:
        if passage.get("kind") in found and isinstance(passage.get("number"), int):
            found[passage["kind"]].append(passage)
    return found
