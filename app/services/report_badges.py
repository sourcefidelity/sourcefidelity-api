"""Margin number badges for citations and references, in PDF page points.

Badges sit in the page margin beside the line they label so they never cover
the paper's text. The left margin is measured from the page's own text block;
badges on the same line stack outward. When the left margin is too narrow the
right margin is used, and failing that a small chip is raised just after the
span. Geometry only: rendering and interaction live in the report renderer.
"""

from __future__ import annotations

BADGE_HEIGHT = 10.0
BADGE_FONT = 7.0
BADGE_GAP = 1.5
MARGIN_GAP = 3.0
PAGE_EDGE = 2.0


def badge_width(number: int) -> float:
    return 4.0 + 4.2 * len(str(int(number)))


def _collides(box, boxes) -> bool:
    return any(box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
               for other in boxes)


def text_block_edges(words, fallback_rectangles=()) -> tuple[float, float] | None:
    """Leftmost and rightmost text on a page, from word boxes or overlays."""
    boxes = [(float(w[0]), float(w[2])) for w in words or [] if len(w) >= 4]
    if not boxes:
        boxes = [(float(r[0]), float(r[2])) for r in fallback_rectangles or []]
    if not boxes:
        return None
    return min(x0 for x0, _ in boxes), max(x1 for _, x1 in boxes)


def place_badges(requests: list[dict], *, text_left: float, text_right: float,
                 page_width: float, occupied=()) -> list[dict]:
    """Place one badge per request on one page.

    Each request is ``{template, kind, number, line: (x0, y0, x1, y1)}``,
    where ``line`` is the first rectangle of the labelled span. Returns the
    requests with ``box`` and ``placement`` (``left``, ``right`` or ``chip``).
    """
    boxes = [tuple(float(v) for v in box) for box in occupied]
    placed = []
    # On a shared line the span furthest along the line is placed first, so it
    # sits nearest the text and earlier spans stack outward: the badges then
    # read left to right in the same order as their spans on the line.
    for request in sorted(requests, key=lambda r: (round(r["line"][1]), -r["line"][0], -r["number"])):
        x0, y0, x1, y1 = (float(v) for v in request["line"])
        width, height = badge_width(request["number"]), BADGE_HEIGHT
        top = (y0 + y1) / 2 - height / 2
        box, placement = None, None
        right = text_left - MARGIN_GAP
        while right - width >= PAGE_EDGE:
            candidate = (right - width, top, right, top + height)
            blockers = [b for b in boxes if _collides(candidate, [b])]
            if not blockers:
                box, placement = candidate, "left"
                break
            right = min(b[0] for b in blockers) - BADGE_GAP
        if box is None:
            left = text_right + MARGIN_GAP
            while left + width <= page_width - PAGE_EDGE:
                candidate = (left, top, left + width, top + height)
                blockers = [b for b in boxes if _collides(candidate, [b])]
                if not blockers:
                    box, placement = candidate, "right"
                    break
                left = max(b[2] for b in blockers) + BADGE_GAP
        if box is None:
            # Last resort: a chip raised into the space above the span's end.
            left = min(x1 + 1.0, page_width - width - PAGE_EDGE)
            chip_top = max(0.0, y0 - height + 2.0)
            box, placement = (max(PAGE_EDGE, left), chip_top, max(PAGE_EDGE, left) + width, chip_top + height), "chip"
        boxes.append(box)
        placed.append({**request, "box": box, "placement": placement})
    return placed
