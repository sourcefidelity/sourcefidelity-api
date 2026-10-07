"""Conservative physical-line URL repair after reference boundaries are known.

Never modify semantic paper text or raw references. Whitespace-only historical
inputs cannot establish a wrap. Ambiguous/opaque continuations remain unchanged.
Existing acquisition, library-permission and SSRF checks still own URL admission.
"""
import hashlib
import re
from urllib.parse import urlsplit

from app.services.schemas import ParsedReference, ReferenceURLRepair
from app.services.text_extractor import _clean_text

_START = re.compile(r"https?://[^\s]+", re.I)
_TOKEN = re.compile(r"[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_text(text)).strip()


def repair_reference_urls(
    references: list[ParsedReference], layout_text: str, *, link_targets: frozenset[str] = frozenset(),
) -> list[ParsedReference]:
    """Return copies only for unique raw-reference-bound, structured wraps.

    A continuation must occupy the entire next physical line and contain URL
    structure (a slash or query assignment). Blank lines, prose, new URLs and
    plain opaque tokens stop reconstruction. Require the complete observed span
    to survive in exactly one parsed raw reference, and require that reference's
    existing locator to be the prefix extracted from the same span.

    A continuation without URL structure ("P ractice_of_...") is accepted only
    when one of the PDF's own link targets begins with the joined pieces and
    the finished URL equals that target (Dena, Paula; 2026-10-02).
    """
    proposals: dict[int, list[tuple[str, ReferenceURLRepair]]] = {}
    for match in _START.finditer(layout_text):
        end = match.end()
        pieces = [match.group()]
        annotated = False
        for _ in range(7):
            next_line = re.match(r"[ \t]*\r?\n[ \t]*([^\r\n]+)", layout_text[end:])
            if not next_line:
                break
            token = next_line[1].strip()
            proven = any(target.startswith("".join(pieces) + token.rstrip('.,;)]>')) for target in link_targets)
            # Also structured (Franchise 2, 2026-10-07): a parameter split in a
            # query the URL already has ("...&utm_sour" / "ce=app_share"),
            # and a whole line that is one long slug of at least four separators
            # ("herlock_holmes_the_awakened_sales_within_10_days"); a short
            # fragment ("ractice_of_Media") still needs a link target's proof.
            query_split = "?" in "".join(pieces) and re.match(r"[\w.-]*=", token)
            slug = len(re.findall(r"[-_]", token)) >= 4 and re.fullmatch(r"[\w-]+/?", token)
            # A line opening with a URL separator cannot start new prose ("_10_days").
            slug = slug or re.match(r"[_\-./?=&%#]\w", token)
            if (
                not _TOKEN.fullmatch(token)
                or _START.search(token)
                or not ("/" in token or re.search(r"[?&][\w.-]+=", token) or query_split or slug or proven)
                or len("".join(pieces)) + len(token) > 8192
            ):
                break
            annotated = annotated or (proven and "/" not in token and not re.search(r"[?&][\w.-]+=", token))
            pieces.append(token)
            end += next_line.end()
        if len(pieces) == 1:
            continue
        observed = layout_text[match.start():end]
        normalized = _normalized(observed)
        prefix = _START.match(normalized).group().rstrip('.,;)]>')
        owners = [i for i, ref in enumerate(references)
                  if ref.url == prefix and not ref.doi and not ref.url_repair
                  and _normalized(ref.raw_ref).count(normalized) == 1]
        if len(owners) != 1:
            continue
        url = "".join(pieces).rstrip('.,;)]>')
        try:
            parsed = urlsplit(url)
            if not parsed.hostname or parsed.username or parsed.password:
                continue
            _ = parsed.port
        except ValueError:
            continue
        if url == prefix or (annotated and url not in link_targets):
            continue
        i = owners[0]
        record = ReferenceURLRepair(
            layout_sha256=hashlib.sha256(layout_text.encode()).hexdigest(),
            raw_reference_sha256=hashlib.sha256(references[i].raw_ref.encode()).hexdigest(),
            original_url=references[i].url, observed_span=observed,
            span_start=match.start(), span_end=end,
        )
        proposals.setdefault(i, []).append((url, record))
    result = list(references)
    for i, choices in proposals.items():
        if len(choices) == 1:
            url, record = choices[0]
            result[i] = references[i].model_copy(update={"url": url, "url_repair": record})
    return result
