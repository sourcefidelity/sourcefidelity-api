"""Script comparability guards, not transliteration or identity judgments."""
import unicodedata


def _letter_scripts(value: str) -> set[str]:
    scripts = set()
    for character in unicodedata.normalize("NFKC", value):
        if not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "")
        if name.startswith(("CJK ", "IDEOGRAPHIC ")):
            scripts.add("HAN")
        elif name:
            scripts.add(name.split()[0])
    return scripts


def cross_script_comparison_unresolved(expected: str | None, observed: str | None) -> bool:
    """Disjoint letter scripts cannot establish a lexical identity mismatch.

    Numerals/punctuation are not shared-script evidence. Mixed-script titles
    with a shared script retain existing comparison behavior; this deliberately
    does not attempt language detection or prove equivalent translated titles.
    """
    left = _letter_scripts(expected or "")
    right = _letter_scripts(observed or "")
    return bool(left and right and left.isdisjoint(right))
