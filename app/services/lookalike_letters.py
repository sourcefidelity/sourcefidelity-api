"""Latin look-alike letters from other scripts inside Latin words.

A Cyrillic "о" in "The cult оf the love goddess" looks identical to the
reader and defeats every exact title search and comparison (Rita Hayworth
reference 17, 2026-10-16). Searches and title comparisons use the folded
form; the reference keeps the student's exact wording. A word with no Latin
letter (a Russian or Greek word) is left alone.
"""
import re
import unicodedata

# Latin look-alike letters from other scripts inside an otherwise Latin word
# ("The cult оf ..." with a Cyrillic o) would defeat an exact phrase search.
_LOOKALIKES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "ο": "o", "Ο": "O", "α": "a", "ν": "v",
})


def fold_lookalikes(text: str) -> str:
    """Replace other-script look-alike letters inside Latin words."""
    def fix(word: re.Match) -> str:
        value = word.group(0)
        latin = any("LATIN" in unicodedata.name(ch, "") for ch in value)
        return value.translate(_LOOKALIKES) if latin else value
    return re.sub(r"\w+", fix, text or "")
