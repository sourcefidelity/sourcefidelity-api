"""Deterministic signs that a reference cites a work in another language.

Shared by discovery (which runs the native-script search route,
`native-script-route-v1`) and reference verification (which keeps a likely
other-language work neutral until that route completed, `reference-verification-v3`).
Nothing here detects a language or translates: it reads characters and
spelling. A signal decides only that a search in the work's own script is
worth running; it is never evidence of identity.
"""
from __future__ import annotations

import re
import unicodedata

# Han, kana and Hangul.
CJK = re.compile('[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]')
HAN = re.compile('[㐀-䶿一-鿿豈-﫿]')

_PINYIN_SYLLABLE = (r'(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?'
                    r'(?:iang|iong|uang|ueng|iao|ian|ing|uai|uan|van|ang|eng|ong|ai|ei|ao|ou|an|en|er'
                    r'|ia|ie|iu|in|ua|uo|ui|un|ue|ve|vn|a|o|e|i|u|v)')
PINYIN_SYLLABLE = re.compile(_PINYIN_SYLLABLE)
_PINYIN_WORD = re.compile(rf'(?:{_PINYIN_SYLLABLE}){{1,4}}')
# Spellings English words rarely have: an initial zh, x or q, or the finals
# uo, iu, iao, iong.
_PINYIN_MARKED = re.compile(r'^(?:zh|x|q)|uo|iu|iao|iong')
_TITLE_WITH_TRANSLATION = re.compile(
    r'^\s*(?P<romanised>[^()\[\]（）]+?)\s*[(\[（](?P<translation>[^()\[\]（）]+)[)\]）]\s*\.?\s*$')


def plain_letters(word: str) -> str:
    """Case-folded, tone marks and other diacritics removed; ü read as v."""
    folded = unicodedata.normalize('NFD', str(word or '').casefold().replace('ü', 'v').replace('u:', 'v'))
    return ''.join(c for c in folded if not unicodedata.combining(c))


def pinyin_word(word: str) -> str | None:
    plain = plain_letters(word)
    return plain if plain.isascii() and _PINYIN_WORD.fullmatch(plain) else None


def pinyin_like(text: str, *, minimum_words: int = 2) -> bool:
    """Every letter word spells one to four pinyin syllables (digits ignored)."""
    words = re.findall(r"[^\W\d_]+", str(text or ''))
    return len(words) >= minimum_words and all(pinyin_word(w) for w in words)


def _romanised_part_qualifies(romanised: str) -> bool:
    """Conservative: three or more romanised words, every one spelled as one to
    four pinyin syllables, at least two with spellings English words rarely
    have. An English title with a bracketed subtitle fails the spelling test."""
    words = re.findall(r"[^\W\d_]+", romanised)
    syllables = [pinyin_word(w) for w in words]
    return (len(words) >= 3 and all(syllables)
            and sum(bool(_PINYIN_MARKED.search(w)) for w in syllables) >= 2)


def split_cited_title(title: str | None) -> dict | None:
    """The cited romanised title and its bracketed translation as separate fields.

    "Wang luo chu ban yan jiu (Research on network publishing)" gives
    {'raw': ..., 'romanised': 'Wang luo chu ban yan jiu', 'translation':
    'Research on network publishing'}. None when the title is not a romanised
    title followed by a translation of two or more words. The raw title is kept.
    """
    raw = str(title or '')
    match = _TITLE_WITH_TRANSLATION.match(raw)
    if not match or len(re.findall(r'[A-Za-z]+', match['translation'])) < 2:
        return None
    romanised = match['romanised'].strip(' .')
    if not _romanised_part_qualifies(romanised):
        return None
    return {'raw': raw, 'romanised': romanised, 'translation': match['translation'].strip(' .')}


def romanised_title(title: str | None) -> str | None:
    """The part of a cited title that romanises a work's own title, or None."""
    parts = split_cited_title(title)
    if parts:
        return parts['romanised']
    value = str(title or '').strip()
    return value if value and not CJK.search(value) and pinyin_like(value) else None


def other_language_signals(reference) -> list[str]:
    """Deterministic signs that a reference cites a work in another language."""
    from app.services.source_type import has_translation_tool_note
    get = (lambda name: reference.get(name)) if isinstance(reference, dict) else (
        lambda name: getattr(reference, name, None))
    raw = re.sub(r'https?://\S+', ' ', str(get('raw_ref') or ''))
    title = str(get('title') or '')
    signals = []
    if CJK.search(raw) or CJK.search(title):
        signals.append('cjk_characters')
    if split_cited_title(title):
        signals.append('romanised_title_with_translation')
    if has_translation_tool_note(raw):
        signals.append('translation_tool_note')
    return signals


_SCRIPT_RANGES = (
    ('han', HAN),
    ('kana', re.compile('[぀-ヿ]')),
    ('hangul', re.compile('[가-힯]')),
    ('cyrillic', re.compile('[Ѐ-ӿ]')),
    ('greek', re.compile('[Ͱ-Ͽ]')),
    ('arabic', re.compile('[؀-ۿ]')),
    ('hebrew', re.compile('[֐-׿]')),
    ('thai', re.compile('[฀-๿]')),
    ('devanagari', re.compile('[ऀ-ॿ]')),
)


def script_counts(text: str, *, limit: int = 20_000) -> dict[str, int]:
    counts: dict[str, int] = {}
    for character in str(text or '')[:limit]:
        if not character.isalpha():
            continue
        name = 'latin' if character.isascii() or 'LATIN' in unicodedata.name(character, '') else None
        if name is None:
            name = next((script for script, pattern in _SCRIPT_RANGES if pattern.match(character)), 'other')
        counts[name] = counts.get(name, 0) + 1
    return counts


def dominant_script(text: str, *, limit: int = 20_000) -> str | None:
    """The writing system most letters of the text use, or None without letters.

    Some writing mixes kana and Han; a text with a substantial share of kana
    is reported as kana so it is not taken for Han-only text.
    """
    counts = script_counts(text, limit=limit)
    total = sum(counts.values())
    if not total:
        return None
    if counts.get('kana', 0) >= 0.15 * total:
        return 'kana'
    return max(counts, key=counts.get)
