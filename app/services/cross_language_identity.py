"""Deterministic identity across writing systems: romanised citations of Han-script works.

`cross-language-identity-v1` (owner decision 2026-10-12). A student who cites
a Han-script work usually gives its title in romanised (pinyin) form, often
followed by a translation. A record or page titled in Han characters cannot be
compared with that by spelling, but it can be romanised deterministically: each
Han character is read as its toneless pinyin syllable (every recorded reading
of a character is allowed, ü read as v or u), and the cited syllables must
spell the same sequence, ignoring spacing, hyphens and punctuation. Names are
compared the same way: a Han-script name's leading character(s) read as the
cited surname, and its next character read with the cited initial.

This replaces no judgment: a title that romanises to the cited one agrees, a
clearly different one conflicts, and anything in between is inconclusive and
left to the existing model comparison (`cross-script-identity-v1`) or to
neutral. Kana, Hangul, Cyrillic and other scripts are not romanised here. A
model-proposed native title is query text only and never passes through this
comparison as an observation.

`pypinyin` is imported lazily; without it every comparison is inconclusive.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

from app.services.other_language import HAN, plain_letters, romanised_title

VERSION = 'cross-language-identity-v1'
# Sequence similarity of the compact romanised strings. At or above the
# first a one-syllable slip ("jiou" for "jiu") is a minor difference; below
# the second the titles are different works (the 0.80 line matches the
# work-identity bound of the bounded review); between them the romanisation
# cannot decide. Measured on the 2026-10-12 smoke candidates: a different
# word of the same sound class scored 0.92, unrelated same-field titles
# 0.55-0.68, a one-syllable slip 0.99.
AGREEMENT_SIMILARITY = 0.95
CONFLICT_SIMILARITY = 0.80
# A cited title this long that the record's title starts with, up to a space
# or separator, names the work with extra words after it (site names,
# editors, keywords).
_PREFIX_MIN_LETTERS = 12
_PREFIX_BOUNDARY = re.compile(r'\s|[【\[(（_|｜《〈]|(?<!-)-(?!-)|–')
_MAX_TITLE_CHARS = 400
_KANA_HANGUL = re.compile('[\u3040-\u30ff\uac00-\ud7af]')
_TITLE_SEPARATORS = re.compile(r'\s*(?:[:：]|——|—|--|－－|\s-\s)\s*')
# Common two-character Han surnames, read before single-character ones.
_COMPOUND_SURNAMES = frozenset({
    '欧阳', '司马', '上官', '诸葛', '东方', '皇甫', '尉迟', '公孙', '慕容', '长孙',
    '宇文', '司徒', '夏侯', '轩辕', '令狐', '端木', '独孤', '南宫', '西门', '万俟',
})


@lru_cache(maxsize=1)
def _pypinyin():
    try:
        import pypinyin  # noqa: F401
        from pypinyin import Style, lazy_pinyin, pinyin
    except Exception:  # noqa: BLE001 - absent library: comparisons are inconclusive
        return None
    return Style, lazy_pinyin, pinyin


def available() -> bool:
    return _pypinyin() is not None


@lru_cache(maxsize=8192)
def character_readings(character: str) -> frozenset[str]:
    """Every toneless reading of one Han character (ü as v and as u)."""
    library = _pypinyin()
    if library is None:
        return frozenset()
    style, lazy, pinyin = library
    readings = set(lazy(character, style=style.NORMAL, errors='ignore'))
    for row in pinyin(character, style=style.NORMAL, heteronym=True, errors='ignore'):
        readings.update(row)
    readings = {plain_letters(r) for r in readings if r}
    readings |= {r.replace('v', 'u') for r in readings if 'v' in r}
    return frozenset(r for r in readings if r.isascii() and r.isalpha())


def _tokens(text: str) -> list[frozenset[str]] | None:
    """Han characters as reading sets, digit/Latin runs as themselves; None
    without the library or without any Han character."""
    value = str(text or '')[:_MAX_TITLE_CHARS]
    if not HAN.search(value) or not available():
        return None
    tokens: list[frozenset[str]] = []
    for match in re.finditer(r'[㐀-䶿一-鿿豈-﫿]|[A-Za-z0-9]+', value):
        piece = match.group(0)
        if HAN.match(piece):
            readings = character_readings(piece)
            if not readings:
                return None
            tokens.append(readings)
        else:
            tokens.append(frozenset({piece.casefold()}))
    return tokens


def romanise(text: str) -> str:
    """Han characters replaced by their usual toneless syllable, space separated.

    Other text is kept as it is. Without the library the text is returned
    unchanged, so callers see no Han-derived matches.
    """
    library = _pypinyin()
    value = str(text or '')
    if library is None or not HAN.search(value):
        return value
    style, lazy, _pinyin = library

    def replace(match: re.Match) -> str:
        syllables = lazy(match.group(0), style=style.NORMAL, errors='ignore')
        return ' ' + ' '.join(plain_letters(s) for s in syllables) + ' '
    return re.sub('[㐀-䶿一-鿿豈-﫿]+', replace, value)


def _compact(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', plain_letters(text))


def _aligns(compact: str, tokens: list[frozenset[str]]) -> bool:
    """Can the cited letters be read exactly as the token sequence?"""
    reachable = {0}
    for readings in tokens:
        reachable = {position + len(r) for position in reachable for r in readings
                     if compact.startswith(r, position)}
        if not reachable:
            return False
    return len(compact) in reachable


def _default_compact(text: str) -> str:
    return _compact(romanise(text))


_SITE_SUFFIX = re.compile(r'\s*(?:\s[-–|]\s|[_｜|]|(?<=[\u3400-\u9fff）)])-(?=[\u3400-\u9fff]))'
                          r'\s*[^-–|_｜]{1,40}$')


def _observed_variants(observed: str) -> list[tuple[str, str]]:
    """The title as given, without a trailing site name, and its main title."""
    variants = [('full', observed)]
    trimmed = _SITE_SUFFIX.sub('', observed).strip()
    if trimmed and trimmed != observed and HAN.search(trimmed):
        variants.append(('full', trimmed))
    for _name, value in list(variants):
        main = _TITLE_SEPARATORS.split(value, maxsplit=1)[0]
        if main and main != value and HAN.search(main):
            variants.append(('main', main))
    # The title before trailing words set off by a space ("... 附录"), for
    # the similarity reading only; an exact reading is the prefix rule.
    head = re.split(r'\s+(?=[\u3400-\u9fff])', observed.strip(), maxsplit=1)[0]
    if head and head != observed.strip() and HAN.search(head):
        variants.append(('head', head))
    return variants


def compare_titles(cited_title: str | None, observed_title: str | None) -> dict | None:
    """Compare a romanised cited title with a Han-script observed title.

    Returns None when the comparison does not apply or cannot decide (no Han
    in the observed title, a cited title that is not romanised (pinyin), the
    library absent, or a similarity between the conflict and agreement
    thresholds). Otherwise a field outcome with its reason and method.
    """
    observed = str(observed_title or '')
    romanised = romanised_title(cited_title)
    if not romanised or not HAN.search(observed) or _KANA_HANGUL.search(observed) or not available():
        # Titles with kana or Hangul are not read in pinyin.
        return None
    cited = _compact(romanised)
    if len(cited) < 6:
        return None
    best = 0.0
    numbered = bool(re.findall(r'\d+', cited))
    same_numbers = not numbered
    for name, variant in _observed_variants(observed):
        tokens = _tokens(variant)
        if not tokens:
            continue
        if _aligns(cited, tokens):
            outcome, reason = {'full': ('agreement', 'romanised_title_match'),
                               'head': ('minor_difference', 'romanised_title_prefix_match')}.get(
                name, ('minor_difference', 'romanised_main_title_match'))
            return {'outcome': outcome, 'reason_code': reason, 'method': VERSION, 'similarity': 1.0}
        compact = _default_compact(variant)
        # A year or number in a title names one work of a series: the 2016
        # report is not the 2017 report, however alike the rest reads.
        if numbered and re.findall(r'\d+', compact) != re.findall(r'\d+', cited):
            continue
        same_numbers = True
        best = max(best, SequenceMatcher(None, cited, compact).ratio())
    if len(cited) >= _PREFIX_MIN_LETTERS:
        for match in _PREFIX_BOUNDARY.finditer(observed):
            prefix = observed[:match.start()].strip()
            tokens = _tokens(prefix) if HAN.search(prefix) else None
            if tokens and _aligns(cited, tokens):
                return {'outcome': 'minor_difference', 'reason_code': 'romanised_title_prefix_match',
                        'method': VERSION, 'similarity': 1.0}
    if not same_numbers:
        return {'outcome': 'material_conflict', 'reason_code': 'romanised_title_conflict',
                'method': VERSION, 'similarity': 0.0}
    if best >= AGREEMENT_SIMILARITY:
        return {'outcome': 'minor_difference', 'reason_code': 'romanised_title_near_match',
                'method': VERSION, 'similarity': round(best, 4)}
    if best and best < CONFLICT_SIMILARITY:
        return {'outcome': 'material_conflict', 'reason_code': 'romanised_title_conflict',
                'method': VERSION, 'similarity': round(best, 4)}
    return None


def latin_title_part_match(cited_title: str | None, observed_title: str | None) -> dict | None:
    """A Latin observed title equal to the cited romanised title or translation.

    "Wang luo chu ban yan jiu (Research on network publishing)" against a
    record titled "Wang luo chu ban yan jiu" agrees; against one titled
    "Research on network publishing" it differs only in form (the record's
    title is the student's translation). None otherwise.
    """
    from app.services.other_language import split_cited_title
    parts = split_cited_title(cited_title)
    if not parts or not observed_title or HAN.search(str(observed_title)):
        return None
    observed = _compact(observed_title)
    if len(observed) < 6:
        return None
    if observed == _compact(parts['romanised']):
        return {'outcome': 'agreement', 'reason_code': 'romanised_title_part_match', 'method': VERSION}
    if observed == _compact(parts['translation']):
        return {'outcome': 'minor_difference', 'reason_code': 'cited_translation_title_match', 'method': VERSION}
    return None


def cited_people(authors) -> list[tuple[str, str]]:
    """(surname, initials) for each person in a cited author list."""
    values = authors if isinstance(authors, (list, tuple)) else [authors]
    people = []
    for value in values:
        text = ' '.join(str(value or '').split())
        text = re.sub(r'(?i),?\s+et\.?\s+al\.?\s*$', '', text)
        for part in re.split(r'(?:,?\s*(?:&|\band\b)\s*|;\s*)', text):
            for segment in re.split(r'(?<=\.)\s*,\s*(?=[^\W\d_][^,]*,)', part):
                segment = segment.strip(' ,')
                if not segment:
                    continue
                if ',' in segment:
                    surname, given = segment.split(',', 1)
                else:
                    words = segment.split()
                    surname, given = (words[-1], ' '.join(words[:-1])) if words else ('', '')
                surname = plain_letters(surname).strip(' .')
                initials = ''.join(w[0] for w in re.findall(r'[^\W\d_]+', plain_letters(given)))
                if surname.isascii() and surname.replace('-', '').isalpha():
                    people.append((surname.replace('-', ''), initials))
    return people


def _han_name_parts(name: str) -> list[tuple[frozenset[str], list[frozenset[str]]]]:
    """Possible (surname readings, given-name readings) splits of a Han name."""
    chars = [c for c in str(name or '') if HAN.match(c)]
    if not 2 <= len(chars) <= 4:
        return []
    splits = []
    if ''.join(chars[:2]) in _COMPOUND_SURNAMES and len(chars) >= 3:
        first, second = character_readings(chars[0]), character_readings(chars[1])
        splits.append((frozenset(a + b for a in first for b in second),
                       [character_readings(c) for c in chars[2:]]))
    splits.append((character_readings(chars[0]), [character_readings(c) for c in chars[1:]]))
    return splits


def _person_matches(person: tuple[str, str], name: str) -> bool:
    surname, initials = person
    for surname_readings, given in _han_name_parts(name):
        if surname not in surname_readings:
            continue
        if not initials or not given:
            return True
        if any(r.startswith(initials[0]) for r in given[0]):
            return True
    return False


def han_name_forms(observed_authors) -> list[str]:
    """Han personal names an author list may hold, in the forms to try.

    One entry may hold several people separated by spaces or list marks
    ("甲乙 丙丁"), or one person written given name first with a space
    ("乙丙 甲", as some indexes display names); each part of two to four
    characters, and a two-part entry joined in both orders, are tried.
    """
    forms: list[str] = []
    for value in observed_authors or []:
        value = str(value or '')
        if not HAN.search(value):
            continue
        parts = [p for p in re.split(r'[\s,，、;；·.]+', value) if HAN.search(p)]
        forms.extend(p for p in parts if 2 <= len(HAN.findall(p)) <= 4)
        if len(parts) == 2 and 2 <= len(HAN.findall(''.join(parts))) <= 4:
            forms.extend([parts[1] + parts[0], parts[0] + parts[1]])
    return list(dict.fromkeys(forms))


def compare_authors(cited_authors, observed_authors) -> dict | None:
    """Compare romanised cited names with Han-script observed names.

    None unless the observed list holds at least one Han personal name of two
    to four characters and the cited list holds parsable personal names.
    """
    han_names = han_name_forms(observed_authors)
    people = cited_people(cited_authors)
    if not han_names or not people or not available():
        return None
    matched = [person for person in people if any(_person_matches(person, n) for n in han_names)]
    if len(matched) == len(people):
        return {'outcome': 'agreement', 'reason_code': 'romanised_author_match', 'method': VERSION}
    if matched:
        return {'outcome': 'minor_difference', 'reason_code': 'romanised_author_partial_match', 'method': VERSION}
    return {'outcome': 'material_conflict', 'reason_code': 'romanised_author_conflict', 'method': VERSION}


# ---- Source text views (admission) -------------------------------------

def applies_to_text(cited_title: str | None, text: str | None) -> bool:
    """The cited title is romanised (pinyin) and the text carries Han characters."""
    return bool(romanised_title(cited_title) and HAN.search(str(text or '')) and available())


def romanised_title_in_text(cited_title: str | None, text: str | None) -> bool:
    """The cited romanised title appears, read character by character, in the text.

    Compared on compact letters (spacing ignored) against the text's default
    reading and, where that misses, against every reading of a window of the
    same number of Han characters.
    """
    romanised = romanised_title(cited_title)
    if not romanised or not available():
        return False
    cited = _compact(romanised)
    if len(cited) < 10:
        return False
    value = str(text or '')
    if cited in _default_compact(value):
        return True
    tokens = _tokens(value[:6000] if len(value) > 6000 else value)
    if not tokens:
        return False
    for start in range(len(tokens)):
        reachable = {0}
        for readings in tokens[start:]:
            reachable = {p + len(r) for p in reachable for r in readings if cited.startswith(r, p)}
            if not reachable:
                break
            if len(cited) in reachable:
                return True
    return False


_HAN_NAME_RUN = re.compile('[㐀-䶿一-鿿豈-﫿]{2,4}')


def byline_names(lines: list[str]) -> list[str]:
    """Han names of two to four characters standing alone in byline lines."""
    names = []
    for line in lines:
        if len(line) > 80:
            continue
        for part in re.split(r'[\s,，、;；·•]+', line.strip()):
            part = re.sub(r'[\d*†‡§¹²³⁴⁵⁶⁷⁸⁹⁰（）()\[\]]+', '', part)
            if _HAN_NAME_RUN.fullmatch(part):
                names.append(part)
    return names[:12]


def credited_people(cited_author: str | None, lines: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Cited people whose romanised names a Han byline shows, and the byline names."""
    names = byline_names(lines)
    people = cited_people(cited_author)
    return [p for p in people if any(_person_matches(p, n) for n in names)], names
