"""Bounded discovery screening, not bibliographic identity or source evidence."""
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.services.relevance import score_title_relevance, _significant_tokens, verify_authors, extract_surnames
from app.services.bibliographic_scripts import cross_script_comparison_unresolved

POLICY = 'bounded-reference-review-v9'
Scope = Literal['outside_bound', 'material', 'unknown', 'resolved_different', 'credible']


def text_key(text):
    text = re.sub(r'<[^>]*>', ' ', text or '')
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def input_hash(title, author):
    return hashlib.sha256((text_key(title) + '\x1f' + text_key(author)).encode()).hexdigest()


def scope_v1(title, candidate_title, author='', candidate_authors=()):
    """Low-affinity leads may fall outside scope; that never proves a different work.

    Existing lexical relevance is used only to allocate review effort. Near
    titles, fragments, author matches, cross-script and generic pages stay open.
    """
    left, right = text_key(title), text_key(candidate_title)
    if not left or not right or cross_script_comparison_unresolved(left, right):
        return 'unknown'
    if re.search(r'\b(?:access denied|captcha|sign in|log in|page not found|just a moment|search results)\b', right):
        return 'unknown'
    if candidate_authors and verify_authors(author, list(candidate_authors))[0]:
        return 'material'
    if len(set(_significant_tokens(left))) < 3 or len(set(_significant_tokens(right))) < 3:
        return 'unknown'
    if (left in right or right in left or '…' in right or '...' in right
            or score_title_relevance(left, right).is_relevant
            or score_title_relevance(right, left).is_relevant
            or SequenceMatcher(None, left, right).ratio() >= .65):
        return 'material'
    return 'outside_bound'


def scope_v2(title, candidate_title, author='', candidate_authors=()):
    """A shared topic alone is not a plausible bibliographic match.

    Keep exact/near title identities and plausible omitted subtitles. A known
    author match remains a reason to inspect metadata; a different author does
    not turn a near-identical title into an unrelated lead.
    """
    left, right = text_key(title), text_key(candidate_title)
    if not left or not right or cross_script_comparison_unresolved(left, right):
        return 'unknown'
    if re.search(r'\b(?:access denied|captcha|sign in|log in|page not found|just a moment|search results)\b',right):
        return 'unknown'
    if candidate_authors:
        if cross_script_comparison_unresolved(author, ' '.join(candidate_authors)):
            return 'unknown'
        # The shared retrieval helper accepts missing/unparseable names by
        # default. That is not affirmative author agreement for review scope.
        if (extract_surnames(author) and any(extract_surnames(a) for a in candidate_authors)
                and verify_authors(author,list(candidate_authors))[0]):
            return 'material'
    if len(set(_significant_tokens(left))) < 3 or len(set(_significant_tokens(right))) < 3:
        return 'unknown'
    normalized = lambda value: re.sub(r'[^\w]+',' ',value).strip()
    lnorm, rnorm = normalized(left), normalized(right)
    if lnorm == rnorm or SequenceMatcher(None,lnorm,rnorm).ratio() >= .80:
        return 'material'
    # Source titles often carry a subtitle or a publisher/site suffix.
    for short,long in ((left,right),(right,left)):
        if long.startswith(short) and re.match(r'^\s*[:|–—-]',long[len(short):]):
            return 'material'
    if ('…' in right or '...' in right) and not candidate_authors:
        # A clipped matching title stays open; unrelated titles do not acquire
        # identity plausibility merely because a provider clipped their ending.
        fragment = re.split(r'…|\.\.\.', right, maxsplit=1)[0]
        fragment = re.sub(r'^\s*[\[(]pdf[\])]\s*', '', fragment)
        fragment = normalized(fragment)
        if fragment and (lnorm.startswith(fragment) or
                SequenceMatcher(None, lnorm[:len(fragment)], fragment).ratio() >= .80):
            return 'material'
    return 'outside_bound'


def _review_author_match(author, candidates):
    """Bibliographic scope, not the looser retrieval-relevance author score."""
    def names(value):
        value = ''.join(c for c in unicodedata.normalize('NFKD', value or '')
                        if not unicodedata.combining(c))
        result = []
        for part in re.split(r'\s*(?:&|;|\band\b)\s*', value):
            if ',' in part:
                surname, given = part.split(',', 1)
            else:
                pieces = part.split()
                surname, given = (pieces[-1], ' '.join(pieces[:-1])) if pieces else ('', '')
            surname = re.sub(r'[^a-z]', '', surname.casefold())
            initial = re.search(r'[a-z]', given.casefold())
            if surname:
                result.append((surname, initial[0] if initial else ''))
        return result
    for expected, initial in names(author):
        for value in candidates:
            for observed, other in names(value):
                if initial and other and initial != other:
                    continue
                if expected == observed:
                    return True
                # One substituted/inserted/deleted letter, not arbitrary fuzzy
                # overlap such as Green/Greenberg. Near titles remain protected
                # independently, even when a student's author is incorrect.
                if min(len(expected), len(observed)) >= 5 and abs(len(expected)-len(observed)) <= 1:
                    edits = sum(max(i2-i1,j2-j1) for tag,i1,i2,j1,j2 in
                                SequenceMatcher(None,expected,observed).get_opcodes() if tag != 'equal')
                    if edits == 1:
                        return True
    return False


def scope_v3(title, candidate_title, author='', candidate_authors=(), *, _preserve_author_exposure=False,
             _author_match_material=True):
    left, right = text_key(title), text_key(candidate_title)
    # Reuse the accepted title protections without the broad fuzzy author veto.
    title_scope = scope_v2(title, candidate_title, '', candidate_authors if _preserve_author_exposure else ())
    if title_scope == 'material':
        return title_scope
    if not left or not right or cross_script_comparison_unresolved(left, right):
        return 'unknown'
    if candidate_authors and cross_script_comparison_unresolved(author, ' '.join(candidate_authors)):
        return 'unknown'
    if _author_match_material and _review_author_match(author, candidate_authors):
        return 'material'
    if re.search(r'\b(?:access denied|captcha|sign in|log in|page not found|just a moment|search results)\b',right):
        return 'unknown'
    normalized = lambda value: re.sub(r'[^\w]+',' ',value).strip()
    lnorm, rnorm = normalized(left), normalized(right)
    if lnorm == rnorm or SequenceMatcher(None,lnorm,rnorm).ratio() >= .80:
        return 'material'
    for short, long in ((left,right),(right,left)):
        if long.startswith(short) and re.match(r'^\s*[:|–—-]',long[len(short):]):
            return 'material'
    if ('…' in right or '...' in right) and (not _preserve_author_exposure or not candidate_authors):
        fragment = normalized(re.split(r'…|\.\.\.',right,maxsplit=1)[0])
        if fragment and lnorm.startswith(fragment):
            return 'material'
    if len(set(_significant_tokens(left))) < 3 or len(set(_significant_tokens(right))) < 2:
        return 'unknown'
    return 'outside_bound'


def scope_v4(title, candidate_title, author='', candidate_authors=()):
    # Catalog displays may end the title with the ISBD responsibility slash.
    # Compare the title, not that terminal separator; preserve ellipses and all
    # internal punctuation. This protects plausible main-title corrections,
    # never confirms the candidate or repairs the submitted reference.
    title = str(title or '').strip().rstrip('/').rstrip()
    candidate_title = str(candidate_title or '').strip().rstrip('/').rstrip()
    return scope_v3(title,candidate_title,author,candidate_authors,_preserve_author_exposure=True)


def scope_v5(title, candidate_title, author='', candidate_authors=()):
    """A real author does not make a clearly different title a possible work.

    Keep all title/correction, short/generic, clipped and cross-script guards.
    Outside the bounded identity search is not a proven different-work identity.
    Journal overlap likewise supplies no override; screen metadata is not source
    admission or a negative vote about the reference as a whole.
    """
    title = str(title or '').strip().rstrip('/').rstrip()
    candidate_title = str(candidate_title or '').strip().rstrip('/').rstrip()
    return scope_v3(title, candidate_title, author, candidate_authors,
                    _preserve_author_exposure=True, _author_match_material=False)


def catalog_title_match(title, candidate_title):
    """Exact substantial title before a recognizable catalog display suffix.

    A discovery hint only: no author/edition agreement or identity promotion.
    Preserve subtitle boundaries and reject arbitrary longer work titles.
    """
    normalize = lambda value: re.sub(r'[^\w]+', ' ', text_key(value)).strip()
    expected = normalize(title)
    if len(set(_significant_tokens(expected))) < 3:
        return False
    candidate = text_key(candidate_title)
    for delimiter in re.finditer(r'[:|–—]', candidate):
        if normalize(candidate[:delimiter.start()]) != expected:
            continue
        tail = candidate[delimiter.end():].strip()
        if (re.search(r'free download,? borrow,? and streaming\s*:\s*internet archive$', tail)
                or re.fullmatch(r'(?:worldcat(?:\.org)?|open library)(?:\s*[-|:]\s*catalog)?', tail)):
            return True
    return False


def scope(title, candidate_title, author='', candidate_authors=()):
    if catalog_title_match(title, candidate_title):
        return 'material'
    return scope_v5(title, candidate_title, author, candidate_authors)


def scope_for(version):
    return {'bounded-reference-review-v1':scope_v1,
            'bounded-reference-review-v2':scope_v2,
            'bounded-reference-review-v3':scope_v3,
            'bounded-reference-review-v4':scope_v4,
            'bounded-reference-review-v5':scope_v5,
            'bounded-reference-review-v6':scope,
            'bounded-reference-review-v7':scope,
            'bounded-reference-review-v8':scope,
            'bounded-reference-review-v9':scope}[version]


class ReviewScreen(BaseModel):
    model_config = ConfigDict(extra='forbid')
    policy_version: Literal['bounded-reference-review-v1','bounded-reference-review-v2','bounded-reference-review-v3','bounded-reference-review-v4','bounded-reference-review-v5','bounded-reference-review-v6','bounded-reference-review-v7','bounded-reference-review-v8','bounded-reference-review-v9'] = POLICY
    input_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    candidate_count: int = Field(ge=0)
    outside_bound: int = Field(default=0, ge=0)
    material: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)
    resolved_different: int = Field(default=0, ge=0)
    credible: int = Field(default=0, ge=0)
    # Metadata adapters retain their small returned set. Brave uses only counts.
    observations: list[dict] | None = Field(default=None, max_length=20)

    @model_validator(mode='after')
    def balanced(self):
        if self.candidate_count != sum(getattr(self, field) for field in
                ('outside_bound', 'material', 'unknown', 'resolved_different', 'credible')):
            raise ValueError('Unbalanced bounded review counts')
        if self.observations is not None:
            if len(self.observations) != self.candidate_count:
                raise ValueError('Missing metadata screening observations')
            counts = Counter(item.get('disposition') for item in self.observations)
            if set(counts) - {'outside_bound', 'material', 'unknown'} or any(
                    counts[field] != getattr(self, field) for field in ('outside_bound','material','unknown')):
                raise ValueError('Metadata screening dispositions disagree')
        return self

    def resolved_for(self, title, author):
        return (self.input_sha256 == input_hash(title, author)
                and not (self.material or self.unknown or self.credible))


def screen_metadata(title, author, results):
    observations = [dict(title=(r.title or '')[:2000], authors=r.authors[:30],
        doi=(r.doi or '')[:255], disposition=scope(title, r.title, author or '', r.authors))
        for r in results]
    counts = Counter(item['disposition'] for item in observations)
    return ReviewScreen(input_sha256=input_hash(title, author or ''), candidate_count=len(results),
                        observations=observations, **counts).model_dump(mode='json')
