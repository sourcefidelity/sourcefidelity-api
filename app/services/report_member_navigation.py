"""Read-only source-member targets over retained paper word geometry."""
import re
import unicodedata
import math


def availability_tone(member):
    coverage = member.get('coverage_level')
    return ('evidence_available' if coverage == 'full_text' else
            'limited_evidence' if coverage == 'abstract_only' else
            'partial_evidence' if coverage == 'partial_text' else 'not_assessed')


def missing_reference_targets(citation, words_by_page):
    """Locate only retained missing-author/year markers using the shared mapper."""
    members=[]
    for label in citation.get('missing_reference_members') or []:
        match=re.fullmatch(r'(.+?),\s*((?:19|20)\d{2}[a-z]?)',str(label))
        if not match:return []
        members.append({'source':{'author':match[1],'year':match[2]},'coverage_level':'unavailable'})
    if not members:return []
    return member_targets({**citation,'members':members},words_by_page)


def _token(value):
    return ''.join(c for c in unicodedata.normalize('NFKC',str(value)).casefold() if c.isalnum())


def _pieces(text):
    """Tokens of one whitespace-delimited word, split at internal commas.

    "(Bennett,2007)." is the author and the year run together; splitting only
    at whitespace left one token that matched neither. The same split is
    applied to the citation marker and the paper words, so a spaced citation
    tokenizes exactly as before.
    """
    parts = [part for part in re.split(r'[,，]', str(text)) if part != ''] or [str(text)]
    return [_token(part) for part in parts]


def marker_words(document):
    """Split joined prose/citation words at real glyph boundaries, display only.

    Native word indices used by annotations must remain unchanged.
    """
    pages = {}
    for page in document:
        words = page.get_text('words', sort=True)
        boundary = r'(?<=\S)[（(]|(?<=[）)])(?=[.!?]?[A-Z])'
        joined = any(re.search(boundary,w[4]) for w in words)
        chars = [c for b in page.get_text('rawdict', sort=True)['blocks'] for line in b.get('lines', [])
                 for span in line.get('spans', []) for c in span.get('chars', [])] if joined else []
        result = []
        for word in words:
            cuts = [m.start() for m in re.finditer(boundary, word[4])]
            glyphs = [c for c in chars if word[0] <= (c['bbox'][0]+c['bbox'][2])/2 <= word[2]
                      and word[1] <= (c['bbox'][1]+c['bbox'][3])/2 <= word[3]] if cuts else []
            if not cuts or ''.join(c['c'] for c in glyphs) != word[4]:
                result.append(word)
                continue
            for start, end in zip([0]+cuts, cuts+[len(glyphs)]):
                boxes = [c['bbox'] for c in glyphs[start:end]]
                result.append((min(b[0] for b in boxes),min(b[1] for b in boxes),
                               max(b[2] for b in boxes),max(b[3] for b in boxes),word[4][start:end]))
        pages[page.number] = result
    return pages


def member_targets(citation, words_by_page):
    """Highlight parenthetical members in full, narrative years narrowly.

    Require the token in the retained citation marker, unique source ownership,
    and exactly one matching paper-word sequence. Shared targets remain unplaced.
    This is presentation geometry, never a source-identity or annotation anchor.
    """
    members = citation.get('members', [])
    marker = str(citation.get('citation_marker') or '')
    marker_tokens = [_token(t) for t in re.findall(r"[\w’'-]+",marker)]
    def contains(tokens, pattern):
        return any(tokens[i:i+len(pattern)] == pattern for i in range(len(tokens)-len(pattern)+1))
    candidates = []
    for member in members:
        source = member.get('source') or {}
        year = _token(source.get('year') or '')
        author = str(source.get('author') or '').split(',')[0].strip()
        author_tokens = [_token(t) for t in author.split() if _token(t)]
        patterns = [[year]] if year else []
        if author_tokens:
            patterns.append(author_tokens)
        if source.get('source_kind') == 'traditional_media':
            title = re.sub(r'\s*\[[^]]+\]\s*$', '', str(source.get('title') or ''))
            title_tokens = [_token(t) for t in title.split() if _token(t)]
            if title_tokens:
                patterns.append(title_tokens)
        candidates.append([pattern for pattern in patterns if contains(marker_tokens,pattern)])
    words = []
    for rect in (citation.get('paper_location') or {}).get('rectangles', []):
        try:
            page = int(rect['page_index'])
            coordinates=[float(rect[key]) for key in ('x0','y0','x1','y1')]
            if not all(math.isfinite(v) for v in coordinates) or not (0 <= coordinates[0] < coordinates[2] and 0 <= coordinates[1] < coordinates[3]):
                continue
        except (KeyError,TypeError,ValueError,OverflowError):
            continue
        for word in words_by_page.get(page, words_by_page.get(str(page), [])):
            x,y = (word[0]+word[2])/2,(word[1]+word[3])/2
            if coordinates[0] <= x <= coordinates[2] and coordinates[1] <= y <= coordinates[3]:
                item=(page,tuple(word[:5]))
                if item not in words:
                    words.append(item)
    targets=[]
    # Positions below index token slots; a comma-joined word supplies several
    # slots that all map back to its one rectangle.
    slots=[(token,index) for index,(page,word) in enumerate(words) for token in _pieces(word[4])]
    text_tokens=[token for token,_ in slots]
    # Locate actual marker segments first; a matching date in the prose is not
    # a citation target when the marker itself is absent or repeated.
    marker_positions=set()
    segments=[]
    depth=0
    for segment in marker.split(';'):
        segment = re.sub(r'(?<=\S)([（(])', r' \1', segment)
        parenthetical=depth > 0 or segment.lstrip().startswith(('(', '（'))
        segment_tokens=[token for piece in segment.split() for token in _pieces(piece)]
        starts=[i for i in range(len(slots)-len(segment_tokens)+1)
                if segment_tokens and text_tokens[i:i+len(segment_tokens)] == segment_tokens]
        if len(starts)==1:
            positions=set(range(starts[0],starts[0]+len(segment_tokens)))
            marker_positions.update(positions)
            # Only uniquely identifying patterns can establish ownership.
            owners={i for i,patterns in enumerate(candidates) if any(
                sum(pattern in others for others in candidates)==1
                and contains(segment_tokens,pattern) for pattern in patterns)}
            segments.append((positions,parenthetical,owners))
        depth=max(0,depth+segment.count('(')+segment.count('（')-segment.count(')')-segment.count('）'))
    for index, tokens in enumerate(candidates):
        for pattern in tokens:
            if sum(pattern in others for others in candidates) != 1:
                continue
            matches=[list(range(i,i+len(pattern))) for i in range(len(slots)-len(pattern)+1)
                     if text_tokens[i:i+len(pattern)] == pattern
                     and set(range(i,i+len(pattern))) <= marker_positions]
            if len(matches) != 1:
                continue
            positions=matches[0]
            enclosing=[(span,parenthetical) for span,parenthetical,owners in segments
                       if owners=={index} and set(positions)<=span]
            if len(enclosing)==1:
                span,parenthetical=enclosing[0]
                if parenthetical:
                    positions=sorted(span)
                else:
                    year=_token((members[index].get('source') or {}).get('year') or '')
                    year_positions=[p for p in span if year and text_tokens[p]==year]
                    if len(year_positions)==1:
                        positions=year_positions
            member_rectangles=[]
            for position in sorted({slots[p][1] for p in positions}):
                page,word=words[position]
                rectangle={'member_index':index,'page_index':page,
                    'x0':word[0],'y0':word[1],'x1':word[2],'y1':word[3],
                    'tone':availability_tone(members[index])}
                previous=member_rectangles[-1] if member_rectangles else None
                if (previous and previous['page_index']==page and word[0]>=previous['x1']-1
                    and min(previous['y1'],word[3])-max(previous['y0'],word[1])
                        > .5*min(previous['y1']-previous['y0'],word[3]-word[1])):
                    previous.update(x1=word[2],y0=min(previous['y0'],word[1]),y1=max(previous['y1'],word[3]))
                else:
                    member_rectangles.append(rectangle)
            targets.extend(member_rectangles)
            break
    return targets
