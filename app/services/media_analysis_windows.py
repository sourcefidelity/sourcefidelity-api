"""Development-only whole-body accounting around the existing candidate route.

Plans do not dispatch calls. Processing all windows is not semantic completeness
or proof that a reference is missing. No title, identity or role is aggregated.
"""
import hashlib
import json
import re

from app.services.llm_input_boundary import estimate_prompt_tokens, redact_direct_identifiers
from app.services.prompts import build_subject_identification_user_prompt
from app.services.schemas import MediaAnalysisPreflight
from app.services.sentence_splitter import split_paragraphs_and_sentences
from app.services.subject_identifier import MEDIA_ANALYSIS_SYSTEM_PROMPT, bind_media_analysis_candidates


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def media_window_prompt_tokens(text: str) -> int:
    redacted = redact_direct_identifiers(text)
    prompt = build_subject_identification_user_prompt(
        redacted.text, [], len(split_paragraphs_and_sentences(text)))
    return estimate_prompt_tokens(MEDIA_ANALYSIS_SYSTEM_PROMPT, prompt)


def plan_media_analysis_windows(body: str, *, max_input_tokens: int, max_windows: int) -> dict:
    """Pack complete paragraphs without lexical/style selection or truncation."""
    if max_input_tokens <= 0 or max_windows < 0:
        raise ValueError('Positive input limit and nonnegative window allowance required')
    spans = []
    start = 0
    for delimiter in re.finditer(r'\n\s*\n', body):
        spans.append((start, delimiter.start()))
        start = delimiter.end()
    spans.append((start, len(body)))
    spans = [(s, e) for s, e in spans if body[s:e].strip()]
    regions = []
    pending = None
    window_count = 0

    def flush():
        nonlocal pending, window_count
        if pending is None:
            return
        s, e = pending
        state = 'planned' if window_count < max_windows else 'window_allowance_exhausted'
        if state == 'planned':
            window_count += 1
        regions.append(dict(start=s, end=e, sha256=_hash(body[s:e]), status=state,
                            prompt_tokens=media_window_prompt_tokens(body[s:e])))
        pending = None

    for s, e in spans:
        tokens = media_window_prompt_tokens(body[s:e])
        if tokens > max_input_tokens:
            flush()
            regions.append(dict(start=s, end=e, sha256=_hash(body[s:e]),
                                status='paragraph_over_budget', prompt_tokens=tokens))
        elif pending is not None and media_window_prompt_tokens(body[pending[0]:e]) <= max_input_tokens:
            pending = (pending[0], e)
        else:
            flush()
            pending = (s, e)
    flush()
    return dict(version='media-window-plan-v1', body_sha256=_hash(body),
                system_sha256=_hash(MEDIA_ANALYSIS_SYSTEM_PROMPT), max_input_tokens=max_input_tokens,
                max_windows=max_windows, regions=regions, references_withheld=True,
                automatic_findings_enabled=False)


def summarize_media_analysis_windows(body: str, plan: dict, results: dict[int, dict]) -> dict:
    """Revalidate frozen coverage and local bindings before projecting offsets."""
    expected = plan_media_analysis_windows(body, max_input_tokens=plan['max_input_tokens'],
                                          max_windows=plan['max_windows'])
    if plan != expected or any(k not in range(len(plan['regions'])) for k in results):
        raise ValueError('Media plan or result-region binding changed')
    regions = []
    candidates = []
    for index, region in enumerate(plan['regions']):
        state = region['status']
        if state != 'planned':
            if index in results:
                raise ValueError('Result provided for an unapproved region')
        elif index not in results:
            state = 'unattempted'
        else:
            window = body[region['start']:region['end']]
            try:
                result = MediaAnalysisPreflight.model_validate(results[index])
                raw = [dict(title=c.title, passage=window[c.passage_start:c.passage_end],
                            proposed_role=c.proposed_role, media_type=c.media_type,
                            **({'title_role': c.title_role} if result.version == 'media-analysis-candidates-v3' else {}))
                       for c in result.candidates]
                rebound = bind_media_analysis_candidates(raw, window, [],
                    require_title_role=result.version == 'media-analysis-candidates-v3')
                if (result.version != rebound.version or result.body_sha256 != _hash(window)
                        or result.reference_inventory_sha256 != rebound.reference_inventory_sha256
                        or result.candidates != rebound.candidates or result.rejected_count
                        or result.status not in {'bound_candidates', 'no_candidates'}
                        or result.status != rebound.status):
                    state = 'invalid_or_unavailable'
                elif len(result.candidates) >= 12:
                    state = 'candidate_cap_reached'
                else:
                    state = 'processed'
                if state in {'processed', 'candidate_cap_reached'}:
                    for c in result.candidates:
                        value = c.model_dump(mode='json')
                        for key in ['title_start', 'title_end', 'passage_start', 'passage_end']:
                            value[key] += region['start']
                        value['title_occurrences'] = [[s+region['start'], e+region['start']]
                                                      for s, e in c.title_occurrences]
                        candidates.append(dict(region_index=index, **value))
            except (ValueError, TypeError, KeyError):
                state = 'invalid_or_unavailable'
        regions.append(dict(region_index=index, **region, processing_status=state))
    return dict(version='media-window-coverage-v1', body_sha256=_hash(body), regions=regions,
                candidates=candidates, coverage_status=('processed_all_windows' if regions and
                    all(r['processing_status']=='processed' for r in regions) else 'incomplete'),
                semantic_accuracy_assessed=False, reference_absence_assessed=False,
                cross_window_identity_assessed=False, automatic_findings_enabled=False)
