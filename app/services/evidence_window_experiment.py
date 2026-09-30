"""Unconnected P2 response-interface experiment; no dispatch or retrieval.

Callers freeze and authorize original inputs. Window IDs bind exact contiguous
sentences in those inputs, not new evidence or claims of source-wide coverage.
"""
from copy import deepcopy
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services import passage_relevance as relevance
from app.services.llm_input_boundary import enforce_complete_prompt_budget, json_data_envelope
from app.services.verification_evidence import _passage_boundary_status

VERSION = "evidence-window-interface-experiment-v1"
COMPACT_VERSION = "evidence-window-interface-experiment-v3-compact"


class _Record(relevance._Assessment):
    rationale: str = Field(max_length=1000)
    basis: relevance._DisplayObservation.model_fields["basis"].annotation
    claim_token_labels: list[tuple[str, str]] = Field(max_length=4)
    source_window_id: str | None


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessments: list[_Record] = Field(min_length=1, max_length=6)


def _hash(value):
    return hashlib.sha256(json_data_envelope(value).encode()).hexdigest()


def inspect_window_choice(passage, choice):
    """Text-free offline diagnosis; never supply a replacement or salvage advice.

    Rebuild the original menu, rather than trusting a model-supplied menu. This
    diagnostic consumes the original labelled passage, not newly fetched text.
    """
    menu = _windows(passage)
    compact = {key[-1] + str(int(key[1:-1])): value for key, value in menu.items()}
    if choice is None:
        return "no_window_selected"
    if not isinstance(choice, str):
        return "invalid_window_syntax"
    if choice in compact:
        return "listed_window"
    match = re.fullmatch(r"([ab])(0|[1-9][0-9]*)", choice)
    if not match:
        return "invalid_window_syntax"
    start, count = int(match[2]), 1 if match[1] == "a" else 2
    if start + count > len(passage["source_sentences"]):
        return "window_exceeds_supplied_sentences"
    return "window_ineligible_boundary_or_length"


def _windows(passage):
    """Retain the complete text; enumerate eligible one/two-sentence windows."""
    text = passage["text"]
    markers = list(re.finditer(r"\[(s\d{3})\] ", text))
    ids = passage["source_sentences"]
    if (not markers or [m[1] for m in markers] != ids
            or len(ids) != len(set(ids))):
        raise ValueError("ambiguous_source_labels")
    chunks = [text[m.end():markers[i+1].start() if i+1<len(markers) else len(text)]
              for i,m in enumerate(markers)]
    windows = {}
    for start in range(len(chunks)):
        for count in (1, 2):
            if start + count > len(chunks):
                continue
            span = ''.join(chunks[start:start+count]).rstrip()
            if 0 < len(span) <= 1400 and _passage_boundary_status(span) == "sentence_complete":
                windows[f"w{start:03d}{'a' if count == 1 else 'b'}"] = ids[start:start+count]
    return windows


def prepare_window_request(system, prompt, *, max_input_tokens=4000, compact=False):
    """Pure preparation: same source text/order, explicit window and claim IDs."""
    system = relevance.single_record_relevance_prompt(system)
    system = system.replace('claim_token_ranges', 'claim_token_labels').replace('source_sentence_ids', 'source_window_id')
    old = ("claim_token_labels is up to four [first_token, last_token] inclusive pairs from\n"
           "the application-labelled source_attributed_text (for example [2, 8] means\n"
           "tokens t2 through t8).")
    new = ("claim_token_labels is up to four inclusive pairs of exact claim labels\n"
           "(for example [\"t2\",\"t8\"]). Use only supplied claim_token_ids, never\n"
           "source-sentence numbers or token counts.")
    if system.count(old) != 1:
        raise ValueError('incompatible_claim_contract')
    system = system.replace(old, new)
    old = ("source_window_id contains one or two\nconsecutive IDs from this passage's supplied source_sentences.")
    if system.count(old) != 1:
        raise ValueError('incompatible_window_contract')
    system = system.replace(old, "source_window_id is one exact ID from this passage's source_windows.\n"
        "wNNNa means sentence sNNN; wNNNb means sNNN plus the immediately following\n"
        "sentence. Choose only listed IDs; the application reconstructs the window.\n"
        "Use null when no listed window is useful, or none is available.")
    original = json.loads(prompt)
    data = deepcopy(original)
    claim_ids = re.findall(r'\[(t\d+)\] ', data['source_attributed_text'])
    if not claim_ids or claim_ids != [f't{i}' for i in range(len(claim_ids))]:
        raise ValueError('ambiguous_claim_labels')
    data['claim_token_ids'] = claim_ids
    menus = {}
    for passage in data['passages']:
        pid = passage['passage_id']
        if pid in menus:
            raise ValueError('duplicate_candidate')
        menus[pid] = _windows(passage)
        del passage['source_sentences']
        passage['source_windows'] = list(menus[pid])
    version = VERSION
    if compact:
        version = COMPACT_VERSION
        system = system.replace('wNNNa means sentence sNNN; wNNNb means sNNN plus the immediately following',
            'aN means sentence sNNN; bN means sNNN plus the immediately following')
        system += '\nsource_windows and claim_token_ids: space-separated IDs.\n'
        data['claim_token_ids'] = ' '.join(claim_ids)
        for passage in data['passages']:
            pid = passage['passage_id']
            menus[pid] = {key[-1]+str(int(key[1:-1])):value for key,value in menus[pid].items()}
            passage['source_windows'] = ' '.join(menus[pid])
    converted = json_data_envelope(data)
    estimate = enforce_complete_prompt_budget(system, converted, max_input_tokens=min(4000,max_input_tokens))
    return {'version':version, 'system':system, 'prompt':converted, 'windows':menus,
        'claim_ids':claim_ids, 'estimated_tokens':estimate,
        'fingerprint':_hash({'version':version, 'original':original, 'system':system,
                             'converted':data, 'menus':menus})}


def bind_window_response(raw, system, prompt, *, expected_fingerprint, max_input_tokens=4000, compact=False):
    """Recompute the exact menu; never repair an unknown/invalid model choice."""
    prepared = prepare_window_request(system, prompt, max_input_tokens=max_input_tokens, compact=compact)
    if prepared['fingerprint'] != expected_fingerprint:
        raise ValueError('stale_window_request')
    parsed = _Response.model_validate(raw)
    ids = [a.passage_id for a in parsed.assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(prepared['windows']):
        raise ValueError('invalid_candidate_ids')
    assessments, observations = [], {}
    claim_map = {value:i for i,value in enumerate(prepared['claim_ids'])}
    for record in parsed.assessments:
        value = record.model_dump()
        labels = value.pop('claim_token_labels')
        ranges = []
        for first,last in labels:
            if first not in claim_map or last not in claim_map or claim_map[first] > claim_map[last]:
                raise ValueError('invalid_claim_labels')
            ranges.append([claim_map[first],claim_map[last]])
        window = value.pop('source_window_id')
        basis = value.pop('basis')
        if window is not None:
            if window not in prepared['windows'][record.passage_id]:
                raise ValueError('invalid_source_window')
            if basis != 'unclear' and not ranges:
                raise ValueError('unbound_material_observation')
            observations[record.passage_id] = {'basis':basis,'claim_token_ranges':ranges,
                'source_sentence_ids':prepared['windows'][record.passage_id][window]}
        assessments.append(value)
    return relevance._Response.model_validate({'assessments':assessments,'display_observations':observations})


UNMAPPED_VERSION = "evidence-window-unmapped-experiment-v1"


class _UnmappedRecord(relevance._Assessment):
    rationale: str = Field(max_length=1000)
    basis: relevance._DisplayObservation.model_fields["basis"].annotation
    source_window_id: str | None


class _UnmappedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessments: list[_UnmappedRecord] = Field(min_length=1, max_length=6)


UNMAPPED_TAGGED_VERSION = "evidence-window-unmapped-tagged-experiment-v2"


class _TaggedUnmappedResponse(_UnmappedResponse):
    type: Literal['json_object'] = 'json_object'


def prepare_unmapped_request(system, prompt, *, max_input_tokens=4000):
    """Separate experiment: source-only clues, never manufactured claim maps."""
    base = prepare_window_request(system, prompt, compact=True)
    instruction = base['system']
    start = instruction.index('claim_token_labels is up to four')
    end = instruction.index('source_window_id is one exact ID', start)
    instruction = instruction[:start] + (
        'Select source wording useful for inspecting the exact attribution, including\n'
        'material partial, qualifying or contrary evidence. Preserve unresolved meaning;\n'
        'do not output claim spans, facets or support judgments.\n') + instruction[end:]
    instruction = instruction.replace('rationale, basis, claim_token_labels,\nsource_window_id.', 'rationale, basis, source_window_id.')
    instruction = instruction.replace('All eight fields', 'All seven fields')
    instruction = instruction.replace('also supply basis, claim_token_labels and\n', 'also supply basis and\n')
    instruction = instruction.replace('Use unclear with an empty claim_token_labels list when uncertain.', 'Use unclear when uncertain.')
    instruction = instruction.replace('source_windows and claim_token_ids: space-separated IDs.', 'source_windows: space-separated IDs.')
    if 'claim_token_labels' in instruction or 'claim_token_ids' in instruction:
        raise ValueError('incompatible_unmapped_contract')
    data = json.loads(base['prompt'])
    del data['claim_token_ids']
    data['source_attributed_text'] = re.sub(r'\[t\d+\] ', '', data['source_attributed_text'])
    converted = json_data_envelope(data)
    count = enforce_complete_prompt_budget(instruction, converted, max_input_tokens=min(4000,max_input_tokens))
    return {'version':UNMAPPED_VERSION, 'system':instruction, 'prompt':converted,
        'windows':base['windows'], 'estimated_tokens':count,
        'fingerprint':_hash({'version':UNMAPPED_VERSION,'base':base['fingerprint'],
                            'system':instruction,'prompt':converted})}


def bind_unmapped_response(raw, system, prompt, *, expected_fingerprint, max_input_tokens=4000, allow_json_object_tag=False):
    prepared = prepare_unmapped_request(system, prompt, max_input_tokens=max_input_tokens)
    if expected_fingerprint != prepared['fingerprint']:
        raise ValueError('stale_unmapped_request')
    parser = _TaggedUnmappedResponse if allow_json_object_tag else _UnmappedResponse
    parsed = parser.model_validate(raw)
    ids = [a.passage_id for a in parsed.assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(prepared['windows']):
        raise ValueError('invalid_candidate_ids')
    assessments, observations = [], {}
    for record in parsed.assessments:
        value = record.model_dump()
        window, basis = value.pop('source_window_id'), value.pop('basis')
        if window is not None:
            if window not in prepared['windows'][record.passage_id]:
                raise ValueError('invalid_source_window')
            observations[record.passage_id] = {'basis':basis,'claim_token_ranges':[],
                'source_sentence_ids':prepared['windows'][record.passage_id][window]}
        assessments.append(value)
    return relevance._Response.model_validate({'assessments':assessments,'display_observations':observations})


def prepare_unmapped_comparison(artifact, pages, *, source_title, max_input_tokens=4000, full_inspected_context=False):
    """Reuse the context selector with source-only clues, never P7 facet input."""
    from app.services import evidence_context_experiment as context, joint_evidence_selection as joint
    if artifact.quotation_check.evidence_passage_ids or artifact.locator_check.evidence_passage_ids:
        # This development alternative has no accepted protected-span projector.
        raise ValueError('protected_projection_not_accepted')
    inputs = joint._inputs(artifact)
    target, _, candidates, _ = joint._prepare(**inputs, source_title=source_title)
    if target != relevance._source_attributed_relevance_text(artifact.claim):
        raise ValueError('unmapped_target_mismatch')
    assessments = {a.passage_id:a for a in artifact.passage_relevance.assessments}
    passages = {p.passage_id:p for p in artifact.passages}
    regions = []
    for candidate in candidates:
        p, a = passages[candidate.passage_id], assessments[candidate.passage_id]
        lo, hi = a.assessed_text_offset_start, a.assessed_text_offset_end
        text = p.text[lo:hi]
        observation = a.display_observation
        span = observation.source_span if observation else ''
        # A repeated string has no unique local position; retain the full input.
        if (not full_inspected_context and span and text.count(span) == 1 and len(span) <= 1400
                and joint._sentence_span(text, span)):
            offset = text.index(span)
            lo += offset
            text = span
        regions.append(context.Region(p.page_index,p.character_start+lo,
            p.character_start+lo+len(text),text,p.passage_role,(p.passage_id,),'source_only_observation'))
    return context.prepare_comparison(claim=artifact.claim,source_title=source_title,
        regions=regions,pages=pages,source_binding={'artifact_sha256':context.digest(artifact.model_dump(mode='json'))},
        mode=UNMAPPED_VERSION+('/full-inspected-context' if full_inspected_context else ''),max_input_tokens=min(4000,max_input_tokens))


def bind_unmapped_comparison(request, raw, artifact, pages, *, source_title, max_input_tokens=4000, full_inspected_context=False):
    from app.services import evidence_context_experiment as context
    fresh = prepare_unmapped_comparison(artifact,pages,source_title=source_title,max_input_tokens=max_input_tokens,full_inspected_context=full_inspected_context)
    if context.digest(fresh) != context.digest(request):
        raise ValueError('stale_unmapped_comparison')
    return context.bind_comparison(request,raw,pages=pages,claim=artifact.claim,
                                   source_binding=request['source_binding'])
