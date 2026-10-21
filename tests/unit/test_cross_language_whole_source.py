"""The whole source for the selector and Judgment across writing systems.

cross-language-whole-source-v1 (owner decision 2026-10-13). Invented fixtures only.
"""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services.candidate_relationship_judgment import attach_verification_candidates
from app.services.citation_use_router import attach_citation_use_routes
from app.services.cross_language_evidence import evidence_record, language_profile
from app.services.cross_language_whole_source import (
    VERSION,
    absolute_key,
    attach_whole_source,
    stored_passages,
)
from app.services.evidence_report import _with_sentence_evidence, evidence_sentence_key
from app.services.evidence_sentence_selection import (
    attach_sentence_evidence,
    prepare_request,
    prepare_request_from_record,
)
from app.services.facet_evidence_judgment import _passage_sentences, attach_facet_evidence_foundation
from app.services.judgment_input import judgment_context_from_payload
from app.services.judgment_report import _sentences
from app.services.verification_evidence import (
    AuthorizedRepresentation,
    ClaimEvidence,
    attach_candidate_passage_retrieval,
    build_passage_evidence,
)
from app.services.verification_report import (
    ReportAuthorizationError,
    _validate_facet_evidence,
    build_inspectable_report_payload,
)
from tests.unit.test_one_layout_evidence import ROUTE

STATEMENT = 'Institutions must ask before collecting personal data (Pan & Lin, 2016).'
# The sentence that matters shares no word with the translated query.
CONSENT = '数据权利保护需要明确的法律规则，个人数据的收集应当取得同意。'


def _han_source_text(paragraphs: int) -> str:
    parts = [''.join(f'第{i}{j}条规定示例机构应当公开第{j}项信息。' for j in range(6)) for i in range(paragraphs)]
    parts.append(CONSENT)
    return '\n\n'.join(parts)


def _source(text):
    content = text.encode('utf-8')
    now = datetime.now(timezone.utc)
    return AuthorizedRepresentation(
        representation_id='han-source', canonical_work_id='han-work', content_object_id='han-object',
        content_sha256=hashlib.sha256(content).hexdigest(), content=content, representation_kind='plain_text',
        media_type='text/plain', provenance='authorized_upload', scope_type='personal_owner', scope_id='owner-1',
        identity_verdict='verified', identity_confidence=0.99, completeness_verdict='complete',
        text_quality='digital', edition_or_version=None, created_at=now, admitted_at=now)


def _retrieved(text, statement=STATEMENT, translated=('公开信息',)):
    source = _source(text)
    claim = ClaimEvidence(claim_id='claim-1', paper_version_id='paper-v1', text=statement, claim_type='paraphrase',
                          reference_ids=['ref-1'], citation_marker='(Pan & Lin, 2016)',
                          citation_marker_type='parenthetical', passage_start=0, passage_end=len(statement))
    profile = language_profile(text, statement)
    record = evidence_record(profile, None) if profile['differs'] else None
    artifact = build_passage_evidence(source, claim=claim, active_reference_id='ref-1',
                                      cited_author_label='Pan & Lin', cross_language=record,
                                      translated_queries=list(translated))
    artifact = attach_citation_use_routes(attach_verification_candidates(artifact))
    return source, attach_candidate_passage_retrieval(source, artifact)


def _choose(fragment, seen=None):
    def call(system_prompt, user_prompt, *, receipt, **_):
        rows = json.loads(user_prompt)['sentences']
        if seen is not None:
            seen.extend(rows)
        alias = next(row['sentence_id'] for row in rows if fragment in row['text'])
        return {'selections': [{'sentence_id': alias, 'reason': 'bears_on_statement'}]}
    return call


def _source_sentences(text):
    return [s for paragraph in text.split('\n\n') for s in paragraph.replace('。', '。\n').split('\n') if s]


def test_a_fitting_source_gives_the_selector_every_sentence_in_order():
    text = _han_source_text(6)
    source, retrieved = _retrieved(text)
    assert retrieved.passages and all(CONSENT not in p.text for p in retrieved.passages)
    artifact = attach_whole_source(source, retrieved)
    record = artifact.cross_language['whole_source']
    assert record['version'] == VERSION and record['status'] == 'applied'
    assert record['sentence_count'] == len(_source_sentences(text))
    assert 0 < record['token_estimate'] <= record['max_input_tokens'] == 16_000
    seen = []
    artifact = attach_sentence_evidence(artifact, route=ROUTE, call=_choose('取得同意', seen))
    assert [row['text'] for row in seen] == _source_sentences(text)
    item = artifact.sentence_evidence.items[0]
    assert artifact.sentence_evidence.status == 'selected' and item.text == CONSENT
    # Retrieved sentences keep their passage ids; the rest bind to whole-source passages.
    request = prepare_request(artifact)
    retrieved_ids = {p.passage_id for p in retrieved.passages}
    whole = {p.passage_id: p for p in artifact._whole_source.passages}
    for ref in request.sentences.values():
        passage = next((p for p in retrieved.passages if p.passage_id == ref.passage_id), None) or whole[ref.passage_id]
        assert passage.text[ref.passage_start:ref.passage_end] == ref.text
    assert retrieved_ids & {ref.passage_id for ref in request.sentences.values()}
    assert item.passage_id in whole


def test_a_source_too_long_keeps_the_retrieved_candidates_and_says_so():
    text = _han_source_text(6)
    source, retrieved = _retrieved(text)
    artifact = attach_whole_source(source, retrieved, max_input_tokens=300)
    record = artifact.cross_language['whole_source']
    assert record['status'] == 'did_not_fit' and record['reason'] == 'over_selector_input_budget'
    assert record['token_estimate'] > 300 and record['sentence_count'] == len(_source_sentences(text))
    assert 'passages' not in record and artifact._whole_source is None
    assert prepare_request(artifact).user_prompt == prepare_request(retrieved).user_prompt
    assert (attach_facet_evidence_foundation(artifact).facet_evidence_foundation
            == attach_facet_evidence_foundation(retrieved).facet_evidence_foundation)


def test_a_same_language_source_is_unchanged():
    text = '\n\n'.join(['Licensing supports competition through provider diversity. Markets vary by region.',
                        'A final paragraph discusses unrelated archival practice.'])
    statement = 'Licensing supports competition through provider diversity (Pan & Lin, 2016).'
    source, retrieved = _retrieved(text, statement, translated=())
    assert retrieved.cross_language is None
    assert attach_whole_source(source, retrieved) is retrieved
    plain = attach_facet_evidence_foundation(attach_sentence_evidence(retrieved, route=ROUTE,
                                                                      call=_choose('Licensing')))
    payload = build_inspectable_report_payload(plain)
    assert 'cross_language' not in payload
    again = attach_facet_evidence_foundation(attach_sentence_evidence(
        attach_whole_source(source, retrieved), route=ROUTE, call=_choose('Licensing')))
    assert json.dumps(build_inspectable_report_payload(again), sort_keys=True, default=str) == json.dumps(
        payload, sort_keys=True, default=str)


def _foundation_keys(artifact):
    passages = {p.passage_id: p for p in [*artifact.passages, *artifact._whole_source.passages]}
    sentences = {s.sentence_id: s for s in artifact.facet_evidence_foundation.source_sentences}
    bundle = artifact.facet_evidence_foundation.candidate_bundles[0]
    return [absolute_key(passages[sentences[sid].passage_id], sentences[sid].passage_start,
                         sentences[sid].passage_end) for sid in bundle.evidence_sentence_ids], sentences, bundle


def test_the_foundation_reads_the_whole_source_when_it_fits_the_cap():
    text = _han_source_text(6)
    source, retrieved = _retrieved(text)
    artifact = attach_facet_evidence_foundation(attach_sentence_evidence(
        attach_whole_source(source, retrieved), route=ROUTE, call=_choose('取得同意')))
    keys, sentences, bundle = _foundation_keys(artifact)
    assert [sentences[sid].text for sid in bundle.evidence_sentence_ids] == _source_sentences(text)
    assert keys == sorted(keys)
    counts = artifact.cross_language['whole_source']['foundation'][bundle.candidate_id]
    assert counts == {'whole_source_sentences': len(keys), 'sentences': len(keys), 'truncated': False,
                      'omitted_sentences': 0, 'selector_sentences_omitted': 0, 'path': 'whole_source'}
    _validate_facet_evidence(artifact)


def test_a_source_longer_than_the_cap_keeps_the_selector_and_retrieval_sentences_and_records_it():
    text = _han_source_text(30)          # 181 sentences, over the 128-sentence cap
    source, retrieved = _retrieved(text)
    artifact = attach_facet_evidence_foundation(attach_sentence_evidence(
        attach_whole_source(source, retrieved), route=ROUTE, call=_choose('取得同意')))
    keys, sentences, bundle = _foundation_keys(artifact)
    texts = [sentences[sid].text for sid in bundle.evidence_sentence_ids]
    assert len(texts) == 128 and CONSENT in texts and keys == sorted(keys)
    retrieved_texts = {s.text for p in retrieved.passages for s in _passage_sentences(p)}
    assert retrieved_texts <= set(texts)
    counts = artifact.cross_language['whole_source']['foundation'][bundle.candidate_id]
    assert counts['truncated'] and counts['whole_source_sentences'] == 181
    assert counts['omitted_sentences'] == 181 - 128 and counts['selector_sentences_omitted'] == 0
    _validate_facet_evidence(artifact)


def test_a_whole_source_prompt_over_the_judgment_budget_keeps_the_retrieved_sentences(monkeypatch):
    text = _han_source_text(6)
    source, retrieved = _retrieved(text)
    ordinary = attach_facet_evidence_foundation(retrieved).facet_evidence_foundation
    monkeypatch.setattr(settings, 'JUDGMENT_MAX_INPUT_TOKENS', 1_000)
    artifact = attach_facet_evidence_foundation(attach_whole_source(source, retrieved))
    bundle = artifact.facet_evidence_foundation.candidate_bundles[0]
    assert bundle.evidence_sentence_ids == ordinary.candidate_bundles[0].evidence_sentence_ids
    assert artifact.cross_language['whole_source']['foundation'] == {
        bundle.candidate_id: {'path': 'retrieved_over_judgment_budget'}}


def test_stored_bindings_resolve_whole_source_sentences_without_source_text():
    text = _han_source_text(6)
    source, retrieved = _retrieved(text)
    artifact = attach_facet_evidence_foundation(attach_sentence_evidence(
        attach_whole_source(source, retrieved), route=ROUTE, call=_choose('取得同意')))
    payload = json.loads(json.dumps(build_inspectable_report_payload(artifact)))
    stored = payload['cross_language']['whole_source']['passages']
    assert stored and all(set(p) == {'passage_id', 'page_index', 'page_label', 'character_start',
                                     'character_end', 'passage_role', 'text_sha256'} for p in stored)
    assert CONSENT not in json.dumps(payload['cross_language'], ensure_ascii=False)
    assert {p['passage_id'] for p in stored_passages(payload)} >= {p['passage_id'] for p in stored}

    item = artifact.sentence_evidence.items[0]
    passage = next(p for p in artifact._whole_source.passages if p.passage_id == item.passage_id)
    key = evidence_sentence_key(*absolute_key(passage, item.passage_start, item.passage_end))
    view = _with_sentence_evidence({'coverage_level': 'full_text'}, SimpleNamespace(id='r', report_payload=payload),
                                   SimpleNamespace(reference_ids=['ref-1']))
    assert view['evidence_sentences'][0]['key'] == key
    assert key in {entry['key'] for entry in _sentences(payload, None).values()}

    context = judgment_context_from_payload(payload)
    assert item.passage_id in {p.passage_id for p in context.passages}
    request = prepare_request_from_record(payload)
    assert [ref.text for ref in request.sentences.values()] == _source_sentences(text)

    # Without the in-memory whole-source passages the binding cannot be checked: fail closed.
    stripped = artifact.model_copy()
    stripped._whole_source = None
    with pytest.raises(ReportAuthorizationError):
        _validate_facet_evidence(stripped)


def test_dense_scripts_count_about_one_token_a_character():
    from app.services.llm_input_boundary import estimate_prompt_tokens
    assert estimate_prompt_tokens("", "a" * 400) == 100
    assert estimate_prompt_tokens("", "数" * 400) == 400
    assert estimate_prompt_tokens("ab", "数据cd") == 3


def test_whole_source_selection_is_on_and_a_whole_read_stands_as_not_supported():
    from app.config import Settings
    from app.services.judgment_runs import _whole_source_read
    assert Settings.model_fields["CROSS_LANGUAGE_WHOLE_SOURCE_ENABLED"].default is True
    payload = {"cross_language": {"whole_source": {"status": "applied", "foundation": {
        "c1": {"whole_source_sentences": 40, "sentences": 40, "truncated": False},
        "c2": {"whole_source_sentences": 181, "sentences": 128, "truncated": True},
        "c3": {"path": "retrieved_over_judgment_budget"}}}}}
    assert _whole_source_read(payload, "c1")
    assert not _whole_source_read(payload, "c2") and not _whole_source_read(payload, "c3")
    assert not _whole_source_read({}, "c1")


def test_whole_source_plan_skips_a_passage_with_an_over_bound_sentence(monkeypatch):
    # 2026-10-18: an over-bound sentence must skip its passage, never fail the run.
    from app.services import cross_language_whole_source as whole
    from app.services import facet_evidence_judgment as facets

    def sentences(passage):
        if passage.passage_id.startswith("long"):
            raise facets._EvidenceSentenceBudgetExceeded
        return [SimpleNamespace(passage_start=0, passage_end=9, text="Sentence.")]

    monkeypatch.setattr(facets, "_passage_sentences", sentences)
    passage = lambda pid, start: SimpleNamespace(passage_id=pid, page_index=1, character_start=start,
                                                 passage_role="body_prose", text="Sentence.")
    artifact = SimpleNamespace(passages=[passage("long-r", 0), passage("ok-r", 100)])
    plan = whole.build_plan(artifact, [passage("long-w", 200), passage("ok-w", 300)])
    assert plan.excluded == 1
