"""Stand-in Judgment judges for development (JUDGMENT_FAKE_PANEL). No network, no spend.

Each claim gets a deterministic scenario from a hash of its candidate ID, so a
report shows every display state: supported, qualified, contradicts, agreed
"no evidence", disagreement and undecided. Answers follow the real response
contract, so the real validation and aggregation code runs on them.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from app.services.llm_service import LLMRoute
from app.services.providers import ProviderConfig

_SCENARIOS = (
    {"deepseek": "supports", "glm": "supports", "qwen": "supports"},
    {"deepseek": "supports", "glm": "qualifies", "qwen": "supports"},
    {"deepseek": "contradicts", "glm": "contradicts", "qwen": "contradicts"},
    {"deepseek": "none", "glm": "none", "qwen": "none"},
    {"deepseek": "supports", "glm": "contradicts", "qwen": "supports"},
    {"deepseek": "uncertain", "glm": "uncertain", "qwen": "uncertain"},
)


def _answer(arm_id: str, user_prompt: str) -> dict:
    payload = json.loads(user_prompt)
    if payload.get("coaching_request"):
        return {"note": f"Stand-in coaching note for a {payload.get('result')} result: development text, "
                        "not advice. Check the evidence sentences against each part of the statement.",
                "facet_ids": [f["facet_id"] for f in payload.get("unsupported_facets") or []][:2],
                "sentence_ids": [s["sentence_id"] for s in payload.get("evidence_sentences") or []][:2]}
    scenario = _SCENARIOS[int(hashlib.sha256(str(payload.get("candidate_id")).encode()).hexdigest(), 16)
                          % len(_SCENARIOS)]
    direction = scenario[arm_id]
    sentences = [s["sentence_id"] for s in payload.get("evidence_sentences") or []
                 if s.get("evidence_use") != "source_discourse_scope_only"]
    evidence = [] if direction == "none" or not sentences else sentences[:1]
    context = ("not_required" if not payload.get("requires_antecedent_context")
               else "resolved" if payload.get("local_context_resolution") == "resolved" else "unresolved")
    if context == "unresolved":
        direction, evidence = "uncertain", []
    return {"context_resolution": context, "mappings": [
        {"facet_id": f["facet_id"], "direction": direction, "confidence": "medium",
         "evidence_sentence_ids": evidence,
         "rationale": f"Stand-in judge ({arm_id}): development answer, not a judgment.",
         "limitations": []}
        for f in payload.get("facets") or []]}


class _FakeClient:
    def __init__(self, arm_id: str):
        self.arm_id = arm_id
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        user = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        content = json.dumps(_answer(self.arm_id, user.split("\n\nIMPORTANT:")[0]))
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0,
                                model_extra={"cost": 0.0})
        return SimpleNamespace(model=f"stand-in-{self.arm_id}", model_extra={"provider": "stand-in"},
                               usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_judge_arms() -> tuple[list[LLMRoute], dict]:
    routes = [LLMRoute(arm_id=arm, model=f"stand-in-{arm}", endpoint_host="stand-in.invalid",
                       client_factory=(lambda a=arm: _FakeClient(a)),
                       provider_config=ProviderConfig(name=f"stand-in {arm}"))
              for arm in ("deepseek", "glm", "qwen")]
    return routes, {"fake_panel": True, "arms": ["deepseek", "glm", "qwen"]}
