"""Student-paper retention policy boundary tests."""

import pytest

from app.services.paper_retention import (
    PaperRetentionMode,
    PaperRetentionPolicyError,
    resolve_paper_retention_policy,
)


def test_temporary_policy_deletes_after_extraction():
    calls = []
    policy = resolve_paper_retention_policy("temporary")
    outcome = policy.after_extraction(lambda: calls.append("deleted") or True)

    assert calls == ["deleted"]
    assert outcome.mode is PaperRetentionMode.TEMPORARY
    assert outcome.action == "delete_after_extraction"
    assert outcome.completed is True


@pytest.mark.parametrize("mode", ["assessment", "course", "institutional"])
def test_future_retention_modes_fail_closed(mode):
    with pytest.raises(PaperRetentionPolicyError) as captured:
        resolve_paper_retention_policy(mode)
    assert captured.value.code == "paper_retention_mode_not_implemented"


def test_unknown_retention_mode_fails_closed():
    with pytest.raises(PaperRetentionPolicyError) as captured:
        resolve_paper_retention_policy("forever")
    assert captured.value.code == "paper_retention_mode_invalid"
