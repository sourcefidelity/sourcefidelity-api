"""Student-paper retention policy boundary.

Only temporary processing is operational in the prototype. The wider modes
are named now so later paper-corpus work can extend this boundary without
silently turning temporary workflow storage into a plagiarism repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class PaperRetentionMode(str, Enum):
    TEMPORARY = "temporary"
    ASSESSMENT = "assessment"
    COURSE = "course"
    INSTITUTIONAL = "institutional"


class PaperRetentionPolicyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PaperRetentionOutcome:
    mode: PaperRetentionMode
    action: str
    completed: bool


class PaperRetentionPolicy(Protocol):
    mode: PaperRetentionMode

    def after_extraction(
        self,
        cleanup_temporary_input: Callable[[], bool],
    ) -> PaperRetentionOutcome: ...


class TemporaryPaperRetentionPolicy:
    mode = PaperRetentionMode.TEMPORARY

    def after_extraction(
        self,
        cleanup_temporary_input: Callable[[], bool],
    ) -> PaperRetentionOutcome:
        completed = cleanup_temporary_input()
        return PaperRetentionOutcome(
            mode=self.mode,
            action="delete_after_extraction",
            completed=completed,
        )


def resolve_paper_retention_policy(value: str) -> PaperRetentionPolicy:
    try:
        mode = PaperRetentionMode(str(value).strip().casefold())
    except ValueError as exc:
        raise PaperRetentionPolicyError(
            "paper_retention_mode_invalid",
            "Paper retention mode is not recognized",
        ) from exc
    if mode is PaperRetentionMode.TEMPORARY:
        return TemporaryPaperRetentionPolicy()
    raise PaperRetentionPolicyError(
        "paper_retention_mode_not_implemented",
        (
            f"Paper retention mode '{mode.value}' requires the separate "
            "student-submission repository, authorization, and lifecycle service"
        ),
    )
