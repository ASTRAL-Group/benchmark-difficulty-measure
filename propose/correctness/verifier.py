"""CorrectnessVerifier protocol. See INTERFACES.md §3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from utils.schemas import CorrectnessResult, ProblemItem


class CorrectnessVerifier(ABC):
    @abstractmethod
    def verify(self, item: ProblemItem) -> CorrectnessResult:
        """Return whether the item is usable; if not, set failure_code and evidence."""


class NonEmptyFieldsVerifier(CorrectnessVerifier):
    """Minimal gate: question must be non-empty; reference_answer required only when
    the benchmark scores against a target string (skipped for ifeval-style program-checked
    items, identified by a non-empty ``instruction_id_list``)."""

    def verify(self, item: ProblemItem) -> CorrectnessResult:
        if not item.question.strip():
            return CorrectnessResult(ok=False, failure_code="PARSE_ERROR", evidence="empty question")
        if item.instruction_id_list:
            return CorrectnessResult(ok=True, evidence="ifeval program-checked item")
        if not item.reference_answer.strip():
            return CorrectnessResult(
                ok=False, failure_code="PARSE_ERROR", evidence="empty reference_answer"
            )
        return CorrectnessResult(ok=True, evidence="non-empty fields")


class ChainedVerifier(CorrectnessVerifier):
    """Run verifiers in order; first failure wins; otherwise last success result."""

    def __init__(self, steps: Sequence[CorrectnessVerifier]) -> None:
        self._steps = list(steps)

    def verify(self, item: ProblemItem) -> CorrectnessResult:
        last = CorrectnessResult(ok=True)
        for v in self._steps:
            last = v.verify(item)
            if not last.ok:
                return last
        return last
