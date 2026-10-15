"""DifficultyEstimator protocol and run helpers. See INTERFACES.md §4."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from utils.schemas import DifficultyResult, ProblemItem


class DifficultyEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        item: ProblemItem,
        *,
        target: float,
        anchors: Sequence[ProblemItem],
    ) -> DifficultyResult:
        """Scalar difficulty on the same scale as ``target`` (default project scale: 0–10)."""


class ConstantDifficultyEstimator(DifficultyEstimator):
    """Always returns a fixed ``estimate`` (smoke tests / demos without a real IRT step)."""

    def __init__(self, estimate: float) -> None:
        self._estimate = float(estimate)

    def estimate(
        self,
        item: ProblemItem,
        *,
        target: float,
        anchors: Sequence[ProblemItem],
    ) -> DifficultyResult:
        _ = item, anchors
        return DifficultyResult(
            estimate=self._estimate,
            target=target,
            confidence="low",
            notes="ConstantDifficultyEstimator stub",
        )


def within_tolerance(estimate: float, target: float, tol: float) -> bool:
    return abs(estimate - target) <= tol

# verifier choose model; tolerance ; verifier (mock first); first go through the full code pipeline for multiple turns

def edit_hints_for_difficulty(estimate: float, target: float) -> list[str]:
    d = estimate - target
    if abs(d) < 1e-6:
        return []
    if d > 0:
        return [
            f"Estimated difficulty above target by ~{d:.2f}: simplify or shorten the task.",
            "Reduce required steps or use more familiar concepts.",
        ]
    return [
        f"Estimated difficulty below target by ~{-d:.2f}: add a non-trivial step or connection.",
        "Require combining ideas or less obvious reasoning.",
    ]


def compress_round_feedback(
    scored_failures: list[tuple[ProblemItem, DifficultyResult | None]],
    max_items: int = 3,
) -> str:
    lines: list[str] = []
    for i, (c, diff) in enumerate(scored_failures[:max_items]):
        snippet = c.question.replace("\n", " ")[:120]
        if diff:
            lines.append(
                f"{i + 1}) {snippet}… | estimate={diff.estimate:.2f} target={diff.target:.2f}"
            )
        else:
            lines.append(f"{i + 1}) {snippet}… | (no difficulty estimate)")
    return "\n".join(lines)
