"""Pick reference anchors around a difficulty gap (default: 2 items per side).

``pool`` difficulties and ``gap_low`` / ``gap_high`` must share one scale (project default: **0–10**).
"""

from __future__ import annotations

from typing import Sequence

from utils.schemas import ProblemItem


def select_gap_end_anchors(
    pool: Sequence[ProblemItem],
    gap_low: float,
    gap_high: float,
    *,
    per_side: int = 2,
) -> list[ProblemItem]:
    """Choose reference items for generation prompts.

    Policy (default per_side=2):
      - **Low side:** among items with difficulty <= gap_low, take the `per_side` items with
        **largest** difficulty (closest to the gap from below).
      - **High side:** among items with difficulty >= gap_high, take the `per_side` items with
        **smallest** difficulty (closest to the gap from above).

    Items with ``difficulty is None`` are skipped. Result order: low-side group first, then
    high-side group (suitable for ``build_prompt``).

    If fewer than ``per_side`` exist on one side, all available on that side are returned.
    """
    if gap_low > gap_high:
        gap_low, gap_high = gap_high, gap_low

    with_b = [p for p in pool if p.difficulty is not None]
    low_side = [p for p in with_b if p.difficulty <= gap_low]
    high_side = [p for p in with_b if p.difficulty >= gap_high]

    low_side.sort(key=lambda p: float(p.difficulty), reverse=True)
    high_side.sort(key=lambda p: float(p.difficulty))

    return low_side[:per_side] + high_side[:per_side]
