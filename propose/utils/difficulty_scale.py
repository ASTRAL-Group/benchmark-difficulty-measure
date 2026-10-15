"""Map IRT item difficulty *b* to a **0–10** scale for prompts and ``ProblemItem.difficulty``.

Uses min–max over a reference set (typically the full item pool):

    s = 10 * (b - b_min) / (b_max - b_min)

When ``b_max == b_min``, every finite *b* maps to ``5.0``. Monotone: ordering andwhich adjacent pair has the largest normalized gap are unchanged from using raw *b*.
"""

from __future__ import annotations

from typing import Sequence

from utils.schemas import ProblemItem

_SPAN_EPS = 1e-12


def b_range_from_pool(pool: Sequence[ProblemItem]) -> tuple[float, float]:
    """Min / max of finite ``difficulty`` values; raises if none present."""
    vals = [float(p.difficulty) for p in pool if p.difficulty is not None]
    if not vals:
        raise ValueError("need at least one item with finite difficulty for b_range_from_pool")
    return min(vals), max(vals)


def b_range_from_values(bs: Sequence[float]) -> tuple[float, float]:
    """Min / max over finite difficulties (raw *b* or already on 0–10)."""
    xs = [float(x) for x in bs]
    if not xs:
        raise ValueError("need at least one value")
    return min(xs), max(xs)


def to_difficulty_0_10(b: float, b_min: float, b_max: float) -> float:
    """Affine map ``[b_min, b_max]`` → ``[0, 10]`` (endpoints inclusive).

    Items outside ``[b_min, b_max]`` are NOT clipped: easier-than-the-easiest
    source item maps to a negative value, harder-than-the-hardest exceeds 10.
    Clipping would collapse all out-of-range items to 0 (or 10) and destroy
    the |diff − target| signal used for parent selection / convergence.
    """
    lo, hi = float(b_min), float(b_max)
    if hi - lo <= _SPAN_EPS:
        return 5.0
    t = (float(b) - lo) / (hi - lo)
    return 10.0 * t


def from_difficulty_0_10(s: float, b_min: float, b_max: float) -> float:
    """Inverse of ``to_difficulty_0_10`` (for export / debugging)."""
    lo, hi = float(b_min), float(b_max)
    if hi - lo <= _SPAN_EPS:
        return 0.5 * (lo + hi)
    t = max(0.0, min(1.0, float(s) / 10.0))
    return lo + t * (hi - lo)


def normalize_items_to_0_10(
    pool: Sequence[ProblemItem],
    b_min: float,
    b_max: float,
) -> list[ProblemItem]:
    """Copy pool with ``difficulty`` replaced by 0–10 scores; ``None`` stays ``None``."""
    out: list[ProblemItem] = []
    for p in pool:
        d = p.difficulty
        d10 = None if d is None else to_difficulty_0_10(float(d), b_min, b_max)
        out.append(
            ProblemItem(
                id=p.id,
                question=p.question,
                reference_answer=p.reference_answer,
                difficulty=d10,
                modification_policy=p.modification_policy,
                parent_id=p.parent_id,
                options=p.options,
                instruction_id_list=p.instruction_id_list,
                kwargs=p.kwargs,
            )
        )
    return out
