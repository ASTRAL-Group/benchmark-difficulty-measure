"""Largest gap on the min–max normalized difficulty scale (Max Gap, Eq. (1) of the paper).

After min–max normalization the smallest and largest items sit at 0 and 1, so the two
boundary intervals of Eq. (1) have length 0 and only interior gaps can be the largest.

**Plots:** min–max to ``[0, 1]`` makes ECDFs comparable and exposes the largest
vertical jump in normalized space.

**Proposal:** normalization is used **only** to decide *which* adjacent pair of items (in
sorted-by-difficulty order) has that largest gap. Returned ``gap_low`` / ``gap_high`` are in
the **same units as the input** sequence (typically raw IRT *b* from a CSV). Map them to the
project’s **0–10** pool scale with ``utils.difficulty_scale.to_difficulty_0_10`` before
``select_gap_end_anchors`` / ``build_prompt`` / ``RunConfig.target_difficulty``.

Because min–max is affine, the arithmetic midpoint ``(lower + upper) / 2`` matches the
midpoint after any affine rescaling (e.g. to 0–10).
"""

from __future__ import annotations

from typing import Sequence


def largest_normalized_gap_bracket_b(
    difficulties: Sequence[float],
) -> tuple[float, float, dict[str, float | int]]:
    """Return **original** ``(b_lower, b_upper)`` for the pair with largest normalized gap.

    Internally compares min–max-normalized gaps between consecutive sorted *b* values;
    the returned endpoints use the same scale as ``difficulties`` (often raw IRT *b*).

    ``b_lower`` and ``b_upper`` are consecutive in **sorted-by-b** order (no other item
    strictly between them on the *b* axis).

    ``meta`` includes ``gap_index`` (lower index in sorted order), ``normalized_gap_width``,
    and copies ``b_sorted_lower`` / ``b_sorted_upper`` for debugging.
    """
    bs = sorted(float(x) for x in difficulties)
    n = len(bs)
    if n < 2:
        raise ValueError("need at least two finite difficulties")
    bmin, bmax = bs[0], bs[-1]
    if bmax == bmin:
        raise ValueError("all difficulties are equal; no gap")

    norm = [(b - bmin) / (bmax - bmin) for b in bs]
    best_i = 0
    best_w = -1.0
    for i in range(n - 1):
        w = norm[i + 1] - norm[i]
        if w > best_w:
            best_w = w
            best_i = i

    meta: dict[str, float | int] = {
        "gap_index": best_i,
        "normalized_gap_width": best_w,
        "b_sorted_lower": bs[best_i],
        "b_sorted_upper": bs[best_i + 1],
    }
    return float(bs[best_i]), float(bs[best_i + 1]), meta


def gap_midpoint_target(gap_low: float, gap_high: float) -> float:
    """Midpoint in the same units as ``gap_low`` / ``gap_high`` (affine-invariant)."""
    return 0.5 * (float(gap_low) + float(gap_high))


def top_k_largest_gap_brackets_b(
    difficulties: Sequence[float],
    k: int,
) -> list[tuple[float, float, dict[str, float | int]]]:
    """Like ``largest_normalized_gap_bracket_b`` but returns the **top K** widest gaps,
    sorted by normalized gap width descending.

    Each entry is ``(b_lower, b_upper, meta)`` in the same units as the input.
    Returns at most ``k`` entries (fewer if the pool has fewer adjacent pairs).
    """
    bs = sorted(float(x) for x in difficulties)
    n = len(bs)
    if n < 2:
        raise ValueError("need at least two finite difficulties")
    bmin, bmax = bs[0], bs[-1]
    if bmax == bmin:
        raise ValueError("all difficulties are equal; no gap")

    norm = [(b - bmin) / (bmax - bmin) for b in bs]
    gaps: list[tuple[float, int]] = [
        (norm[i + 1] - norm[i], i) for i in range(n - 1)
    ]
    gaps.sort(key=lambda t: t[0], reverse=True)

    out: list[tuple[float, float, dict[str, float | int]]] = []
    for w, i in gaps[: max(0, int(k))]:
        meta: dict[str, float | int] = {
            "gap_index": i,
            "normalized_gap_width": w,
            "b_sorted_lower": bs[i],
            "b_sorted_upper": bs[i + 1],
        }
        out.append((float(bs[i]), float(bs[i + 1]), meta))
    return out
