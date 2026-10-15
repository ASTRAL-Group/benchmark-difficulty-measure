#!/usr/bin/env python3
"""Export gap-fill summary CSV for an e2e_split_* result tree.

For each ``split_*`` directory (sorted by name), reads ``final_accepted.jsonl``, picks the
row with smallest ``abs_delta_full``, and writes one CSV line.

**Scale**

All gap columns use the **same** IRT reference pool: ``irt_results/``
merged with ``data/pools/<bm>/test.jsonl``, min–max affine to ``[0, 10]``
(``utils.difficulty_scale.to_difficulty_0_10``).

**gap_after**

Same construction as before, but pseudo-``b`` for accepted items is the affine inverse of
the pool map: ``b_new = b_min + (difficulty_full_0_10 / 10) * (b_max - b_min)``.

Run from repo ``propose/`` (so ``utils`` imports work). Default CSV path::

    python scripts/export_gap_fill_summary.py --e2e-dir results/e2e_split_ifeval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# propose/ on path; this script lives in propose/scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.difficulty_scale import (  # noqa: E402
    b_range_from_values,
    to_difficulty_0_10,
)
from utils.irt_join import merge_q_indexed_irt_csv_with_eval_jsonl  # noqa: E402
from utils.max_gap import top_k_largest_gap_brackets_b  # noqa: E402


def _difficulty_0_10_to_b(s: float, b_min: float, b_max: float) -> float:
    """Inverse of ``to_difficulty_0_10`` (unclipped affine)."""
    lo, hi = float(b_min), float(b_max)
    span = hi - lo
    if span <= 1e-12:
        return 0.5 * (lo + hi)
    return lo + (float(s) / 10.0) * span


def _max_subgap_after_on_0_10(
    bs_merged: list[float],
    b_lo: float,
    b_hi: float,
    b_min: float,
    b_max: float,
) -> tuple[float, float, float]:
    """Return (span_0_10, seg_low_0_10, seg_high_0_10) for the widest inner consecutive pair."""
    denom = b_max - b_min
    if denom <= 0.0:
        raise ValueError("empty b span")
    best_w = -1.0
    best_lo_b, best_hi_b = b_lo, b_hi
    for i in range(len(bs_merged) - 1):
        lo_b, hi_b = bs_merged[i], bs_merged[i + 1]
        if lo_b >= b_lo and hi_b <= b_hi:
            w = (hi_b - lo_b) / denom
            if w > best_w:
                best_w = w
                best_lo_b, best_hi_b = lo_b, hi_b
    if best_w < 0.0:
        best_lo_b, best_hi_b = b_lo, b_hi
    s0 = to_difficulty_0_10(best_lo_b, b_min, b_max)
    s1 = to_difficulty_0_10(best_hi_b, b_min, b_max)
    return float(s1 - s0), float(s0), float(s1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root containing data/pools/ (default: this repo)",
    )
    ap.add_argument(
        "--gap-root",
        type=Path,
        default=None,
        help="Gap-filling tree with irt_results/ (default: <workspace>)",
    )
    ap.add_argument(
        "--e2e-dir",
        type=Path,
        required=True,
        help="Directory with split_* subdirs and final_accepted.jsonl in each",
    )
    ap.add_argument("--benchmark", default="ifeval")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: this script's directory / gap_fill_summary_{benchmark}.csv)",
    )
    args = ap.parse_args()

    workspace = args.workspace.resolve()
    gap_root = (args.gap_root or workspace).resolve()
    e2e = args.e2e_dir.resolve()
    bench = args.benchmark
    out_path = args.out
    if out_path is None:
        out_path = _SCRIPT_DIR / f"gap_fill_summary_{bench}.csv"
    out_path = out_path.resolve()

    irt_csv = gap_root / "irt_results" / f"{bench}_difficulty.csv"
    eval_jsonl = workspace / "data" / "pools" / bench / "test.jsonl"

    pool = merge_q_indexed_irt_csv_with_eval_jsonl(irt_csv, eval_jsonl)
    bs0 = sorted(float(p.difficulty) for p in pool if p.difficulty is not None)
    b_min, b_max = b_range_from_values(bs0)

    ranked_before = top_k_largest_gap_brackets_b(bs0, 10)
    brackets: list[dict] = []
    for rank, (blo, bhi, meta) in enumerate(ranked_before[:5], start=1):
        g0 = to_difficulty_0_10(blo, b_min, b_max)
        g1 = to_difficulty_0_10(bhi, b_min, b_max)
        brackets.append(
            {
                "rank": rank,
                "b_lo": blo,
                "b_hi": bhi,
                "gap_before_0_10": float(g1 - g0),
                "gap_endpoint_low_0_10": float(g0),
                "gap_endpoint_high_0_10": float(g1),
            }
        )

    splits = sorted(e2e.glob("split_*"))
    if len(splits) != len(brackets):
        print(
            f"warning: {len(splits)} splits vs {len(brackets)} gap brackets; "
            "pairing by order (min count).",
            file=sys.stderr,
        )
    n = min(len(splits), len(brackets))

    best_rows: list[dict] = []
    b_news: list[float] = []
    for si in range(n):
        d = splits[si]
        acc_path = d / "final_accepted.jsonl"
        acc = [
            json.loads(line)
            for line in acc_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        acc.sort(key=lambda x: float(x.get("abs_delta_full", 1e9)))
        best = acc[0]
        df = float(best["estimated_difficulty_full"])
        tf = float(best["target_full"])
        b_new = _difficulty_0_10_to_b(df, b_min, b_max)
        b_news.append(b_new)
        br = brackets[si]
        best_rows.append(
            {
                "gap_before_0_10": br["gap_before_0_10"],
                "gap_endpoint_low_0_10": br["gap_endpoint_low_0_10"],
                "gap_endpoint_high_0_10": br["gap_endpoint_high_0_10"],
                "b_lo": br["b_lo"],
                "b_hi": br["b_hi"],
                "target_difficulty_full_0_10": tf,
                "difficulty_full_0_10": df,
                "delta_full_0_10": df - tf,
                "id": best["id"],
                "item": best.get("question") or "",
            }
        )

    bs_merged = sorted(bs0 + b_news)
    for row in best_rows:
        span_a, lo_a, hi_a = _max_subgap_after_on_0_10(
            bs_merged, row["b_lo"], row["b_hi"], b_min, b_max
        )
        row["gap_after_0_10"] = span_a
        row["gap_after_low_0_10"] = lo_a
        row["gap_after_high_0_10"] = hi_a

    headers = [
        "gap_before (IRT pool 0-10)",
        "gap_endpoint_low (IRT pool 0-10)",
        "gap_endpoint_high (IRT pool 0-10)",
        "gap_after (IRT pool 0-10)",
        "gap_after_segment_low (IRT pool 0-10)",
        "gap_after_segment_high (IRT pool 0-10)",
        "target difficulty (full 0-10)",
        "difficulty (full 0-10)",
        "delta (full 0-10)",
        "id",
        "item",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in best_rows:
            w.writerow(
                [
                    row["gap_before_0_10"],
                    row["gap_endpoint_low_0_10"],
                    row["gap_endpoint_high_0_10"],
                    row["gap_after_0_10"],
                    row["gap_after_low_0_10"],
                    row["gap_after_high_0_10"],
                    row["target_difficulty_full_0_10"],
                    row["difficulty_full_0_10"],
                    row["delta_full_0_10"],
                    row["id"],
                    row["item"],
                ]
            )
    print(f"wrote {out_path} ({len(best_rows)} rows)")


if __name__ == "__main__":
    main()
