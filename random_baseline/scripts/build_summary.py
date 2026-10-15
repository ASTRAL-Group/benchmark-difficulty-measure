#!/usr/bin/env python3
"""For each benchmark, find the closest-to-target item per gap and write a summary CSV.

Reads:
  random_baseline/results/<bench>/gap_rank_<r>/generated.jsonl
  random_baseline/results/<bench>/difficulty/<bench>_gap<r>_difficulty.csv

Writes:
  random_baseline/results/<bench>/<bench>_baseline_summary.csv

Output columns:
  gap_rank, item_id, parent_id, target_difficulty,
  item_difficulty_12m, delta

delta = |item_difficulty_12m - target_difficulty|
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_GAP = Path(__file__).resolve().parents[1].parent  # repo root
_RESULTS = _GAP / "random_baseline" / "results"

# math: difficulty.csv `item` is ProblemItem.id (e.g. "rand_g1_p0_r0_c0")
# lmms: difficulty.csv `item` is the row index in the candidate jsonl (which equals
#       the row index in generated.jsonl since build_lmms_candidates.py uses key=row_idx)
_MATH = {"aime24", "aime25", "aime26", "amc23", "gaokao2023en", "math-500", "gsm8k"}


def _summarize_gap(bench: str, gap: int, gen_path: Path, diff_path: Path) -> dict | None:
    gen_rows: list[dict] = []
    with gen_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gen_rows.append(json.loads(line))
    if not gen_rows:
        return None
    target = float(gen_rows[0]["target_difficulty"])

    by_item: dict[str, dict] = {}
    with diff_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = str(r["item"])
            b12 = r.get("difficulty")
            if b12 in (None, "", "nan"):
                b12 = r.get("b_hat")
            if b12 in (None, "", "nan"):
                continue
            by_item[key] = {"b12": float(b12)}

    use_index = bench not in _MATH
    # lmms batch verify keeps *global* combined.jsonl indices in difficulty.csv
    # (gap2/3 start at 16/32/…), while generated.jsonl is local 0..n-1 per gap.
    # Prefer local index when present; otherwise map by sorted global order.
    global_keys: list[str] | None = None
    if use_index:
        global_keys = sorted(
            by_item.keys(),
            key=lambda k: int(k) if str(k).isdigit() else str(k),
        )
    cands: list[tuple] = []
    for i, item in enumerate(gen_rows):
        if use_index:
            if str(i) in by_item:
                key = str(i)
            elif global_keys is not None and i < len(global_keys):
                key = global_keys[i]
            else:
                continue
        else:
            key = str(item.get("id"))
        d = by_item.get(key)
        if d is None:
            continue
        delta = abs(d["b12"] - target)
        cands.append((delta, str(item.get("id")), str(item.get("parent_id")), d["b12"]))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    delta, item_id, parent_id, b12 = cands[0]
    return {
        "gap_rank": gap,
        "item_id": item_id,
        "parent_id": parent_id,
        "target_difficulty": target,
        "item_difficulty_12m": b12,
        "delta": delta,
    }


def summarize_benchmark(bench: str) -> Path | None:
    bench_dir = _RESULTS / bench
    if not bench_dir.is_dir():
        print(f"[summary] {bench_dir} not found"); return None

    gaps: list[int] = sorted(
        int(p.name.removeprefix("gap_rank_"))
        for p in bench_dir.iterdir()
        if p.is_dir() and p.name.startswith("gap_rank_")
    )

    rows: list[dict] = []
    for g in gaps:
        gen_path = bench_dir / f"gap_rank_{g}" / "generated.jsonl"
        diff_path = bench_dir / "difficulty" / f"{bench}_gap{g}_difficulty.csv"
        if not gen_path.is_file():
            print(f"[summary {bench} gap{g}] missing {gen_path.name}"); continue
        if not diff_path.is_file():
            print(f"[summary {bench} gap{g}] missing {diff_path.name}"); continue
        row = _summarize_gap(bench, g, gen_path, diff_path)
        if row is None:
            print(f"[summary {bench} gap{g}] no matchable items"); continue
        rows.append(row)
        print(f"  gap_rank={g} target={row['target_difficulty']:.3f} "
              f"best={row['item_id']} parent={row['parent_id']} "
              f"b12={row['item_difficulty_12m']:.3f} delta={row['delta']:.3f}")

    if not rows:
        print(f"[summary {bench}] no complete gaps yet"); return None

    out_path = bench_dir / f"{bench}_baseline_summary.csv"
    fieldnames = [
        "gap_rank", "item_id", "parent_id", "target_difficulty",
        "item_difficulty_12m", "delta",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[summary {bench}] wrote {len(rows)} rows -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks", nargs="+", help="benchmark names (one or more)")
    args = ap.parse_args()
    for b in args.benchmarks:
        summarize_benchmark(b)


if __name__ == "__main__":
    main()
