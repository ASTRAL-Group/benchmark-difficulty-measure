#!/usr/bin/env python3
"""Build budget-match baseline spec from MiniMax BENCHFILL 24-gap package.

For each gap g: T_g = #candidates_round_*.jsonl in the method run;
proposal budget = 16 * T_g (paper: T_g rounds × 4 parents × 4 children).

Writes:
  random_baseline/results/_budget_spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_GAP = Path(__file__).resolve().parents[1].parent  # repo root
_RECORDS = _GAP / "new_item_records.csv"
_RESULTS = _GAP / "propose" / "results"
_OUT = _GAP / "random_baseline" / "results" / "_budget_spec.json"

_MATH = {"aime24", "aime25", "aime26", "amc23", "gaokao2023en"}
_LMMS = {"mmlu_bio", "ifeval", "bbh_3obj"}


def _load_records(path: Path) -> list[dict]:
    import csv

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = json.loads(r["item"])
            rows.append(
                {
                    "benchmark": r["benchmark"],
                    "top_gap": r["top_gap"],
                    "gap_rank": int(str(r["top_gap"]).removeprefix("gap")),
                    "run_id": item["run_id"],
                    "target_from_record": float(r["target"]) if r.get("target") not in (None, "") else None,
                    "item": item,
                }
            )
    return rows


def _target_from_run(run_dir: Path) -> float | None:
    for cf in sorted(run_dir.glob("candidates_round_*.jsonl")):
        for line in cf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            t = o.get("estimated_difficulty", {}).get("target")
            if t is not None:
                return float(t)
    return None


def build_spec(*, records_path: Path = _RECORDS) -> dict:
    gaps: list[dict] = []
    for rec in _load_records(records_path):
        bm = rec["benchmark"]
        run_dir = _RESULTS / f"e2e_split_{bm}" / rec["run_id"]
        if not run_dir.is_dir():
            raise SystemExit(f"missing run dir: {run_dir}")
        round_files = sorted(run_dir.glob("candidates_round_*.jsonl"))
        t_g = len(round_files)
        if t_g < 1:
            raise SystemExit(f"no candidates_round_*.jsonl in {run_dir}")
        method_n = 0
        for cf in round_files:
            method_n += sum(1 for line in cf.read_text(encoding="utf-8").splitlines() if line.strip())
        target = _target_from_run(run_dir)
        if target is None and rec["target_from_record"] is not None:
            target = float(rec["target_from_record"])
        if target is None:
            raise SystemExit(f"cannot resolve target for {bm} gap{rec['gap_rank']}")
        framework = "lmms" if bm in _LMMS else "math"
        n_sampling = 4 if framework == "lmms" else 32
        budget = 16 * t_g
        gaps.append(
            {
                "benchmark": bm,
                "gap_rank": rec["gap_rank"],
                "top_gap": rec["top_gap"],
                "run_id": rec["run_id"],
                "T_g": t_g,
                "budget": budget,
                "children_per_parent": 4 * t_g,
                "n_parents": 4,
                "method_annotated": method_n,
                "target_difficulty": target,
                "framework": framework,
                "n_sampling": n_sampling,
            }
        )

    gaps.sort(key=lambda g: (g["benchmark"], g["gap_rank"]))
    return {
        "source_records": str(records_path),
        "paper_budget_formula": "16 * T_g  (T_g rounds × 4 parents × 4 children)",
        "generator_model": "minimax-m2.7",
        "global_seed": 42,
        "n_gaps": len(gaps),
        "total_budget": sum(g["budget"] for g in gaps),
        "total_method_annotated": sum(g["method_annotated"] for g in gaps),
        "gaps": gaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=_RECORDS)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args()
    spec = build_spec(records_path=args.records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  n_gaps={spec['n_gaps']} total_budget={spec['total_budget']} "
          f"method_annotated={spec['total_method_annotated']}")
    for g in spec["gaps"]:
        print(f"  {g['benchmark']:12} gap{g['gap_rank']} T_g={g['T_g']} "
              f"budget={g['budget']:3} method_n={g['method_annotated']:3} "
              f"target={g['target_difficulty']:.4f}")


if __name__ == "__main__":
    main()
