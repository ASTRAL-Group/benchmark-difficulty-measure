"""Recompute the paper's difficulty metrics from the released files.

Table 1 (App. C.4 definitions), for every irt_results/<benchmark>_difficulty.csv:
  - difficulties are b_mean, min-max normalized to [0, 1];
  - DDM is the mean |z_(i) - (i - 1/2)/n|, the 1-Wasserstein distance to a
    same-size empirical uniform reference;
  - Max Gap is the largest spacing of 0 = z_(0) <= z_(1) <= ... <= z_(n) <= z_(n+1) = 1,
    so the two boundary intervals are included (under min-max normalization they
    have length 0);
  - Top-5 Sum is the sum of the five largest such spacings.

Table 2, BENCHFILL and Oracle columns, for the eight gap-filling benchmarks:
  - Top-3 Gap Sum Reduction (%) after inserting the three released items of
    new_item_records.csv (BENCHFILL) or the three gap midpoints (Oracle);
  - Target Dev. = mean |difficulty - target| / 10 over the three released items.

Each value is printed next to the number published in the paper. The Baseline
column of Table 2 is not recomputed: baseline run outputs are not released.

Usage:
    python scripts/check_paper_metrics.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Table 1 of the paper: (DDM, Max Gap, Top-5 Sum, #items).
PAPER_TABLE1 = {
    "math-500": (0.164, 0.058, 0.175, 500),
    "minerva_math": (0.122, 0.031, 0.149, 272),
    "olympiadbench": (0.064, 0.048, 0.142, 675),
    "gaokao2023en": (0.088, 0.055, 0.183, 385),
    "amc23": (0.088, 0.175, 0.528, 40),
    "aime24": (0.077, 0.179, 0.502, 30),
    "aime25": (0.079, 0.158, 0.492, 30),
    "aime26": (0.085, 0.119, 0.507, 30),
    "mmlu_bio": (0.125, 0.043, 0.176, 717),
    "humaneval": (0.148, 0.165, 0.423, 164),
    "mbpp": (0.163, 0.049, 0.214, 378),
    "ifeval": (0.159, 0.046, 0.190, 541),
    "bbh_3obj": (0.090, 0.088, 0.354, 250),
}

# Table 2 of the paper: (BENCHFILL reduction %, Oracle reduction %, BENCHFILL Target Dev.).
PAPER_TABLE2 = {
    "aime24": (35.57, 35.57, 0.005),
    "aime25": (37.56, 37.56, 0.005),
    "aime26": (33.04, 33.04, 0.002),
    "amc23": (35.19, 35.78, 0.003),
    "gaokao2023en": (38.88, 45.47, 0.004),
    "bbh_3obj": (29.88, 32.80, 0.006),
    "mmlu_bio": (30.13, 36.84, 0.006),
    "ifeval": (21.09, 23.44, 0.007),
}


def normalized_difficulties(benchmark: str) -> np.ndarray:
    with (ROOT / "irt_results" / f"{benchmark}_difficulty.csv").open(encoding="utf-8") as f:
        b = np.array([float(row["b_mean"]) for row in csv.DictReader(f)])
    return (b - b.min()) / (b.max() - b.min())


def spacings(z: np.ndarray) -> np.ndarray:
    return np.diff(np.concatenate([[0.0], np.sort(z), [1.0]]))


def ddm(z: np.ndarray) -> float:
    n = len(z)
    return float(np.abs(np.sort(z) - (np.arange(1, n + 1) - 0.5) / n).mean())


def top_k_sum(z: np.ndarray, k: int) -> float:
    return float(np.sort(spacings(z))[-k:].sum())


def top3_midpoints(z: np.ndarray) -> np.ndarray:
    edges = np.concatenate([[0.0], np.sort(z), [1.0]])
    idx = np.argsort(np.diff(edges))[-3:]
    return (edges[idx] + edges[idx + 1]) / 2


def mark(got: float, paper: float, digits: int) -> str:
    return "ok" if round(got, digits) == round(paper, digits) else "DIFFERS"


def main() -> None:
    print("Table 1  (recomputed / paper)")
    for benchmark, (p_ddm, p_gap, p_top5, p_n) in PAPER_TABLE1.items():
        z = normalized_difficulties(benchmark)
        got = (ddm(z), float(spacings(z).max()), top_k_sum(z, 5))
        cells = [f"{g:.4f}/{p:.3f} {mark(g, p, 3)}" for g, p in zip(got, (p_ddm, p_gap, p_top5))]
        print(f"  {benchmark:14s} n={len(z):4d}/{p_n:<4d} DDM {cells[0]:22s} MaxGap {cells[1]:22s} Top5 {cells[2]}")

    with (ROOT / "new_item_records.csv").open(encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    print("\nTable 2  (recomputed / paper)")
    for benchmark, (p_fill, p_oracle, p_dev) in PAPER_TABLE2.items():
        z = normalized_difficulties(benchmark)
        rows = [r for r in records if r["benchmark"] == benchmark]
        new = np.array([float(r["difficulty"]) for r in rows]) / 10
        target = np.array([float(r["target"]) for r in rows]) / 10
        before = top_k_sum(z, 3)
        fill = 100 * (before - top_k_sum(np.concatenate([z, new]), 3)) / before
        oracle = 100 * (before - top_k_sum(np.concatenate([z, top3_midpoints(z)]), 3)) / before
        dev = float(np.abs(new - target).mean())
        print(f"  {benchmark:14s} BENCHFILL {fill:6.2f}/{p_fill:<6.2f} {mark(fill, p_fill, 2):8s}"
              f" Oracle {oracle:6.2f}/{p_oracle:<6.2f} {mark(oracle, p_oracle, 2):8s}"
              f" TargetDev {dev:.4f}/{p_dev:.3f} {mark(dev, p_dev, 3)}")


if __name__ == "__main__":
    main()
