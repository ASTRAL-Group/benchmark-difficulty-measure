"""Plot per-split convergence paths: round vs. best candidate's delta to target.

Both per-round best (best within that round) and historical best (best up to
and including that round) are drawn on the same figure.

Usage:
    python plot_convergence.py <results_dir> [--out <out_dir>] [--tol 0.15] [--abs]

Example:
    python propose/scripts/plot_convergence.py \
        propose/results/e2e_split_bbh_3obj   # from the repo root
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROUND_RE = re.compile(r"candidates_round_(\d+)\.jsonl$")


def collect_round_deltas(split_dir: Path) -> list[dict]:
    """Return per-round records sorted by round.

    Each record:
        {
          "round":        int,
          "round_signed": float,  # this-round best (signed)
          "round_abs":    float,  # this-round best |delta|
          "hist_signed":  float,  # cumulative best (signed) up to this round
          "hist_abs":     float,  # cumulative best |delta| up to this round
        }
    """
    per_round: list[tuple[int, float, float]] = []
    for f in sorted(split_dir.glob("candidates_round_*.jsonl")):
        m = ROUND_RE.search(f.name)
        if not m:
            continue
        r = int(m.group(1))
        round_signed = None
        round_abs = float("inf")
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ed = d.get("estimated_difficulty") or {}
                if "estimate" not in ed or "target" not in ed:
                    continue
                delta = ed["estimate"] - ed["target"]
                if abs(delta) < round_abs:
                    round_abs = abs(delta)
                    round_signed = delta
        if round_signed is not None:
            per_round.append((r, round_signed, round_abs))

    per_round.sort(key=lambda x: x[0])

    out: list[dict] = []
    hist_signed = None
    hist_abs = float("inf")
    for r, s, a in per_round:
        if a < hist_abs:
            hist_abs = a
            hist_signed = s
        out.append({
            "round": r,
            "round_signed": s,
            "round_abs": a,
            "hist_signed": hist_signed,
            "hist_abs": hist_abs,
        })
    return out


def plot_split(split_dir: Path, out_path: Path, tol: float, use_abs: bool) -> bool:
    rounds = collect_round_deltas(split_dir)
    if not rounds:
        return False

    xs = [d["round"] for d in rounds]
    if use_abs:
        ys_round = [d["round_abs"] for d in rounds]
        ys_hist = [d["hist_abs"] for d in rounds]
        y_label = "|delta|"
    else:
        ys_round = [d["round_signed"] for d in rounds]
        ys_hist = [d["hist_signed"] for d in rounds]
        y_label = "delta = estimate - target"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys_round, marker="o", linewidth=1.2, color="C0", alpha=0.85,
            label="per-round best")
    ax.plot(xs, ys_hist, marker="s", linewidth=1.8, color="C3",
            label="historical best")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(tol, color="gray", linewidth=0.8, linestyle="--",
               label=f"+/- {tol} tol")
    if not use_abs:
        ax.axhline(-tol, color="gray", linewidth=0.8, linestyle="--")

    if use_abs:
        ax.axhspan(0, tol, color="gray", alpha=0.08)
    else:
        ax.axhspan(-tol, tol, color="gray", alpha=0.08)

    ax.set_xlabel("round")
    ax.set_ylabel(y_label)
    ax.set_title(f"Convergence: {split_dir.name}")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="Dir containing split_<TS>/ subdirs")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <results_dir>/convergence_plots/)")
    ap.add_argument("--tol", type=float, default=0.15, help="Tolerance band (default 0.15)")
    ap.add_argument("--abs", action="store_true",
                    help="Plot |delta| instead of signed delta")
    args = ap.parse_args()

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"not a directory: {results_dir}")

    out_dir = args.out.resolve() if args.out else results_dir / "convergence_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = sorted(p for p in results_dir.iterdir()
                    if p.is_dir() and p.name.startswith("split_"))
    if not splits:
        raise SystemExit(f"no split_* dirs under {results_dir}")

    n_ok = 0
    for split in splits:
        out_path = out_dir / f"convergence_{split.name}.png"
        if plot_split(split, out_path, tol=args.tol, use_abs=args.abs):
            print(f"  wrote {out_path}")
            n_ok += 1
        else:
            print(f"  skipped {split.name} (no candidates_round_*.jsonl)")
    print(f"done: {n_ok}/{len(splits)} splits plotted into {out_dir}")


if __name__ == "__main__":
    main()
