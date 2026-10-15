"""Validation: compare verifier grid-posterior ``b`` estimates against PyMC ground truth.

Usage (``cwd`` = repo root, env ``diff``):

    conda run --no-capture-output -n diff python -m verifier.validate --benchmark aime24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import irt
from .config import BENCHMARK_META, MATRIX_DIR, VerifierConfig
from .estimators import OfflineEstimator


def run_validation(benchmark: str) -> pd.DataFrame:
    """Run full-rollout validation for every item in a benchmark."""
    cfg = VerifierConfig(benchmark=benchmark)

    b_true_arr, item_ids = irt.load_difficulty(cfg)

    matrix_path = MATRIX_DIR / f"{benchmark}_response_matrix.csv"
    df_matrix = pd.read_csv(matrix_path, index_col=0)
    response_matrix = df_matrix.to_numpy()  # (M, N)

    estimator = OfflineEstimator(cfg)

    rows = []
    for j, item_id in enumerate(item_ids):
        y_m = response_matrix[:, j]
        b_true = b_true_arr[j]

        res = estimator.estimate_from_vector(y_m, target=b_true)
        rows.append({
            "item": item_id,
            "b_true": b_true,
            "b_hat": res.estimate,
            "b_sd": res.raw["b_sd"],
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate verifier IRT estimates")
    parser.add_argument("--benchmark", default="gpqa", help="Benchmark name")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    benchmark = args.benchmark
    if benchmark not in BENCHMARK_META:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_META.keys())}")

    print(f"Validating verifier on {benchmark}...")
    df = run_validation(benchmark)

    b_true = df["b_true"].to_numpy()
    b_hat = df["b_hat"].to_numpy()
    b_sd = df["b_sd"].to_numpy()
    err = b_hat - b_true

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(b_true, b_hat)[0, 1])
    mean_sd = float(np.mean(b_sd))
    coverage = float(np.mean(np.abs(err) <= 1.96 * b_sd))

    cfg = VerifierConfig(benchmark=benchmark)
    theta_arr, _ = irt.load_theta(cfg)

    print(f"\n{'=' * 50}")
    print(f"  {benchmark}: {len(df)} items, {len(theta_arr)} models")
    print(f"{'=' * 50}")
    print(f"  RMSE:            {rmse:.4f}")
    print(f"  MAE:             {mae:.4f}")
    print(f"  Correlation:     {corr:.4f}")
    print(f"  Mean post. sd:   {mean_sd:.4f}")
    print(f"  95% CI coverage: {coverage:.2%}")

    out_dir = Path(args.out_dir) if args.out_dir else cfg.irt_results_dir.parent / "verifier_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{benchmark}_validation.csv", index=False)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
