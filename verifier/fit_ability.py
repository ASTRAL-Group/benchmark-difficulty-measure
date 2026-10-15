"""Estimate local model abilities from the released response matrices.

The release ships item difficulties and the 12-model response counts. It does
not ship model abilities. This module recovers one ability per model by a
conditional MAP, holding the released item difficulties fixed:

    y_{m,j} ~ Binomial(n, σ(θ_m − b_j)),   θ_m ~ Normal(0, 1)

b_j is ``b_mean`` from ``irt_results/<benchmark>_difficulty.csv``. n is the
trial count of that benchmark's response matrix (32 for AIME/AMC, 8 for
Gaokao2023en, 4 for BBH-3 / MMLU-Pro-Biology / IFEval). The guessing floor
is 0, matching the 1PL binomial fit that produced the released difficulties.

The result is written to ``irt_results/<benchmark>_ability.csv`` with columns
``model,theta_mean,theta_sd``. That file is gitignored. It is this machine's
panel, not the posterior means behind the paper numbers. New-item difficulties
scored with it sit on the same 1PL scale as the released item difficulties.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .config import IRT_RESULTS_DIR, MATRIX_DIR

# Trial counts of the released matrices, not the new-item rollout length.
MATRIX_N_TRIALS: dict[str, int] = {
    "aime24": 32,
    "aime25": 32,
    "aime26": 32,
    "amc23": 32,
    "gaokao2023en": 8,
    "bbh_3obj": 4,
    "mmlu_bio": 4,
    "ifeval": 4,
}


def ability_csv_path(benchmark: str, irt_results_dir: Path | None = None) -> Path:
    root = Path(irt_results_dir) if irt_results_dir is not None else IRT_RESULTS_DIR
    return root / f"{benchmark}_ability.csv"


def _map_theta(y: np.ndarray, b: np.ndarray, n_trials: int) -> tuple[float, float]:
    """MAP of one model's ability and a Laplace standard deviation."""

    def neg_log_post(theta: float) -> float:
        p = np.clip(expit(theta - b), 1e-12, 1.0 - 1e-12)
        ll = np.sum(y * np.log(p) + (n_trials - y) * np.log(1.0 - p))
        return float(-(ll - 0.5 * theta * theta))

    res = minimize_scalar(neg_log_post, bounds=(-8.0, 8.0), method="bounded")
    if not res.success:
        raise RuntimeError(f"ability MAP failed: {res.message}")
    theta_hat = float(res.x)
    eps = 1e-4
    d2 = (
        neg_log_post(theta_hat + eps)
        - 2.0 * neg_log_post(theta_hat)
        + neg_log_post(theta_hat - eps)
    ) / eps**2
    theta_sd = float(1.0 / np.sqrt(max(d2, 1e-8)))
    return theta_hat, theta_sd


def fit_benchmark(
    benchmark: str,
    irt_results_dir: Path | None = None,
    matrix_dir: Path | None = None,
    out_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Write ``<benchmark>_ability.csv``. Skip an existing file unless ``force``."""
    if benchmark not in MATRIX_N_TRIALS:
        raise ValueError(
            f"No released response matrix trial count for {benchmark}. "
            f"Known: {sorted(MATRIX_N_TRIALS)}"
        )
    out = Path(out_path) if out_path is not None else ability_csv_path(benchmark, irt_results_dir)
    if out.exists() and not force:
        return out

    matrix_root = Path(matrix_dir) if matrix_dir is not None else MATRIX_DIR
    irt_root = Path(irt_results_dir) if irt_results_dir is not None else IRT_RESULTS_DIR
    matrix_path = matrix_root / f"{benchmark}_response_matrix.csv"
    diff_path = irt_root / f"{benchmark}_difficulty.csv"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Response matrix not found: {matrix_path}")
    if not diff_path.is_file():
        raise FileNotFoundError(f"Item difficulties not found: {diff_path}")

    counts = pd.read_csv(matrix_path, index_col=0)
    difficulty = pd.read_csv(diff_path).set_index("item")["b_mean"]
    missing = [col for col in counts.columns if col not in difficulty.index]
    extra = [item for item in difficulty.index if item not in counts.columns]
    if missing or extra:
        raise ValueError(
            f"{benchmark}: matrix columns and difficulty items must match. "
            f"only in matrix={missing[:6]} only in difficulty={extra[:6]}"
        )
    b = difficulty.loc[list(counts.columns)].to_numpy(dtype=float)
    n_trials = MATRIX_N_TRIALS[benchmark]
    observed = counts.to_numpy(dtype=float)
    if np.any(observed > n_trials) or np.any(observed < 0):
        raise ValueError(
            f"{benchmark}: counts must lie in 0..{n_trials} "
            f"(got min={observed.min()}, max={observed.max()})"
        )

    rows = []
    for model, y in zip(counts.index.astype(str), observed, strict=True):
        theta_mean, theta_sd = _map_theta(y, b, n_trials)
        rows.append({"model": model, "theta_mean": theta_mean, "theta_sd": theta_sd})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[fit_ability] {benchmark}: wrote {len(rows)} model abilities → {out}", flush=True)
    return out


def fit_available(
    irt_results_dir: Path | None = None,
    matrix_dir: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Fit every benchmark that has both a released matrix and a difficulty table."""
    matrix_root = Path(matrix_dir) if matrix_dir is not None else MATRIX_DIR
    written: list[Path] = []
    for matrix_path in sorted(matrix_root.glob("*_response_matrix.csv")):
        benchmark = matrix_path.name[: -len("_response_matrix.csv")]
        if benchmark not in MATRIX_N_TRIALS:
            print(f"[fit_ability] skip {benchmark}: no trial count", flush=True)
            continue
        written.append(
            fit_benchmark(
                benchmark,
                irt_results_dir=irt_results_dir,
                matrix_dir=matrix_root,
                force=force,
            )
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate local model abilities from released matrices and item difficulties."
    )
    parser.add_argument("--benchmark", default=None, help="One benchmark. Default: every released matrix.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing ability CSV.")
    args = parser.parse_args()
    if args.benchmark:
        fit_benchmark(args.benchmark, force=args.force)
    else:
        fit_available(force=args.force)


if __name__ == "__main__":
    main()
