"""Fit joint Bayesian 1PL IRT from model-by-item correct counts.

Paper Appendix C.3: theta_m, b_i ~ Normal(0, 1), independently;
y_mi ~ Binomial(n_trials, g_i + (1 - g_i) * sigmoid(theta_m - b_i)).
Guessing probabilities g_i are fixed, defaulting to zero. NUTS defaults:
4 chains, 1000 tuning steps and 1000 draws per chain, target_accept=0.95.

Input: CSV with model names in the first column (header: model), item IDs
in subsequent headers, and integer correct counts. IDs and order are preserved.
Output: difficulty.csv, ability.csv, and diagnostics.json in --output-dir.
The Gaussian priors anchor the latent location; outputs are not min-max scaled.

Install: python -m pip install -r requirements_irt.txt
Run: python scripts/fit_irt.py responses_matrix.csv --n-trials 32 --output-dir fit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def read_matrix(path: Path, n_trials: int) -> tuple[list[str], list[str], list[list[int]]]:
    """Read a complete count matrix without converting string IDs to numbers."""
    if n_trials < 1:
        raise ValueError("--n-trials must be positive")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if len(header) < 2 or header[0] != "model":
            raise ValueError("Matrix header must be model,<item ID>,<item ID>,...")
        items = header[1:]
        if any(not item.strip() for item in items) or len(set(items)) != len(items):
            raise ValueError("Item IDs must be nonempty and unique")
        models, counts = [], []
        for row in reader:
            if len(row) != len(header) or not row[0].strip():
                raise ValueError(f"CSV line {reader.line_num}: incomplete model/item row")
            if row[0] in models:
                raise ValueError(f"Duplicate model ID: {row[0]!r}")
            try:
                values = [int(value) for value in row[1:]]
            except ValueError:
                raise ValueError(f"CSV line {reader.line_num}: counts must be integers") from None
            if any(value < 0 or value > n_trials for value in values):
                raise ValueError(f"CSV line {reader.line_num}: counts must be in 0..{n_trials}")
            models.append(row[0])
            counts.append(values)
    if len(models) < 2 or len(items) < 2:
        raise ValueError("Joint IRT fitting requires at least two models and two items")
    return models, items, counts


def read_guessing(items: list[str], value: float = 0.0, path: Path | None = None) -> list[float]:
    """Use one fixed chance level, or align item_id,guessing CSV rows by ID."""
    if path is None:
        values = [value] * len(items)
    else:
        lookup = {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames or any(
                reader.fieldnames.count(key) != 1 for key in ("item_id", "guessing")
            ):
                raise ValueError("Guessing CSV must contain item_id,guessing exactly once each")
            for row in reader:
                item = row["item_id"]
                if None in row or not item or item in lookup:
                    raise ValueError("Guessing CSV contains missing, duplicate, or malformed item rows")
                try:
                    lookup[item] = float(row["guessing"])
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid guessing probability for {item!r}") from None
        if set(lookup) != set(items):
            raise ValueError("Guessing CSV item IDs must match the matrix columns exactly")
        values = [lookup[item] for item in items]
    if any(not math.isfinite(g) or not 0 <= g < 1 for g in values):
        raise ValueError("Guessing probabilities must be finite and in [0, 1)")
    return values


def build_model(models, items, counts, n_trials, guessing):
    """Build the Appendix C.3 model, jointly fitting abilities and difficulties."""
    import numpy as np
    import pymc as pm

    with pm.Model(coords={"model_id": models, "item_id": items}) as model:
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="model_id")
        b = pm.Normal("b", mu=0.0, sigma=1.0, dims="item_id")
        g = np.asarray(guessing, dtype=float)
        p = g[None, :] + (1.0 - g[None, :]) * pm.math.sigmoid(theta[:, None] - b[None, :])
        pm.Binomial("correct", n=n_trials, p=p, observed=np.asarray(counts, dtype="int64"),
                    dims=("model_id", "item_id"))
    return model


def summarize(trace, models, items):
    """Return unrounded posterior summaries and joint sampling diagnostics."""
    import arviz as az
    import numpy as np

    rhat = az.rhat(trace, var_names=["theta", "b"], method="rank")
    ess = az.ess(trace, var_names=["theta", "b"], method="bulk")
    tables = []
    for variable, dimension, ids in (("b", "item_id", items), ("theta", "model_id", models)):
        samples = trace.posterior[variable].transpose("chain", "draw", dimension).values
        means = samples.mean(axis=(0, 1))
        sds = samples.reshape(-1, len(ids)).std(axis=0, ddof=1)
        if not np.isfinite(means).all() or not np.isfinite(sds).all():
            raise ValueError("Sampler returned nonfinite posterior summaries")
        tables.append(list(zip(ids, means, sds, rhat[variable].values, ess[variable].values)))
    all_rhat = np.concatenate([rhat[v].values for v in ("theta", "b")])
    all_ess = np.concatenate([ess[v].values for v in ("theta", "b")])
    diagnostics = {
        "max_rhat": float(all_rhat.max()) if np.isfinite(all_rhat).all() else None,
        "min_bulk_ess": float(all_ess.min()) if np.isfinite(all_ess).all() else None,
        "divergences": int(trace.sample_stats["diverging"].sum().item()),
    }
    return tables[0], tables[1], diagnostics


def write_results(output_dir, difficulty, ability, metadata):
    """Stage all three outputs before replacing existing result files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".irt-", dir=output_dir) as directory:
        staging = Path(directory)
        for filename, columns, rows in (
            ("difficulty.csv", ["item", "b_mean", "b_sd", "r_hat", "ess_bulk"], difficulty),
            ("ability.csv", ["model", "theta_mean", "theta_sd", "r_hat", "ess_bulk"], ability),
        ):
            with (staging / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows)
        (staging / "diagnostics.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        for filename in ("difficulty.csv", "ability.csv", "diagnostics.json"):
            os.replace(staging / filename, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix_path", type=Path)
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--guessing", type=float, default=0.0, help="Fixed chance level; e.g. 0.25 for four choices")
    group.add_argument("--guessing-csv", type=Path, help="Per-item chance levels: item_id,guessing")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain")
    parser.add_argument("--tune", type=int, default=1000, help="Warmup steps per chain")
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--cores", type=int, default=1, help="Concurrent chains (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Hide the sampling progress bar")
    args = parser.parse_args()
    try:
        if args.chains < 2 or args.draws < 4 or args.tune < 0 or args.cores < 1 or args.seed < 0:
            raise ValueError("Need chains >= 2, draws >= 4, tune >= 0, cores >= 1, seed >= 0")
        if not 0 < args.target_accept < 1:
            raise ValueError("--target-accept must be in (0, 1)")
        destinations = [(args.output_dir / name).resolve() for name in
                        ("difficulty.csv", "ability.csv", "diagnostics.json")]
        inputs = [args.matrix_path] + ([args.guessing_csv] if args.guessing_csv else [])
        if any(path.resolve() in destinations for path in inputs):
            raise ValueError("Input paths must not overlap output files")
        models, items, counts = read_matrix(args.matrix_path, args.n_trials)
        guessing = read_guessing(items, args.guessing, args.guessing_csv)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.error(str(error))

    try:
        import arviz as az
        import pymc as pm
    except ImportError:
        parser.error("Install IRT dependencies: python -m pip install -r requirements_irt.txt")
    model = build_model(models, items, counts, args.n_trials, guessing)
    with model:
        trace = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=min(args.cores, args.chains),
            target_accept=args.target_accept, random_seed=args.seed, progressbar=not args.quiet,
            return_inferencedata=True, nuts_sampler="pymc",
        )
    difficulty, ability, diagnostics = summarize(trace, models, items)
    metadata = {
        "model": "1PL with fixed item guessing probabilities",
        "priors": {"theta": "Normal(0, 1)", "b": "Normal(0, 1)"},
        "estimator": "NUTS posterior mean and standard deviation",
        "matrix_sha256": hashlib.sha256(args.matrix_path.read_bytes()).hexdigest(),
        "models": models, "items": items, "n_trials": args.n_trials, "guessing": guessing,
        "chains": args.chains, "draws": args.draws, "tune": args.tune,
        "target_accept": args.target_accept, "seed": args.seed,
        "versions": {"pymc": pm.__version__, "arviz": az.__version__},
        **diagnostics,
    }
    write_results(args.output_dir, difficulty, ability, metadata)
    print(f"Saved {len(items)} item difficulties and {len(models)} model abilities to {args.output_dir}")
    print(json.dumps(diagnostics, indent=2, allow_nan=False))
    if diagnostics["max_rhat"] is None or diagnostics["min_bulk_ess"] is None:
        print("Some sampling diagnostics are unavailable; inspect the chains before using estimates.", file=sys.stderr)


if __name__ == "__main__":
    main()
