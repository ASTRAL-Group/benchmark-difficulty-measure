"""Diagnose a benchmark from a CSV containing one difficulty per item.

Uses the paper's empirical uniform reference for DDM (midpoints of n equal
bins), and includes the boundary intervals in Maximum Gap. Requires only the
Python standard library; it reads existing difficulty estimates, not responses.

Usage:
    python3 scripts/diagnose_benchmark.py data/benchmarks/AIME24/difficulty.csv
    python3 scripts/diagnose_benchmark.py my_difficulties.csv --column difficulty
    python3 scripts/diagnose_benchmark.py normalized.csv --column z --normalized
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


def read_difficulties(path: Path, column: str = "b_mean") -> list[float]:
    """Read every item; reject missing, nonnumeric, or nonfinite values."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"CSV must contain a {column!r} column")
        if reader.fieldnames.count(column) != 1:
            raise ValueError(f"CSV contains duplicate {column!r} columns")
        values = []
        for row in reader:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                raise ValueError(
                    f"CSV line {reader.line_num}: {column!r} must be a finite number"
                ) from None
            if not math.isfinite(value):
                raise ValueError(
                    f"CSV line {reader.line_num}: {column!r} must be a finite number"
                )
            values.append(value)
    if not values:
        raise ValueError("CSV contains no item difficulties")
    return values


def diagnose(values: Sequence[float], *, normalized: bool = False) -> dict:
    """Return DDM and Maximum Gap on the [0, 1] scale.

    Raw difficulties are min-max normalized within this benchmark. Pass
    normalized=True only for values already on the intended [0, 1] scale;
    this preserves uncovered intervals next to the boundaries.
    """
    if len(values) == 0 or not all(math.isfinite(value) for value in values):
        raise ValueError("Provide at least one finite difficulty and no nonfinite values")
    lower, upper = min(values), max(values)
    if normalized:
        if lower < 0.0 or upper > 1.0:
            raise ValueError("With --normalized, every difficulty must be in [0, 1]")
        z = sorted(values)
    else:
        span = upper - lower
        if span == 0.0:
            raise ValueError(
                "Min-max normalization needs at least two distinct difficulties; "
                "use --normalized only if your values already represent a [0, 1] scale"
            )
        if not math.isfinite(span):
            raise ValueError("Difficulty range is too large for min-max normalization")
        z = sorted((value - lower) / span for value in values)

    n = len(z)
    ddm = math.fsum(abs(value - (i + 0.5) / n) for i, value in enumerate(z)) / n
    edges = [0.0, *z, 1.0]
    return {
        "items": n,
        "normalization": "already_normalized" if normalized else "per_benchmark_min_max",
        "input_min": lower,
        "input_max": upper,
        "ddm_reference": "same_size_uniform_midpoints",
        "ddm": ddm,
        "max_gap": max(right - left for left, right in zip(edges, edges[1:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="CSV with one row per benchmark item")
    parser.add_argument("--column", default="b_mean", help="Difficulty column (default: b_mean)")
    parser.add_argument(
        "--normalized", action="store_true",
        help="Use an existing [0, 1] scale without min-max normalization",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    try:
        result = diagnose(read_difficulties(args.csv_path, args.column), normalized=args.normalized)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(f"Items: {result['items']}")
        scale = "already in [0, 1]" if args.normalized else "per-benchmark min-max to [0, 1]"
        print(f"Normalization: {scale}")
        print(f"DDM: {result['ddm']:.6f}")
        print(f"Max Gap: {result['max_gap']:.6f}")


if __name__ == "__main__":
    main()
