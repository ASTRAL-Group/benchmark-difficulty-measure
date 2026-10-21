"""Aggregate graded responses into a model-by-item matrix of correct counts.

Input CSV: model,item_id,trial,correct (one row per response; correct is 0 or 1).
Trial IDs must be unique within each model/item pair. Every pair must have
exactly --n-trials responses. Models and item IDs keep their first-seen order;
IDs are not renumbered. Extra input columns are ignored.

Output CSV: model,<item IDs...>. Each cell is an integer in [0, n_trials],
matching the count-matrix format read by verifier/fit_ability.py.
Requires only Python 3.9+; model inference, grading, and IRT fitting are separate.

Usage:
    python3 scripts/prepare_response_matrix.py responses.csv --n-trials 32 \
        --output data/matrix/aime24_response_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile


def prepare_matrix(path: Path, n_trials: int) -> tuple[list[str], list[str], list[list[int]]]:
    """Validate the full observed model/item grid before returning any counts."""
    if n_trials < 1:
        raise ValueError("--n-trials must be a positive integer")
    models: dict[str, None] = {}
    items: dict[str, None] = {}
    responses: dict[tuple[str, str], dict[str, int]] = {}
    required = ("model", "item_id", "trial", "correct")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not reader.fieldnames or any(reader.fieldnames.count(key) != 1 for key in required):
            raise ValueError("CSV must contain model,item_id,trial,correct exactly once each")
        for row in reader:
            prefix = f"CSV line {reader.line_num}"
            if None in row or any(row[key] is None or not row[key].strip() for key in required):
                raise ValueError(f"{prefix}: missing fields or unexpected extra values")
            model, item, trial, correct = (row[key].strip() for key in required)
            if item == "model":
                raise ValueError(f"{prefix}: item_id 'model' conflicts with the output header")
            if correct not in ("0", "1"):
                raise ValueError(f"{prefix}: correct must be 0 or 1")
            cell = responses.setdefault((model, item), {})
            if trial in cell:
                raise ValueError(f"{prefix}: duplicate trial {trial!r} for {model!r}, {item!r}")
            cell[trial] = int(correct)
            models[model] = None
            items[item] = None
    if not responses:
        raise ValueError("CSV contains no graded responses")

    matrix = []
    for model in models:
        counts = []
        for item in items:
            cell = responses.get((model, item), {})
            if len(cell) != n_trials:
                raise ValueError(
                    f"{model!r}, {item!r}: expected {n_trials} trials, found {len(cell)}"
                )
            counts.append(sum(cell.values()))
        matrix.append(counts)
    return list(models), list(items), matrix


def write_matrix(path: Path, models: list[str], items: list[str], matrix: list[list[int]]) -> None:
    """Replace the output only after the complete CSV has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(["model", *items])
            writer.writerows([model, *counts] for model, counts in zip(models, matrix))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="CSV of graded individual responses")
    parser.add_argument("--n-trials", type=int, required=True, help="Responses per model/item pair")
    parser.add_argument("--output", type=Path, required=True, help="Destination count-matrix CSV")
    args = parser.parse_args()
    try:
        if args.csv_path.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files")
        models, items, matrix = prepare_matrix(args.csv_path, args.n_trials)
        write_matrix(args.output, models, items, matrix)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.error(str(error))
    print(f"Saved {len(models)} models × {len(items)} items to {args.output}")
    print(f"Each cell is the number correct out of {args.n_trials} trials.")


if __name__ == "__main__":
    main()
