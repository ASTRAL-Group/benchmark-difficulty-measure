"""Bridge ``verifier`` IRT rollout difficulty → ``propose`` :class:`DifficultyEstimator`.

``_repo_root()`` resolves to the repo root (3 levels above this file), so
``verifier/`` is importable after ``sys.path`` insertion.

Endpoint JSON: a list of objects with ``name``, ``base_url``, ``model`` (optional, defaults to
``name``), and optional ``temperature``, ``top_p``, ``max_tokens``. Every model name in
``irt_results_math_coding/<benchmark>_ability.csv`` must appear as ``name``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from correctness.difficulty import DifficultyEstimator
from utils.schemas import DifficultyResult, ProblemItem


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _ensure_verifier_importable() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_endpoints_json(path: Path) -> list[Any]:
    from verifier.rollout import ModelEndpoint

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"endpoints JSON must be a list, got {type(data)}")
    out: list[ModelEndpoint] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"endpoints[{i}] must be an object")
        name = row.get("name")
        base_url = row.get("base_url")
        if not name or not base_url:
            raise ValueError(f"endpoints[{i}] needs non-empty name and base_url")
        model = row.get("model") or name
        out.append(
            ModelEndpoint(
                name=str(name),
                base_url=str(base_url),
                model=str(model),
                api_key=str(row.get("api_key", "EMPTY")),
                temperature=float(row.get("temperature", 0.6)),
                top_p=float(row.get("top_p", 0.95)),
                max_tokens=int(row.get("max_tokens", 10240)),
            )
        )
    return out


class VerifierDifficultyAdapter(DifficultyEstimator):
    """Wraps ``verifier.estimators.FullRolloutEstimator`` (IRT after multi-model rollout).

    ``b_min`` / ``b_max`` are the min/max b_hat values from the reference pool's difficulty CSV.
    The raw IRT b_hat returned by the verifier is normalized to [0, 10] using them so the
    pipeline's ``within_tolerance`` comparison operates on the same scale as ``target``.
    """

    def __init__(self, inner: Any, b_min: float, b_max: float) -> None:
        self._inner = inner
        self._b_min = b_min
        self._b_span = max(b_max - b_min, 1e-12)

    def estimate(
        self,
        item: ProblemItem,
        *,
        target: float,
        anchors: Sequence[ProblemItem],
    ) -> DifficultyResult:
        del anchors  # verifier API does not use anchor text
        vr = self._inner.estimate(
            item.question,
            item.reference_answer,
            target=target,
        )
        # Affine-map raw IRT b_hat to the 0-10 scale anchored by the reference
        # pool's b range. Out-of-range items are NOT clipped (negative or >10
        # carries real "outside the source distribution" signal — see
        # utils/difficulty_scale.to_difficulty_0_10).
        estimate_0_10 = 10.0 * (vr.estimate - self._b_min) / self._b_span
        conf: Any = vr.confidence
        if conf not in ("low", "medium", "high"):
            conf = "low"
        return DifficultyResult(
            estimate=estimate_0_10,
            target=float(vr.target),  # already on 0-10 scale from the caller
            confidence=conf,
            notes=vr.notes,
            raw=vr.raw,
        )


def build_verifier_difficulty_estimator(
    workspace: Path,
    benchmark: str,
    endpoints_json: Path,
) -> VerifierDifficultyAdapter:
    """Build :class:`VerifierDifficultyAdapter` from repo IRT tables + endpoint JSON."""
    _ensure_verifier_importable()
    import pandas as pd
    from verifier import irt as v_irt
    from verifier.config import VerifierConfig
    from verifier.estimators import FullRolloutEstimator
    from verifier.rollout import RolloutRunner

    irt_dir = (workspace / "irt_results_math_coding").resolve()
    if not irt_dir.is_dir():
        raise FileNotFoundError(f"IRT directory not found: {irt_dir}")

    cfg = VerifierConfig(benchmark=benchmark, irt_results_dir=irt_dir)
    _, model_names = v_irt.load_theta(cfg)

    # Load b_min / b_max from the reference pool for 0-10 normalization.
    diff_csv = irt_dir / f"{benchmark}_difficulty.csv"
    if not diff_csv.is_file():
        raise FileNotFoundError(f"IRT difficulty CSV not found: {diff_csv}")
    df_diff = pd.read_csv(diff_csv)
    b_col = "b_mean" if "b_mean" in df_diff.columns else "b_hat"
    b_min = float(df_diff[b_col].min())
    b_max = float(df_diff[b_col].max())

    endpoints = _parse_endpoints_json(endpoints_json.resolve())
    by_name = {e.name: e for e in endpoints}
    missing = [m for m in model_names if m not in by_name]
    if missing:
        raise ValueError(
            "endpoints JSON is missing entries for every model in "
            f"{benchmark}_ability.csv. Missing: {missing!r}"
        )
    ordered = [by_name[m] for m in model_names]
    runner = RolloutRunner(ordered, cfg)
    inner = FullRolloutEstimator(runner, cfg)
    return VerifierDifficultyAdapter(inner, b_min, b_max)
