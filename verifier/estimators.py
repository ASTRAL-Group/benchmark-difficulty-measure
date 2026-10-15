"""Difficulty estimators.

FullRolloutEstimator  — all M models × n_trials, then IRT grid posterior
OfflineEstimator      — works with pre-computed response vectors (for validation)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from . import irt
from .config import VerifierConfig
from .rollout import RolloutResult, RolloutRunner


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class DifficultyResult:
    """Scalar difficulty on the IRT b scale."""

    estimate: float
    target: float
    confidence: str = "low"  # "low" | "medium" | "high"
    notes: str = ""
    raw: dict[str, Any] | None = None


def _confidence_label(b_sd: float, cfg: VerifierConfig) -> str:
    if b_sd < cfg.high_conf_sd:
        return "high"
    if b_sd < cfg.medium_conf_sd:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DifficultyEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        question: str,
        reference_answer: str,
        *,
        target: float,
        anchors: Sequence[dict] | None = None,
    ) -> DifficultyResult:
        """Estimate difficulty of a candidate item."""


# ---------------------------------------------------------------------------
# Full Rollout
# ---------------------------------------------------------------------------


class FullRolloutEstimator(DifficultyEstimator):
    """Run all M models × n_trials, then IRT posterior for b."""

    def __init__(self, runner: RolloutRunner, cfg: VerifierConfig):
        self.runner = runner
        self.cfg = cfg
        self.theta, self.model_names = irt.load_theta(cfg)
        self.meta = cfg.meta

    def estimate(
        self,
        question: str,
        reference_answer: str,
        *,
        target: float,
        anchors: Sequence[dict] | None = None,
    ) -> DifficultyResult:
        rollout = self.runner.run_single_item(question, reference_answer)
        y_m = rollout.to_y_vector(self.model_names)

        b_hat, b_sd = irt.estimate_b_grid(
            self.theta,
            y_m,
            self.meta.n_trials,
            self.meta.c,
            grid_lo=self.cfg.grid_lo,
            grid_hi=self.cfg.grid_hi,
            grid_n=self.cfg.grid_n,
            prior_mu=self.cfg.prior_mu,
            prior_sd=self.cfg.prior_sd,
        )

        return DifficultyResult(
            estimate=b_hat,
            target=target,
            confidence=_confidence_label(b_sd, self.cfg),
            notes=f"full rollout M={len(self.theta)}",
            raw={
                "b_sd": b_sd,
                "response_vector": y_m.tolist(),
                "models": self.model_names,
                "n_trials": self.meta.n_trials,
                "c": self.meta.c,
            },
        )


# ---------------------------------------------------------------------------
# Offline estimator (for validation with pre-computed response matrix)
# ---------------------------------------------------------------------------


class OfflineEstimator:
    """Estimate b from a pre-existing response matrix row."""

    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.theta, self.model_names = irt.load_theta(cfg)
        self.meta = cfg.meta

    def estimate_from_vector(
        self,
        y_m: NDArray,
        target: float,
    ) -> DifficultyResult:
        """Estimate b given a response vector (from response matrix)."""
        b_hat, b_sd = irt.estimate_b_grid(
            self.theta,
            y_m,
            self.meta.n_trials,
            self.meta.c,
            grid_lo=self.cfg.grid_lo,
            grid_hi=self.cfg.grid_hi,
            grid_n=self.cfg.grid_n,
            prior_mu=self.cfg.prior_mu,
            prior_sd=self.cfg.prior_sd,
        )

        return DifficultyResult(
            estimate=b_hat,
            target=target,
            confidence=_confidence_label(b_sd, self.cfg),
            notes=f"offline full M={len(self.theta)}",
            raw={"b_sd": b_sd, "y_m": y_m.tolist()},
        )
