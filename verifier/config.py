"""Benchmark metadata and verifier configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------

JudgeType = Literal["numeric", "letter", "medqa", "string"]


@dataclass(frozen=True)
class BenchmarkMeta:
    """Static properties of a benchmark needed by the verifier."""

    n_trials: int  # rollouts per model (must match IRT training)
    c: float  # guessing floor = 1/K for K-way MC, 0 for free-response
    judge: JudgeType  # how to score a response


BENCHMARK_META: dict[str, BenchmarkMeta] = {
    # --- MC benchmarks ---
    "gpqa": BenchmarkMeta(n_trials=4, c=1 / 4, judge="letter"),
    "mmlu_bio": BenchmarkMeta(n_trials=4, c=1 / 10, judge="letter"),
    "mmlu_chem": BenchmarkMeta(n_trials=4, c=1 / 10, judge="letter"),
    "mmlu_phys": BenchmarkMeta(n_trials=4, c=1 / 10, judge="letter"),
    "medqa": BenchmarkMeta(n_trials=4, c=1 / 5, judge="medqa"),
    "scienceqa": BenchmarkMeta(n_trials=4, c=1 / 4, judge="letter"),
    "bbh_3obj": BenchmarkMeta(n_trials=4, c=1 / 3, judge="letter"),
    "bbh_5obj": BenchmarkMeta(n_trials=4, c=1 / 5, judge="letter"),
    "bbh_7obj": BenchmarkMeta(n_trials=4, c=1 / 7, judge="letter"),
    # --- free-response benchmarks ---
    "ifeval": BenchmarkMeta(n_trials=4, c=0.0, judge="string"),
    "zebra": BenchmarkMeta(n_trials=4, c=0.0, judge="string"),
    # --- math (32 trials) ---
    "aime24": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "aime25": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "aime25_new": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "aime26": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "amc23": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "math-500": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "gsm8k": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "olympiadbench": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "minerva_math": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
    "gaokao2023en": BenchmarkMeta(n_trials=32, c=0.0, judge="numeric"),
}

# ---------------------------------------------------------------------------
# Verifier configuration
# ---------------------------------------------------------------------------

IRT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "irt_results"
MATRIX_DIR = Path(__file__).resolve().parent.parent / "data" / "matrix"
# Default CSV output for ``verifier.main`` when ``--out`` is omitted (under this package).
VERIFIER_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class VerifierConfig:
    """Runtime knobs for the difficulty verifier."""

    benchmark: str = "gpqa"

    # IRT prior
    prior_mu: float = 0.0
    prior_sd: float = 1.0

    # Posterior grid
    grid_lo: float = -4.0
    grid_hi: float = 4.0
    grid_n: int = 2000

    # Confidence thresholds on posterior sd
    high_conf_sd: float = 0.15
    medium_conf_sd: float = 0.30

    # Paths
    irt_results_dir: Path = field(default_factory=lambda: IRT_RESULTS_DIR)

    @property
    def meta(self) -> BenchmarkMeta:
        return BENCHMARK_META[self.benchmark]
