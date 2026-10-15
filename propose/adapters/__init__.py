"""Integration adapters (e.g. repo ``verifier`` → ``propose`` pipelines)."""

from .verifier_difficulty import (
    VerifierDifficultyAdapter,
    build_verifier_difficulty_estimator,
)

__all__ = [
    "VerifierDifficultyAdapter",
    "build_verifier_difficulty_estimator",
]
