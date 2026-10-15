"""Tests for failure → next-round feedback summarization."""

from __future__ import annotations

import unittest

from run.pipeline import summarize_failures_for_next_round
from utils.schemas import CorrectnessResult, DifficultyResult, ProblemItem, ScoredCandidate


class TestSummarizeFailures(unittest.TestCase):
    def test_correctness_failure_adds_evidence(self) -> None:
        item = ProblemItem("1", "q", "a", 5.0)
        sc = ScoredCandidate(
            candidate=item,
            correctness=CorrectnessResult(ok=False, failure_code="PARSE_ERROR", evidence="bad"),
            difficulty=None,
            accepted=False,
        )
        fb = summarize_failures_for_next_round([sc], target=5.0, tol=1.0)
        text = fb.to_prompt_bullets()
        self.assertIn("PARSE_ERROR", text)
        self.assertIn("Correctness: bad", text)

    def test_difficulty_failure_adds_hints(self) -> None:
        item = ProblemItem("1", "q", "a", 5.0)
        d = DifficultyResult(estimate=9.0, target=5.0, confidence="low")
        sc = ScoredCandidate(
            candidate=item,
            correctness=CorrectnessResult(ok=True),
            difficulty=d,
            accepted=False,
        )
        fb = summarize_failures_for_next_round([sc], target=5.0, tol=0.5)
        text = fb.to_prompt_bullets()
        self.assertIn("simplify", text.lower())


if __name__ == "__main__":
    unittest.main()
