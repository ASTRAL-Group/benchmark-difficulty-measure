"""Tests for the normalized max-gap helper."""

from __future__ import annotations

import unittest

from utils.max_gap import gap_midpoint_target, largest_normalized_gap_bracket_b


class TestLargestNormalizedGap(unittest.TestCase):
    def test_sparse_tail(self) -> None:
        # Normalized: 0, 0.1, 0.2, 0.3, 1.0 → largest step 0.7 between 3 and 10
        lo, hi, meta = largest_normalized_gap_bracket_b([0.0, 1.0, 2.0, 3.0, 10.0])
        self.assertAlmostEqual(lo, 3.0)
        self.assertAlmostEqual(hi, 10.0)
        self.assertGreater(meta["normalized_gap_width"], 0.69)

    def test_uniform_almost_equal_gaps(self) -> None:
        lo, hi, meta = largest_normalized_gap_bracket_b([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(meta["normalized_gap_width"], 0.25)

    def test_midpoint(self) -> None:
        self.assertAlmostEqual(gap_midpoint_target(0.0, 10.0), 5.0)


if __name__ == "__main__":
    unittest.main()
