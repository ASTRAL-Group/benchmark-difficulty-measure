"""Tests for IRT *b* → 0–10 min–max mapping."""

from __future__ import annotations

import unittest

from utils.difficulty_scale import (
    b_range_from_values,
    from_difficulty_0_10,
    normalize_items_to_0_10,
    to_difficulty_0_10,
)
from utils.max_gap import gap_midpoint_target
from utils.schemas import ProblemItem


class TestDifficultyScale(unittest.TestCase):
    def test_endpoints(self) -> None:
        self.assertAlmostEqual(to_difficulty_0_10(-2.0, -2.0, 8.0), 0.0)
        self.assertAlmostEqual(to_difficulty_0_10(8.0, -2.0, 8.0), 10.0)
        self.assertAlmostEqual(to_difficulty_0_10(3.0, -2.0, 8.0), 5.0)

    def test_out_of_range_not_clipped(self) -> None:
        # Out-of-range items extrapolate linearly (see to_difficulty_0_10).
        self.assertAlmostEqual(to_difficulty_0_10(-99.0, -2.0, 8.0), -97.0)
        self.assertAlmostEqual(to_difficulty_0_10(99.0, -2.0, 8.0), 101.0)

    def test_flat_pool_maps_to_five(self) -> None:
        self.assertAlmostEqual(to_difficulty_0_10(3.0, 3.0, 3.0), 5.0)

    def test_inverse_roundtrip(self) -> None:
        lo, hi = -1.5, 4.5
        for b in (-1.5, 0.0, 2.25, 4.5):
            s = to_difficulty_0_10(b, lo, hi)
            b2 = from_difficulty_0_10(s, lo, hi)
            self.assertAlmostEqual(b2, b, places=6)

    def test_normalize_items(self) -> None:
        raw = [
            ProblemItem("a", "q", "a", -2.0),
            ProblemItem("b", "q", "a", 8.0),
        ]
        out = normalize_items_to_0_10(raw, -2.0, 8.0)
        self.assertIsNotNone(out[0].difficulty)
        self.assertIsNotNone(out[1].difficulty)
        self.assertAlmostEqual(out[0].difficulty, 0.0)
        self.assertAlmostEqual(out[1].difficulty, 10.0)
        self.assertEqual(raw[0].difficulty, -2.0)

    def test_midpoint_affine_matches_b_to_10(self) -> None:
        b_lo, b_hi = 1.0, 5.0
        m_b = gap_midpoint_target(b_lo, b_hi)
        s_lo = to_difficulty_0_10(b_lo, b_lo, b_hi)
        s_hi = to_difficulty_0_10(b_hi, b_lo, b_hi)
        m_s = gap_midpoint_target(s_lo, s_hi)
        self.assertAlmostEqual(m_s, to_difficulty_0_10(m_b, b_lo, b_hi))

    def test_b_range_from_values(self) -> None:
        self.assertEqual(b_range_from_values([3.0, 1.0, 2.0]), (1.0, 3.0))


if __name__ == "__main__":
    unittest.main()
