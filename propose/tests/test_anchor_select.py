"""Tests for gap-end anchor selection (no mocks, small fixed pools)."""

from __future__ import annotations

import unittest

from utils.anchor_select import select_gap_end_anchors
from utils.schemas import ProblemItem


class TestSelectGapEndAnchors(unittest.TestCase):
    def test_two_per_side(self) -> None:
        # Difficulties on 0–10-style scale (same logic as any monotone difficulty axis).
        pool = [
            ProblemItem("a", "q", "a", 3.0),
            ProblemItem("b", "q", "a", 5.0),
            ProblemItem("c", "q", "a", 5.2),
            ProblemItem("d", "q", "a", 8.0),
            ProblemItem("e", "q", "a", 8.2),
            ProblemItem("f", "q", "a", 9.0),
        ]
        sel = select_gap_end_anchors(pool, 5.5, 7.0, per_side=2)
        self.assertEqual([x.id for x in sel], ["c", "b", "d", "e"])

    def test_skips_none_difficulty(self) -> None:
        pool = [
            ProblemItem("x", "q", "a", None),
            ProblemItem("y", "q", "a", 5.0),
            ProblemItem("z", "q", "a", 7.5),
        ]
        sel = select_gap_end_anchors(pool, 5.5, 7.0, per_side=1)
        self.assertEqual([x.id for x in sel], ["y", "z"])


if __name__ == "__main__":
    unittest.main()
