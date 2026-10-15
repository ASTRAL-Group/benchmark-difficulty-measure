"""Tests for IRT csv + eval jsonl merge."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.irt_join import merge_q_indexed_irt_csv_with_eval_jsonl


class TestIrtJoin(unittest.TestCase):
    def test_merge_two_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            csv_p = base / "d.csv"
            csv_p.write_text(
                "item,b_mean,b_sd\nq0,1.0,0.1\nq1,2.0,0.2\n",
                encoding="utf-8",
            )
            j = base / "t.jsonl"
            j.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "problem": "P1?",
                        "answer": "A1",
                    },
                )
                + "\n"
                + json.dumps(
                    {
                        "id": 2,
                        "problem": "P2?",
                        "answer": "A2",
                    },
                )
                + "\n",
                encoding="utf-8",
            )
            items = merge_q_indexed_irt_csv_with_eval_jsonl(csv_p, j)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].id, "1")
            self.assertEqual(items[0].question, "P1?")
            self.assertEqual(items[0].reference_answer, "A1")
            self.assertEqual(items[0].difficulty, 1.0)
            self.assertEqual(items[1].difficulty, 2.0)


class TestIrtJoinAime24RealPaths(unittest.TestCase):
    def test_aime24_files_align(self) -> None:
        root = Path(__file__).resolve().parents[2]  # workspace root (diff)
        csv_p = root / "irt_results_math_coding" / "aime24_difficulty.csv"
        j = root / "evaluate" / "evaluation" / "data" / "aime24" / "test.jsonl"
        if not csv_p.is_file() or not j.is_file():
            self.skipTest("repo irt/eval paths not present")
        items = merge_q_indexed_irt_csv_with_eval_jsonl(csv_p, j)
        self.assertEqual(len(items), 30)
        self.assertEqual(items[0].id, "60")
        self.assertIn("Aya", items[0].question)
        self.assertAlmostEqual(items[0].difficulty, -3.2133474791724717)


if __name__ == "__main__":
    unittest.main()
