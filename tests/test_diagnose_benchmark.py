"""Metric and CLI checks: python3 -m unittest discover -s tests -v."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.diagnose_benchmark import diagnose, read_difficulties

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_benchmark.py"


class DiagnoseTests(unittest.TestCase):
    def test_uniform_reference_has_zero_ddm_and_boundary_gaps(self):
        result = diagnose([0.75, 0.25], normalized=True)
        self.assertEqual(result["ddm"], 0.0)
        self.assertEqual(result["max_gap"], 0.5)
        self.assertEqual(result["max_gap_interval"], [0.25, 0.75])

    def test_largest_gap_can_be_a_boundary_interval(self):
        result = diagnose([0.1, 0.3], normalized=True)
        self.assertAlmostEqual(result["ddm"], 0.3)
        self.assertAlmostEqual(result["max_gap"], 0.7)
        self.assertEqual(result["max_gap_interval"], [0.3, 1.0])

    def test_raw_values_are_normalized_and_duplicates_count_as_items(self):
        result = diagnose([4.0, -2.0, -2.0])
        self.assertEqual(result["items"], 3)
        self.assertAlmostEqual(result["ddm"], 5 / 18)
        self.assertEqual(result["max_gap"], 1.0)
        self.assertEqual(result["max_gap_interval"], [0.0, 1.0])

    def test_normalized_singleton_and_constant_values(self):
        self.assertEqual(diagnose([0.5], normalized=True)["ddm"], 0.0)
        self.assertEqual(diagnose([0.5, 0.5], normalized=True)["max_gap"], 0.5)

    def test_invalid_difficulties_are_rejected(self):
        for values in ([], [1.0], [2.0, 2.0], [0.0, float("nan")],
                       [0.0, float("inf")], [-1e308, 1e308]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                diagnose(values)
        for values in ([-0.1, 0.5], [0.5, 1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                diagnose(values, normalized=True)

    def test_published_aime24_metrics(self):
        result = diagnose(read_difficulties(ROOT / "data/benchmarks/AIME24/difficulty.csv"))
        self.assertEqual(result["items"], 30)
        self.assertEqual(round(result["ddm"], 3), 0.077)
        self.assertEqual(round(result["max_gap"], 3), 0.179)

    def test_cli_custom_column_normalized_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my benchmark.csv"
            path.write_text("\ufeffitem_id,z\n0,0.1\n1,0.3\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), "--column", "z", "--normalized", "--json"],
                capture_output=True, text=True, cwd=directory,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["normalization"], "already_normalized")
            self.assertAlmostEqual(result["max_gap"], 0.7)

    def test_invalid_csv_fails_without_partial_metrics(self):
        cases = ["", "other\n1\n", "b_mean\n", "b_mean\n1\n",
                 "item_id,b_mean\n0,0\n1,\n", "b_mean\n0\nnan\n",
                 "b_mean\n0\ninf\n", "b_mean\n0\ninvalid\n", "b_mean,b_mean\n0,1\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for contents in cases:
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    proc = subprocess.run(
                        [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True,
                    )
                    self.assertEqual(proc.returncode, 2)
                    self.assertEqual(proc.stdout, "")
                    self.assertIn("error:", proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
