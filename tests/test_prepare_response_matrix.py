"""Response aggregation and CLI checks: python3 -m unittest discover -s tests -v."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.prepare_response_matrix import prepare_matrix

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_response_matrix.py"
HEADER = "model,item_id,trial,correct\n"
EXAMPLE = HEADER + (
    "model-a,q0,0,1\nmodel-a,q0,1,0\nmodel-a,q1,0,1\nmodel-a,q1,1,1\n"
    "model-b,q0,0,0\nmodel-b,q0,1,0\nmodel-b,q1,0,1\nmodel-b,q1,1,0\n"
)


class ResponseMatrixTests(unittest.TestCase):
    def test_counts_preserve_ids_and_first_seen_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            path.write_text(
                "\ufeffmodel,item_id,trial,correct,response\n"
                "002,q10,seed-a,0,answer\n001,q2,seed-b,1,answer\n"
                "001,q10,seed-b,1,answer\n002,q2,seed-a,0,answer\n"
                "001,q2,seed-a,1,answer\n002,q10,seed-b,1,answer\n"
                "002,q2,seed-b,0,answer\n001,q10,seed-a,0,answer\n",
                encoding="utf-8",
            )
            models, items, matrix = prepare_matrix(path, 2)
            self.assertEqual(models, ["002", "001"])
            self.assertEqual(items, ["q10", "q2"])
            self.assertEqual(matrix, [[1, 0], [1, 2]])

    def test_cli_example_writes_count_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            output = Path(directory) / "matrix" / "response_matrix.csv"
            path.write_text(EXAMPLE, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), "--n-trials", "2", "--output", str(output)],
                capture_output=True, text=True, cwd=directory,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.reader(handle)), [
                    ["model", "q0", "q1"], ["model-a", "1", "2"], ["model-b", "0", "1"],
                ])
            self.assertIn("2 models × 2 items", proc.stdout)
            self.assertIn("out of 2 trials", proc.stdout)

    def test_incomplete_grid_duplicate_and_wrong_trial_counts_are_rejected(self):
        cases = [
            (EXAMPLE.replace("model-b,q1,0,1\nmodel-b,q1,1,0\n", ""), "found 0"),
            (EXAMPLE.replace("model-b,q1,1,0\n", ""), "found 1"),
            (EXAMPLE + "model-a,q0,2,0\n", "found 3"),
            (EXAMPLE + "model-a,q0,0,1\n", "duplicate trial"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            for contents, message in cases:
                with self.subTest(message=message):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        prepare_matrix(path, 2)

    def test_invalid_input_never_replaces_existing_output(self):
        cases = [
            "", HEADER, "model,item_id,correct\na,q0,1\n",
            "model,item_id,trial,correct,correct\na,q0,0,1,0\n",
            HEADER + "a,q0,0,\n", HEADER + ",q0,0,1\n",
            HEADER + "a,,0,1\n", HEADER + "a,q0,,1\n",
            HEADER + "a,q0,0,1,extra\n", HEADER + "a,q0,0\n",
            HEADER + "a,model,0,1\n",
        ] + [HEADER + f"a,q0,0,{value}\n" for value in ("NaN", "0.5", "2", "-1", "True")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            output = Path(directory) / "matrix.csv"
            output.write_text("existing result\n", encoding="utf-8")
            for contents in cases:
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    proc = subprocess.run(
                        [sys.executable, str(SCRIPT), str(path), "--n-trials", "1", "--output", str(output)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(proc.returncode, 2)
                    self.assertEqual(proc.stdout, "")
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertEqual(output.read_text(encoding="utf-8"), "existing result\n")

    def test_invalid_trial_count_and_same_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            path.write_text(EXAMPLE, encoding="utf-8")
            for n in (0, -1):
                with self.subTest(n=n), self.assertRaisesRegex(ValueError, "positive integer"):
                    prepare_matrix(path, n)
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(path), "--n-trials", "2", "--output", str(path)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("different files", proc.stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), EXAMPLE)


if __name__ == "__main__":
    unittest.main()
