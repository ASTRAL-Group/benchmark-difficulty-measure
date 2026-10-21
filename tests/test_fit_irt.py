"""Install requirements_irt.txt, then run python -m unittest discover -s tests -v."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.fit_irt import build_model, read_guessing, read_matrix, summarize

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_irt.py"
HAS_IRT = all(importlib.util.find_spec(name) is not None for name in ("pymc", "arviz", "numpy", "scipy"))


class IRTInputTests(unittest.TestCase):
    def test_matrix_preserves_string_ids_and_zero_full_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            path.write_text("\ufeffmodel,002,001\n02,0,4\n01,2,3\n", encoding="utf-8")
            self.assertEqual(read_matrix(path, 4), (["02", "01"], ["002", "001"], [[0, 4], [2, 3]]))

    def test_invalid_matrices(self):
        cases = ["", "model,q0,q1\n", "name,q0,q1\na,1,0\nb,0,1\n",
                 "model,q0,q0\na,1,0\nb,0,1\n", "model,,q1\na,1,0\nb,0,1\n",
                 "model,q0,q1\na,1,0\na,0,1\n", "model,q0,q1\n,1,0\nb,0,1\n",
                 "model,q0,q1\na,1\nb,0,1\n", "model,q0,q1\na,1,0,1\nb,0,1\n",
                 "model,q0,q1\na,1,0\n", "model,q0\na,1\nb,0\n"]
        cases += [f"model,q0,q1\na,1,{value}\nb,0,1\n" for value in ("", "0.5", "NaN", "inf", "-1", "5")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_matrix(path, 4)
            with self.assertRaisesRegex(ValueError, "positive"):
                read_matrix(path, 0)

    def test_guessing_csv_is_aligned_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guessing.csv"
            path.write_text("item_id,guessing\n002,0.25\n001,0.5\n", encoding="utf-8")
            self.assertEqual(read_guessing(["001", "002"], path=path), [0.5, 0.25])
        self.assertEqual(read_guessing(["a", "b"]), [0.0, 0.0])

    def test_invalid_guessing(self):
        for value in (-0.1, 1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_guessing(["q0", "q1"], value)
        cases = ["item_id,guessing\nq0,0\n", "item_id,guessing\nq0,0\nq2,0\n",
                 "item_id,guessing\nq0,0\nq0,0.5\nq1,0\n",
                 "item_id,guessing\nq0,NaN\nq1,0\n", "item_id,guessing\nq0,1\nq1,0\n",
                 "item_id,guessing\nq0,\nq1,0\n", "item_id,guessing\nq0,0,extra\nq1,0\n"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guessing.csv"
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_guessing(["q0", "q1"], path=path)

    def test_cli_validates_before_sampling_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            output = Path(directory) / "fit"
            output.mkdir()
            existing = output / "difficulty.csv"
            existing.write_text("existing\n", encoding="utf-8")
            path.write_text("model,q0,q1\na,1,0\nb,0,1\n", encoding="utf-8")
            invalid_options = [["--chains", "1"], ["--n-trials", "0"], ["--guessing", "nan"],
                               ["--target-accept", "1"], ["--draws", "0"], ["--seed", "-1"]]
            for options in invalid_options:
                with self.subTest(options=options):
                    proc = subprocess.run(
                        [sys.executable, str(SCRIPT), str(path), "--n-trials", "1",
                         "--output-dir", str(output), *options], capture_output=True, text=True,
                    )
                    self.assertEqual(proc.returncode, 2)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertEqual(existing.read_text(), "existing\n")


@unittest.skipUnless(HAS_IRT, "Install requirements_irt.txt to run model and sampling tests")
class IRTModelTests(unittest.TestCase):
    def test_log_posterior_matches_binomial_and_normal_densities(self):
        import numpy as np
        from scipy.special import expit
        from scipy.stats import binom, norm

        theta, b = np.array([-0.8, 1.1]), np.array([0.4, -1.2])
        counts = np.array([[1, 3], [2, 4]])
        for guessing in ([0.0, 0.0], [0.25, 0.5]):
            with self.subTest(guessing=guessing):
                model = build_model(["a", "b"], ["q0", "q1"], counts, 4, guessing)
                actual = model.compile_logp()({"theta": theta, "b": b})
                g = np.array(guessing)
                p = g + (1 - g) * expit(theta[:, None] - b[None, :])
                expected = binom.logpmf(counts, 4, p).sum() + norm.logpdf(theta).sum() + norm.logpdf(b).sum()
                self.assertAlmostEqual(float(actual), float(expected), places=8)

    def test_cli_recovers_synthetic_order_and_feeds_diagnostics(self):
        import numpy as np
        from scipy.special import expit
        from scripts.diagnose_benchmark import diagnose, read_difficulties

        theta = np.array([-1.6, -0.5, 0.5, 1.6])
        b = np.array([-1.5, -0.6, 0.0, 0.6, 1.5])
        g = np.array([0, 0.25, 0, 0.25, 0])
        counts = np.random.default_rng(2026).binomial(128, g + (1 - g) * expit(theta[:, None] - b))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix, guessing, output = root / "matrix.csv", root / "guessing.csv", root / "fit"
            with matrix.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["model", "004", "003", "002", "001", "000"])
                writer.writerows([f"model-{i}", *row] for i, row in enumerate(counts))
            guessing.write_text("item_id,guessing\n000,0\n001,0.25\n002,0\n003,0.25\n004,0\n")
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(matrix), "--n-trials", "128", "--output-dir", str(output),
                 "--guessing-csv", str(guessing), "--chains", "4", "--draws", "300", "--tune", "300", "--quiet"],
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with (output / "difficulty.csv").open() as handle:
                difficulties = list(csv.DictReader(handle))
            with (output / "ability.csv").open() as handle:
                abilities = list(csv.DictReader(handle))
            self.assertEqual([row["item"] for row in difficulties], ["004", "003", "002", "001", "000"])
            fitted_b = np.array([float(row["b_mean"]) for row in difficulties])
            fitted_theta = np.array([float(row["theta_mean"]) for row in abilities])
            self.assertTrue(np.all(np.diff(fitted_b) > 0))
            self.assertTrue(np.all(np.diff(fitted_theta) > 0))
            np.testing.assert_allclose(fitted_b, b, atol=0.45)
            self.assertTrue(all(float(row["b_sd"]) > 0 for row in difficulties))
            metadata = json.loads((output / "diagnostics.json").read_text())
            self.assertEqual(metadata["n_trials"], 128)
            self.assertEqual(metadata["guessing"], list(g))
            self.assertEqual(metadata["chains"], 4)
            self.assertIsNotNone(metadata["max_rhat"])
            self.assertIsNotNone(metadata["min_bulk_ess"])
            result = diagnose(read_difficulties(output / "difficulty.csv"))
            self.assertEqual(result["items"], 5)
            self.assertTrue(0 <= result["ddm"] <= 1 and 0 <= result["max_gap"] <= 1)


if __name__ == "__main__":
    unittest.main()
