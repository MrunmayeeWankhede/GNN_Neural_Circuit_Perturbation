"""Offline checks for the CLI and the numerical evidence shipped with the pilot."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from circuit_pruning import response_error
from connectome_model_v2 import SENSORY


ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def check_exports(self, folder):
        manifest = json.loads((folder / "manifest.json").read_text())
        summaries = json.loads((folder / "summary.json").read_text())
        traces = json.loads((folder / "traces.json").read_text())
        with (folder / "curves.csv").open(newline="") as handle:
            curves = list(csv.DictReader(handle))
        self.assertEqual(set(summaries), set(traces))
        self.assertEqual(set(summaries), {row["run"] for row in curves})
        protected = set(manifest["input_names"]) | set(manifest["output_names"])
        for label, summary in summaries.items():
            with self.subTest(run=label):
                trace = traces[label]
                removed = trace["removed_names"]
                retained = trace["retained_names"]
                steps = len(removed) + 1
                self.assertEqual(len(set(removed)), len(removed))
                self.assertFalse(set(removed) & set(retained))
                self.assertTrue(protected <= set(retained))
                self.assertEqual(len(set(removed) | set(retained)), manifest["node_count"])
                self.assertEqual(summary["removed_count"], len(removed))
                self.assertEqual(summary["retained_count"], len(retained))
                self.assertEqual(trace["stop_reason"], summary["stop_reason"])
                self.assertEqual(trace["search_trajectories"], summary["search_trajectories"])
                self.assertEqual([row["removed_neuron"] for row in trace["trace"]][1:], removed)
                rows = [row for row in curves if row["run"] == label]
                self.assertEqual(len(rows), steps)
                self.assertEqual(len(trace["trace"]), steps)
                self.assertEqual(len(trace["heldout_trace"]), steps)
                with np.load(folder / f"{label}_responses.npz", allow_pickle=False) as vectors:
                    self.assertEqual(vectors["outputs"].tolist(), manifest["output_names"])
                    for split in ("fit", "heldout"):
                        responses = vectors[split]
                        self.assertEqual(responses.shape, (
                            steps, len(manifest["stimuli"][f"{split}_names"]),
                            manifest["output_count"],
                        ))
                        for step, row in enumerate(rows):
                            self.assertEqual(int(row["removed_count"]), step)
                            self.assertEqual(int(row["retained_count"]), manifest["node_count"] - step)
                            record = trace["trace" if split == "fit" else "heldout_trace"][step]
                            self.assertEqual(record["step"], step)
                            if split == "heldout" and not record["converged"]:
                                self.assertIsNone(record["errors"])
                                self.assertTrue(np.isnan(responses[step]).all())
                                self.assertEqual(row["heldout_max"], "")
                                continue
                            errors = response_error(responses[0], responses[step], manifest["error_floor"])
                            np.testing.assert_allclose(
                                errors, record["fit_errors" if split == "fit" else "errors"],
                                rtol=1e-10, atol=1e-12,
                            )
                            self.assertAlmostEqual(float(row[f"{split}_max"]), float(errors.max()))
                            self.assertAlmostEqual(float(row[f"{split}_mean"]), float(errors.mean()))
                            if split == "fit":
                                self.assertLessEqual(float(errors.max()), manifest["max_error"] + 1e-12)
                            if step == steps - 1:
                                self.assertAlmostEqual(summary[f"{split}_max"], float(errors.max()))
                                self.assertAlmostEqual(summary[f"{split}_mean"], float(errors.mean()))
        self.assertGreater((folder / "fidelity_curve.png").stat().st_size, 0)

    def test_pilot_scores_match_saved_output_vectors(self):
        self.check_exports(ROOT / "results/pruning_pilot")

    def test_cli_exports_and_refuses_to_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            edge_list = folder / "edges.csv"
            with edge_list.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Source", "Target", "Type", "Weight"])
                for neuron in SENSORY:
                    writer.writerow([neuron, "RELAY", "chemical", 1])
                writer.writerow(["RELAY", "DA01", "chemical", 1])
                writer.writerow(["UNUSED", "UNUSED", "chemical", 1])
            output = folder / "results"
            command = [sys.executable, str(ROOT / "run_pruning.py"),
                       "--edge-list", str(edge_list), "--output", str(output),
                       "--max-removals", "2", "--random-runs", "1"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=folder)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.check_exports(output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["source_data"]["matches_pinned_source"])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=folder)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Output directory is not empty", repeated.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
