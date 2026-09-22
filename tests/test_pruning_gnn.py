"""GNN-specific scientific invariants; optional training dependencies required."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from circuit_pruning import Circuit, response_error

HAS_TORCH = (importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("torch_geometric") is not None)
if HAS_TORCH:
    import torch
    from torch_geometric.data import Batch
    from train_pruning_gnn import PruningGCN, features, guided_prune, load_samples


@unittest.skipUnless(HAS_TORCH, "Install requirements-pruning-gnn.txt for GNN tests")
class GNNPruningTests(unittest.TestCase):
    def circuit(self):
        w = np.zeros((5, 5))
        w[1, 0] = w[2, 1] = 1
        w[4, 3] = 1
        return Circuit(w, ["input", "essential", "output", "unused_a", "unused_b"], [0], [2])

    def test_messages_flow_downstream_to_sender_and_deleted_edges_are_absent(self):
        circuit = self.circuit()
        intact = circuit.simulate([[1]])
        current = circuit.simulate([[1]], [3])
        _, edges, candidates = features(circuit, [3], current, intact)
        self.assertEqual(set(map(tuple, edges.T)), {(1, 0), (2, 1)})
        np.testing.assert_array_equal(candidates, [False, True, False, False, True])

    def test_wrong_gnn_rankings_cannot_bypass_simulator_or_budget(self):
        class BadRanker(torch.nn.Module):
            def forward(self, x, edge_index):
                return torch.arange(len(x), dtype=torch.float32)

        circuit = self.circuit()
        result = guided_prune(circuit, np.array([[1.0]]), "gnn", budget=4,
                              model=BadRanker(), mean=torch.zeros(10), scale=torch.ones(10))
        self.assertEqual(result["simulation_calls"], 4)
        self.assertEqual(result["stop_reason"], "simulation_budget")
        self.assertNotIn(1, result["removed"])
        self.assertFalse(circuit.protected & set(result["removed"]))
        self.assertEqual(result["fit_max"], 0)

    def test_static_ranking_cost_counts_against_budget(self):
        circuit = self.circuit()
        result = guided_prune(circuit, np.array([[1.0]]), "static_severity", budget=4)
        self.assertEqual(result["simulation_calls"], 4)
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["stop_reason"], "simulation_budget")


PILOT = Path(__file__).resolve().parents[1] / "results/pruning_gnn_pilot"


@unittest.skipUnless(HAS_TORCH and (PILOT / "pruning_comparison.json").exists(),
                     "Requires optional GNN dependencies and the completed pilot")
class SavedGNNTests(unittest.TestCase):
    def test_removal_sets_and_trajectory_seeds_do_not_cross_splits(self):
        metadata = json.loads((PILOT / "dataset.json").read_text())
        states, trajectories = {}, {}
        for record in metadata["records"]:
            key = tuple(sorted(record["removed_names"]))
            self.assertNotIn(key, states)
            states[key] = record["split"]
            seed = record["trajectory_seed"]
            if seed is not None:
                self.assertEqual(trajectories.setdefault(seed, record["split"]), record["split"])

    def test_saved_model_reproduces_test_predictions_with_training_only_scaling(self):
        metadata = json.loads((PILOT / "dataset.json").read_text())
        samples = load_samples(PILOT, metadata)
        saved = torch.load(PILOT / "model.pt", map_location="cpu", weights_only=True)
        train_x = torch.cat([data.x for data in samples["train"]])
        torch.testing.assert_close(saved["mean"], train_x.mean(0))
        torch.testing.assert_close(saved["scale"], train_x.std(0).clamp_min(1e-6))
        for data in samples["test"]:
            data.x = (data.x - saved["mean"]) / saved["scale"]
        batch = Batch.from_data_list(samples["test"])
        model = PruningGCN(saved["num_features"], saved["hidden"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            predicted = model(batch.x, batch.edge_index)[batch.mask].numpy()
        with np.load(PILOT / "predictions.npz", allow_pickle=False) as vectors:
            np.testing.assert_allclose(predicted, vectors["test_prediction"], rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(batch.y[batch.mask].numpy(), vectors["test_target"])

    def test_pruning_metrics_match_outputs_and_obey_online_budget(self):
        report = json.loads((PILOT / "pruning_comparison.json").read_text())
        with np.load(PILOT / "evaluation_responses.npz", allow_pickle=False) as vectors:
            for method, result in report["methods"].items():
                with self.subTest(method=method):
                    self.assertLessEqual(result["simulation_calls"], report["online_simulation_budget"])
                    self.assertEqual(result["fitting_trajectories"], result["simulation_calls"] * report["stimuli_per_call"])
                    self.assertEqual(result["removed_count"], len(set(result["removed"])))
                    for row in result["trace"]:
                        self.assertLessEqual(max(row["fit_errors"]), 0.05)
                    self.assertTrue(result["heldout_converged"])
                    errors = response_error(vectors["intact"], vectors[method])
                    np.testing.assert_allclose(errors, result["heldout_errors"])
                    self.assertAlmostEqual(float(errors.max()), result["heldout_max"])
                    self.assertAlmostEqual(float(errors.mean()), result["heldout_mean"])
                    self.assertEqual(int((errors <= 0.05).sum()), result["heldout_within_budget"])


if __name__ == "__main__":
    unittest.main()
