"""Scientific invariants for pruning; no downloads or trained models needed."""
import unittest

import numpy as np

from circuit_pruning import Circuit, ConvergenceError, evaluate_trace, prune, response_error
from connectome_model_v2 import run_dynamics
from run_pruning import stimulus_suite


class PruningTests(unittest.TestCase):
    def redundant_circuit(self):
        # Input -> two parallel relays -> protected output.
        w = np.zeros((4, 4))
        w[1, 0] = w[2, 0] = 1
        w[3, 1] = w[3, 2] = 0.5
        return Circuit(w, ["input", "relay_a", "relay_b", "output"], [0], [3])

    def test_clamp_every_input_including_inactive_ones(self):
        w = np.zeros((3, 3))
        w[1, 0] = w[2, 1] = 1
        circuit = Circuit(w, ["input_a", "input_b", "output"], [0, 1], [2])
        states = circuit.simulate([[1, 0], [0, 1]])
        self.assertEqual(states[0, 1], 0)
        self.assertEqual(states[0, 2], 0)
        self.assertAlmostEqual(states[1, 2], np.tanh(1))

    def test_removal_preserves_surviving_weights(self):
        circuit = self.redundant_circuit()
        before = circuit.weights.toarray().copy()
        x = circuit.simulate([[1]], removed=[1])
        self.assertEqual(x[0, 1], 0)
        self.assertAlmostEqual(x[0, 3], np.tanh(0.5 * np.tanh(1)))
        np.testing.assert_array_equal(before, circuit.weights.toarray())

    def test_protected_neurons_cannot_be_removed(self):
        circuit = self.redundant_circuit()
        for protected in ([0], [3], [1, 3]):
            with self.assertRaises(ValueError):
                circuit.simulate([[1]], protected)

    def test_two_individually_feasible_deletions_are_not_jointly_feasible(self):
        circuit = self.redundant_circuit()
        baseline = circuit.simulate([[1]])[:, circuit.outputs]
        for node in [1, 2]:
            error = response_error(baseline, circuit.simulate([[1]], [node])[:, circuit.outputs])
            self.assertLess(error[0], 0.5)
        for method in ("greedy", "static_severity", "out_strength", "random"):
            result = prune(circuit, [[1]], method=method, max_error=0.5, max_removals=2)
            self.assertEqual(len(result.removed), 1)
            self.assertEqual(result.stop_reason, "no_feasible_single_removal")

    def test_error_budget_applies_to_every_stimulus(self):
        # Relay is irrelevant when its input is off, but essential when on.
        w = np.zeros((4, 4))
        w[2, 1] = w[3, 2] = 1
        circuit = Circuit(w, ["unused_input", "input", "relay", "output"], [0, 1], [3])
        result = prune(circuit, [[1, 0], [0, 1]], max_error=0.6, max_removals=1)
        self.assertEqual(result.removed, [])  # Mean error would incorrectly accept 0.5.

    def test_different_heldout_inputs_can_reveal_failure(self):
        w = np.zeros((4, 4))
        w[2, 1] = w[3, 2] = 1
        circuit = Circuit(w, ["unused_input", "input", "relay", "output"], [0, 1], [3])
        result = prune(circuit, [[1, 0]], max_error=0.0, max_removals=1)
        self.assertEqual(result.removed, [2])
        frozen_order = result.removed.copy()
        report, _ = evaluate_trace(circuit, [[0, 1]], result.removed)
        self.assertAlmostEqual(report[-1]["errors"][0], 1.0)
        self.assertEqual(result.removed, frozen_order)

    def test_nonconvergence_is_not_silently_scored(self):
        circuit = self.redundant_circuit()
        circuit.max_iterations = 1
        with self.assertRaises(ConvergenceError):
            circuit.simulate([[1]])
        with self.assertRaises(ConvergenceError):
            prune(circuit, [[1]])

    def test_heldout_nonconvergence_is_explicit(self):
        circuit = self.redundant_circuit()
        original = circuit.simulate

        def reject_candidate(stimuli, removed=()):
            if removed:
                raise ConvergenceError("Deliberate candidate convergence failure")
            return original(stimuli, removed)

        circuit.simulate = reject_candidate
        report, responses = evaluate_trace(circuit, [[1]], [1])
        self.assertFalse(report[-1]["converged"])
        self.assertIsNone(report[-1]["errors"])
        self.assertTrue(np.isnan(responses[-1]).all())
        result = prune(circuit, [[1]], max_removals=1)
        self.assertEqual(result.removed, [])
        self.assertEqual(result.rejected_nonconvergent, 2)

    def test_matches_existing_v2_all_on_dynamics(self):
        circuit = self.redundant_circuit()
        for removed in ([], [1], [1, 2]):
            expected, converged = run_dynamics(circuit.weights.toarray(), [0],
                                               knockout_idx=removed or None)
            self.assertIsNotNone(converged)
            np.testing.assert_allclose(circuit.simulate([[1]], removed)[0], expected, atol=1e-9)

    def test_random_seed_reproduces_order(self):
        a = prune(self.redundant_circuit(), [[1]], method="random", max_error=1, seed=123)
        b = prune(self.redundant_circuit(), [[1]], method="random", max_error=1, seed=123)
        self.assertEqual(a.removed, b.removed)

    def test_zero_reference_has_explicit_absolute_scale(self):
        np.testing.assert_allclose(response_error([[0, 0]], [[1e-6, 1e-6]]), [1])

    def test_absent_posterior_group_is_omitted_not_a_silent_probe(self):
        from connectome_model_v2 import SENSORY
        present = [n for n in SENSORY if n not in {"PLML", "PLMR"}]
        fit, heldout, metadata = stimulus_suite(present)
        self.assertEqual(metadata["omitted_empty_groups"], ["posterior_touch"])
        self.assertEqual(fit.shape, (6, len(present)))
        self.assertTrue(np.all(fit.sum(axis=1) > 0))
        np.testing.assert_array_equal(fit.sum(axis=0), np.ones(len(present)))
        for stimulus in heldout:
            self.assertFalse(any(np.array_equal(stimulus, row) for row in fit))


if __name__ == "__main__":
    unittest.main()
