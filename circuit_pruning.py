"""Circuit reduction under the v2 smooth recurrent-map assumptions.

Weights use W[receiver, sender]. Inputs and outputs are never removed; surviving
weights are never renormalized. This is model fidelity, not biological function.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


class ConvergenceError(RuntimeError):
    """A candidate cannot be scored as a steady state."""


@dataclass
class PruningResult:
    method: str
    removed: list
    trace: list
    stop_reason: str
    search_trajectories: int
    search_updates: int
    rejected_nonconvergent: int


class Circuit:
    def __init__(self, weights, nodes, inputs, outputs, gain=1.0,
                 tolerance=1e-9, max_iterations=2000):
        self.weights = csr_matrix(weights, dtype=float, copy=True)
        self.nodes = tuple(nodes)
        n = len(self.nodes)
        if n == 0 or len(set(nodes)) != n or self.weights.shape != (n, n):
            raise ValueError("Need unique nodes and a matching square weight matrix")
        if not np.isfinite(self.weights.data).all():
            raise ValueError("Weights must be finite")
        for label, indices in (("inputs", inputs), ("outputs", outputs)):
            if not len(indices) or any(i < 0 or i >= n for i in indices):
                raise ValueError(f"{label} must contain valid node indices")
            if len(set(indices)) != len(indices):
                raise ValueError(f"Duplicate {label}")
        self.inputs = np.array(inputs, dtype=int)
        self.outputs = np.array(outputs, dtype=int)
        self.protected = set(inputs) | set(outputs)
        if not np.isfinite(gain) or gain <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("gain and tolerance must be positive and finite")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.gain, self.tolerance = gain, tolerance
        self.max_iterations = max_iterations
        self.trajectories = 0
        self.updates = 0

    def simulate(self, stimuli, removed=()):
        """Return [stimulus, node] states; clamp ALL inputs, including zeros.

        Each stimulus is [input] with amplitudes in [0, 1]. Independent trajectories
        stop only after their own L1 change is below tolerance. Nonconvergence is
        an error, never an acceptable low-fidelity circuit.
        """
        stimuli = np.asarray(stimuli, dtype=float)
        if (stimuli.ndim != 2 or stimuli.shape[1] != len(self.inputs)
                or not len(stimuli) or not np.isfinite(stimuli).all()
                or np.any((stimuli < 0) | (stimuli > 1))):
            raise ValueError("Stimuli must be a nonempty [stimulus, input] array in [0, 1]")
        removed = sorted(set(removed))
        if any(i < 0 or i >= len(self.nodes) for i in removed):
            raise ValueError("Invalid removed neuron index")
        if self.protected.intersection(removed):
            raise ValueError("Input and output neurons are protected")
        x = np.zeros((len(self.nodes), len(stimuli)))
        x[self.inputs, :] = stimuli.T
        active = np.ones(len(stimuli), dtype=bool)
        self.trajectories += len(stimuli)
        for _ in range(self.max_iterations):
            columns = np.flatnonzero(active)
            new = np.tanh(self.gain * (self.weights @ x[:, columns]))
            new[self.inputs, :] = stimuli[columns].T
            new[removed, :] = 0
            delta = np.abs(new - x[:, columns]).sum(axis=0)
            if not np.isfinite(new).all():
                raise ConvergenceError("Nonfinite state")
            x[:, columns] = new
            self.updates += len(columns)
            active[columns[delta < self.tolerance]] = False
            if not active.any():
                return x.T
        raise ConvergenceError(f"{active.sum()} stimuli did not converge")


def response_error(reference, response, floor=1e-6):
    """Per-stimulus relative L1 error over the SAME fixed output population.

    Denominator is max(sum(abs(reference)), n_outputs * floor), so an almost
    silent reference has an explicit absolute error scale. No output is dropped.
    """
    reference, response = np.asarray(reference), np.asarray(response)
    if reference.ndim != 2 or reference.shape != response.shape or reference.shape[1] == 0:
        raise ValueError("Expected matching [stimulus, output] arrays")
    if not np.isfinite(reference).all() or not np.isfinite(response).all():
        raise ValueError("Responses must be finite")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Error floor must be positive and finite")
    scale = np.maximum(np.abs(reference).sum(axis=1), reference.shape[1] * floor)
    return np.abs(response - reference).sum(axis=1) / scale


def prune(circuit, fit_stimuli, method="greedy", max_error=0.05,
          max_removals=50, seed=42, progress=None):
    """Search using fitting stimuli ONLY. Held-out inputs are evaluated later.

    Greedy evaluates every remaining candidate at every step. Baselines keep a
    fixed priority order and accept the first feasible candidate. Failed candidates
    are retried on the next step because errors need not be monotone under removal.
    Static severity ranks single deletions from the intact graph. Low out-strength
    uses absolute outgoing weights in the original normalized dynamical matrix.
    """
    if method not in {"greedy", "static_severity", "out_strength", "random"}:
        raise ValueError("Unknown method")
    if not np.isfinite(max_error) or max_error < 0 or max_removals < 0:
        raise ValueError("Invalid error tolerance or removal budget")
    start_count, start_updates = circuit.trajectories, circuit.updates
    reference = circuit.simulate(fit_stimuli)[:, circuit.outputs]
    candidates = [i for i in range(len(circuit.nodes)) if i not in circuit.protected]
    removed, trace = [], [{"step": 0, "removed_neuron": None,
                           "fit_errors": [0.0] * len(fit_stimuli)}]
    rejected = 0

    def evaluate(deletions):
        nonlocal rejected
        try:
            return response_error(reference, circuit.simulate(fit_stimuli, deletions)[:, circuit.outputs])
        except ConvergenceError:
            rejected += 1
            return None

    if method == "static_severity":
        scores = {}
        for i in candidates:
            error = evaluate([i])
            scores[i] = np.inf if error is None else float(error.max())
        candidates.sort(key=lambda i: (scores[i], circuit.nodes[i]))
    elif method == "out_strength":
        strength = np.asarray(abs(circuit.weights).sum(axis=0)).ravel()
        candidates.sort(key=lambda i: (strength[i], circuit.nodes[i]))
    elif method == "random":
        np.random.default_rng(seed).shuffle(candidates)

    stop_reason = "removal_budget"
    while len(removed) < max_removals:
        if not candidates:
            stop_reason = "all_unprotected_removed"
            break
        best, best_error, best_key = None, None, None
        for i in candidates:
            error = evaluate(removed + [i])
            if error is None or error.max() > max_error:
                continue
            key = (float(error.max()), float(error.mean()), circuit.nodes[i])
            if best is None or key < best_key:
                best, best_error, best_key = i, error, key
            if method != "greedy":
                break
        if best is None:
            stop_reason = "no_feasible_single_removal"
            break
        removed.append(best)
        candidates.remove(best)
        trace.append({"step": len(removed), "removed_neuron": circuit.nodes[best],
                      "fit_errors": best_error.tolist()})
        if progress is not None:
            progress(trace[-1])
    return PruningResult(method, removed, trace, stop_reason,
                         circuit.trajectories - start_count,
                         circuit.updates - start_updates, rejected)


def evaluate_trace(circuit, stimuli, removed):
    """Evaluate a frozen deletion order; never feed these results into search.

    Nonconvergent held-out circuits get null errors and NaN response arrays, and
    an explicit flag. The intact reference itself must converge.
    """
    reference = circuit.simulate(stimuli)[:, circuit.outputs]
    rows, responses = [], []
    for step in range(len(removed) + 1):
        try:
            response = reference if step == 0 else circuit.simulate(stimuli, removed[:step])[:, circuit.outputs]
            errors = response_error(reference, response).tolist()
            converged = True
        except ConvergenceError:
            response = np.full_like(reference, np.nan)
            errors, converged = None, False
        rows.append({"step": step, "errors": errors, "converged": converged})
        responses.append(response)
    return rows, np.array(responses)
