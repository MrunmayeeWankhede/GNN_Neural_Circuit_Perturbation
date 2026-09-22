"""Train a downstream GCN on cumulative knockout errors in partially pruned graphs."""
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

from circuit_pruning import Circuit, ConvergenceError, prune, response_error
from connectome_model_v2 import (GABAERGIC, SENSORY, apply_signs,
                                 build_connectome_graph, graph_to_matrix,
                                 motor_neurons, normalize_weights)
from run_pruning import ROOT, git_revision, source_data, stimulus_suite


ERROR_BUDGET = 0.05
DEPTHS = (8, 16, 24)
SPLIT_SEEDS = {"train": list(range(100, 106)), "validation": [200, 201],
               "test": [300, 301]}


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_circuit(edge_list):
    path, provenance = source_data(edge_list)
    graph = build_connectome_graph(pd.read_csv(path))
    weights, nodes = graph_to_matrix(graph)
    weights = normalize_weights(apply_signs(weights, nodes))
    circuit = Circuit(weights, nodes, [i for i, n in enumerate(nodes) if n in SENSORY],
                      motor_neurons(nodes))
    fit, _, suite = stimulus_suite([nodes[i] for i in circuit.inputs])
    return circuit, fit, suite, provenance


def features(circuit, removed, current, intact):
    """Features use pre-candidate states; candidate knockout responses are labels only."""
    n = len(circuit.nodes)
    active = np.ones(n, dtype=bool)
    active[list(removed)] = False
    w = circuit.weights.toarray() * active[:, None] * active[None, :]
    is_input, is_output = np.zeros(n), np.zeros(n)
    is_input[circuit.inputs], is_output[circuit.outputs] = 1, 1
    flags = np.column_stack((active, is_input, is_output,
                             [name in GABAERGIC for name in circuit.nodes]))
    strengths = np.column_stack((np.abs(w).sum(1), np.abs(w).sum(0),
                                 (w != 0).sum(1) / n, (w != 0).sum(0) / n))
    x = np.column_stack((flags, strengths, current.T, intact.T)).astype(np.float32)
    # W[receiver, sender]: reverse anatomical edges to aggregate downstream targets.
    receiver, sender = np.nonzero(w)
    edge_index = np.stack((receiver, sender)).astype(np.int64)
    candidate = active.copy()
    candidate[list(circuit.protected)] = False
    return x, edge_index, candidate


def generate_dataset(circuit, fit, out, suite, provenance):
    if (out / "dataset.json").exists():
        raise ValueError("Dataset exists; use --stage train or choose another --output")
    started = time.perf_counter()
    intact = circuit.simulate(fit)
    reference = intact[:, circuit.outputs]
    records, seen = [], set()

    def snapshot(split, seed, removed):
        key = tuple(sorted(removed))
        if key in seen:
            return
        seen.add(key)
        current = intact if not removed else circuit.simulate(fit, removed)
        x, edges, mask = features(circuit, removed, current, intact)
        errors = np.full(len(circuit.nodes), np.nan)
        for i in np.flatnonzero(mask):
            try:
                responses = circuit.simulate(fit, list(removed) + [int(i)])[:, circuit.outputs]
                errors[i] = response_error(reference, responses).max()
            except ConvergenceError:
                pass
        name = f"state_{len(records):03d}.npz"
        np.savez_compressed(out / name, x=x, edge_index=edges, candidate=mask,
                            errors=errors, removed=np.array(removed, dtype=np.int64))
        records.append({"file": name, "split": split, "trajectory_seed": seed,
                        "removed_count": len(removed), "removed_names": [circuit.nodes[i] for i in removed],
                        "labels": int(np.isfinite(errors).sum()),
                        "nonconvergent_labels": int((mask & ~np.isfinite(errors)).sum())})
        print(f"  {split} seed={seed}: {len(removed)} removed, "
              f"{np.isfinite(errors).sum()} candidate labels", flush=True)

    snapshot("train", None, [])
    for split, seeds in SPLIT_SEEDS.items():
        for seed in seeds:
            path = prune(circuit, fit, method="random", max_error=ERROR_BUDGET,
                         max_removals=max(DEPTHS), seed=seed)
            for depth in DEPTHS:
                if depth <= len(path.removed):
                    snapshot(split, seed, path.removed[:depth])
    for split in SPLIT_SEEDS:
        if not any(record["split"] == split for record in records):
            raise ValueError(f"No states for {split}")
    metadata = {
        "schema_version": 1, "task": "worst cumulative fitting-input error after one further deletion",
        "target": "log1p(error / 0.05)", "split_unit": "whole seeded removal trajectory",
        "interpretation": "same-connectome state generalization; neurons recur across splits",
        "error_budget": ERROR_BUDGET, "depths": list(DEPTHS), "split_seeds": SPLIT_SEEDS,
        "nodes": list(circuit.nodes), "source_data": provenance, "stimuli": suite,
        "features": ["active", "input", "output", "GABAergic", "in_strength", "out_strength",
                     "in_degree_fraction", "out_degree_fraction"]
                    + [f"current_{s}" for s in suite["fit_names"]]
                    + [f"intact_{s}" for s in suite["fit_names"]],
        "records": records, "simulation_trajectories": circuit.trajectories,
        "seconds": time.perf_counter() - started, "source_code": git_revision(),
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("train_pruning_gnn.py", "circuit_pruning.py",
                                       "run_pruning.py", "connectome_model_v2.py")},
        "environment": {name: version(name) for name in
                        ("torch", "torch-geometric", "numpy", "scipy", "networkx", "pandas")},
        "state_hashes": {r["file"]: hashlib.sha256((out / r["file"]).read_bytes()).hexdigest()
                         for r in records},
    }
    save_json(out / "dataset.json", metadata)
    return metadata


class PruningGCN(nn.Module):
    def __init__(self, num_features, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + num_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = nn.functional.dropout(h, p=0.1, training=self.training)
        h = self.conv2(h, edge_index).relu()
        return self.head(torch.cat((x, h), dim=-1)).squeeze(-1)


def load_samples(out, metadata):
    samples = {split: [] for split in SPLIT_SEEDS}
    for record in metadata["records"]:
        path = out / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["state_hashes"][record["file"]]:
            raise ValueError(f"Dataset checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as saved:
            mask = saved["candidate"] & np.isfinite(saved["errors"])
            y = np.log1p(np.nan_to_num(saved["errors"], nan=0) / ERROR_BUDGET)
            data = Data(x=torch.from_numpy(saved["x"]), edge_index=torch.from_numpy(saved["edge_index"]),
                        y=torch.tensor(y, dtype=torch.float32), mask=torch.from_numpy(mask))
            samples[record["split"]].append(data)
    return samples


def prediction_metrics(model, batch):
    model.eval()
    with torch.no_grad():
        p = model(batch.x, batch.edge_index)[batch.mask].numpy()
    y = batch.y[batch.mask].numpy()
    residual = np.sum((y - p) ** 2)
    variance = np.sum((y - y.mean()) ** 2)
    return {"log_target_mse": float(np.mean((y - p) ** 2)),
            "log_target_r2": float(1 - residual / variance),
            "spearman": float(spearmanr(y, p).statistic), "candidate_labels": len(y)}, p, y


def train_model(out, metadata, epochs, seed):
    if (out / "model.pt").exists():
        raise ValueError("Model exists; choose another --output to retrain")
    torch.manual_seed(seed)
    samples = load_samples(out, metadata)
    # Fit preprocessing on training states only, including their unlabeled context nodes.
    train_x = torch.cat([data.x for data in samples["train"]])
    mean, scale = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    for split in samples.values():
        for data in split:
            data.x = (data.x - mean) / scale
    batches = {split: Batch.from_data_list(data) for split, data in samples.items()}
    model = PruningGCN(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        batch = batches["train"]
        prediction = model(batch.x, batch.edge_index)
        loss = nn.functional.mse_loss(prediction[batch.mask], batch.y[batch.mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = batches["validation"]
            validation_loss = nn.functional.mse_loss(model(val.x, val.edge_index)[val.mask], val.y[val.mask]).item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch % 25 == 0:
            print(f"epoch {epoch}: train={loss.item():.5f}, validation={validation_loss:.5f}", flush=True)
        if epoch - best_epoch >= 50:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "mean": mean, "scale": scale,
                "num_features": train_x.shape[1], "hidden": 32, "seed": seed,
                "best_epoch": best_epoch, "nodes": metadata["nodes"],
                "dataset_sha256": hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest()}, out / "model.pt")
    metrics = {"seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
               "seconds": time.perf_counter() - started, "splits": {}}
    predictions = {}
    for split, batch in batches.items():
        metrics["splits"][split], p, y = prediction_metrics(model, batch)
        predictions[f"{split}_prediction"], predictions[f"{split}_target"] = p, y
    np.savez_compressed(out / "predictions.npz", **predictions)
    save_json(out / "training_history.json", history)
    save_json(out / "training_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return model, mean, scale


def guided_prune(circuit, fit, method, budget, model=None, mean=None, scale=None):
    """Equal online simulation budgets include references and static ranking costs."""
    if budget < 1:
        raise ValueError("Simulation budget must be positive")
    start = circuit.trajectories
    intact = circuit.simulate(fit)
    current = intact
    reference = intact[:, circuit.outputs]
    removed, trace = [], []
    candidates = [i for i in range(len(circuit.nodes)) if i not in circuit.protected]

    def calls():
        return (circuit.trajectories - start) // len(fit)

    def evaluate(deletions):
        try:
            state = circuit.simulate(fit, deletions)
            error = response_error(reference, state[:, circuit.outputs])
            return state, error
        except ConvergenceError:
            return None, None

    if method == "static_severity":
        if budget < 1 + len(candidates):
            raise ValueError("Budget cannot cover static single-knockout ranking")
        scores = {}
        for i in candidates:
            _, errors = evaluate([i])
            scores[i] = float("inf") if errors is None else float(errors.max())
    elif method != "gnn":
        raise ValueError("Unknown guided pruning method")
    reason = "all_unprotected_removed"
    while candidates:
        if method == "gnn":
            x, edges, _ = features(circuit, removed, current, intact)
            model.eval()
            with torch.no_grad():
                scores = model((torch.from_numpy(x) - mean) / scale,
                               torch.from_numpy(edges)).numpy()
        order = sorted(candidates, key=lambda i: (float(scores[i]), circuit.nodes[i]))
        accepted = False
        for i in order:
            if calls() >= budget:
                reason = "simulation_budget"
                break
            state, errors = evaluate(removed + [i])
            if errors is not None and errors.max() <= ERROR_BUDGET:
                removed.append(i)
                candidates.remove(i)
                current = state
                trace.append({"step": len(removed), "neuron": circuit.nodes[i],
                              "fit_errors": errors.tolist(), "simulation_calls": calls()})
                accepted = True
                break
        if not accepted:
            if reason != "simulation_budget":
                reason = "no_feasible_single_removal"
            break
    return {"method": method, "removed": removed, "removed_names": [circuit.nodes[i] for i in removed],
            "removed_count": len(removed), "retained_count": len(circuit.nodes) - len(removed),
            "simulation_calls": calls(), "fitting_trajectories": circuit.trajectories - start,
            "stop_reason": reason, "trace": trace,
            "fit_max": float(response_error(reference, current[:, circuit.outputs]).max())}


def evaluate_model(circuit, fit, out, model, mean, scale, budget):
    if (out / "pruning_comparison.json").exists():
        raise ValueError("Evaluation exists; do not retune on the same test results")
    comparison = {}
    for method in ("static_severity", "gnn"):
        started = time.perf_counter()
        comparison[method] = guided_prune(circuit, fit, method, budget, model, mean, scale)
        comparison[method]["online_seconds"] = time.perf_counter() - started
        print(f"{method}: {comparison[method]['removed_count']} removed; "
              f"{comparison[method]['simulation_calls']} simulation calls", flush=True)
    # A prespecified new suite, accessed only after both deletion orders are frozen.
    rng = np.random.default_rng(20260922)
    heldout = np.vstack((rng.uniform(0, 1, (16, len(fit))) @ fit,
                         rng.uniform(0, 1, (16, len(circuit.inputs)))))
    reference = circuit.simulate(heldout)[:, circuit.outputs]
    arrays = {"stimuli": heldout, "intact": reference,
              "outputs": np.array([circuit.nodes[i] for i in circuit.outputs])}
    for method, result in comparison.items():
        try:
            response = circuit.simulate(heldout, result["removed"])[:, circuit.outputs]
            errors = response_error(reference, response)
            result.update(heldout_converged=True, heldout_max=float(errors.max()),
                          heldout_mean=float(errors.mean()), heldout_errors=errors.tolist(),
                          heldout_within_budget=int((errors <= ERROR_BUDGET).sum()))
        except ConvergenceError:
            response = np.full_like(reference, np.nan)
            result.update(heldout_converged=False, heldout_max=None, heldout_mean=None,
                          heldout_errors=None, heldout_within_budget=None)
        arrays[method] = response
    np.savez_compressed(out / "evaluation_responses.npz", **arrays)
    save_json(out / "pruning_comparison.json", {
        "online_simulation_budget": budget, "stimuli_per_call": len(fit),
        "heldout_seed": 20260922, "heldout_count": len(heldout), "methods": comparison,
        "cost_caveat": "GNN offline label generation and training are additional, not amortized here",
        "scope": "one worm graph, one training seed; no biological or cross-connectome generalization claim",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list")
    parser.add_argument("--output", default=str(ROOT / "results/pruning_gnn_pilot"))
    parser.add_argument("--stage", choices=("all", "generate", "train", "evaluate"), default="all")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulation-budget", type=int, default=550)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    circuit, fit, suite, provenance = load_circuit(args.edge_list)
    if args.stage in ("all", "generate"):
        metadata = generate_dataset(circuit, fit, out, suite, provenance)
    else:
        metadata = json.loads((out / "dataset.json").read_text())
        if provenance["sha256"] != metadata["source_data"]["sha256"]:
            raise ValueError("Connectome differs from the training dataset")
    if args.stage in ("all", "train"):
        model, mean, scale = train_model(out, metadata, args.epochs, args.seed)
    if args.stage == "evaluate":
        saved = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
        if saved["dataset_sha256"] != hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest():
            raise ValueError("Model and dataset manifests do not match")
        model = PruningGCN(saved["num_features"], saved["hidden"])
        model.load_state_dict(saved["state_dict"])
        mean, scale = saved["mean"], saved["scale"]
    if args.stage in ("all", "evaluate"):
        evaluate_model(circuit, fit, out, model, mean, scale, args.simulation_budget)


if __name__ == "__main__":
    main()
