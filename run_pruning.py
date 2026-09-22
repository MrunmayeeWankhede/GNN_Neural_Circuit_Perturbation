"""Run a reproducible first circuit-reduction experiment: python run_pruning.py."""
import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd
import networkx as nx

from circuit_pruning import Circuit, evaluate_trace, prune
from connectome_model_v2 import (
    SENSORY, apply_signs, build_connectome_graph, graph_to_matrix,
    motor_neurons, normalize_weights,
)

ROOT = Path(__file__).resolve().parent
DATA_COMMIT = "b36380a36d2a6dda0f03c946c433524b25ea2268"
DATA_URL = f"https://raw.githubusercontent.com/openworm/CElegansNeuroML/{DATA_COMMIT}/herm_full_edgelist.csv"
DATA_SHA256 = "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca"


def source_data(custom_path=None):
    path = Path(custom_path) if custom_path else ROOT / "data/pruning/herm_full_edgelist.csv"
    if not path.exists():
        if custom_path:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded connectome checksum does not match the pinned source")
        path.write_bytes(content)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if custom_path is None and sha != DATA_SHA256:
        raise ValueError("Cached connectome differs from pinned data; use --edge-list for custom data")
    return path, {"sha256": sha, "matches_pinned_source": sha == DATA_SHA256,
                  "path": str(path), "pinned_url": DATA_URL}


def stimulus_suite(input_names):
    """Synthetic input probes, not measured sensory responses or behaviors.

    Groups partition the present v2 inputs. All sensory neurons are clamped on
    every run; inactive groups are explicitly clamped to zero.
    """
    groups = {
        "anterior_touch": ["ALML", "ALMR", "AVM"],
        "posterior_touch": ["PLML", "PLMR"],
        "ASE_ASH": ["ASEL", "ASER", "ASHL", "ASHR"],
        "AWA_AWB_AWC": ["AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR"],
        "AFD": ["AFDL", "AFDR"],
        "oxygen_group": ["AQR", "PQR", "URXL", "URXR"],
        "ADL_ASJ_ASK": ["ADLL", "ADLR", "ASJL", "ASJR", "ASKL", "ASKR"],
    }
    names = list(input_names)
    probes = {label: np.array([float(n in group) for n in names])
              for label, group in groups.items()}
    omitted = [label for label, probe in probes.items() if not probe.any()]
    probes = {label: probe for label, probe in probes.items() if probe.any()}
    required = set(groups) - {"posterior_touch"}
    if not required.issubset(probes):
        raise ValueError("Default probes require the worm model's sensory groups")
    fit = np.array(list(probes.values()))
    if not np.all(fit.sum(axis=0) == 1):
        raise ValueError("Every modeled input must belong to exactly one probe group")
    heldout = np.array([
        0.6 * probes["anterior_touch"] + 0.4 * probes["ASE_ASH"],
        0.4 * probes["ADL_ASJ_ASK"] + 0.6 * probes["oxygen_group"],
        0.7 * probes["AWA_AWB_AWC"] + 0.3 * probes["AFD"],
        0.5 * probes["ASE_ASH"] + 0.5 * probes["ADL_ASJ_ASK"],
        np.full(len(names), 0.5),
        np.ones(len(names)),
    ])
    return fit, heldout, {
        "kind": "synthetic steady-state input probes; no behavioral ground truth",
        "input_names": names,
        "fit_names": list(probes), "fit_amplitudes": fit.tolist(),
        "omitted_empty_groups": omitted,
        "heldout_names": ["touch_chem_mix", "oxygen_misc_mix", "odor_AFD_mix",
                          "chem_misc_mix", "all_half", "all_one"],
        "heldout_amplitudes": heldout.tolist(),
    }


def git_revision():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def plot_curves(frame, max_error, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name, rows in frame.groupby("run", sort=False):
        for ax, key in zip(axes, ("fit_max", "heldout_max")):
            ax.plot(rows["removed_count"], 100 * rows[key], label=name, linewidth=1.8)
    for ax, title in zip(axes, ("Inputs used to select removals", "Held-out inputs: evaluated afterward")):
        ax.axhline(100 * max_error, color="#8b2525", linestyle="--", linewidth=1)
        ax.set(xlabel="Neurons removed (inputs and outputs protected)",
               ylabel="Worst input: relative L1 output error (%)", title=title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Circuit reduction in the C. elegans recurrent-map model", fontsize=13)
    fig.savefig(out / "fidelity_curve.png", dpi=160)
    plt.close(fig)


def run(args):
    if args.max_removals < 0 or args.random_runs < 1:
        raise ValueError("max-removals must be nonnegative and random-runs positive")
    if not np.isfinite(args.max_error) or args.max_error < 0:
        raise ValueError("max-error must be finite and nonnegative")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new --output")
    path, provenance = source_data(args.edge_list)
    graph = build_connectome_graph(pd.read_csv(path))
    weights, nodes = graph_to_matrix(graph)
    weights = normalize_weights(apply_signs(weights, nodes))
    inputs = [i for i, n in enumerate(nodes) if n in SENSORY]
    outputs = motor_neurons(nodes)
    reaches_output = {nodes[i] for i in outputs}
    for i in outputs:
        reaches_output.update(nx.ancestors(graph, nodes[i]))
    circuit = Circuit(weights, nodes, inputs, outputs, gain=args.gain,
                      max_iterations=args.max_iterations)
    fit, heldout, suite = stimulus_suite([nodes[i] for i in inputs])
    # Abort before expensive search if the held-out intact reference is invalid.
    # This is only a convergence check; responses are not supplied to prune().
    circuit.simulate(heldout)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "model": "v2 signed tanh recurrent map; chemical synapses only",
        "interpretation": "fidelity to this model; not validated behavior or a proven global minimum",
        "gain": args.gain, "max_error": args.max_error, "error_floor": 1e-6,
        "error_metric": "maximum across stimuli of relative L1 error on the fixed motor readout",
        "convergence_tolerance": circuit.tolerance, "max_iterations": args.max_iterations,
        "max_removals": args.max_removals, "seed": args.seed, "random_runs": args.random_runs,
        "node_count": len(nodes), "directed_edge_count": graph.number_of_edges(),
        "input_count": len(inputs), "output_count": len(outputs),
        "protected_count": len(circuit.protected),
        "unprotected_count": len(nodes) - len(circuit.protected),
        "unprotected_without_output_path": [n for i, n in enumerate(nodes)
                                              if i not in circuit.protected and n not in reaches_output],
        "missing_v2_sensory_names": sorted(set(SENSORY) - set(nodes)),
        "input_names": [nodes[i] for i in inputs], "output_names": [nodes[i] for i in outputs],
        "source_data": provenance, "source_code": git_revision(),
        "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                        for p in ("circuit_pruning.py", "run_pruning.py", "connectome_model_v2.py")},
        "environment": {"python": platform.python_version(),
                        **{p: version(p) for p in ("numpy", "scipy", "pandas", "networkx", "matplotlib")}},
        "stimuli": suite,
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(nodes)} neurons; protecting {len(circuit.protected)} inputs/outputs; "
          f"{len(fit)} fitting and {len(heldout)} held-out input patterns", flush=True)
    runs = [(m, args.seed) for m in ("greedy", "static_severity", "out_strength")]
    runs += [("random", args.seed + i) for i in range(args.random_runs)]
    traces, summaries, curves = {}, {}, []
    for method, seed in runs:
        label = f"random_{seed}" if method == "random" else method
        print(f"Running {label}", flush=True)
        started = time.perf_counter()

        def progress(row):
            if row["step"] % 10 == 0:
                print(f"  {row['step']} removed; fit error {max(row['fit_errors']):.4%}", flush=True)

        result = prune(circuit, fit, method, args.max_error, args.max_removals, seed, progress)
        search_seconds = time.perf_counter() - started
        search_end = circuit.trajectories
        test_trace, test_responses = evaluate_trace(circuit, heldout, result.removed)
        heldout_trajectories = circuit.trajectories - search_end
        # Save output vectors, not only scalar scores, so fidelity can be audited.
        _, fit_responses = evaluate_trace(circuit, fit, result.removed)
        np.savez_compressed(out / f"{label}_responses.npz", fit=fit_responses, heldout=test_responses,
                            outputs=np.array([nodes[i] for i in outputs]))
        payload = asdict(result)
        payload["removed_names"] = [nodes[i] for i in result.removed]
        payload["retained_names"] = [n for i, n in enumerate(nodes) if i not in result.removed]
        payload["heldout_trace"] = test_trace
        traces[label] = payload
        for fit_row, test_row in zip(result.trace, test_trace):
            test_error = test_row["errors"]
            curves.append({
                "run": label, "removed_count": fit_row["step"],
                "retained_count": len(nodes) - fit_row["step"],
                "fit_max": max(fit_row["fit_errors"]), "fit_mean": float(np.mean(fit_row["fit_errors"])),
                "heldout_max": max(test_error) if test_error is not None else None,
                "heldout_mean": float(np.mean(test_error)) if test_error is not None else None,
                "heldout_converged": test_row["converged"],
            })
        final = curves[-1]
        summaries[label] = {
            **final, "stop_reason": result.stop_reason,
            "search_trajectories": result.search_trajectories,
            "search_neuron_vector_updates": result.search_updates,
            "heldout_trajectories": heldout_trajectories,
            "fit_export_trajectories": circuit.trajectories - search_end - heldout_trajectories,
            "search_seconds": search_seconds,
            "rejected_nonconvergent_candidates": result.rejected_nonconvergent,
            "removed_without_output_path": sum(nodes[i] not in reaches_output for i in result.removed),
            "removed_with_output_path": sum(nodes[i] in reaches_output for i in result.removed),
        }
        # Checkpoint completed methods before starting the next one.
        (out / "traces.json").write_text(json.dumps(traces, indent=2, allow_nan=False) + "\n")
        (out / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
        with (out / "curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
        test_text = f"{final['heldout_max']:.3%}" if final["heldout_converged"] else "NONCONVERGENT"
        print(f"  retained {final['retained_count']}; fit {final['fit_max']:.3%}; "
              f"held-out {test_text}; {result.stop_reason}", flush=True)
    plot_curves(pd.DataFrame(curves), args.max_error, out)
    print(f"Saved manifest, curves, traces, output vectors and plot to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list", help="Local OpenWorm-format CSV; otherwise use a pinned download")
    parser.add_argument("--output", default=str(ROOT / "results/pruning_run"))
    parser.add_argument("--max-error", type=float, default=0.05, help="Worst fitting-input relative L1 error budget")
    parser.add_argument("--max-removals", type=int, default=50, help="Search cap, not a claim of minimality")
    parser.add_argument("--random-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=2000)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Pruning experiment failed: {error}\n")


if __name__ == "__main__":
    main()
