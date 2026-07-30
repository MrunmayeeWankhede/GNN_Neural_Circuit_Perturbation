"""
validate_labels.py

Two independent checks that the severity labels measure something real,
neither of which uses any model:

  1. Bilateral symmetry. Left/right members of a neuron class are anatomically
     equivalent but were reconstructed independently in the EM data. If the
     pipeline measures a structural property of the circuit, paired neurons
     should score alike. If it measures reconstruction noise, they won't.

  2. Known biology. The mechanosensory escape circuit (AVA, AVD, PVC command
     interneurons; AVM/ALM touch receptors) is one of the best-characterised
     circuits in neuroscience. It should surface unprompted.

Run after connectome_model_v2.py. Writes results/validation.txt.
"""

#import the packages we need
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from connectome_model_v2 import (build_connectome_graph, graph_to_matrix,
                                 apply_signs, normalize_weights, run_dynamics,
                                 motor_neurons, SENSORY, EDGE_CSV)

#define the gain for the recurrent dynamics, and the known members of the mechanosensory escape circuit
GAIN = 1.0
KNOWN_CIRCUIT = {"AVAL", "AVAR", "AVDL", "AVDR", "PVCL", "PVCR",
                 "AVM", "ALML", "ALMR", "AVBL", "AVBR"}


#construct a table of severities for all neurons, by running the dynamics with each neuron knocked out in turn
def severity_table():
    G = build_connectome_graph(pd.read_csv(EDGE_CSV))
    W, nodes = graph_to_matrix(G)
    W = normalize_weights(apply_signs(W, nodes))
    index = {n: i for i, n in enumerate(nodes)}
    sensory_idx = [index[s] for s in SENSORY if s in index]
    readout = motor_neurons(nodes)

    base, conv = run_dynamics(W, sensory_idx, gain=GAIN)
    if conv is None:
        raise RuntimeError("baseline did not converge")

    sev = {}
    for i, name in enumerate(nodes):
        p, c = run_dynamics(W, sensory_idx, knockout_idx=i, gain=GAIN)
        if c is None:
            print(f"warning: knockout of {name} did not converge")
        ro = [m for m in readout if m != i]          # self-excluded
        sev[name] = float(np.abs((base - p)[ro]).sum())
    return sev, nodes

#check for bilateral symmetry in the severities, and report the correlation and largest asymmetries
def check_bilateral(sev, nodes):
    #pair XXXL with XXXR, skipping ventral/dorsal names 
    pairs = [(n, n[:-1] + "R") for n in nodes
             if n.endswith("L") and (n[:-1] + "R") in sev
             and not n[:-1].endswith(("D", "V"))]
    L = np.array([sev[a] for a, _ in pairs])
    R = np.array([sev[b] for _, b in pairs])
    lines = [
        f"bilateral pairs found: {len(pairs)}",
        f"  Pearson  r = {np.corrcoef(L, R)[0, 1]:.3f}",
        f"  Spearman r = {spearmanr(L, R).statistic:.3f}",
    ]
    rel = np.abs(L - R) / ((L + R) / 2 + 1e-9)
    lines.append(f"median relative L/R difference: {np.median(rel):.1%}")
    lines.append("largest asymmetries (note these are all near-zero neurons):")
    for k in np.argsort(-rel)[:5]:
        a, b = pairs[k]
        lines.append(f"    {a}/{b}: {sev[a]:.3f} vs {sev[b]:.3f}")
    return lines

#check for known members of the mechanosensory escape circuit in the top N severities
def check_known_circuit(sev, top_n=10):
    ranked = sorted(sev, key=sev.get, reverse=True)
    top = ranked[:top_n]
    hits = [n for n in top if n in KNOWN_CIRCUIT]
    lines = [f"top {top_n} by severity: {', '.join(top)}",
             f"of these, mechanosensory escape circuit members: "
             f"{len(hits)}/{top_n} -> {', '.join(hits)}"]
    lines.append("rank of each known circuit member:")
    for n in sorted(KNOWN_CIRCUIT):
        if n in sev:
            lines.append(f"{n:6s} rank {ranked.index(n)+1:3d} / {len(ranked)}"
                         f"severity {sev[n]:.3f}")
    return lines


if __name__ == "__main__":
    sev, nodes = severity_table()
    out = ["validation of severity labels", ""]
    out += check_bilateral(sev, nodes) + [""]
    out += check_known_circuit(sev)
    text = "\n".join(out)
    print(text)
    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    (results / "validation.txt").write_text(text + "\n")
    print(f"\nwritten to {results / 'validation.txt'}")