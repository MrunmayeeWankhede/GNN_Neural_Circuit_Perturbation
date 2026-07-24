"""
connectome_model.py  (v2)

Changes from v1, and why:
  1. signed edges      - GABAergic neurons get negative outgoing weight, so the
                         system is no longer monotone and knockouts can INCREASE
                         downstream activity (disinhibition).
  2. |w| row-normalize - mixed-sign rows must be normalized by sum of absolute
                         value, or rows with cancelling inputs blow up.
  3. no clip           - clip(drop, 0, None) silently deletes every disinhibition
                         effect. It was a no-op in v1 (the model was monotone) but
                         would destroy the signal now.
  4. readout set       - severity is measured at a defined OUTPUT population
                         (motor neurons), not summed over the whole network.
                         This is the change that makes the task non-trivial.
  5. gain parameter    - lets you push tanh out of its near-linear regime.
  6. honest name       - this is a smooth recurrent map, not linear-threshold
                         dynamics.
"""
import re
import numpy as np
import pandas as pd
import networkx as nx
import urllib.request
from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data"
data_path.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/herm_full_edgelist.csv"
EDGE_CSV = data_path / "herm_full_edgelist.csv"


#building graph and matrix from csv
def is_non_neuronal(cell_name):
    n = str(cell_name).strip()
    if re.fullmatch(r'M[DV][LR]\d+', n):
        return True
    if "BWM" in n.upper():
        return True
    if any(c.islower() for c in n):
        return True
    return False


def build_connectome_graph(df):
    df = df.copy()
    for c in ("Source", "Target", "Type"):
        df[c] = df[c].astype(str).str.strip()
    sub = df[df["Type"] == "chemical"]
    mask = ~sub["Source"].apply(is_non_neuronal) & ~sub["Target"].apply(is_non_neuronal)
    sub = sub[mask]
    G = nx.from_pandas_edgelist(sub, "Source", "Target", edge_attr="Weight",
                                create_using=nx.DiGraph())
    for u, v, d in G.edges(data=True):
        d["weight"] = d.pop("Weight")
    return G


def graph_to_matrix(G):
    """W[i, j] = weight of synapse from j (sender) to i (receiver)."""
    nodes = sorted(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))
    for u, v, d in G.edges(data=True):
        W[index[v], index[u]] = d["weight"]
    return W, nodes


# ------------------------------------------------------------ neuron annotation
# Hermaphrodite GABAergic neurons (McIntire et al. 1993; consistent with CeNGEN).
# NOTE the zero-padded names - the OpenWorm edge list uses DD01 not DD1.
GABAERGIC = (["DD%02d" % i for i in range(1, 7)]
             + ["VD%02d" % i for i in range(1, 14)]
             + ["AVL", "DVB", "RIS", "RMED", "RMEV", "RMEL", "RMER"])

SENSORY = ["ALML", "ALMR", "AVM", "PLML", "PLMR", "ASEL", "ASER", "ASHL", "ASHR",
           "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR", "AFDL", "AFDR",
           "AQR", "PQR", "URXL", "URXR", "ADLL", "ADLR", "ASJL", "ASJR",
           "ASKL", "ASKR"]


def motor_neurons(nodes):
    """Ventral-cord and head motor neurons - the output population."""
    vc = ("VA", "VB", "DA", "DB", "AS", "VC", "DD", "VD")
    head = ("RMD", "RME", "SMD", "SMB")
    out = []
    for i, n in enumerate(nodes):
        if n[:2] in vc and n[2:].isdigit():
            out.append(i)
        elif n[:3] in head:
            out.append(i)
    return sorted(set(out))


def apply_signs(W, nodes, inhibitory=GABAERGIC):
    """Neuron j's outgoing column carries its transmitter sign."""
    index = {n: i for i, n in enumerate(nodes)}
    sign = np.ones(len(nodes))
    found = 0
    for name in inhibitory:
        if name in index:
            sign[index[name]] = -1.0
            found += 1
    missing = [n for n in inhibitory if n not in index]
    if missing:
        print(f"  warning: {len(missing)} inhibitory neurons not in graph: {missing}")
    print(f"  applied negative sign to {found}/{len(inhibitory)} inhibitory neurons")
    return W * sign[None, :]


def normalize_weights(W):
    """Row-normalize by sum of ABSOLUTE weight (required once signs exist)."""
    rs = np.abs(W).sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


# ---------------------------------------------------------------- dynamics
def run_dynamics(W, sensory_idx, knockout_idx=None, gain=1.0,
                 max_iter=2000, tol=1e-9, verbose=False):
    """
    Smooth recurrent map: x <- tanh(gain * W x), with sensory neurons clamped
    to 1 and knocked-out neurons clamped to 0. Returns (x, iters_to_converge).
    iters is None if it never converged - check this, do not assume.
    """
    n = W.shape[0]
    x = np.zeros(n)
    x[sensory_idx] = 1.0
    for i in range(max_iter):
        x_new = np.tanh(gain * (W @ x))
        x_new[sensory_idx] = 1.0
        if knockout_idx is not None:
            x_new[knockout_idx] = 0.0
        delta = np.abs(x_new - x).sum()
        x = x_new
        if delta < tol:
            return x, i
    if verbose:
        print(f"  WARNING: did not converge, final delta={delta:.2e}")
    return x, None


def cascade_severity(baseline, perturbed, readout_idx, fail_threshold=0.1):
    """
    Severity measured ONLY at the readout population, and as absolute
    deviation so that disinhibition (activity going UP) counts.
    """
    drop = (baseline - perturbed)[readout_idx]
    b, p = baseline[readout_idx], perturbed[readout_idx]
    return {
        "severity":       float(np.abs(drop).sum()),   # magnitude, either direction
        "activity_lost":  float(np.clip(drop, 0, None).sum()),
        "activity_gained": float(np.clip(-drop, 0, None).sum()),
        "failure_count":  int(np.sum((b >= fail_threshold) & (p < fail_threshold))),
    }


def generate_dataset(W, nodes, sensory_idx, readout_idx, baseline, gain=1.0):
    rows, n_bad = [], 0
    for i, name in enumerate(nodes):
        perturbed, conv = run_dynamics(W, sensory_idx, knockout_idx=i, gain=gain)
        if conv is None:
            n_bad += 1
        rec = {"neuron": name}
        rec.update(cascade_severity(baseline, perturbed, readout_idx))
        rows.append(rec)
    if n_bad:
        print(f"  WARNING: {n_bad}/{len(nodes)} knockouts failed to converge")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    GAIN = 1.0        # try 3.0 to push tanh into saturation
    SIGNED = True

    if not EDGE_CSV.exists():
        urllib.request.urlretrieve(URL, EDGE_CSV)
    G = build_connectome_graph(pd.read_csv(EDGE_CSV))
    W, nodes = graph_to_matrix(G)
    print(f"graph: {G.number_of_nodes()} neurons, {G.number_of_edges()} synapses")

    if SIGNED:
        W = apply_signs(W, nodes)
    W = normalize_weights(W)

    index = {n: i for i, n in enumerate(nodes)}
    sensory_idx = [index[s] for s in SENSORY if s in index]
    readout_idx = motor_neurons(nodes)
    print(f"sensory (clamped inputs): {len(sensory_idx)}")
    print(f"motor   (readout set)   : {len(readout_idx)}")

    baseline, conv = run_dynamics(W, sensory_idx, gain=GAIN, verbose=True)
    print(f"baseline converged at iter {conv}, "
          f"activity range [{baseline.min():.3f}, {baseline.max():.3f}]")
    print(f"neurons driven negative by inhibition: {(baseline < 0).sum()}")

    ds = generate_dataset(W, nodes, sensory_idx, readout_idx, baseline, gain=GAIN)
    ds["baseline_activity"] = baseline          # <- use this as a node feature
    ds.to_csv(data_path / "knockout_severity_motor.csv", index=False)

    print("\ntop 10 by severity at motor readout:")
    print(ds.sort_values("severity", ascending=False).head(10).to_string(index=False))
    print(f"\nknockouts producing net disinhibition somewhere: "
          f"{(ds['activity_gained'] > 1e-6).sum()}/{len(ds)}")
