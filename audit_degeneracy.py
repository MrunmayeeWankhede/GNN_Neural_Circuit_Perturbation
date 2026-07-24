"""
audit_degeneracy.py

Asks one question: is the cascade-severity label a closed-form function of the
graph, rather than something a model has to learn?

For a row-normalized non-negative W in the near-linear regime of tanh, knocking
out neuron j reduces activity at each target i by roughly x_j * W[i,j]. Summed
over all targets:

    severity(j)  ~=  x_j * sum_i W[i,j]  =  baseline_activity(j) * colsum(j)

If that closed form predicts the label well, the benchmark is degenerate: no
model can beat it by learning, because there is nothing left to learn.

Run this against each variant of the simulator to find a task setup where the
closed form FAILS - that is where a GNN has room to do something.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from connectome_model_v2 import (build_connectome_graph, graph_to_matrix,
                                 apply_signs, normalize_weights, run_dynamics,
                                 motor_neurons, SENSORY, EDGE_CSV)

KF = KFold(5, shuffle=True, random_state=0)


def cv_r2(X, y):
    X = np.asarray(X)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    return cross_val_score(LinearRegression(), X, y, cv=KF, scoring="r2").mean()


def build(signed):
    G = build_connectome_graph(pd.read_csv(EDGE_CSV))
    W, nodes = graph_to_matrix(G)
    if signed:
        W = apply_signs(W, nodes)
    return normalize_weights(W), nodes, G


def run_variant(signed, gain, readout, label):
    W, nodes, _ = build(signed)
    index = {n: i for i, n in enumerate(nodes)}
    sensory_idx = [index[s] for s in SENSORY if s in index]
    ro = list(range(len(nodes))) if readout == "global" else motor_neurons(nodes)

    base, _ = run_dynamics(W, sensory_idx, gain=gain)
    y = np.array([np.abs(base - run_dynamics(W, sensory_idx, knockout_idx=i,
                                             gain=gain)[0])[ro].sum()
                  for i in range(len(nodes))])

    analytic = base * np.abs(W).sum(axis=0)
    feats = np.column_stack([base, np.abs(W).sum(axis=0), (W != 0).sum(axis=0),
                             np.abs(W).sum(axis=1)])
    gbm = cross_val_score(GradientBoostingRegressor(random_state=0),
                          feats, y, cv=KF, scoring="r2").mean()
    sat = np.mean(np.abs(np.tanh(gain * (W @ base))) > 0.9)
    return {"variant": label, "closed_form_R2": cv_r2(analytic, y),
            "GBM_R2": gbm, "saturated": sat}


if __name__ == "__main__":
    rows = [
        run_variant(False, 1.0, "global", "unsigned, gain 1, global  [ORIGINAL]"),
        run_variant(True,  1.0, "global", "signed,   gain 1, global"),
        run_variant(False, 1.0, "motor",  "unsigned, gain 1, motor"),
        run_variant(True,  1.0, "motor",  "signed,   gain 1, motor   [PROPOSED]"),
        run_variant(True,  3.0, "motor",  "signed,   gain 3, motor"),
    ]
    df = pd.DataFrame(rows)
    print("closed_form_R2 HIGH  -> task is degenerate, nothing to learn")
    print("closed_form_R2 LOW   -> label is not a simple function of the graph")
    print(df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
