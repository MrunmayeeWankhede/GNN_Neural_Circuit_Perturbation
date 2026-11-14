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
from pathlib import Path
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from connectome_model_v2 import (build_connectome_graph, graph_to_matrix,
                                 apply_signs, normalize_weights, run_dynamics,
                                 motor_neurons, SENSORY, EDGE_CSV)

KF = KFold(5, shuffle=True, random_state=0)
#the folds every model script uses - needed to put the closed form in the same table as the models
CV_MODELS = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)


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


def run_variant(signed, gain, readout, label, self_exclude=False, save_scores=None):
    W, nodes, _ = build(signed)
    index = {n: i for i, n in enumerate(nodes)}
    sensory_idx = [index[s] for s in SENSORY if s in index]
    ro = list(range(len(nodes))) if readout == "global" else motor_neurons(nodes)

    base, _ = run_dynamics(W, sensory_idx, gain=gain)
    y = np.empty(len(nodes))
    for i in range(len(nodes)):
        d = np.abs(base - run_dynamics(W, sensory_idx, knockout_idx=i, gain=gain)[0])
        #self-exclusion: a knocked-out readout neuron's own drop is not damage it caused
        keep = [m for m in ro if m != i] if self_exclude else ro
        y[i] = d[keep].sum()

    analytic = base * np.abs(W).sum(axis=0)
    feats = np.column_stack([base, np.abs(W).sum(axis=0), (W != 0).sum(axis=0),
                             np.abs(W).sum(axis=1)])
    gbm = cross_val_score(GradientBoostingRegressor(random_state=0),
                          feats, y, cv=KF, scoring="r2").mean()
    sat = np.mean(np.abs(np.tanh(gain * (W @ base))) > 0.9)
    #same closed form, scored the way the models are scored: log1p target, shared folds
    log_scores = cross_val_score(LinearRegression(), analytic.reshape(-1, 1), np.log1p(y),
                                 cv=CV_MODELS, scoring="r2")
    if save_scores is not None:
        np.save(save_scores, log_scores)
    return {"variant": label, "closed_form_R2": cv_r2(analytic, y),
            "closed_form_R2_log": log_scores.mean(),
            "GBM_R2": gbm, "saturated": sat}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    rows = [
        run_variant(False, 1.0, "global", "unsigned, gain 1, global  [ORIGINAL]"),
        run_variant(True,  1.0, "global", "signed,   gain 1, global"),
        run_variant(False, 1.0, "motor",  "unsigned, gain 1, motor"),
        run_variant(True,  1.0, "motor",  "signed,   gain 1, motor   [PROPOSED]"),
        run_variant(True,  1.0, "motor",  "signed,   gain 1, motor, self-excluded [FINAL]",
                    self_exclude=True, save_scores=root / "data" / "scores_closed_form.npy"),
        run_variant(True,  3.0, "motor",  "signed,   gain 3, motor"),
    ]
    df = pd.DataFrame(rows)
    text = "\n".join([
        "closed_form_R2      raw target, KFold(5, shuffle, seed 0). HIGH -> task is degenerate",
        "closed_form_R2_log  log1p target, RepeatedKFold(5x10, seed 42) - the protocol the models",
        "                    are scored with, so THIS is the column comparable to model R2",
        "GBM_R2              raw target, GBM on [baseline, |W| col sum, out-degree, |W| row sum]",
        "",
        df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"),
    ])
    print(text)
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "degeneracy_audit.txt").write_text(text + "\n")
