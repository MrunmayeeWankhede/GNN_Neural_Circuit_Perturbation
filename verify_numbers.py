

"""
verify_report_numbers.py
 
Recomputes every value marked [sandbox] in the project report, so they can be
quoted as verified. Run from the repository root:
 
    python verify_report_numbers.py > results/verified_numbers.txt
 
Output is organised by the report chapter each number appears in.
 
Two classes of number are produced:
 
  MATCH   - should reproduce the report's value closely. Small differences in
            the last decimal are expected across platforms for tree models
            (floating-point split ties) but not for linear/analytic quantities.
 
  YOURS   - the report's value came from a reimplementation rather than this
            repository's code, so your number will differ. The EFFECT should
            replicate (same sign, same rough size); the exact figure to quote
            is the one this script prints.
 
Runtime: roughly 2-4 minutes, dominated by the epistasis section (Chapter 4.6)
and the GCN section (Chapter 8.6).
"""
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
 
warnings.filterwarnings("ignore")
 

#import the simulator from whichever module name this repo uses
try:
    from connectome_model_v2 import (build_connectome_graph, graph_to_matrix,
                                     apply_signs, normalize_weights, run_dynamics,
                                     motor_neurons, SENSORY, GABAERGIC, EDGE_CSV)
except ImportError:
    from connectome_model import (build_connectome_graph, graph_to_matrix,
                                  apply_signs, normalize_weights, run_dynamics,
                                  motor_neurons, SENSORY, GABAERGIC, EDGE_CSV)
 
GAIN = 1.0
CV_SEED = 42
 
def hdr(chapter, what):
    print(f"\n{'='*74}\nCHAPTER {chapter}  |  {what}\n{'='*74}")
 
G = build_connectome_graph(pd.read_csv(EDGE_CSV))
W_raw, nodes = graph_to_matrix(G)
index = {n: i for i, n in enumerate(nodes)}
sensory_idx = [index[s] for s in SENSORY if s in index]
motor_idx = motor_neurons(nodes)
motor_set = set(motor_idx)
N = len(nodes)
 
W_signed = normalize_weights(apply_signs(W_raw.copy(), nodes))
W_unsigned = normalize_weights(W_raw.copy())
 
base_signed, conv_s = run_dynamics(W_signed, sensory_idx, gain=GAIN)
base_unsigned, conv_u = run_dynamics(W_unsigned, sensory_idx, gain=GAIN)
 
print(f"graph: {N} neurons, {G.number_of_edges()} synapses")
print(f"sensory clamped: {len(sensory_idx)}   motor readout: {len(motor_idx)}")
print(f"baseline convergence: signed iter {conv_s}, unsigned iter {conv_u}")
 
 
def severity_vector(W, base, readout, self_exclude=True, gain=GAIN):
    out = np.empty(N)
    for i in range(N):
        p, _ = run_dynamics(W, sensory_idx, knockout_idx=i, gain=gain)
        ro = [m for m in readout if m != i] if self_exclude else list(readout)
        out[i] = np.abs((base - p)[ro]).sum()
    return out
 
 
ALL = list(range(N))
y_motor = severity_vector(W_signed, base_signed, motor_idx, self_exclude=True)
y_log = np.log1p(y_motor)
 
 
# ---------------------------------------------------------------- 2.3
hdr("2.3", "How nonlinear is tanh actually being? (report: median 0.49, ~9% bend)")
z = W_unsigned @ base_unsigned
nz = z[z > 1e-9]
bend = np.abs(np.tanh(nz) - nz) / nz
print(f"MATCH  median tanh input        = {np.median(nz):.4f}")
print(f"MATCH  mean relative bend       = {bend.mean():.4f}  ({bend.mean()*100:.1f}%)")
print(f"       90th pct input           = {np.percentile(nz, 90):.4f}")
 
 
# ---------------------------------------------------------------- 4.5
hdr("4.5", "Did clip(drop, 0, None) discard anything? (report: exactly zero)")
worst = 0.0
for i in range(N):
    p, _ = run_dynamics(W_unsigned, sensory_idx, knockout_idx=i, gain=GAIN)
    d = base_unsigned - p
    worst = max(worst, float(np.clip(-d, 0, None).sum()))
print(f"MATCH  max activity INCREASE anywhere, unsigned model = {worst:.3e}")
print(f"       (zero confirms the v1 model was provably monotone)")
 
 
# ---------------------------------------------------------------- 4.6
hdr("4.6", "Pairwise epistasis (report: median 0.52% unsigned, 0.59% signed)")
rng = np.random.default_rng(0)
hubs = np.argsort(-W_raw.sum(axis=0))[:40]
all_pairs = list(combinations(hubs, 2))
pairs = [all_pairs[k] for k in rng.choice(len(all_pairs), 200, replace=False)]
 
for lab, W, base in (("unsigned", W_unsigned, base_unsigned),
                     ("signed  ", W_signed, base_signed)):
    def sev(ko):
        p, _ = run_dynamics(W, sensory_idx, knockout_idx=ko, gain=GAIN)
        return float(np.abs(base - p).sum())
    singles = {int(i): sev(int(i)) for i in hubs}
    eps, denom = [], []
    for a, b in pairs:
        eps.append(sev([int(a), int(b)]) - (singles[int(a)] + singles[int(b)]))
        denom.append(singles[int(a)] + singles[int(b)])
    eps, denom = np.array(eps), np.array(denom)
    print(f"MATCH  {lab}: median |interaction| as % of additive = "
          f"{np.median(np.abs(eps)/denom)*100:.3f}%   (max {np.abs(eps).max():.4f})")
 
 
# ---------------------------------------------------------------- 7.2 / 7.3
hdr("7.2-7.3", "Self-inclusion inflation (report: 62.6% of motor severity, r=0.96)")
y_incl = severity_vector(W_signed, base_signed, motor_idx, self_exclude=False)
mot_mask = np.array([i in motor_set for i in range(N)])
share = (y_incl - y_motor)[mot_mask] / np.maximum(y_incl[mot_mask], 1e-12)
print(f"MATCH  median self-contribution, motor neurons = {np.median(share)*100:.1f}%")
print(f"MATCH  correlation between the two labels      = "
      f"{np.corrcoef(y_incl, y_motor)[0,1]:.4f}")
 
kf = KFold(5, shuffle=True, random_state=CV_SEED)
flag = mot_mask.astype(float).reshape(-1, 1)
print(f"MATCH  'is_motor' flag ALONE predicts severity: CV R2 = "
      f"{cross_val_score(LinearRegression(), flag, y_incl, cv=kf, scoring='r2').mean():+.3f}")
 
 
# ---------------------------------------------------------------- 8.4
hdr("8.4", "failure_count is unusable (report: 96/297 -> 14/297, 283 zeros)")
bm = base_signed[motor_idx]
print(f"MATCH  median motor baseline activity = {np.median(bm):.3f}")
print(f"MATCH  drop needed to cross 0.1 threshold = {(1 - 0.1/np.median(bm))*100:.0f}%")
inc_f, exc_f = [], []
for i in range(N):
    p, _ = run_dynamics(W_signed, sensory_idx, knockout_idx=i, gain=GAIN)
    b_all, p_all = base_signed[motor_idx], p[motor_idx]
    inc_f.append(int(((b_all >= 0.1) & (p_all < 0.1)).sum()))
    others = [m for m in motor_idx if m != i]
    exc_f.append(int(((base_signed[others] >= 0.1) & (p[others] < 0.1)).sum()))
inc_f, exc_f = np.array(inc_f), np.array(exc_f)
print(f"MATCH  knockouts with >=1 failure, self INCLUDED = {(inc_f>0).sum()}/{N}")
print(f"MATCH  knockouts with >=1 failure, self EXCLUDED = {(exc_f>0).sum()}/{N}")
print(f"MATCH  neurons with failure_count == 0          = {(exc_f==0).sum()}/{N}")
 
 
# ---------------------------------------------------------------- 9.2 / 9.3
hdr("9.2-9.3", "Direct vs indirect damage (report: 40.3%/59.7%, r=0.764)")
direct, indirect = [], []
for i, name in enumerate(nodes):
    p, _ = run_dynamics(W_unsigned, sensory_idx, knockout_idx=i, gain=GAIN)
    d = np.abs(base_unsigned - p); d[i] = 0.0
    nb = [index[t] for t in G.successors(name)]
    direct.append(d[nb].sum()); indirect.append(d.sum() - d[nb].sum())
direct, indirect = np.array(direct), np.array(indirect)
tot = direct + indirect; m = tot > 1e-6
print(f"MATCH  share of damage at DIRECT neighbours = "
      f"{direct[m].sum()/tot[m].sum()*100:.1f}%")
print(f"MATCH  share further downstream            = "
      f"{indirect[m].sum()/tot[m].sum()*100:.1f}%")
print(f"MATCH  corr(direct, indirect): Pearson {np.corrcoef(direct[m], indirect[m])[0,1]:.3f}"
      f"  Spearman {spearmanr(direct[m], indirect[m]).statistic:.3f}")
 
 
# ---------------------------------------------------------------- 9.5
hdr("9.5", "Network depth (report: 93/138/46/1, 19 unreachable)")
sp = dict(nx.shortest_path_length(G))
hop = np.array([0 if i in motor_set else
                min([sp[nodes[i]].get(nodes[m], 99) for m in motor_idx] + [99])
                for i in range(N)])
for k in range(0, 4):
    print(f"MATCH  {k} hops from motor set: {int((hop==k).sum()):3d} neurons")
print(f"MATCH  unreachable                : {int((hop==99).sum()):3d} neurons")
nm = hop[[i for i in range(N) if i not in motor_set]]
nm = nm[nm < 99]
print(f"MATCH  non-motor median = {np.median(nm):.0f} hops, max = {nm.max()}")
 
 
# ---------------------------------------------------------------- 8.1
hdr("8.1", "Unshuffled CV folds (report: 0.278 vs 0.538, on RAW target)")
scc = max(nx.strongly_connected_components(G), key=len)
eig_dir = nx.eigenvector_centrality_numpy(G.subgraph(scc).copy())
btw = nx.betweenness_centrality(G); pr = nx.pagerank(G)
X = np.nan_to_num(np.column_stack([
    [G.in_degree(n) for n in nodes], [G.out_degree(n) for n in nodes],
    [btw[n] for n in nodes], [eig_dir.get(n, 0.0) for n in nodes],
    [pr[n] for n in nodes], base_signed]))
for lab, cv, target in (("cv=5 alphabetical, RAW  ", KFold(5, shuffle=False), y_motor),
                        ("KFold shuffled,    RAW  ", KFold(5, shuffle=True, random_state=CV_SEED), y_motor),
                        ("cv=5 alphabetical, LOG  ", KFold(5, shuffle=False), y_log),
                        ("KFold shuffled,    LOG  ", KFold(5, shuffle=True, random_state=CV_SEED), y_log)):
    print(f"MATCH  {lab} linear CV R2 = "
          f"{cross_val_score(LinearRegression(), X, target, cv=cv, scoring='r2').mean():+.3f}")
 
 
# ---------------------------------------------------------------- 8.3
hdr("8.3", "Outlier-dominated metric (report: 0.622 with AVAL/AVAR, 0.093 without)")
big = {index["AVAL"], index["AVAR"]}
CVR = RepeatedKFold(n_splits=5, n_repeats=10, random_state=CV_SEED)
with_, without = [], []
for tr, te in CVR.split(y_motor):
    s = r2_score(y_motor[te], LinearRegression().fit(X[tr], y_motor[tr]).predict(X[te]))
    (with_ if big & set(te) else without).append(s)
print(f"MATCH  folds CONTAINING AVAL/AVAR: mean R2 = {np.mean(with_):+.3f}  (n={len(with_)})")
print(f"MATCH  folds without them        : mean R2 = {np.mean(without):+.3f}  (n={len(without)})")
print(f"       median severity = {np.median(y_motor):.3f}, AVAL = {y_motor[index['AVAL']]:.3f},"
      f" AVAR = {y_motor[index['AVAR']]:.3f}")
 
 
# ---------------------------------------------------------------- 8.5
hdr("8.5", "Eigenvector centrality variants (report: 0.28-0.56 spread)")
und = G.to_undirected()
lcc = und.subgraph(max(nx.connected_components(und), key=len))
variants = {
    "directed SCC, in-edges, unweighted (repo)": eig_dir,
    "directed SCC, in-edges, weighted":
        nx.eigenvector_centrality_numpy(G.subgraph(scc).copy(), weight="weight"),
    "directed SCC, out-edges, unweighted":
        nx.eigenvector_centrality_numpy(G.subgraph(scc).copy().reverse()),
    "undirected LCC, unweighted": nx.eigenvector_centrality_numpy(lcc),
    "undirected LCC, weighted": nx.eigenvector_centrality_numpy(lcc, weight="weight"),
}
CVR2 = RepeatedKFold(n_splits=5, n_repeats=10, random_state=CV_SEED)
for lab, ev in variants.items():
    col = np.array([ev.get(n, 0.0) for n in nodes])
    Xv = X.copy(); Xv[:, 3] = col
    r = cross_val_score(LinearRegression(), Xv, y_log, cv=CVR2, scoring="r2").mean()
    print(f"MATCH  {lab:42s} corr={np.corrcoef(col, y_log)[0,1]:+.3f}  linear R2={r:.3f}")
print("       NOTE: report's table used the RAW target; these use LOG. Quote these.")
 
 
# ---------------------------------------------------------------- 12.5
hdr("12.5", "Residual structure (report: sensory +0.006, posterior sensory +0.356)")
pred = cross_val_predict(LinearRegression(), X, y_log,
                         cv=KFold(5, shuffle=True, random_state=CV_SEED))
res = y_log - pred
is_sens = np.array([n in SENSORY for n in nodes])
tail = [i for i, n in enumerate(nodes)
        if n[:3] in ("PHA", "PHB", "PHC", "PVD", "PQR", "PLM", "PVM", "PDE")]
print(f"MATCH  mean residual, clamped sensory ({int(is_sens.sum()):2d}) = {res[is_sens].mean():+.3f}")
print(f"MATCH  mean residual, all others      ({int((~is_sens).sum()):3d}) = {res[~is_sens].mean():+.3f}")
print(f"MATCH  mean residual, posterior sensory ({len(tail):2d}) = {res[tail].mean():+.3f}")
print(f"MATCH  corr(residual, hop distance)          = "
      f"{np.corrcoef(res[hop<99], hop[hop<99])[0,1]:+.3f}")
print(f"       posterior set: {sorted(nodes[i] for i in tail)}")
 
 
# ---------------------------------------------------------------- 8.6
hdr("8.6", "GCN feature standardisation (report: 0.246 -> 0.457 -- YOURS WILL DIFFER)")
try:
    import torch
    import torch.nn.functional as Fn
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from sklearn.preprocessing import StandardScaler
 
    # reversed edges: the corrected orientation from Chapter 11
    el = [(index[v], index[u]) for u, v in G.edges()]
    ei = torch.tensor(el, dtype=torch.long).t().contiguous()
 
    class GCN(torch.nn.Module):
        def __init__(s, f, h=16, dp=0.3):
            super().__init__()
            s.c1 = GCNConv(f, h); s.c2 = GCNConv(h, h)
            s.o = torch.nn.Linear(h, 1); s.dp = dp
        def forward(s, x, ei):
            h = Fn.dropout(Fn.relu(s.c1(x, ei)), p=s.dp, training=s.training)
            h = Fn.dropout(Fn.relu(s.c2(h, ei)), p=s.dp, training=s.training)
            return s.o(h).squeeze(-1)
 
    yt = torch.tensor(y_log, dtype=torch.float32)
    def gcn_cv(scale):
        out = []
        for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=CV_SEED).split(y_log)):
            Xf = StandardScaler().fit(X[tr]).transform(X) if scale else X
            xt = torch.tensor(Xf, dtype=torch.float32)
            torch.manual_seed(i)
            m = GCN(X.shape[1])
            opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
            for _ in range(200):
                m.train(); opt.zero_grad()
                Fn.mse_loss(m(xt, ei)[tr], yt[tr]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                out.append(r2_score(y_log[te], m(xt, ei).numpy()[te]))
        return np.mean(out)
    print(f"YOURS  GCN, UNSTANDARDISED features = {gcn_cv(False):.3f}")
    print(f"YOURS  GCN, standardised features   = {gcn_cv(True):.3f}")
    print("       (effect should replicate; quote YOUR numbers, not the report's)")
except ImportError:
    print("SKIP   torch/torch_geometric not importable in this environment")
 
print(f"\n{'='*74}\ndone. Numbers marked MATCH should agree with the report.")
print("Numbers marked YOURS supersede the report's values.\n" + "="*74)
 
