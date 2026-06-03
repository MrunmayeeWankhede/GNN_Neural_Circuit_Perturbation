#this is a script to compare the GCN and linear baseline errors neuron by neuron, to see if there are specific neurons where the GCN does better or worse than the linear model
#we wanna see if they are making the same mistakes or different

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

from connectome_model import load_connectome, build_connectome_graph

#paths
project_dir = Path(__file__).resolve().parent
data_dir = project_dir /"data"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

#load everything 
raw = load_connectome(data_dir / "herm_full_edgelist.csv")
G = build_connectome_graph(raw)

severity = pd.read_csv(data_dir / "knockout_severity_dataset.csv")
centralities = pd.read_csv(data_dir / "neuron_centralities.csv")
df = severity.merge(centralities, on="neuron")

nodes = sorted(G.nodes())
node_idx = {node: i for i, node in enumerate(nodes)}
df_indexed = df.set_index("neuron").loc[nodes] #align with node order

feature_cols = ["in_degree", "out_degree", "betweenness", "eigenvector", "pagerank"]
features = df_indexed[feature_cols].fillna(0).to_numpy(dtype=np.float32) #num_nodes x num_features
labels = df_indexed["total_activity_lost"].to_numpy(dtype=np.float32) #num_nodes x 1

print("loaded", len(nodes), "neurons", G.number_of_edges(), "edges")

#linear regression - 5 fold CV predictions
print()
print("getting 5-fold CV predictions from linear regression-")
y_pred_lr = cross_val_predict(LinearRegression(), features, labels, cv=KFold(n_splits=5, shuffle=True, random_state=42))
print("Linear regression R^2:", r2_score(labels, y_pred_lr), 3)

#GCN - 5 fold CV predictions
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size=16, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h).squeeze(-1)

# build the static PyG data once
edge_list = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
x = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)
data = Data(x=x, edge_index=edge_index, y=y) #num_nodes x num_features, edge_index shape (2, num_edges), y shape (num_nodes,)


def train_and_predict_fold(train_idx, test_idx, seed):
    #train a GCN with these training indices, return predictions on test_idx
    torch.manual_seed(seed)
    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    model = GCN(num_features=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = loss_fn(pred[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
    return pred[test_mask].numpy(), test_idx

print()
print("getting 5-fold GCN predictions...")
y_pred_gcn = np.zeros(len(nodes), dtype=np.float32)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(nodes)))):
    test_preds, idx = train_and_predict_fold(train_idx, test_idx, seed=fold)
    y_pred_gcn[idx] = test_preds
    print("  fold", fold + 1, "done")
print("GCN R²:", round(r2_score(labels, y_pred_gcn), 3))

#combine results into a comparison table
results = pd.DataFrame({
    "neuron": nodes,
    "actual": labels,
    "lin_pred": y_pred_lr,
    "gcn_pred": y_pred_gcn,
    "lin_residual": labels - y_pred_lr,
    "gcn_residual": labels - y_pred_gcn,
})
results["lin_abs_err"] = results["lin_residual"].abs()
results["gcn_abs_err"] = results["gcn_residual"].abs()
results.to_csv(data_dir / "error_analysis.csv", index=False)

#top-10 worst from each model
print()
print("top 10 worst-predicted by LINEAR model:")
print(results.sort_values("lin_abs_err", ascending=False)
      .head(10)[["neuron", "actual", "lin_pred", "gcn_pred"]]
      .to_string(index=False))

print()
print("top 10 worst-predicted by GCN model:")
print(results.sort_values("gcn_abs_err", ascending=False)
      .head(10)[["neuron", "actual", "lin_pred", "gcn_pred"]]
      .to_string(index=False))


#correlation between residuals 
corr = results["lin_residual"].corr(results["gcn_residual"])
print()
print(f"correlation of residuals (linear vs GCN): {corr:.3f}")
print("interpretation:")
print("  ~1.0 = same errors, same neurons (no complementary info)")
print("  ~0.5 = partly overlapping errors")
print("  ~0.0 = independent errors (would benefit from ensembling)")


#scatter plot of residuals
plt.figure(figsize=(8, 8))
plt.scatter(results["lin_residual"], results["gcn_residual"],
            alpha=0.5, s=30)
lim = max(results["lin_residual"].abs().max(),
          results["gcn_residual"].abs().max()) * 1.1
plt.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y = x")
plt.axhline(0, color="gray", alpha=0.3)
plt.axvline(0, color="gray", alpha=0.3)
plt.xlabel("Linear regression residual (actual - predicted)")
plt.ylabel("GCN residual (actual - predicted)")
plt.title("where each model errs")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "residual_comparison.png", dpi=150)
plt.close()
print()
print("saved residual comparison plot to", results_dir / "residual_comparison.png")