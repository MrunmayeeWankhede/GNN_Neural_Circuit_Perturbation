#this script trains a Graph Convolutional Network to predict cascade severity
#severity determined from local neighborhood structure
#compares R^2 score against the linear baseline (see train_baseline_models.py)
#
#usage:
#    python train_gnn.py                        # downstream messages (the corrected orientation)
#    python train_gnn.py --direction upstream   # PyG default orientation, for the README Section 8 comparison

#CHUNK 1: imports and data loading
#import packages
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

#reuse the graph construction code from the connectome_model.py
from connectome_model import load_connectome, build_connectome_graph

#same folds as train_baseline_models.py, so per-fold scores can be paired
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

#paths
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

FEATURES = ["in_degree", "out_degree", "betweenness", "eigenvector", "pagerank", "baseline_activity"]


#CHUNK 2: convert to PyTorch Geometric (PyG) format
def load_data(direction="downstream", severity_csv="knockout_severity_motor.csv",
              feature_cols=FEATURES, label_col="severity", log_target=True):
    #load graph + features + labels and package them into a PyG Data object
    #returns (data, nodes) where nodes is the canonical (lexicographical) neuron order
    raw = load_connectome(data_dir / "herm_full_edgelist.csv")
    G = build_connectome_graph(raw)

    severity = pd.read_csv(data_dir / severity_csv)
    centralities = pd.read_csv(data_dir / "neuron_centralities.csv")
    df = severity.merge(centralities, on="neuron")

    #we will use one canonical ordering of neurons (lexicographical)
    nodes = sorted(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}

    #edge index: 2 rows, 1 column per edge
    #PyG passes messages from edge_index[0] (source) to edge_index[1] (target)
    #downstream: reverse the edges, so each neuron aggregates from the neurons it synapses ONTO
    #            (knockout damage depends on what a neuron SENDS, not what it receives)
    #upstream:   keep the synaptic direction, so each neuron aggregates from its presynaptic partners
    if direction == "downstream":
        edge_list = [(node_idx[v], node_idx[u]) for u, v in G.edges()]
    elif direction == "upstream":
        edge_list = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    else:
        raise ValueError("direction must be 'downstream' or 'upstream'")
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    #x: node features matrix (num_nodes x num_features), 1 row per neuron
    #reindex the merged df by neuron and pull neurons in the canonical order of nodes
    df_indexed = df.set_index("neuron").loc[nodes] #align with node order
    #eigenvector centrality is NaN outside the largest strongly connected component -> 0
    features = df_indexed[list(feature_cols)].fillna(0).to_numpy(dtype=np.float32)
    x = torch.tensor(features, dtype=torch.float32)

    #y: severity label per neuron (num_nodes,)
    labels = df_indexed[label_col].to_numpy(dtype=np.float32)
    if log_target:
        labels = np.log1p(labels) #log-transform the severity labels to reduce skew
    y = torch.tensor(labels, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, y=y), nodes


#CHUNK 3: define GNN model
class GCN(torch.nn.Module):
    #this is a simple 2-layer GCN that maps node features to predicted severity
    def __init__(self, num_features, hidden_size=16, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size) #graph convolution layer: aggregate neighbor features and apply linear transformation, output shape (num_nodes x hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size) #second graph convolution layer, takes output of first layer as input, output shape (num_nodes x hidden_size)
        self.out = torch.nn.Linear(hidden_size, 1) #output layer for regression
        self.dropout = dropout #dropout rate for regularization

    def forward(self, x, edge_index):
        #push the whole graph through the model in one forward pass, returns predicted severity for each node
        #297 predictions, one per neuron
        #layer 1: aggregate + transform + non-linearity + dropout
        h = self.conv1(x, edge_index)
        h = F.relu(h) #turns negative values to 0, keeps positive values unchanged
        h = F.dropout(h, p=self.dropout, training=self.training) #randomly zeroes 30% of the hidden units during training, helps prevent overfitting

        #layer 2: aggregate + transform + non-linearity + dropout
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        #final projection to 1 output per node
        output = self.out(h).squeeze(-1) #shape (num_nodes,)
        return output


#CHUNK 4: training
def train_one_split(data, train_mask, test_mask, seed=42, n_epochs=200, lr=0.01,
                    dropout=0.3, standardize=True, return_pred=False, verbose=False):
    #train on train_mask nodes, return test R^2 (or the test-node predictions if return_pred)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if standardize:
        #standardize features using ONLY training nodes, so test statistics don't leak
        X_np = data.x.numpy()
        scaler = StandardScaler().fit(X_np[train_mask.numpy()])
        x_in = torch.tensor(scaler.transform(X_np), dtype=torch.float32)
    else:
        x_in = data.x

    model = GCN(num_features=data.x.shape[1], dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_in, data.edge_index)
        loss = loss_fn(predictions[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                p = model(x_in, data.edge_index)
                print("Epoch:", epoch + 1, "Loss:", round(loss.item(), 4),
                      "Test R^2:", round(r2_score(data.y[test_mask].numpy(), p[test_mask].numpy()), 3))

    model.eval()
    with torch.no_grad():
        pred = model(x_in, data.edge_index)
    if return_pred:
        return pred[test_mask].numpy()
    return r2_score(data.y[test_mask].numpy(), pred[test_mask].numpy())


def masks_from_indices(n, train_idx, test_idx):
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[test_idx] = True
    return train_mask, test_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["downstream", "upstream"], default="downstream",
                        help="which neighbours each neuron aggregates from (default: downstream targets)")
    args = parser.parse_args()

    data, nodes = load_data(direction=args.direction)
    print("PyG Data object:", data)
    print("message passing direction:", args.direction)

    #check the model can process the data: one forward pass, look at output shape
    torch.manual_seed(42)
    model = GCN(num_features=data.x.shape[1])
    print(model)
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
    print("untrained prediction shape:", pred.shape)

    #5-fold CV repeated 10 times (50 splits), same folds as the baselines
    print("Training GCN on RepeatedKFold splits:")
    scores = []
    for i, (tr, te) in enumerate(CV.split(np.arange(data.num_nodes))):
        train_mask, test_mask = masks_from_indices(data.num_nodes, tr, te)
        scores.append(train_one_split(data, train_mask, test_mask, seed=i))
    scores = np.array(scores)
    print(f"GCN ({args.direction}) test R^2: {scores.mean():.3f} +/- {scores.std():.3f}  (n={len(scores)})")

    #downstream is the main model -> scores_gcn.npy; upstream is kept for the direction comparison
    out_name = "scores_gcn.npy" if args.direction == "downstream" else "scores_gcn_upstream.npy"
    np.save(data_dir / out_name, scores)
    print("saved", data_dir / out_name)
