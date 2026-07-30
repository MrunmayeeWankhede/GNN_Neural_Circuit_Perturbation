#this script trauns a Graph Conolutiona Network to predict cascade severity
#severity determined from local neighborhood structure
#comapres R^2 score against linear baseline (0.492)

#CHUNK 1: imports and data loading
#import packages
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sklearn
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

#reuse the graph construction code from the connectome_model.py
from connectome_model import load_connectome, build_connectome_graph

CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

#paths
project_dir = Path(__file__).resolve().parent
data_dir = project_dir /"data"

#load graph + features + labels
raw = load_connectome(data_dir / "herm_full_edgelist.csv")
G = build_connectome_graph(raw)

severity = pd.read_csv(data_dir / "knockout_severity_motor.csv")
centralities = pd.read_csv(data_dir / "neuron_centralities.csv")

#merge like we did before
df = severity.merge(centralities, on="neuron")
print("graph:", G.number_of_nodes(), "nodes", G.number_of_edges(), "edges")
print("merged data shape:", df.shape)

#CHUNK 2: convert to PyTorch Geometric (PyG) format
#we will use one canonical ordering of neurons (lexicographical)

nodes = sorted(G.nodes())
node_idx = {node: i for i, node in enumerate(nodes)}

#edge index: 2 rows, 1 column per edge
#reverse edges: messages flow from downstream targets back to the sender,
#because knockout damage depends on what a neuron SENDS, not what it receives
edge_list = [(node_idx[v], node_idx[u]) for u, v in G.edges()]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
print("edge_index shape:", edge_index.shape) 

#x: node features matrix (num_nodes x num_features), 1 row per neuron
#columns = centrality measures
#reindex the merged df by neuron
#pull neurons in the canonical order of nodes
df_indexed = df.set_index("neuron").loc[nodes] #align with node order

features = ["in_degree", "out_degree", "betweenness", "eigenvector", "pagerank", "baseline_activity"]
features = df_indexed[features].fillna(0).to_numpy(dtype=np.float32) #num_nodes x num_features
x = torch.tensor(features, dtype=torch.float32)
print("x shape:", x.shape)

#y: severity label per neuron (num_nodes x 1)
labels = np.log1p(df_indexed["severity"].to_numpy(dtype=np.float32)) #log-transform the severity labels to reduce skew
y = torch.tensor(labels, dtype=torch.float32)
print("y shape:", y.shape)

#package into PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
print("PyG Data object:", data)

def train_one_split(data, train_mask, test_mask, seed=42, n_epochs=200, lr=0.01, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    #standardize features using ONLY training nodes, so test statistics don't leak
    X_np = data.x.numpy()
    scaler = StandardScaler().fit(X_np[train_mask.numpy()])
    x_scaled = torch.tensor(scaler.transform(X_np), dtype=torch.float32)

    model = GCN(num_features=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_scaled, data.edge_index)
        loss = loss_fn(predictions[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                p = model(x_scaled, data.edge_index)
                print("Epoch:", epoch + 1, "Loss:", round(loss.item(), 4),
                      "Test R^2:", round(r2_score(data.y[test_mask].numpy(), p[test_mask].numpy()), 3))

    model.eval()
    with torch.no_grad():
        pred = model(x_scaled, data.edge_index)
        return r2_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
    
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
        h = F.dropout(h, p=self.dropout, training=self.training) #masking - 30% of the nodes are randomly set to 0 during training, helps prevent overfitting

        #layer 2: aggregate + transform + non-linearity + dropout
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training) 

        #final projection to 1 output per node
        output = self.out(h).squeeze(-1) #shape (num_nodes,)
        return output
    
#check the model can process the data
torch.manual_seed(42)
model = GCN(num_features=x.shape[1])
print(model)

#check: run one forward pass and look at output shape
with torch.no_grad():
    pred = model(data.x, data.edge_index)
print("untrained prediction shape:", pred.shape) 
print("sample predictions (untrained):", pred[:5].tolist())

#run 5 random splits and average test R^2
print("Training GCN on RepeatedKFold splits:")
scores = []
for i, (tr, te) in enumerate(CV.split(np.arange(data.num_nodes))):
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool); train_mask[tr] = True
    test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool); test_mask[te]  = True
    scores.append(train_one_split(data, train_mask, test_mask, seed=i))
scores = np.array(scores)
print(f"GCN test R^2: {scores.mean():.3f} +/- {scores.std():.3f}  (n={len(scores)})")

np.save(data_dir / "scores_gcn.npy", scores)