#this script trauns a Graph Conolutiona Network to predict cascade severity
#severity determined from local neighborhood structure
#comapres R^2 score against linear baseline (0.425)

'''Regularization experiments testing why the default GCN underperforms.
Currently configured for Experiment C. To reproduce others, modify
the indicated section in Chunks 2-3.'''

#CHUNK 1: imports and data loading
#import packages
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import r2_score

#reuse the graph construction code from the connectome_model.py
from connectome_model import load_connectome, build_connectome_graph

#paths
project_dir = Path(__file__).resolve().parent
data_dir = project_dir /"data"

#load graph + features + labels
raw = load_connectome(data_dir / "herm_full_edgelist.csv")
G = build_connectome_graph(raw)

severity = pd.read_csv(data_dir / "knockout_severity_dataset.csv")
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
edge_list = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
print("edge_index shape:", edge_index.shape) 

#x: node features matrix (num_nodes x num_features), 1 row per neuron
#columns = centrality measures
#reindex the merged df by neuron
#pull neurons in the canonical order of nodes
df_indexed = df.set_index("neuron").loc[nodes] #align with node order

feature_cols = ["in_degree", "out_degree", "betweenness", "eigenvector", "pagerank"]
features = df_indexed[feature_cols].fillna(0).to_numpy(dtype=np.float32) #num_nodes x num_features
x = torch.tensor(features, dtype=torch.float32)
print("x shape:", x.shape)

#y: severity label per neuron (num_nodes x 1)
labels = df_indexed["total_activity_lost"].to_numpy(dtype=np.float32) #num_nodes x 1
y = torch.tensor(labels, dtype=torch.float32)
print("y shape:", y.shape)

#package into PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
print("PyG Data object:", data)


#CHUNK 3: define GNN model
class GCN(torch.nn.Module):
    #this is a simple 2-layer GCN that maps node features to predicted severity
    def __init__(self, num_features, hidden_size=16, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size) #graph convolution layer: aggregate neighbor features and apply linear transformation, output shape (num_nodes x hidden_size)
        #self.conv2 = GCNConv(hidden_size, hidden_size) #second graph convolution layer, takes output of first layer as input, output shape (num_nodes x hidden_size)
        self.out = torch.nn.Linear(hidden_size, 1) #output layer for regression
        self.dropout = dropout #dropout rate for regularization

    def forward(self, x, edge_index): 
        #push the whole graph through the model in one forward pass, returns predicted severity for each node
        #297 predictions, one per neuron
        #layer 1: aggregate + transform + non-linearity + dropout
        h = self.conv1(x, edge_index)
        h = F.relu(h) #turns negative values to 0, keeps positive values unchanged
        h = F.dropout(h, p=self.dropout, training=self.training) #masking - 10% of the nodes are randomly set to 0 during training, helps prevent overfitting

        #layer 2: aggregate + transform + non-linearity + dropout
        #h = self.conv2(h, edge_index)
        #h = F.relu(h)
        #h = F.dropout(h, p=self.dropout, training=self.training) 

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


#CHUNK 4: training and evaluation loop

#train a fresh GCN model on one random 80/20 split 
#return test R^2 score and predictions
def train_one_split(data, seed, n_epochs=200, lr=0.01, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    #80/20 random split into train and test masks
    n = data.num_nodes
    permutation = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[permutation[:n_train]] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[permutation[n_train:]] = True

    #fresh model and optimizer
    model = GCN(num_features=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    #training loop
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(data.x, data.edge_index)
        loss = loss_fn(predictions[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if verbose and (epoch+1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index)
                test_r2 = r2_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
            print("Epoch:", epoch+1, "Loss:", loss.item(), "Test R^2:", test_r2)

    #final evaluation on test set
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        test_r2 = r2_score(data.y[test_mask].numpy(), pred[test_mask].numpy())
    return test_r2

print("EXPERIMENT C: 1 GCN layer, dropout=0.1")
#run 5 random splits and average test R^2
print()
print("Training GCN on 5 random splits...")
scores = []
for i in range(5):
    r2 = train_one_split(data, seed=i, verbose=(i==0)) #verbose for first split only
    print(f"Split{i}: Test R^2 = {r2:.4f}")
    scores.append(r2)

scores = np.array(scores)
print()
print(f"GCN test R^2: {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"Linear baseline (LR) = 0.425")
print(f"Difference: {scores.mean()-0.425:+.3f}")
