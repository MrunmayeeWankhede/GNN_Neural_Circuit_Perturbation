#this script trauns a Graph Conolutiona Network to predict cascade severity
#severity determined from local neighborhood structure
#comapres R^2 score against linear baseline (0.425)

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
