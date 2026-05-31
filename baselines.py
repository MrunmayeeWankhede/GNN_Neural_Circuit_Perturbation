#this script computes the classical graph centralities
#trains simple baseline models (linear regression, gradient boosting) to predict cascade severity from structure alone
#evaluates how well these models predict cascade severity compared to the GNN model

#import packages
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

#reuse the work from connectome_model.py 
from connectome_model import load_connectome, build_connectome_graph  

#define paths
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data"
edgelist_path = data_path / "herm_full_edgelist.csv"

#build the graph 
raw = load_connectome(edgelist_path)
G = build_connectome_graph(raw)
print("graph built:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

#compute centralities
def compute_centralities(G):
    #compute 4 classical centrality measures for every neuron (node)
    #returns a dataframe eith 1 row per neuron and columns for each centrality measure
    #eigenvector centrality is computed only on the largest connected component
    #it is mathematically undefined for disconnected graphs

    #find the largest "strongly" connected component 
    #strongly connected means that there is a directed path from every node to every other node in the component
    #this is needed for eigenvector centrality, which is undefined for disconnected graphs
    components = list(nx.strongly_connected_components(G))
    largest_cc = max(components, key=len)
    print("graph has", len(components), "strongly connected components")
    print("largest has", len(largest_cc), "out of", G.number_of_nodes(), "nodes")

    #restrict to the largest connected component for eigenvector centrality
    G_main = G.subgraph(largest_cc).copy()

    #eigenvector on the main component, other neurons get NaN
    eig_main = nx.eigenvector_centrality_numpy(G_main)
    eig_full = {n: eig_main.get(n, float("nan")) for n in G.nodes()}

    centralities = {"in_degree": dict(G.in_degree()), "out_degree": dict(G.out_degree()), 
                    "betweenness": nx.betweenness_centrality(G), "eigenvector": eig_full, "pagerank": nx.pagerank(G)}
    
    centrality_df = pd.DataFrame(centralities)
    centrality_df.index.name = "neuron"

    return centrality_df.reset_index()

#run the function and save the results
print()
print("computing centralities")
centralities = compute_centralities(G)
print("done. shape:", centralities.shape)

print()
print("first 10 rows:")
print(centralities.head(10))

print()
print("top 10 neurons by in degree:")
print(centralities.sort_values("in_degree", ascending=False).head(10))

print()
print("top 10 neurons by out degree:")
print(centralities.sort_values("out_degree", ascending=False).head(10))

print()
print("top 10 neurons by betweenness centrality:")
print(centralities.sort_values("betweenness", ascending=False).head(10))

print()
print("top 10 neurons by eigenvector centrality:")
print(centralities.sort_values("eigenvector", ascending=False).head(10))

print()
print("top 10 neurons by pagerank:")
print(centralities.sort_values("pagerank", ascending=False).head(10))

#save the centralities to a csv file for later use
centralities.to_csv(data_path/"neuron_centralities.csv", index=False)
print()
print("centralities saved to", data_path/"neuron_centralities.csv")