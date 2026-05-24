#import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
from pathlib import Path

#set up the paths
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data"
data_path.mkdir(exist_ok=True)


import urllib.request
url = "https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/herm_full_edgelist.csv"
output_path = data_path / "herm_full_edgelist.csv"

if not output_path.exists():
    print("downloading to ", output_path)
    urllib.request.urlretrieve(url, output_path)
    print("done")
else:
    print("already downloaded to ", output_path)

#load/check the data
df = pd.read_csv(output_path)
print("shape of the data:", df.shape)
print("columns:", df.columns.tolist())
print() #need this for .head() to work properly
print(df.head(10))
print()
print("type values:", df["Type"].unique())
print("weight range:", df["Weight"].min(), "to", df["Weight"].max())
print("unique source neurons:", df["Source"].nunique())
print("unique target neurons:", df["Target"].nunique())

#this would also include non-neuronal cells - 449 cell names but only 302 unique neurons
#but we will only focus on neurons for this project
#real nueron names are short and ALL CAPS
#1. muscle cells - either have M, then D, V (dorsal or ventral), then L or R (left or right), then digits
#2. some have string "BWM" (body wall muscle) 
#3. muscle and organ cells aslo have lower case letters in their names
#function checks for all 3 and if even 1 true - returns True (NOT a neuron)

#while builfing the grapg, filter to chemical synapses only (these are directed and make a cascade flow)
#drop ant edge that touches a non-neuronal cell on either end (we want to focus on neurons only)  
#make a DiGraph (directed graph) using networkx

#this next chunk of code loads the C. elegans connectome edge list
#build a DAG of CHEMICAL synapses only
#and only those that connect neurons (not muscle or organ cells)

import re 

#decide whether a cell is a neuron or not
def is_non_neuronal(cell_name):
    n = str(cell_name).strip() #remove whitespace

    if re.fullmatch(r'M[DV][LR]\d+', n): #matches muscle cells
        return True
    
    if "BWM" in n.upper(): #matches body wall muscle cells
        return True
    
    if any(c.islower() for c in n): #matches any cell with lowercase letters (muscle and organ cells)
        return True
    
    return False

#load the raw CSV data
def load_connectome(edge_list_path):
    df = pd.read_csv(edge_list_path)

    df["Source"] = df["Source"].astype(str).str.strip() #remove whitespace
    df["Target"] = df["Target"].astype(str).str.strip()
    return df

#build the graph
def build_connectome_graph(df):
    #1. keep only chemical synapses
    sub = df[df["Type"] == "chemical"].copy()

    #2. drop any edge that touches a non-neuronal cell on either end
    neuron_mask = ~sub["Source"].apply(is_non_neuronal) & ~sub["Target"].apply(is_non_neuronal)
    sub = sub[neuron_mask]

    #3. build a directed graph using networkx
    G = nx.from_pandas_edgelist(sub, source="Source", target="Target", edge_attr="Weight", create_using=nx.DiGraph())

    for u, v, d in G.edges(data=True):
        d["weight"] = d.pop("Weight")

    return G

#convert the graph to an adjacency matrix
def graph_to_matrix(G):
    nodes = sorted(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    W = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        #egde u -> v: u sends, v receives
        #receiver = row, sender = column
        W[index[v], index[u]] = d["weight"]

    return W, nodes


df = load_connectome(output_path)
G = build_connectome_graph(df)
W, nodes = graph_to_matrix(G)

print("Matrix shape:", W.shape)
print("Number of nodes:", len(nodes))
print("Non-zero entries in W:", np.count_nonzero(W))
print("Edges in graph:", G.number_of_edges())

