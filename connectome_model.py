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

#row-normalize so each neuron's total input sums to 1
#rows with zero input will stay all zeros (no division by zero)
def normalize_weights(W):
    row_sums = W.sum(axis=1, keepdims=True) #shape (n, 1)
    row_sums[row_sums == 0] = 1.0 #avoid division by zero for neurons with no inputs
    W_normalized = W / row_sums

    return W_normalized

df = load_connectome(output_path)
G = build_connectome_graph(df)
W, nodes = graph_to_matrix(G)
W = normalize_weights(W)
import matplotlib.pyplot as plt

#visualize the graph - this is a dense graph so just showing nodes and edges without labels
plt.figure(figsize=(14, 14))
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)  #invent node positions

nx.draw_networkx_nodes(G, pos, node_size=60, node_color="steelblue", alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.2, arrows=False, width=0.5)

plt.title(f"C. elegans chemical connectome ({G.number_of_nodes()} neurons, "
          f"{G.number_of_edges()} synapses)")
plt.axis("off")

results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)
plt.savefig(results_dir / "connectome_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved connectome_full.png")

#choose a focus neuron and visualize its neighborhood (directly connected neurons)
focus = "AVAL"
neighbors = set(G.successors(focus)) | set(G.predecessors(focus))
sub = G.subgraph({focus} | neighbors)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(sub, k=0.5, seed=42)

#colour the focus neuron differently from its neighbours
colors = ["crimson" if n == focus else "steelblue" for n in sub.nodes()]
sizes  = [400 if n == focus else 120 for n in sub.nodes()]

nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=sizes, alpha=0.9)
nx.draw_networkx_edges(sub, pos, edge_color="gray", alpha=0.4, arrows=True,
                       arrowsize=8, width=0.6)
nx.draw_networkx_labels(sub, pos, font_size=7)

plt.title(f"{focus} and its {len(neighbors)} direct neighbors")
plt.axis("off")
plt.savefig(results_dir / "neighborhood_AVAL.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved neighborhood_AVAL.png ({len(neighbors)} neighbors)")

#THE SIMULATOR
#define all sensory neurons in the connectome
#TODO: verify this list with the literature 
sensory_neurons = [
    "ALML", "ALMR", "AVM", "PLML", "PLMR",            #gentle touch
    "ASEL", "ASER", "ASHL", "ASHR",                    #chemosensation
    "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR",     #odor sensing
    "AFDL", "AFDR",                                     #thermosensation
    "AQR", "PQR", "URXL", "URXR",                       #oxygen sensing
    "ADLL", "ADLR", "ASJL", "ASJR", "ASKL", "ASKR",    #other sensory
]

#iterate linear-threshold dynamics until activity converges
def run_dynamics(W, nodes, sensory_idx, knockout_idx=None, max_iter=500, tol=1e-6):
    #returns the steady-state activity vector
    #pass knockput_idx=None for healthy condition
    #an index for knockout

    n = len(nodes)
    x = np.zeros(n) #initial activity is zero
    x[sensory_idx] = 1.0 #activate sensory neurons

    for i in range(max_iter):
        #weigjted input, squashed to range [0, 1] by tanh nonlinearity
        x_new = np.tanh(W @ x) #update activity based on inputs and weights
        x_new[sensory_idx] = 1.0 #keep sensory neurons active

        if knockout_idx is not None:
            x_new[knockout_idx] = 0.0 #dead neuron sends no signal

        if np.sum(np.abs(x_new - x)) < tol: #converged or not?
            x = x_new
            break
        x = x_new

    return x

#comapre a knocked-out steady state to the healthy one
def cascade_severity(baseline, perturbed, fail_threshold=0.1):
    #returns both severity measures as a dictionary
    drop = baseline - perturbed
    total_activity_lost = float(np.sum(np.clip(drop, 0, None))) #total activity lost across all neurons

    failed_neurons = (baseline >= fail_threshold) & (perturbed < fail_threshold)
    failure_count = int(np.sum(failed_neurons)) #number of neurons that fail (drop below threshold
    
    return {"total_activity_lost": total_activity_lost, "failure_count": failure_count}


index = {n: i for i, n in enumerate(nodes)}

print("Matrix shape:", W.shape, "Non-zero entries in W:", np.count_nonzero(W), "Edges in graph:", G.number_of_edges())

#sensory neurons present in our graph
sensory_idx = [index[s] for s in sensory_neurons if s in index]
print("Sensory neurons in graph:", [nodes[i] for i in sensory_idx], "Count:", len(sensory_idx))

#healthy steady state
baseline = run_dynamics(W, nodes, sensory_idx)
print("Baseline total activity:", np.sum(baseline))

#knock out AVAL and see the effect
k = index["AVAL"]
perturbed = run_dynamics(W, nodes, sensory_idx, knockout_idx=k)
print("AVAL knockout severity:", cascade_severity(baseline, perturbed))

#compare knocout of other neurons
print()
print("Comparing knockouts of other neurons:")
for name in ["AVAL", "AVAR", "AVBL", "PVCL", "ASEL", "RIH", "RIS", "VA01", "DA01"]:
    if name not in index:
        print(f"{name} not in graph, skipping")
        continue
    m = run_dynamics(W, nodes, sensory_idx, knockout_idx=index[name])
    s = cascade_severity(baseline, m)
    print(" ", name, "- activity_lost =", round(s["total_activity_lost"], 2),
          ", failures =", s["failure_count"])