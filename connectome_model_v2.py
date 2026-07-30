"""
connectome_model.py  (v2)

Changes from v1, and why:
  1. signed edges      - GABAergic neurons get negative outgoing weight, so the
                         system is no longer monotone and knockouts can INCREASE
                         downstream activity (disinhibition).
  2. |w| row-normalize - mixed-sign rows must be normalized by sum of absolute
                         value, or rows with cancelling inputs blow up.
  3. no clip           - clip(drop, 0, None) silently deletes every disinhibition
                         effect. v1 was monotone, so clipping 
                         would destroy the signal now.
  4. readout set       - severity is measured at a defined OUTPUT population
                         (motor neurons), not summed over the whole network.
                         This is the change that makes the task non-trivial.
  5. gain parameter    - lets you push tanh out of its near-linear regime.
  6. honest name       - this is a smooth recurrent map, not linear-threshold
                         dynamics.
"""

#import the packages
import re
import numpy as np
import pandas as pd
import networkx as nx
import urllib.request
from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data"
data_path.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/herm_full_edgelist.csv"
EDGE_CSV = data_path / "herm_full_edgelist.csv"


#filter the dataset to remove non-neuronal cells
def is_non_neuronal(cell_name):
    n = str(cell_name).strip()
    if re.fullmatch(r'M[DV][LR]\d+', n):
        return True
    if "BWM" in n.upper():
        return True
    if any(c.islower() for c in n):
        return True
    return False

#only consider chemical synapeses (directed)
#cannot include gap junctiond coz they are electrical and birectional
#cannot add them in our directed graph model
def build_connectome_graph(df):
    df = df.copy()
    for c in ("Source", "Target", "Type"):
        df[c] = df[c].astype(str).str.strip()
    sub = df[df["Type"] == "chemical"]
    #keep edges only if both source and target are neurons (not muscles or glia)
    mask = ~sub["Source"].apply(is_non_neuronal) & ~sub["Target"].apply(is_non_neuronal)
    sub = sub[mask]
    #build the directed graph from the filtered edge list
    G = nx.from_pandas_edgelist(sub, "Source", "Target", edge_attr="Weight", create_using=nx.DiGraph())
    #resolve upper/lower case issues in node names 
    for u, v, d in G.edges(data=True):
        d["weight"] = d.pop("Weight")
    return G

#create the weight matrix W from the directed graph G
def graph_to_matrix(G):
    #W[i, j] = weight of synapse from j (sender) to i (receiver)
    #this means info flowinf from jo to i not i to j (IMP!!!!)
    nodes = sorted(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))
    for u, v, d in G.edges(data=True):
        #edge is from u to v, but row index is v and column index is u
        W[index[v], index[u]] = d["weight"]
    return W, nodes

#neuron annotation
#hermaphrodite GABAergic neurons (McIntire et al. 1993; consistent with CeNGEN).
#the zero-padded names - the OpenWorm edge list uses DD01 not DD1.
#define the GABAergic neurons in the C. elegans connectome
GABAERGIC = (["DD%02d" % i for i in range(1, 7)]
             + ["VD%02d" % i for i in range(1, 14)]
             + ["AVL", "DVB", "RIS", "RMED", "RMEV", "RMEL", "RMER"])
#define the sensory neurons in the C. elegans connectome
SENSORY = ["ALML", "ALMR", "AVM", "PLML", "PLMR", "ASEL", "ASER", "ASHL", "ASHR",
           "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR", "AFDL", "AFDR",
           "AQR", "PQR", "URXL", "URXR", "ADLL", "ADLR", "ASJL", "ASJR",
           "ASKL", "ASKR"]

#define the motor neurons in the C. elegans connectome
def motor_neurons(nodes):
    #ventral-cord and head motor neurons - the output population.
    #we are gonna measure activity lost/severity lost at these neurons only, not the whole network (in v2)
    vc = ("VA", "VB", "DA", "DB", "AS", "VC", "DD", "VD")
    head = ("RMD", "RME", "SMD", "SMB")
    out = []
    for i, n in enumerate(nodes):
        #check if the neuron name starts with a ventral-cord or head motor neuron prefix
        #also check if the rest of the name is a number 
        #so we dont accidentally include other neurons that start with the same prefix but are not motor neurons
        #just pattern matching at this point, need to verify with the actual neuron names in the connectome
        if n[:2] in vc and n[2:].isdigit():
            out.append(i)
        elif n[:3] in head:
            out.append(i)
    return sorted(set(out))

#add the sign of the synapse based on the transmitter type of the presynaptic neuron
def apply_signs(W, nodes, inhibitory=GABAERGIC):
    #neuron j's outgoing column carries its transmitter sign
    index = {n: i for i, n in enumerate(nodes)}
    sign = np.ones(len(nodes))
    found = 0
    #if it is a GABAergic neuron, then its outgoing synapses are inhibitory, so we set the sign to -1
    for name in inhibitory:
        if name in index:
            sign[index[name]] = -1.0
            found += 1
    #warn if any inhibitory neurons are missing from the graph
    missing = [n for n in inhibitory if n not in index]
    if missing:
        print(f"warning: {len(missing)} inhibitory neurons not in graph: {missing}")
    print(f"applied negative sign to {found}/{len(inhibitory)} inhibitory neurons")
    #return the weight matrix with the signs applied to the outgoing synapses of inhibitory neurons
    return W * sign[None, :]

#normalize the weight matrix by the sum of absolute weights in each row
def normalize_weights(W):
    #row-normalize by sum of ABSOLUTE weight (coz negative weights exist)
    #axis=1 sums across each row
    rs = np.abs(W).sum(axis=1, keepdims=True)
    #avoid division by zero for neurons with no outgoing synapses (isolated neurons)
    rs[rs == 0] = 1.0
    #return the normalized weight matrix
    return W/rs


#dynamics of the recurrent network: x <- tanh(gain * W x)
def run_dynamics(W, sensory_idx, knockout_idx=None, gain=1.0,
                 max_iter=2000, tol=1e-9, verbose=False):
    """
    Run the dynamics of the recurrent network.
    Smooth recurrent map: x <- tanh(gain * W x), with sensory neurons clamped
    to 1 and knocked-out neurons clamped to 0. Returns (x, iters_to_converge).
    iters is None if it never converged - check this, do not assume.
    """
    #initialize the activity vector x with zeros
    n = W.shape[0]
    x = np.zeros(n)
    #set the activity of sensory neurons to 1.0 (clamped)
    #everything is silent and sensory neurons are on (1.0)
    x[sensory_idx] = 1.0
    for i in range(max_iter):
        #one update step of the recurrent dynamics
        #W @ x gathers the inputs to each neuron from all other neurons
        #gain scales the input to the tanh nonlinearity
        #tanh squashes the input to the range [-1, 1]
        #sensory neurons are reclamped to 1.0, so we set their activity
        x_new = np.tanh(gain * (W @ x))
        x_new[sensory_idx] = 1.0
        #knocknout aooied after sensory clamp
        #if we knockout sensory neruron, it stays dead
        if knockout_idx is not None:
            x_new[knockout_idx] = 0.0
        #compute how much the activity vector has changed from the previous iteration
        #delta is the sum of absolute differences between the new and old activity vectors
        delta = np.abs(x_new - x).sum()
        x = x_new
        #if the change is below the tolerance, we consider the dynamics to have converged
        if delta < tol:
            return x, i
    #if we reach here, the dynamics did not converge within the maximum number of iterations
    #might need to increase the number of itertaions or adjust the gain to get convergence
    if verbose:
        print(f"warning: did not converge, final delta={delta:.2e}")
    return x, None

#in this version, we measure the severity of the knockout effect at a defined readout population (motor neurons)
def cascade_severity(baseline, perturbed, readout_idx, knockout_idx=None):
    ro = readout_idx
    if knockout_idx is not None:
        ko = {knockout_idx} if isinstance(knockout_idx, int) else set(knockout_idx)
        ro = [i for i in readout_idx if i not in ko]
    #do not consider the knocked-out neurons in the readout population when measuring severity
    drop = (baseline - perturbed)[ro]
    return {
        "severity":        float(np.abs(drop).sum()),
        "activity_lost":   float(np.clip(drop, 0, None).sum()),
        "activity_gained": float(np.clip(-drop, 0, None).sum()),
    }


#loops over all 297 neurons, knocks them out one by one, and measures the severity of the effect on the readout population
def generate_dataset(W, nodes, sensory_idx, readout_idx, baseline, gain=1.0):
    rows, n_bad = [], 0
    for i, name in enumerate(nodes):
        perturbed, conv = run_dynamics(W, sensory_idx, knockout_idx=i, gain=gain)
        if conv is None:
            n_bad += 1
        #merges the 4 severity metrics into a single record for the neuron, and appends it to the dataset
        rec = {"neuron": name}
        rec.update(cascade_severity(baseline, perturbed, readout_idx, knockout_idx=i))
        rows.append(rec)
    #warn if any knockouts failed to converge, and return the dataset as a dataframe
    if n_bad:
        print(f"warning: {n_bad}/{len(nodes)} knockouts failed to converge")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    #set the gain for the recurrent dynamics
    GAIN = 1.0    
    SIGNED = True

    #download the edge list if it does not exist, build the connectome graph, and convert it to a weight matrix
    if not EDGE_CSV.exists():
        urllib.request.urlretrieve(URL, EDGE_CSV)
    G = build_connectome_graph(pd.read_csv(EDGE_CSV))
    W, nodes = graph_to_matrix(G)
    print(f"graph: {G.number_of_nodes()} neurons, {G.number_of_edges()} synapses")

    #apply the signs before normalizing, so that inhibitory neurons have negative outgoing weights
    if SIGNED:
        W = apply_signs(W, nodes)
    W = normalize_weights(W)

    #find the indices of the sensory and motor neurons in the nodes list
    index = {n: i for i, n in enumerate(nodes)}
    sensory_idx = [index[s] for s in SENSORY if s in index]
    readout_idx = motor_neurons(nodes)
    print(f"sensory (clamped inputs): {len(sensory_idx)}")
    print(f"motor (readout set): {len(readout_idx)}")

    #run the dynamics with the sensory neurons clamped to 1.0, and measure the baseline activity of the network
    baseline, conv = run_dynamics(W, sensory_idx, gain=GAIN, verbose=True)
    print(f"baseline converged at iter {conv}, "
          f"activity range [{baseline.min():.3f}, {baseline.max():.3f}]")
    print(f"neurons driven negative by inhibition: {(baseline < 0).sum()}")

    ds = generate_dataset(W, nodes, sensory_idx, readout_idx, baseline, gain=GAIN)
    #bolts the healthy activity onto the output as a column
    ds["baseline_activity"] = baseline          # <- use this as a node feature
    ds.to_csv(data_path / "knockout_severity_motor.csv", index=False)

    print("\ntop 10 by severity at motor readout:")
    print(ds.sort_values("severity", ascending=False).head(10).to_string(index=False))
    print(f"\nknockouts producing net disinhibition somewhere: "
          f"{(ds['activity_gained'] > 1e-6).sum()}/{len(ds)}")
