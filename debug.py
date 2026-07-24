import pandas as pd
raw = pd.read_csv("data/herm_full_edgelist.csv")
raw['Source'] = raw['Source'].str.strip()
raw['Target'] = raw['Target'].str.strip()

all_cells = set(raw['Source'].unique()) | set(raw['Target'].unique())
def looks_like_neuron(n):
    if not n or n[0].islower(): return False
    if n in ['anal', 'hyp', 'sph', 'intestine']: return False
    return True
all_neurons_in_data = set(c for c in all_cells if looks_like_neuron(c))

chemical = raw[raw['Type'] == 'chemical']
chemical_cells = set(chemical['Source'].unique()) | set(chemical['Target'].unique())
chemical_neurons = set(c for c in chemical_cells if looks_like_neuron(c))

non_chemical_only = all_neurons_in_data - chemical_neurons
print("Neurons with NO chemical edges:", sorted(non_chemical_only))
print("Count:", len(non_chemical_only))