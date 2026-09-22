# GNN-guided circuit pruning

This experiment trains a GNN to rank the next neuron to remove from an already
pruned circuit. Its target is the worst fitting-input output error **after the
additional removal**, measured against the intact circuit. Every proposed
deletion is checked by the simulator before acceptance.

This is separate from `train_gnn.py`, which predicts individual knockout severity
in the intact graph. The original script is unchanged.

## Run

In an activated Python 3.11+ environment at the repository root:

```bash
python -m pip install -r requirements-pruning-gnn.txt
python -m unittest discover -s tests -v
python train_pruning_gnn.py --output results/pruning_gnn_run
```

The pipeline downloads the pinned worm edge list if needed. For an offline run,
add `--edge-list data/herm_full_edgelist.csv`. Training runs on CPU with two threads;
no GPU is required. Label generation is usually more expensive than model fitting.
The curated `results/pruning_gnn_pilot` directory already contains a trained model
and results, so use a new output directory for a fresh run.

Stages can also be run separately in the same output directory:

```bash
python train_pruning_gnn.py --stage generate --output results/pruning_gnn_run
python train_pruning_gnn.py --stage train --output results/pruning_gnn_run
python train_pruning_gnn.py --stage evaluate --output results/pruning_gnn_run
```

Existing dataset manifests, trained checkpoints and final comparisons are not
overwritten by their respective stages. A failed generation stage can be restarted
before its manifest has been written; completed state files may then be replaced.

## Training data and split

- Six fitting input patterns, 25 protected sensory neurons and 93 protected motor
  neurons use the same definitions as the initial pruning experiment.
- Ten random-priority feasible pruning trajectories use seeds 100-105 for training,
  200-201 for validation and 300-301 for testing.
- States are sampled after 8, 16 and 24 accepted removals, if reached. The intact
  graph is added to training only. Identical removal sets are deduplicated across
  all splits.
- Each state labels every remaining unprotected neuron by simulating its removal.
  Nonconvergent candidates have missing labels and are excluded from regression.
- The target is `log1p(worst_relative_L1_error / 0.05)`. The model predicts the
  transformed score; its ordering is used during pruning.

The split unit is a whole removal trajectory, not random rows from one state.
However, **the same neurons and underlying worm connectome occur in every split**.
This tests generalization to new removal states within one graph. It does not test
new animals, new connectomes, neuron identities or different fitting-input suites.

## Model

The network has two PyTorch Geometric `GCNConv` layers with 32 hidden units, ReLU,
and 0.1 dropout between layers. A small regression head receives both the final
graph embedding and the original standardized node features.

Messages follow reversed anatomical edges, so senders aggregate their downstream
targets. Deleted neurons have no remaining incident edges. The GCN uses unweighted
adjacency; signed weights remain in the simulator, while inhibitory identity and
remaining incoming/outgoing strengths are node features.

Features include active/input/output/inhibitory flags, remaining strengths and
degree fractions, plus intact and current activity under each fitting stimulus.
No candidate knockout response enters its features. Activity features do require
simulation: the initial state and accepted candidate states are reused during
pruning, and their simulation costs are counted.

Feature means and scales are fitted on training states only. Adam uses learning
rate 0.003 and weight decay 0.0005. Training runs for at most 300 epochs, stopping
after 50 epochs without validation improvement. The best validation-MSE checkpoint
is restored before reporting test metrics. The pilot uses one training seed, 42.

## Evaluation and cost

Static severity and the GNN each receive a cap of 550 online simulator calls, each
evaluating all six fitting stimuli. Initial intact references, static ranking
calls and rejected candidate checks count against this cap. GNN rankings are
recomputed after each accepted removal. Both methods verify cumulative deletions
against the same 5% fitting-error budget and may stop early if no deletion works.

Once both removal orders are frozen, the final circuits are evaluated on 32 new
input patterns from seed 20260922: 16 continuous mixtures of the fitting groups and
16 patterns with independent amplitudes for each sensory neuron. These responses
are not used for model fitting, checkpoint selection or pruning. Held-out failures
are reported without revising the circuit. This new suite is separate from the
six held-out patterns in the original baseline pilot.

**Online cost excludes offline label generation and training.** The dataset
manifest and training report record those additional costs. A reduction in online
calls alone is not an end-to-end speedup; repeated use would be needed to amortize
training. Wall-clock times depend on hardware. Equal caps do not require equal
actual usage if a method stops early.

Regression R-squared is computed on the transformed target, and Spearman measures
candidate ranking. Good regression metrics do not by themselves show that pruning
beats static severity. The pruning comparison is the relevant downstream result.

## Saved artifacts

| File | Contents |
|---|---|
| `dataset.json` | Source/data hashes, split seeds, features, states, environment and label-generation cost |
| `state_*.npz` | Per-state features, message edges, candidate masks, raw error labels and removed indices |
| `model.pt` | Model weights, architecture dimensions, feature normalization, seed and dataset-manifest hash |
| `training_history.json` | Training and validation loss per epoch |
| `training_metrics.json` | Best epoch, fitting time and split-specific metrics |
| `predictions.npz` | Predictions and transformed targets for each split |
| `pruning_comparison.json` | Online costs, removal traces, fitting errors and final held-out errors |
| `evaluation_responses.npz` | Fresh held-out stimuli and intact/reduced motor output vectors |

The checkpoint can be loaded on CPU with `torch.load(path, map_location="cpu",
weights_only=True)`. Instantiate `PruningGCN` with its saved dimensions, load
`state_dict`, and standardize features with its saved `mean` and `scale`.

One graph, one training seed and a small set of related removal states constitute
a pilot. A stronger follow-up needs multiple training seeds, a feature-only model
ablation, deeper removal states and a new final test suite chosen before tuning.
The outputs describe this signed recurrent-map model, not preserved worm behavior
or a proven minimum biological circuit.
