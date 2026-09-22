# First trained pruning GNN

The GNN predicts cumulative knockout error well on held-out removal states, but
does not produce a smaller circuit than static severity in this pilot. It uses
fewer online simulations to reach its own stopping point. Offline training costs
are substantial relative to either individual pruning run.

## Reproduce

```bash
python -m pip install -r requirements-pruning-gnn.txt
python train_pruning_gnn.py --edge-list data/herm_full_edgelist.csv --output results/pruning_gnn_reproduction
python -m unittest discover -s tests -v
```

Without `--edge-list`, the pinned source downloads automatically. The existing
pilot is preserved; the command writes a new run. The tests audit the curated
pilot, including checkpoint reload and predictions, rather than the new directory.

## Training

The 297-neuron worm model has 118 protected inputs/outputs. Training uses the six
fitting patterns from the original pruning experiment and one model seed, 42.
Whole random-priority removal trajectories are split before training. The intact
graph appears only in training; additional states have 8, 16 or 24 removals.
Identical removal sets do not cross splits. All candidate labels converged.

| Split | Circuit states | Candidate labels | R-squared on log target | Spearman |
|---|---:|---:|---:|---:|
| Training | 19 | 3,113 | 0.8898 | 0.9314 |
| Validation | 6 | 978 | 0.9334 | 0.9490 |
| Test | 6 | 978 | 0.9256 | 0.9360 |

The target is `log1p(worst_fitting_error / 0.05)`. R-squared is not a percentage
accuracy and is not directly comparable with the original intact-graph GNN's
scores. These observations are related states of the same graph, with recurring
neuron identities, not independent animals or unseen connectomes. Training
includes the intact graph while validation/test contain only partially pruned
graphs, so their score distributions differ.

The best validation checkpoint was epoch 300 of 300. Label generation used
96,012 individual stimulus trajectories and took 250.48 seconds on the training
machine. Network fitting took another 11.43 seconds. These costs are separate
from the online search costs below. No hyperparameter search was performed after
viewing test results.

## Simulator-verified pruning

Both methods had a cap of 550 online simulator calls. Each call evaluates all six
fitting stimuli, and the count includes intact references, static ranking and
rejected proposals. The GNN recomputes its candidate order after every accepted
removal. All accepted removals satisfy the 5% worst fitting-input error limit.

| Method | Removed | Retained | Online calls | Worst fitting error | Worst fresh held-out error |
|---|---:|---:|---:|---:|---:|
| Static severity | 48 | 249 | 550 | 4.9716% | 3.8254% |
| GNN | 37 | 260 | 198 | 4.9996% | 3.2281% |

Static severity reached the call cap. The GNN stopped because no further single
removal was feasible in its current circuit, despite unused budget. Online wall
times were 8.95 seconds and 3.32 seconds respectively. Fewer calls at a different
final circuit size are not evidence of equal-quality acceleration.

Both final circuits stayed below 5% on all 32 fresh held-out inputs. These include
16 random group mixtures and 16 independently varied sensory patterns, with seed
20260922 fixed before evaluation. These are different from the original pilot's
six held-out inputs; their errors should not be compared as if the suites were
identical. The lower GNN error also accompanies fewer removals.

This result illustrates why regression quality alone is insufficient: a different
removal order can reach a less compact circuit even when its candidate predictions
are accurate on sampled states. Training states also stop at 24 removals, while
the deployed pruning policy travels beyond that depth.

## Evidence and next experiment

The directory contains all 31 labeled states, the trained `model.pt`, feature
normalization, per-split predictions, loss history, removal traces and final motor
response vectors. The manifest records the pre-commit working tree and hashes of
the exact implementation files used. Checkpoint reload reproduces saved test
predictions; the tests verify split isolation, training-only preprocessing and
reported held-out errors from the saved vectors.

This is one training seed on one connectome. The next study should compare
multiple training seeds and a feature-only predictor, add deeper training states,
and evaluate on a new final test suite chosen before tuning. This pilot does not
establish a GNN advantage or preserved biological behavior.

See [the training guide](../../docs/pruning_gnn.md) for model and protocol details.
