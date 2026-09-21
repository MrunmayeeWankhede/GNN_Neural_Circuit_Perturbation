# Minimum viable circuit: first experiment

The question is: **how many internal neurons can we remove while keeping the
model's output responses close to those of the intact network?** This is the
next step from predicting single-neuron knockout severity. The first version
establishes baselines before adding a GNN to accelerate candidate selection.

## Run on a Mac

From this repository's directory, with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pruning.txt
python -m unittest discover -s tests -v
python run_pruning.py
open results/pruning_run/fidelity_curve.png
```

The connectome CSV downloads on the first run from a pinned OpenWorm commit.
The default cache is `data/pruning/herm_full_edgelist.csv`; the script checks its
SHA-256. No PyTorch installation or trained model is required for this experiment.
Existing output directories must be empty, so each new experiment needs a new
`--output` path. This avoids overwriting a previous result.

The default searches for up to 50 removals. That is a runtime cap, not a minimum.
To allow the search to continue through all 179 unprotected neurons in this dataset:

```bash
python run_pruning.py --max-removals 179 --output results/pruning_full
```

Runtime depends on hardware and the number of feasible candidates. Exact greedy
search evaluates every remaining candidate at each step and is intentionally
expensive: it provides the reference against which a future GNN can save work.
Sparse matrix operations keep the dynamics small, but no M2 runtime is promised.

You can run offline with an already downloaded OpenWorm-format CSV:

```bash
python run_pruning.py --edge-list data/herm_full_edgelist.csv --output results/pruning_local
```

Custom data is hashed and labeled in the manifest. The default sensory suite and
motor-name rules are specific to this worm model; this is not yet a FlyWire loader.

## What stays fixed

- **Model:** the v2 signed, row-normalized `tanh` recurrent map, gain 1 by default,
  with chemical synapses only. `W[receiver, sender]` retains the original convention.
- **Inputs:** all 25 sensory neurons present in the chemical graph. Every input
  is clamped on every iteration; inactive inputs are clamped to zero.
- **Outputs:** the same 93 motor neurons from the v2 name-based definition.
- **Protected neurons:** the union of inputs and outputs (118 in the pinned data).
- **Weights:** normalized once on the intact graph. Removing a neuron clamps it
  to zero without strengthening surviving synapses or retraining the dynamics.
- **Reference:** the intact network's output vector for each stimulus. Every
  accepted deletion is compared to that reference, not to the previous step.

The v2 single-knockout metric excludes a knocked-out output from its readout.
This experiment instead prohibits output deletion, so the optimization cannot
improve its score by shrinking the population being measured.

## Inputs and error budget

The fitting probes activate one group at amplitude 1, with other inputs at 0:
anterior-touch neurons, ASE/ASH, AWA/AWB/AWC, AFD, the oxygen group, and ADL/ASJ/ASK.
The old sensory list also names PLML and PLMR; they are absent from this chemical
graph. The empty posterior-touch probe is omitted and recorded in the manifest.

Six held-out patterns use mixtures of those groups and all-input activation at
amplitudes 0.5 and 1.0. These are **synthetic model probes**, not measured sensory
responses. Held-out mixtures test a limited form of input generalization within
the same model; they do not establish preserved locomotion or other behavior.

For stimulus `s`, the error is:

```text
sum_outputs(abs(reduced_response[s] - intact_response[s]))
----------------------------------------------------------------
max(sum_outputs(abs(intact_response[s])), number_of_outputs * 1e-6)
```

Default acceptance requires this error to be at most 0.05 for **every fitting
stimulus**. A good mean cannot hide a failing input. The denominator floor gives
an almost-silent intact response an explicit absolute error scale. This metric
still aggregates across outputs: a small aggregate error can hide a large change
in one weakly active motor neuron. Per-output vectors are saved for inspection.

The complete removal order is frozen before evaluating held-out responses.
Held-out errors do not choose removals, tune thresholds, or select a stopping step.
An intact held-out convergence check occurs before search to catch invalid runs.
Do not repeatedly tune the experiment against these same held-out inputs and then
claim they remain an untouched test set.

## Four search methods

| Method | Candidate priority | Re-evaluation |
|---|---|---|
| Greedy | Lowest current worst-input error; mean error and name break ties | Every remaining candidate at every step |
| Static severity | Intact single-knockout error, computed once | First currently feasible candidate in that fixed order |
| Outgoing strength | Lowest absolute outgoing weight in the normalized intact matrix | First currently feasible candidate in that fixed order |
| Random | Fixed random priority order; seeds 42, 43, 44 by default | First currently feasible candidate in that fixed order |

All methods protect the same nodes, use the same stimuli and error threshold, and
check each proposed cumulative deletion with the simulator. Previously infeasible
deletions are retried after a different deletion because inhibition and cancellation
can make error nonmonotonic. The three random runs are descriptive baselines, not
a powered statistical comparison.

The search stops at its removal budget, when no unprotected nodes remain, or when
no single remaining deletion is feasible. It does not search all combinations,
and cannot prove a global minimum. It may also miss beneficial joint deletions
whose individual intermediate steps exceed the error budget.

Convergence uses the v2 L1 state-change tolerance of 1e-9 independently per stimulus.
Nonconvergent candidates are rejected and counted. If a reduced circuit fails on
held-out convergence, the report records a failure and null errors, not a success.

## Reading the outputs

| File | Purpose |
|---|---|
| `manifest.json` | Data checksum/source, code hashes, package versions, model settings, protected nodes, exact input amplitudes |
| `summary.json` | Final sizes, fitting/held-out errors, stop reasons, search cost and convergence failures |
| `curves.csv` | Size and fitting/held-out error at every accepted removal |
| `traces.json` | Removal order, retained neuron names, per-stimulus errors |
| `*_responses.npz` | Output vectors shaped `[step, stimulus, output]`, plus output neuron names |
| `fidelity_curve.png` | Fitting and held-out error versus number of removed neurons |

Search cost counts individual stimulus trajectories, including intact references,
static-ranking evaluations, rejected candidates, and accepted-candidate checks.
Held-out evaluation and response-export work are reported separately. Iteration
counts provide an additional cost measure; wall times depend on the computer.

The manifest lists unprotected neurons with no directed path to the readout. Final
summaries separate their removal from removals of output-connected neurons. Easy
structural exclusions should not be mistaken for discovering a compact functional
core. The graph still includes all 297 neurons at the start.

## What this establishes, and what comes next

The result is a **smaller subnetwork preserving selected responses of this model**.
It inherits the original model's limits: approximate signs, no gap junctions, no
spikes or measured timing, name-based motor labels, and gain-dependent behavior.
Sparse multiplication changes implementation cost, not those assumptions.

Next, train a predictor on *partially pruned graphs* to shortlist candidate deletions,
then verify shortlisted candidates with the same simulator. Compare fidelity and
retained size at matched simulation budgets and across seeds. The old intact-graph
GNN is a useful baseline; its performance cannot be assumed to transfer after
successive removals. Use independent stimuli or circuits for final evaluation.

Only after that baseline is understood should this move to an annotated fly circuit
and anatomical 3D rendering. A graph layout alone would not represent real neuron
positions.
