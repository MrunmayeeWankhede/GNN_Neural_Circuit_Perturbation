#this is a script to compare the GCN and linear baseline errors neuron by neuron, to see if there are specific neurons where the GCN does better or worse than the linear model
#we wanna see if they are making the same mistakes or different
#
#usage:
#    python error_analysis.py        # current task: motor readout, log target, 6 features, downstream GCN
#    python error_analysis.py --v1   # the original task (README Section 4): global readout, raw target,
#                                    # 5 centralities, unstandardised upstream GCN. Needs connectome_model.py output.

#import packages
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

from train_gnn import load_data, train_one_split, masks_from_indices, FEATURES, project_dir, data_dir

results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--v1", action="store_true", help="reproduce the version-1 error analysis")
args = parser.parse_args()

if args.v1:
    data, nodes = load_data(direction="upstream", severity_csv="knockout_severity_dataset.csv",
                            feature_cols=FEATURES[:5], label_col="total_activity_lost", log_target=False)
    gcn_kwargs = dict(dropout=0.1, standardize=False)
    tag, target_name = "_v1", "total activity lost (v1)"
else:
    data, nodes = load_data(direction="downstream")
    gcn_kwargs = dict(dropout=0.3, standardize=True)
    tag, target_name = "", "log(1 + motor severity)"

features = data.x.numpy()
labels = data.y.numpy()
print("loaded", len(nodes), "neurons,", data.edge_index.shape[1], "edges, target:", target_name)

#both models get the same 5 folds, so residuals are comparable neuron by neuron
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#linear regression - 5 fold CV predictions
print()
print("getting 5-fold CV predictions from linear regression-")
y_pred_lr = cross_val_predict(LinearRegression(), features, labels, cv=kf)
print("Linear regression R^2:", round(r2_score(labels, y_pred_lr), 3))

#GCN - 5 fold CV predictions
print()
print("getting 5-fold GCN predictions...")
y_pred_gcn = np.zeros(len(nodes), dtype=np.float32)
for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(nodes)))):
    train_mask, test_mask = masks_from_indices(len(nodes), train_idx, test_idx)
    y_pred_gcn[test_idx] = train_one_split(data, train_mask, test_mask, seed=fold,
                                           return_pred=True, **gcn_kwargs)
    print("  fold", fold + 1, "done")
print("GCN R^2:", round(r2_score(labels, y_pred_gcn), 3))

#combine results into a comparison table
results = pd.DataFrame({
    "neuron": nodes,
    "actual": labels,
    "lin_pred": y_pred_lr,
    "gcn_pred": y_pred_gcn,
    "lin_residual": labels - y_pred_lr,
    "gcn_residual": labels - y_pred_gcn,
})
results["lin_abs_err"] = results["lin_residual"].abs()
results["gcn_abs_err"] = results["gcn_residual"].abs()
results.to_csv(data_dir / f"error_analysis{tag}.csv", index=False)

#top-10 worst from each model
print()
print("top 10 worst-predicted by LINEAR model:")
print(results.sort_values("lin_abs_err", ascending=False)
      .head(10)[["neuron", "actual", "lin_pred", "gcn_pred"]]
      .to_string(index=False))

print()
print("top 10 worst-predicted by GCN model:")
print(results.sort_values("gcn_abs_err", ascending=False)
      .head(10)[["neuron", "actual", "lin_pred", "gcn_pred"]]
      .to_string(index=False))


#correlation between residuals
corr = results["lin_residual"].corr(results["gcn_residual"])
print()
print(f"correlation of residuals (linear vs GCN): {corr:.3f}")
print("interpretation:")
print("  ~1.0 = same errors, same neurons (no complementary info)")
print("  ~0.5 = partly overlapping errors")
print("  ~0.0 = independent errors (would benefit from ensembling)")


#scatter plot of residuals
plt.figure(figsize=(8, 8))
plt.scatter(results["lin_residual"], results["gcn_residual"],
            alpha=0.5, s=30)
lim = max(results["lin_residual"].abs().max(),
          results["gcn_residual"].abs().max()) * 1.1
plt.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y = x")
plt.axhline(0, color="gray", alpha=0.3)
plt.axvline(0, color="gray", alpha=0.3)
plt.xlabel("Linear regression residual (actual - predicted)")
plt.ylabel("GCN residual (actual - predicted)")
plt.title(f"where each model errs - {target_name} (residual r = {corr:.2f})")
plt.legend()
plt.tight_layout()
plot_path = results_dir / f"residual_comparison{tag}.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print()
print("saved residual comparison plot to", plot_path)
