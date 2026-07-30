"""Paired comparison across models. All scripts use RepeatedKFold(5, 10, random_state=42),
so fold i means the same train/test split everywhere and the differences can be paired."""
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

d = Path(__file__).resolve().parent / "data"
names = ["lin", "gbm_default", "gbm_tuned", "gcn"]
scores = {n: np.load(d / f"scores_{n}.npy") for n in names}

print(f"{'model':14s} {'mean R2':>9s} {'std':>7s}")
for n in names:
    print(f"{n:14s} {scores[n].mean():9.3f} {scores[n].std():7.3f}")

print(f"\npaired differences against GCN (n={len(scores['gcn'])} folds):")
for n in names:
    if n == "gcn":
        continue
    diff = scores["gcn"] - scores[n]
    p = wilcoxon(diff).pvalue
    print(f"  GCN - {n:12s} mean {diff.mean():+.3f}   "
          f"GCN wins {int((diff > 0).sum())}/{len(diff)}   p = {p:.4f}")