"""Paired comparison across models. All scripts use RepeatedKFold(5, 10, random_state=42),
so fold i means the same train/test split everywhere and the differences can be paired.

Two p-values are printed:

  wilcoxon   - paired Wilcoxon signed-rank over the 50 fold scores. It treats the folds
               as independent, which they are not (training sets overlap, and the 10
               repeats reuse the same 297 neurons), so it is anti-conservative.
  corrected  - Nadeau & Bengio (2003) corrected resampled t-test, which inflates the
               variance by the test/train ratio to account for that overlap. Quote this one.
"""
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, t as t_dist

d = Path(__file__).resolve().parent / "data"
TEST_TRAIN_RATIO = 1 / 4          # 5-fold CV: each fold tests on 1/5, trains on 4/5

required = ["lin", "gbm_default", "gbm_v1", "gcn"]
optional = ["closed_form", "gcn_upstream"]   # audit_degeneracy.py / train_gnn.py --direction upstream
names = required + [n for n in optional if (d / f"scores_{n}.npy").exists()]
scores = {n: np.load(d / f"scores_{n}.npy") for n in names}


def corrected_ttest(diff):
    n = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return float("nan")
    t = diff.mean() / np.sqrt((1 / n + TEST_TRAIN_RATIO) * var)
    return 2 * t_dist.sf(abs(t), df=n - 1)


print(f"{'model':14s} {'mean R2':>9s} {'std':>7s}")
for n in names:
    print(f"{n:14s} {scores[n].mean():9.3f} {scores[n].std():7.3f}")

print(f"\npaired differences against GCN (n={len(scores['gcn'])} folds):")
for n in names:
    if n == "gcn":
        continue
    diff = scores["gcn"] - scores[n]
    print(f"  GCN - {n:12s} mean {diff.mean():+.3f}   "
          f"GCN wins {int((diff > 0).sum())}/{len(diff)}   "
          f"wilcoxon p = {wilcoxon(diff).pvalue:.4f}   corrected p = {corrected_ttest(diff):.4f}")
