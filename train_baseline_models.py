#this script evaluates the performance of simple baseline models (linear regression, gradient boosting) 
#from graph centralities alone
#the R^2 scores here are the "bar" that the GNN model would need to beat

#import packages
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

#paths
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data"

#load and merge the data
severity = pd.read_csv(data_path / "knockout_severity_motor.csv")
centralities = pd.read_csv(data_path / "neuron_centralities.csv")
df = severity.merge(centralities, on="neuron")
features = ["in_degree", "out_degree", "betweenness", "eigenvector", "pagerank", "baseline_activity"]
x = df[features].fillna(0) #fill NaN with 0 for the baseline models
y = np.log1p(df["severity"]) #log-transform the severity labels to reduce skew
print("merged shape:", df.shape)
print(df.head())
print("features:", features)
print("target:", "severity (log-transformed)")

#linear regression model
#we will fill the NaN values with 0 
#LR cant handle NaN
lin = LinearRegression()
lin_scores = cross_val_score(lin, x, y, cv=CV, scoring="r2")
print()
print("Linear Regression")
print("R^2 per fold:", np.round(lin_scores, 3))
print("mean R^2:", np.round(lin_scores.mean(), 3), "+/-", np.round(lin_scores.std(), 3))

#gradient boosting model
gb_default = GradientBoostingRegressor(random_state=42)
gb_tuned   = GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                       learning_rate=0.05,
                                       min_samples_leaf=10, random_state=42)
print()
print("Gradient Boosting")
for name, model in [("default", gb_default), ("tuned (v1 settings)", gb_tuned)]:
    s = cross_val_score(model, x, y, cv=CV, scoring="r2")
    print(f"  {name:20s} mean R^2 = {s.mean():.3f} +/- {s.std():.3f}  (n={len(s)})")

    np.save(data_path / f"scores_gbm_{name.split()[0]}.npy", s)

#check which features matter most for the gradient boosting model
gb_full = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, min_samples_leaf=10, random_state=42).fit(x, y)
importances = pd.DataFrame({"feature": features, "importance": gb_full.feature_importances_}).sort_values("importance", ascending=False)
print()
print("Feature importances for Gradient Boosting:")
print(importances.to_string(index=False))


#diagbostic test
#see which neurons are the worst predicted by the linear regression model 
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(LinearRegression(), x, y, cv=KFold(5, shuffle=True, random_state=42))
df_diag = df.copy()
df_diag["log_severity"] = np.log1p(df_diag["severity"])
df_diag["predicted"] = y_pred
df_diag["residual"] = df_diag["log_severity"] - df_diag["predicted"]
df_diag["abs_residual"] = df_diag["residual"].abs()
print()
print("spearman:", round(spearmanr(y, y_pred).statistic, 3))

print("top 10 worst predicted neurons by linear regression:")
worst = df_diag.sort_values("abs_residual", ascending=False).head(10)
print(worst[["neuron", "log_severity", "predicted", "residual"]].to_string(index=False))

np.save(data_path / "scores_lin.npy", lin_scores)
