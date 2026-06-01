#this script evaluates the performance of simple baseline models (linear regression, gradient boosting) 
#from graph centralities alone
#the R^2 scores here are the "bar" that the GNN model would need to beat

#import packages
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

#paths
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data"

#load and merge the data
severity = pd.read_csv(data_path / "knockout_severity_dataset.csv")
centralities = pd.read_csv(data_path / "neuron_centralities.csv")
df = severity.merge(centralities, on="neuron")
print("merged shape:", df.shape)
print(df.head())

#set up the features and target
features = ["in_degree", "out_degree", "betweenness", "eigenvector", "pagerank"]
x = df[features]
y = df["total_activity_lost"]
print()
print("features:", features)
print("target: total_activity_lost")
print("x shape:", x.shape)
print("y shape:", y.shape)

#linear regression model
#we will fill the NaN values with 0 
#LR cant handle NaN
x_lin = x.fillna(0)
lin = LinearRegression()
lin_scores = cross_val_score(lin, x_lin, y, cv=5, scoring="r2")
print()
print("Linear Regression")
print("R^2 per fold:", np.round(lin_scores, 3))
print("mean R^2:", np.round(lin_scores.mean(), 3), "+/-", np.round(lin_scores.std(), 3))

#gradient boosting model
gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, min_samples_leaf=10, random_state=42)
gb_scores = cross_val_score(gb, x_lin, y, cv=5, scoring="r2")
print()
print("Gradient Boosting")
print("R^2 per fold:", np.round(gb_scores, 3))
print("mean R^2:", np.round(gb_scores.mean(), 3), "+/-", np.round(gb_scores.std(), 3))

#check which features matter most for the gradient boosting model
gb_full = GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, min_samples_leaf=10, random_state=42).fit(x_lin, y)
importances = pd.DataFrame({"feature": features, "importance": gb_full.feature_importances_}).sort_values("importance", ascending=False)
print()
print("Feature importances for Gradient Boosting:")
print(importances.to_string(index=False))


#diagbostic test
#see which neurons are the worst predicted by the linear regression model 
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(LinearRegression(), x_lin, y, cv=5)
df_diag = df.copy()
df_diag["predicted"] = y_pred
df_diag["residual"] = df_diag["total_activity_lost"] - df_diag["predicted"]
df_diag["abs_residual"] = df_diag["residual"].abs()
print()
print("top 10 worst predicted neurons by linear regression:")
worst = df_diag.sort_values("abs_residual", ascending=False).head(10)
print(worst[["neuron", "total_activity_lost", "predicted", "residual", "failure_count"]].to_string(index=False))