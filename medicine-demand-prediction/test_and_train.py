#!/usr/bin/env python3
"""
Train & evaluate a Regularized RandomForest model to predict district-level **medicine** demand
with feature engineering, per-district cross-validation, and evaluation metrics.
(This file now follows the SAME STRUCTURE and STYLE as File 2.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load CSV ---
csv_path = Path("medicine-demand-prediction/bangladesh_medicine_demand.csv")
df = pd.read_csv(csv_path)

# --- Feature Engineering & Data Cleaning ---
medicine_cols = [
    "antibiotics",
    "painkillers",
    "antacids",
    "vitamins",
    "antihistamines",
    "insulin",
]
num_cols = ["population", "pop_density"] + medicine_cols

# 1. Handle missing values (median imputation)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 2. Outlier capping (IQR method)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# 3. Additional Features
df["pop_density_per_1000"] = df["pop_density"] / 1000
df["year_scaled"] = (df["year"] - df["year"].min()) / (
    df["year"].max() - df["year"].min()
)

# 4. Lag features: previous year medicine demand per district + area
df_sorted = df.sort_values(["district", "area", "year"])
for med in medicine_cols:
    df_sorted[f"{med}_prev"] = df_sorted.groupby(["district", "area"])[med].shift(1)

# 5. Fill NaNs for lag features using median
lag_cols = [f"{m}_prev" for m in medicine_cols]
df_sorted[lag_cols] = df_sorted[lag_cols].fillna(df_sorted[lag_cols].median())

# --- Features & Targets ---
features = [
    "population",
    "pop_density",
    "pop_density_per_1000",
    "year_scaled",
] + lag_cols
targets = medicine_cols

X = df_sorted[features]
y = df_sorted[targets]
groups = df_sorted["district"]

# --- Grouped Cross-Validation (per district) ---
gkf = GroupKFold(n_splits=len(df_sorted["district"].unique()))
print("=== Per-District Cross-Validation Metrics (Medicine) ===")

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    district_name = groups.iloc[test_idx].unique()[0]

    # --- Train Regularized Random Forest ---
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Predict & Evaluate ---
    y_pred = model.predict(X_test)
    print(f"District: {district_name}")

    for i, med in enumerate(targets):
        mse = mean_squared_error(y_test[med], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[med], y_pred[:, i])
        r2 = r2_score(y_test[med], y_pred[:, i])
        print(f"{med}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

# --- Train final model on full dataset ---
final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
)
final_model.fit(X, y)

# --- Predict on full dataset ---
df_pred = df_sorted.copy()
preds = final_model.predict(X)
for i, med in enumerate(targets):
    df_pred[f"{med}_pred"] = preds[:, i]

# --- Aggregate by district & year ---
actual_agg = df_sorted.groupby(["district", "year"])[targets].sum().reset_index()
pred_agg = (
    df_pred.groupby(["district", "year"])[[f"{m}_pred" for m in targets]]
    .sum()
    .reset_index()
)

# --- Plot per district per medicine ---
for med in targets:
    for district in actual_agg["district"].unique():
        sub_actual = actual_agg[actual_agg["district"] == district]
        sub_pred = pred_agg[pred_agg["district"] == district]

        mse = mean_squared_error(sub_actual[med], sub_pred[f"{med}_pred"])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(sub_actual[med], sub_pred[f"{med}_pred"])
        r2 = r2_score(sub_actual[med], sub_pred[f"{med}_pred"])

        print(f"Final Metrics - District: {district}, Medicine: {med}")
        print(f"MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        plt.figure(figsize=(8, 5))
        plt.plot(
            sub_actual["year"], sub_actual[med], "o--", color="blue", label="Actual"
        )
        plt.plot(
            sub_pred["year"],
            sub_pred[f"{med}_pred"],
            "o-",
            color="orange",
            label="Predicted",
        )
        plt.title(f"{med.capitalize()} Demand: {district} (2010-2025)")
        plt.xlabel("Year")
        plt.ylabel(f"{med.capitalize()} Demand (units)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
