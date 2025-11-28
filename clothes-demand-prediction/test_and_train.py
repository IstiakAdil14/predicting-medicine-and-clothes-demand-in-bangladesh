
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

csv_path = r"clothes-demand-prediction/bangladesh_clothes_demand.csv"
df = pd.read_csv(csv_path)

num_cols = [
    "population",
    "pop_density",
    "shirts",
    "pants",
    "jackets",
    "sarees",
    "dresses",
    "coats",
]

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

df["pop_density_per_1000"] = df["pop_density"] / 1000
df["year_scaled"] = (df["year"] - df["year"].min()) / (
    df["year"].max() - df["year"].min()
)

df_sorted = df.sort_values(["district", "area", "year"])
clothes_cols = ["shirts", "pants", "jackets", "sarees", "dresses", "coats"]

for item in clothes_cols:
    df_sorted[f"{item}_prev"] = df_sorted.groupby(["district", "area"])[item].shift(1)

numeric_cols = num_cols + [f"{item}_prev" for item in clothes_cols]
df_sorted[numeric_cols] = df_sorted[numeric_cols].fillna(
    df_sorted[numeric_cols].median()
)

features = [
    "population",
    "pop_density",
    "pop_density_per_1000",
    "year_scaled",
    "shirts_prev",
    "pants_prev",
    "jackets_prev",
    "sarees_prev",
    "dresses_prev",
    "coats_prev",
]

targets = clothes_cols

X = df_sorted[features]
y = df_sorted[targets]
groups = df_sorted["district"]

gkf = GroupKFold(n_splits=len(df["district"].unique()))
print("=== Per-District Cross-Validation Metrics ===")

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    district_name = groups.iloc[test_idx].unique()[0]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nDistrict: {district_name}")
    for i, item in enumerate(targets):
        mse = mean_squared_error(y_test[item], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[item], y_pred[:, i])
        r2 = r2_score(y_test[item], y_pred[:, i])
        print(f"{item}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
)
final_model.fit(X, y)

df_pred = df_sorted.copy()
df_pred[targets] = final_model.predict(X)

df_actual = df_sorted.groupby(["district", "year"])[targets].sum().reset_index()
df_district = df_pred.groupby(["district", "year"])[targets].sum().reset_index()

for item in targets:
    for district in df_actual["district"].unique():
        subset_actual = df_actual[df_actual["district"] == district]
        subset_pred = df_district[df_district["district"] == district]

        mse = mean_squared_error(subset_actual[item], subset_pred[item])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(subset_actual[item], subset_pred[item])
        r2 = r2_score(subset_actual[item], subset_pred[item])
        print(f"\nFinal Metrics - District: {district}, Item: {item}")
        print(f"MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        plt.figure(figsize=(8, 5))
        plt.plot(
            subset_actual["year"],
            subset_actual[item],
            "o--",
            color="blue",
            label="Actual",
        )
        plt.plot(
            subset_pred["year"],
            subset_pred[item],
            "o-",
            color="orange",
            label="Predicted",
        )
        plt.title(f"{item.capitalize()} Demand: {district} (2010-2025)")
        plt.xlabel("Year")
        plt.ylabel(f"{item.capitalize()} Demand (units/year)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
