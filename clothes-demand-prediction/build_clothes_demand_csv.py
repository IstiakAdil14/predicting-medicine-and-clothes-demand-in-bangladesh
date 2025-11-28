#!/usr/bin/env python3
"""
Clean and preprocess the synthetic Bangladesh clothes-demand dataset.
Includes:
- column standardization
- numeric conversion
- missing value imputation
- anomaly handling
- feature engineering
- output of cleaned CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Paths ---
INPUT_CSV = Path("clothes-demand-prediction/bangladesh_clothes_demand.csv")
OUTPUT_CSV = Path("clothes-demand-prediction/cleaned_clothes_demand.csv")

# --- Clothes columns ---
clothes_cols = ["shirts", "pants", "jackets", "sarees", "dresses", "coats"]
numeric_cols = ["population", "pop_density", "year"] + clothes_cols


# -------------------------------
#            MAIN PIPELINE
# -------------------------------
def main():
    # --- Load dataset ---
    df = pd.read_csv(INPUT_CSV)

    # --- Standardize column names ---
    df.columns = [c.strip() for c in df.columns]

    # --- Convert numeric columns ---
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Impute population ---
    df["population"] = (
        df.groupby(["district", "area"])["population"]
        .transform(lambda x: x.fillna(x.median()))
        .fillna(df["population"].median())
    )

    # --- Impute population density ---
    df["pop_density"] = (
        df.groupby(["district", "area"])["pop_density"]
        .transform(lambda x: x.fillna(x.median()))
        .fillna(df["pop_density"].median())
    )

    # --- Replace impossible clothes values (negative → NaN) ---
    for col in clothes_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # --- Impute clothes-demand columns ---
    for col in clothes_cols:
        if col in df.columns:
            # Median per district-area-year
            df[col] = df.groupby(["district", "area", "year"])[col].transform(
                lambda x: x.fillna(x.median())
            )

            # Fallback: median per district-area
            df[col] = df.groupby(["district", "area"])[col].transform(
                lambda x: x.fillna(x.median())
            )

            # Fallback: global median
            df[col] = df[col].fillna(df[col].median())

    # --- Vulnerability flag ---
    if "vulnerable" in df.columns:
        df["vulnerable_flag"] = df["vulnerable"].astype(bool).astype(int)
    else:
        df["vulnerable_flag"] = (
            (df["pop_density"] < df["pop_density"].median())
            & (df["population"] < df["population"].median())
        ).astype(int)

    # -------------------------
    #     FEATURE ENGINEERING
    # -------------------------
    df["pop_density_per_1000"] = df["pop_density"] / 1000.0

    # year scaled 0–1
    df["year_scaled"] = (df["year"] - df["year"].min()) / (
        df["year"].max() - df["year"].min()
    )

    # Sort for lag features
    df = df.sort_values(["district", "area", "year"])

    # Lagged features (previous year)
    for col in clothes_cols:
        if col in df.columns:
            df[f"{col}_prev"] = df.groupby(["district", "area"])[col].shift(1)
            df[f"{col}_prev"] = df[f"{col}_prev"].fillna(df[f"{col}_prev"].median())

    # --- Save cleaned dataset ---
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Cleaned clothes-demand CSV saved to:\n{OUTPUT_CSV}")


# -------------------------------
if __name__ == "__main__":
    main()
