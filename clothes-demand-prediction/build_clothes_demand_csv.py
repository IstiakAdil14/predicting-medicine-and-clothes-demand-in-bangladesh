
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path("clothes-demand-prediction/bangladesh_clothes_demand.csv")
OUTPUT_CSV = Path("clothes-demand-prediction/cleaned_clothes_demand.csv")

clothes_cols = ["shirts", "pants", "jackets", "sarees", "dresses", "coats"]
numeric_cols = ["population", "pop_density", "year"] + clothes_cols

def main():
    df = pd.read_csv(INPUT_CSV)

    df.columns = [c.strip() for c in df.columns]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["population"] = (
        df.groupby(["district", "area"])["population"]
        .transform(lambda x: x.fillna(x.median()))
        .fillna(df["population"].median())
    )

    df["pop_density"] = (
        df.groupby(["district", "area"])["pop_density"]
        .transform(lambda x: x.fillna(x.median()))
        .fillna(df["pop_density"].median())
    )

    for col in clothes_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    for col in clothes_cols:
        if col in df.columns:
            df[col] = df.groupby(["district", "area", "year"])[col].transform(
                lambda x: x.fillna(x.median())
            )

            df[col] = df.groupby(["district", "area"])[col].transform(
                lambda x: x.fillna(x.median())
            )

            df[col] = df[col].fillna(df[col].median())

    if "vulnerable" in df.columns:
        df["vulnerable_flag"] = df["vulnerable"].astype(bool).astype(int)
    else:
        df["vulnerable_flag"] = (
            (df["pop_density"] < df["pop_density"].median())
            & (df["population"] < df["population"].median())
        ).astype(int)

    df["pop_density_per_1000"] = df["pop_density"] / 1000.0

    df["year_scaled"] = (df["year"] - df["year"].min()) / (
        df["year"].max() - df["year"].min()
    )

    df = df.sort_values(["district", "area", "year"])

    for col in clothes_cols:
        if col in df.columns:
            df[f"{col}_prev"] = df.groupby(["district", "area"])[col].shift(1)
            df[f"{col}_prev"] = df[f"{col}_prev"].fillna(df[f"{col}_prev"].median())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Cleaned clothes-demand CSV saved to:\n{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
