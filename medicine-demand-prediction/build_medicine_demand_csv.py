# Bangladesh Medicine Demand Data Preprocessing Pipeline
# This script cleans and prepares raw medicine demand data for machine learning

import pandas as pd
import numpy as np
from pathlib import Path

# Define input and output file paths
INPUT_CSV = Path("bangladesh_medicine_demand.csv")
OUTPUT_CSV = Path("cleaned_medicine_demand.csv")

# Define medicine types and numeric columns for processing
medicine_cols = ["antibiotics", "painkillers", "antacids", "vitamins", "antihistamines", "insulin"]
numeric_cols = ["population", "pop_density", "year"] + medicine_cols

def main():
    # Load raw data
    df = pd.read_csv(INPUT_CSV)

    # Clean column names (remove whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Convert columns to numeric, handling invalid values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing population data using hierarchical approach
    # First try district-area median, then overall median
    df["population"] = (
        df.groupby(["district", "area"])["population"]
        .transform(lambda x: x.fillna(x.median()))
        .fillna(df["population"].median())
    )

    # Fill missing population density using same hierarchical approach
    df["pop_density"] = (
        df.groupby(["district", "area"])["pop_density"]
        .transform(lambda x: x.fillna(x.median()))
        .fillna(df["pop_density"].median())
    )

    # Remove negative values in medicine demand (data quality issue)
    for col in medicine_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Fill missing medicine demand using hierarchical median imputation
    for col in medicine_cols:
        if col in df.columns:
            # Try district-area-year median first
            df[col] = df.groupby(["district", "area", "year"])[col].transform(
                lambda x: x.fillna(x.median())
            )

            # Then district-area median
            df[col] = df.groupby(["district", "area"])[col].transform(
                lambda x: x.fillna(x.median())
            )

            # Finally overall median
            df[col] = df[col].fillna(df[col].median())

    # Create vulnerability flag (areas with low population and density)
    if "vulnerable" in df.columns:
        df["vulnerable_flag"] = df["vulnerable"].astype(bool).astype(int)
    else:
        df["vulnerable_flag"] = (
            (df["pop_density"] < df["pop_density"].median())
            & (df["population"] < df["population"].median())
        ).astype(int)

    # Create engineered features for better model performance
    df["pop_density_per_1000"] = df["pop_density"] / 1000.0  # Scale density

    # Normalize year to 0-1 range for better model convergence
    df["year_scaled"] = (df["year"] - df["year"].min()) / (
        df["year"].max() - df["year"].min()
    )

    # Sort data chronologically for lag feature creation
    df = df.sort_values(["district", "area", "year"])

    # Create lag features (previous year demand) for time series modeling
    for col in medicine_cols:
        if col in df.columns:
            df[f"{col}_prev"] = df.groupby(["district", "area"])[col].shift(1)
            # Fill missing lag values with median
            df[f"{col}_prev"] = df[f"{col}_prev"].fillna(df[f"{col}_prev"].median())

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Cleaned medicine-demand CSV saved to:\n{OUTPUT_CSV}")

# Execute preprocessing pipeline
if __name__ == "__main__":
    main()
