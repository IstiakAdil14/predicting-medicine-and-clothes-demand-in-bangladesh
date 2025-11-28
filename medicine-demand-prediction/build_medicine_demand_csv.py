#!/usr/bin/env python3
"""
Generate synthetic district-level population & medicine demand data for Bangladesh (2010-2025)
with district and area columns, including missing values and anomalies
for cleaning/feature engineering practice.
"""

import pandas as pd
import numpy as np

# --- Districts & corresponding areas ---
districts_areas = {
    "Dhaka": ["Dhaka North", "Dhaka South", "Gazipur", "Narsingdi"],
    "Chittagong": ["Chittagong City", "Cox's Bazar", "Feni", "Comilla"],
    "Khulna": ["Khulna City", "Jessore", "Satkhira", "Bagerhat"],
    "Rajshahi": ["Rajshahi City", "Pabna", "Natore", "Bogra"],
    "Barisal": ["Barisal City", "Patuakhali", "Bhola", "Jhalokathi"],
    "Sylhet": ["Sylhet City", "Moulvibazar", "Habiganj", "Sunamganj"],
    "Rangpur": ["Rangpur City", "Dinajpur", "Thakurgaon", "Lalmonirhat"],
    "Mymensingh": ["Mymensingh City", "Netrokona", "Jamalpur", "Sherpur"],
}

medicine_cols = [
    "antibiotics",
    "painkillers",
    "antacids",
    "vitamins",
    "antihistamines",
    "insulin",
]
years = list(range(2010, 2026))

# --- Generate synthetic data ---
data = []
np.random.seed(42)

for district, areas in districts_areas.items():
    for area in areas:
        base_pop = np.random.randint(100_000, 1_000_000)
        base_density = np.random.uniform(500, 5000)

        for year in years:
            growth_rate = np.random.uniform(0.011, 0.015)
            pop = base_pop * ((1 + growth_rate) ** (year - 2010))

            # Medicine demand (units/year/person) with variation
            medicine_demand = {}
            for med in medicine_cols:
                value = np.random.normal(
                    loc=base_density / 1000 + np.random.uniform(10, 50), scale=5
                )

                # Outliers (5%)
                if np.random.rand() < 0.05:
                    value *= np.random.choice([0.1, 2, 4, 6])

                # Missing values (3%)
                if np.random.rand() < 0.03:
                    value = np.nan

                medicine_demand[med] = max(0, value)

            # Occasional missing population
            population_val = int(pop)
            if np.random.rand() < 0.02:
                population_val = np.nan

            # Occasional missing density
            density_val = round(base_density, 2)
            if np.random.rand() < 0.02:
                density_val = np.nan

            data.append(
                {
                    "district": district,
                    "area": area,
                    "year": year,
                    "population": population_val,
                    "pop_density": density_val,
                    **medicine_demand,
                }
            )

# --- Create DataFrame and save CSV ---
df = pd.DataFrame(data)
csv_path = "medicine-demand-prediction/bangladesh_medicine_demand.csv"
df.to_csv(csv_path, index=False)

print(f"Synthetic medicine demand CSV written to: {csv_path}")
