# Bangladesh Clothes Demand Prediction - Training and Testing Module
# This script trains a Random Forest model to predict clothes demand across Bangladesh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the clothes demand dataset
csv_path = r"bangladesh_clothes_demand.csv"
df = pd.read_csv(csv_path)

# Define numeric columns for data preprocessing
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

# Fill missing values with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Remove outliers using IQR method
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# Create engineered features
df["pop_density_per_1000"] = df["pop_density"] / 1000  # Scale population density
df["year_scaled"] = (df["year"] - df["year"].min()) / (
    df["year"].max() - df["year"].min()
)  # Normalize year to 0-1 range

# Sort data for lag feature creation
df_sorted = df.sort_values(["district", "area", "year"])
clothes_cols = ["shirts", "pants", "jackets", "sarees", "dresses", "coats"]

# Create lag features (previous year demand)
for item in clothes_cols:
    df_sorted[f"{item}_prev"] = df_sorted.groupby(["district", "area"])[item].shift(1)

# Fill missing lag features with median
numeric_cols = num_cols + [f"{item}_prev" for item in clothes_cols]
df_sorted[numeric_cols] = df_sorted[numeric_cols].fillna(
    df_sorted[numeric_cols].median()
)

# Define features and targets for machine learning
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

# Prepare data for cross-validation
X = df_sorted[features]
y = df_sorted[targets]
groups = df_sorted["district"]

# Perform GroupKFold cross-validation to prevent data leakage
gkf = GroupKFold(n_splits=len(df["district"].unique()))
print("=== Per-District Cross-Validation Metrics ===")

# Track performance metrics
all_mae_values = []
all_rmse_values = []
for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    district_name = groups.iloc[test_idx].unique()[0]

    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    print(f"\nDistrict: {district_name}")
    for i, item in enumerate(targets):
        mse = mean_squared_error(y_test[item], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[item], y_pred[:, i])
        r2 = r2_score(y_test[item], y_pred[:, i])
        all_mae_values.append(mae)
        all_rmse_values.append(rmse)
        print(f"{item}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

# Calculate overall performance metrics
avg_mae = np.mean(all_mae_values)
avg_rmse = np.mean(all_rmse_values)

print(f"\n📊 Overall Model Performance:")
print(f"   Average MAE: {avg_mae:.2f} units")
print(f"   Average RMSE: {avg_rmse:.2f} units")

# Train final model on all data
final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
)
final_model.fit(X, y)

print("✅ Model trained successfully!")

# Clothes Demand Predictor Class
# Provides interface for making predictions and visualizing trends
class ClothesPredictor:
    def __init__(self):
        self.model = final_model
        self.df = df_sorted
        self.clothes_items = clothes_cols
        self.avg_mae = avg_mae
        self.avg_rmse = avg_rmse
        
    def predict_item(self, year, item, area):
        """Predict demand for specific item in specific area for given year"""
        # Validate item input
        if item.lower() not in [c.lower() for c in self.clothes_items]:
            return f"❌ Item '{item}' not available. Choose from: {', '.join(self.clothes_items)}"
        
        # Get area-specific demographic data
        area_data = self.df[self.df['area'].str.contains(area, case=False, na=False)]
        
        if area_data.empty:
            # Fallback to overall averages if area not found
            avg_pop = self.df['population'].mean()
            avg_density = self.df['pop_density'].mean()
            print(f"⚠️  Area '{area}' not found. Using average values.")
        else:
            avg_pop = area_data['population'].mean()
            avg_density = area_data['pop_density'].mean()
        
        # Prepare input features for prediction
        pop_density_per_1000 = avg_density / 1000
        year_scaled = (year - self.df['year'].min()) / (self.df['year'].max() - self.df['year'].min())
        
        # Use median values for lag features (previous year demand)
        lag_features = [self.df[f"{c}_prev"].median() for c in self.clothes_items]
        
        # Create feature vector matching training data format
        feature_names = ['population', 'pop_density', 'pop_density_per_1000', 'year_scaled'] + [f"{c}_prev" for c in self.clothes_items]
        prediction_input = pd.DataFrame([[avg_pop, avg_density, pop_density_per_1000, year_scaled] + lag_features], columns=feature_names)
        
        # Generate predictions for all clothing items
        all_predictions = self.model.predict(prediction_input)[0]
        
        # Extract prediction for requested item
        item_index = [c.lower() for c in self.clothes_items].index(item.lower())
        predicted_demand = all_predictions[item_index]
        
        return {
            'item': item,
            'area': area,
            'year': year,
            'predicted_demand': round(max(predicted_demand, 0), 1),  # Ensure non-negative
            'population': int(avg_pop),
            'density': int(avg_density)
        }
    
    def show_trend(self, item, area):
        """Show actual vs predicted demand trend for specific item in specific area from 2010-2025"""
        # Validate item input
        if item.lower() not in [c.lower() for c in self.clothes_items]:
            print(f"❌ Item '{item}' not available.")
            return
        
        # Get area-specific data
        area_data = self.df[self.df['area'].str.contains(area, case=False, na=False)]
        
        if area_data.empty:
            print(f"⚠️  Area '{area}' not found.")
            return
        
        # Extract historical data for visualization
        actual_data = area_data[['year', item]].sort_values('year')
        years = actual_data['year'].values
        actual_values = actual_data[item].values
        
        # Generate model predictions for comparison
        avg_pop = area_data['population'].mean()
        avg_density = area_data['pop_density'].mean()
        item_index = [c.lower() for c in self.clothes_items].index(item.lower())
        
        predicted_values = []
        for year in years:
            # Use year-specific demographics if available
            year_data = area_data[area_data['year'] == year]
            if not year_data.empty:
                year_pop = year_data['population'].iloc[0]
                year_density = year_data['pop_density'].iloc[0]
            else:
                year_pop = avg_pop
                year_density = avg_density
            
            # Prepare features for prediction
            pop_density_per_1000 = year_density / 1000
            year_scaled = (year - self.df['year'].min()) / (self.df['year'].max() - self.df['year'].min())
            lag_features = [self.df[f"{c}_prev"].median() for c in self.clothes_items]
            
            feature_names = ['population', 'pop_density', 'pop_density_per_1000', 'year_scaled'] + [f"{c}_prev" for c in self.clothes_items]
            pred_input = pd.DataFrame([[year_pop, year_density, pop_density_per_1000, year_scaled] + lag_features], columns=feature_names)
            all_preds = self.model.predict(pred_input)[0]
            predicted_values.append(max(all_preds[item_index], 0))
        
        # Create visualization comparing actual vs predicted trends
        plt.figure(figsize=(10, 6))
        plt.plot(years, actual_values, marker='o', linewidth=2, markersize=6, color='blue', label='Actual')
        plt.plot(years, predicted_values, marker='s', linewidth=2, markersize=6, color='red', label='Predicted')
        plt.title(f'{item.title()} Demand: Actual vs Predicted in {area} (2010-2025)', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel(f'{item.title()} Demand (units)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(years[::2])  # Show every 2nd year to avoid crowding
        plt.tight_layout()
        plt.show()

# Interactive User Interface
# Provides command-line interface for making predictions and viewing trends
def user_interface():
    predictor = ClothesPredictor()
    
    print("\n🎯 Bangladesh Clothes Demand Predictor")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Predict demand for specific item")
        print("2. Show demand trend (2010-2025)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            # Single prediction workflow
            year = int(input("Enter year: "))
            item = input("Enter item (shirts/pants/jackets/sarees/dresses/coats): ").strip()
            area = input("Enter area (e.g., Dhaka North): ").strip()
            
            result = predictor.predict_item(year, item, area)
            
            if isinstance(result, dict):
                # Display prediction results
                print(f"\n📊 Prediction Result:")
                print(f"   Item: {result['item'].title()}")
                print(f"   Area: {result['area']}")
                print(f"   Year: {result['year']}")
                print(f"   Predicted Demand: {result['predicted_demand']} units")
                print(f"   Population: {result['population']:,}")
                print(f"   Density: {result['density']:,}/km²")
                print("📈 Model Performance:")
                print(f"   Average MAE: {predictor.avg_mae:.2f} units")
                print(f"   Average RMSE: {predictor.avg_rmse:.2f} units")
            else:
                print(result)  # Error message
        
        elif choice == '2':
            # Trend visualization workflow
            item = input("Enter item (shirts/pants/jackets/sarees/dresses/coats): ").strip()
            area = input("Enter area (e.g., Dhaka North): ").strip()
            predictor.show_trend(item, area)
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

# Execute the interactive interface
user_interface()
