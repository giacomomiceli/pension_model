# Usage Examples for TimeSeriesPredictor
# Training Phase

import pandas as pd
import logging

from pension_model.predictor import TimeSeriesPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the predictor object
predictor = TimeSeriesPredictor(target_col='D_PIB', model_dir='models')

# Load your data
dtypes = {
    "Annee": "int64",
    "D_PIB": "float64"
}
raw_df = pd.read_csv("data/dataset_COR_norm.csv", index_col=False, dtype=dtypes)
raw_df = raw_df.loc[:, ~raw_df.columns.str.contains("^Unnamed")]

#############################################################################
# Remove repeating data from 2002 to 2024
#############################################################################
# Check if the length is a multiple of 69
n_years = 69
assert len(raw_df) % n_years == 0, "Data length must be a multiple of 69"

# Create dictionary of DataFrames
grouped_dfs = {
    f"group_{i}": raw_df.iloc[i*n_years:(i+1)*n_years].reset_index(drop=True)
    for i in range(len(raw_df) // n_years)
}
# Retail only real values
df_historical_data = grouped_dfs["group_0"]
df_historical_data = df_historical_data[df_historical_data['Annee'] < 2024]

# Remaining all scenario data
# Filter each DataFrame, then concatenate
filtered_dfs = [df[df['Annee'] >= 2024] for df in grouped_dfs.values()]
df_scenarios = pd.concat(filtered_dfs, ignore_index=True)

# If you want all unique vales (real data + all scenarions), concatenate `df_historical_data` and `df_scenarios`
raw_df = pd.concat([df_historical_data, df_scenarios], ignore_index=True)

#############################################################################

# Complete training pipeline

## Step 1: Load and explore data
df_clean = predictor.load_and_explore_data(raw_df)

# ## Step 2: Visualize data
# correlation_matrix = predictor.visualize_data()

# # Step 3: Statistical tests
# predictor.statistical_tests()

# Step 4: Feature engineering
X, y = predictor.feature_engineering()

# Step 5: Prepare data
X_scaled, y = predictor.prepare_data_for_modeling()

# Step 6: Train models
results = predictor.train_models()

# Step 7: Evaluate best model
evaluation = predictor.evaluate_model()

# Step 8: Model comparison
comparison = predictor.model_comparison_summary()

# Step 9: Save best model
model_path = predictor.save_best_model("production_model_v1")
print(f" Model saved to: {model_path}")

