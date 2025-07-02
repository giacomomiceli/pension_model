# Usage Examples for TimeSeriesPredictor
# Training Phase

import pandas as pd
import logging

from revamp_predictor import TimeSeriesPredictor

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

