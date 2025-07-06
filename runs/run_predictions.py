import pandas as pd
import logging

from pension_model.predictor import TimeSeriesPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a new predictor instance
new_predictor = TimeSeriesPredictor()

# Load the saved model
success = new_predictor.load_model("models/production_model_v1")
if success:
    print("Model loaded successfully!")
    
    # Check what data is needed for predictions
    requirements = new_predictor.get_prediction_requirements()
    print("Prediction requirements:", requirements)
    
    # ================================
    # Scenario 1: Multiple rows prediction
    # ================================
    
    # Load your data
    dtypes = {
        "Annee": "int64",
        "D_PIB": "float64"
    }
    raw_df = pd.read_csv("data/dataset_COR_norm.csv", index_col=False, dtype=dtypes)
    new_data_multi = raw_df.loc[:, ~raw_df.columns.str.contains("^Unnamed")]

    
    # Make predictions
    predictions_multi = new_predictor.predict_new_data(new_data_multi)
    print(f"Predictions for multiple rows:\n{predictions_multi}")
    
    # Get predictions with confidence (if supported)
    pred_with_conf = new_predictor.predict_new_data(new_data_multi, return_confidence=True)
    if isinstance(pred_with_conf, tuple):
        predictions, confidence = pred_with_conf
        print(f"Predictions:\n{predictions}")
        print(f"Confidence:\n{confidence}")