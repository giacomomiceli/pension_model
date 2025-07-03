import numpy as np
import pandas as pd
import logging

from revamp_predictor import TimeSeriesPredictor

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
    print(f"Predictions for multiple rows: {predictions_multi}")
    
    # Get predictions with confidence (if supported)
    pred_with_conf = new_predictor.predict_new_data(new_data_multi, return_confidence=True)
    if isinstance(pred_with_conf, tuple):
        predictions, confidence = pred_with_conf
        print(f"Predictions: {predictions}")
        print(f"Confidence: {confidence}")
    
    # ================================
    # Scenario 2: Single row prediction (more challenging)
    # ================================
    
    # For single row, you need historical context for lag features
    # Option 1: Provide recent historical data + new row
    historical_context = pd.DataFrame({
        'Y': [2023, 2023, 2023, 2024],  # Including history
        'F1': [1.0, 1.1, 1.15, 1.2],   # Last value is the "new" data
        'F2': [0.7, 0.75, 0.8, 0.85],
        'F3': [1.9, 2.0, 2.05, 2.1],
        'F4': [0.4, 0.45, 0.5, 0.55]
    })
    
    # Predict only the last row (but provide context)
    single_prediction = new_predictor.predict_new_data(historical_context)
    print(f"Single prediction (with context): {single_prediction[-1]}")  # Last prediction
    
    # ================================
    # Scenario 3: Batch prediction from file
    # ================================
    
    # Load new data from file
    new_batch = pd.read_csv('new_predictions_data.csv')  # Must have Y, F1, F2, F3, F4
    
    try:
        batch_predictions = new_predictor.predict_new_data(new_batch)
        
        # Save predictions
        results_df = new_batch.copy()
        results_df['predictions'] = batch_predictions
        results_df.to_csv('predictions_output.csv', index=False)
        print("Batch predictions saved to predictions_output.csv")
        
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        # Check requirements
        req = new_predictor.get_prediction_requirements()
        print("Make sure your data meets these requirements:", req)

# ================================
# Handling Edge Cases
# ================================

def handle_insufficient_data(predictor, new_data):
    """
    Handle cases where new data doesn't have enough history for lag features
    """
    requirements = predictor.get_prediction_requirements()
    min_rows = requirements['minimum_rows_needed']
    
    if len(new_data) < min_rows:
        print(f"Warning: Need at least {min_rows} rows, got {len(new_data)}")
        print("Consider:")
        print("1. Providing more historical data")
        print("2. Using feature imputation")
        print("3. Retraining model with fewer lag features")
        return None
    
    return predictor.predict_new_data(new_data)

# ================================
# Production Monitoring
# ================================

def production_predict_with_monitoring(predictor, new_data):
    """
    Production-ready prediction with monitoring and error handling
    """
    import time
    
    start_time = time.time()
    
    try:
        # Validate data quality
        if new_data.isnull().sum().sum() > 0:
            print("Warning: Input data contains NaN values")
        
        # Make prediction
        predictions = predictor.predict_new_data(new_data, return_confidence=True)
        
        # Log performance
        prediction_time = time.time() - start_time
        print(f"Prediction completed in {prediction_time:.3f} seconds")
        
        # Additional monitoring could include:
        # - Data drift detection
        # - Prediction distribution analysis
        # - Model performance tracking
        
        return predictions
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        # Log error, send alert, etc.
        return None

# Example usage
# predictions = production_predict_with_monitoring(new_predictor, new_data_multi)