def _validate_prediction_data(self, new_data: pd.DataFrame):
    """Validate input data for prediction"""
    # Check required columns
    required_base_cols = ['Y'] + [col for col in self.data_schema.get('columns', []) 
                                    if col.startswith('F')]
    missing_cols = [col for col in required_base_cols if col not in new_data.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    for col in new_data.columns:
        if col in self.data_schema.get('dtypes', {}):
            expected_dtype = self.data_schema['dtypes'][col]
            if new_data[col].dtype != expected_dtype:
                logger.warning(f"Column {col} has dtype {new_data[col].dtype}, "
                                f"expected {expected_dtype}")
    
    logger.info("Input data validation passed")
    
def _apply_feature_engineering(self, new_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as during training"""
    df = new_data.copy()
    
    # Get feature engineering parameters
    lag_periods = self.feature_engineering_params.get('lag_periods', [])
    rolling_windows = self.feature_engineering_params.get('rolling_windows', [])
    
    # Create lag features
    feature_cols = [col for col in df.columns if col.startswith('F')]
    for col in feature_cols:
        for lag in lag_periods:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df[col].shift(lag)
    
    # Create rolling window features
    for col in feature_cols:
        for window in rolling_windows:
            roll_col = f"{col}_roll_mean_{window}"
            df[roll_col] = df[col].rolling(window=window).mean()
    
    # Handle NaN values for predictions
    # For lag features, we need historical data or imputation
    if df.isnull().any().any():
        logger.warning("NaN values found after feature engineering. "
                        "Consider providing more historical data or using imputation")
        
        # Simple forward fill for demonstration - customize as needed
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def get_prediction_requirements(self) -> Dict[str, Any]:
    """
    Get information about what data is needed for predictions
    
    Returns:
        Dictionary with prediction requirements
    """
    if not self.feature_engineering_params:
        return {"error": "No model trained yet"}
    
    return {
        "required_base_columns": ['Y'] + [col for col in self.data_schema.get('columns', []) 
                                        if col.startswith('F')],
        "max_lag_required": self.max_lag,
        "minimum_rows_needed": max(self.max_lag, 
                                    max(self.feature_engineering_params.get('rolling_windows', [1]))),
        "feature_engineering_applied": {
            "lag_periods": self.feature_engineering_params.get('lag_periods'),
            "rolling_windows": self.feature_engineering_params.get('rolling_windows')
        },
        "total_features_created": len(self.feature_names) if self.feature_names else 0
    }