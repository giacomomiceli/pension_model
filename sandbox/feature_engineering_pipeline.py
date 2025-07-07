from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline"""
    # Lag features
    lag_periods: List[int] = field(default_factory=list)
    
    # Rolling window features
    rolling_windows: List[int] = field(default_factory=list)
    
    # Feature selection
    max_features: int = 20
    min_features: int = 5
    feature_selection_method: str = 'selectkbest'  # 'selectkbest', 'rfe', 'none'
    
    # Missing value handling
    missing_value_strategy: str = 'median'  # 'median', 'mean', 'forward_fill'
    
    # Advanced features
    polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_features: bool = False
    
    # Scaling
    scale_features: bool = True
    scaler_type: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # Time series specific
    time_column: str = 'Annee'
    target_column: str = 'target'
    
    # Validation
    validate_features: bool = True
    feature_drift_threshold: float = 0.1

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create lag features for time series data"""
    
    def __init__(self, lag_periods: List[int] = None, time_column: str = 'Annee'):
        self.lag_periods = lag_periods or []
        self.time_column = time_column
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for lag features)"""
        self.feature_names_ = [col for col in X.columns if col != self.time_column]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        if not self.lag_periods:
            return X
            
        X_transformed = X.copy()
        
        for col in self.feature_names_:
            for lag in self.lag_periods:
                lag_col = f"{col}_lag_{lag}"
                X_transformed[lag_col] = X_transformed[col].shift(lag)
        
        logger.info(f"Created {len(self.feature_names_) * len(self.lag_periods)} lag features")
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if input_features is None:
            input_features = self.feature_names_
        
        output_features = list(input_features)
        for col in self.feature_names_:
            for lag in self.lag_periods:
                output_features.append(f"{col}_lag_{lag}")
        
        return output_features

class RollingFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create rolling window features"""
    
    def __init__(self, rolling_windows: List[int] = None, time_column: str = 'Annee'):
        self.rolling_windows = rolling_windows or []
        self.time_column = time_column
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer"""
        self.feature_names_ = [col for col in X.columns if col != self.time_column]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        if not self.rolling_windows:
            return X
            
        X_transformed = X.copy()
        
        for col in self.feature_names_:
            for window in self.rolling_windows:
                # Rolling mean
                roll_col = f"{col}_roll_mean_{window}"
                X_transformed[roll_col] = X_transformed[col].rolling(window=window).mean()
                
                # Rolling std (optional)
                roll_std_col = f"{col}_roll_std_{window}"
                X_transformed[roll_std_col] = X_transformed[col].rolling(window=window).std()
        
        logger.info(f"Created {len(self.feature_names_) * len(self.rolling_windows) * 2} rolling features")
        return X_transformed

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values with various strategies"""
    
    def __init__(self, strategy: str = 'median'):
        self.strategy = strategy
        self.fill_values_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Learn fill values from training data"""
        if self.strategy == 'median':
            self.fill_values_ = X.median().to_dict()
        elif self.strategy == 'mean':
            self.fill_values_ = X.mean().to_dict()
        elif self.strategy == 'forward_fill':
            # For forward fill, we don't need to store values
            pass
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value handling"""
        X_transformed = X.copy()
        
        if self.strategy in ['median', 'mean']:
            for col in X_transformed.columns:
                if col in self.fill_values_:
                    X_transformed[col] = X_transformed[col].fillna(self.fill_values_[col])
        elif self.strategy == 'forward_fill':
            X_transformed = X_transformed.fillna(method='ffill').fillna(method='bfill')
        
        return X_transformed

class FeatureValidator(BaseEstimator, TransformerMixin):
    """Validate features for consistency between training and prediction"""
    
    def __init__(self, drift_threshold: float = 0.1):
        self.drift_threshold = drift_threshold
        self.training_stats_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Store training statistics"""
        self.training_stats_ = {
            'feature_names': set(X.columns),
            'means': X.mean().to_dict(),
            'stds': X.std().to_dict(),
            'mins': X.min().to_dict(),
            'maxs': X.max().to_dict()
        }
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate features and check for drift"""
        if not self.training_stats_:
            logger.warning("FeatureValidator not fitted. Skipping validation.")
            return X
        
        # Check for missing features
        missing_features = self.training_stats_['feature_names'] - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Check for feature drift
        for col in X.columns:
            if col in self.training_stats_['means']:
                current_mean = X[col].mean()
                training_mean = self.training_stats_['means'][col]
                
                if abs(current_mean - training_mean) > self.drift_threshold * abs(training_mean):
                    logger.warning(f"Feature drift detected in {col}: "
                                 f"training_mean={training_mean:.3f}, "
                                 f"current_mean={current_mean:.3f}")
        
        return X

class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.pipeline = None
        self.feature_selector = None
        self.is_fitted = False
        
    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline"""
        steps = []
        
        # 1. Missing value handling (initial)
        steps.append(('missing_initial', MissingValueHandler(strategy='forward_fill')))
        
        # 2. Lag features
        if self.config.lag_periods:
            steps.append(('lag_features', LagFeatureTransformer(
                lag_periods=self.config.lag_periods,
                time_column=self.config.time_column
            )))
        
        # 3. Rolling features
        if self.config.rolling_windows:
            steps.append(('rolling_features', RollingFeatureTransformer(
                rolling_windows=self.config.rolling_windows,
                time_column=self.config.time_column
            )))
        
        # 4. Missing value handling (after feature engineering)
        steps.append(('missing_final', MissingValueHandler(strategy=self.config.missing_value_strategy)))
        
        # 5. Feature validation
        if self.config.validate_features:
            steps.append(('validator', FeatureValidator(drift_threshold=self.config.feature_drift_threshold)))
        
        # 6. Scaling
        if self.config.scale_features:
            if self.config.scaler_type == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif self.config.scaler_type == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif self.config.scaler_type == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            steps.append(('scaler', scaler))
        
        return Pipeline(steps)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the feature engineering pipeline"""
        logger.info("Fitting feature engineering pipeline...")
        
        # Build pipeline
        self.pipeline = self._build_pipeline()
        
        # Fit the pipeline
        X_transformed = self.pipeline.fit_transform(X)
        
        # Feature selection (separate from pipeline for flexibility)
        if self.config.feature_selection_method != 'none':
            self._fit_feature_selector(X_transformed, y)
        
        self.is_fitted = True
        logger.info("Feature engineering pipeline fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        logger.info("Transforming data with feature engineering pipeline...")
        
        # Apply main pipeline
        X_transformed = self.pipeline.transform(X)
        
        # Convert back to DataFrame if needed
        if isinstance(X_transformed, np.ndarray):
            feature_names = self._get_feature_names()
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _fit_feature_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selector"""
        if y is None:
            logger.warning("No target provided. Skipping feature selection.")
            return
        
        # Calculate max features based on sample size
        max_features = max(self.config.min_features, 
                          min(self.config.max_features, len(X) // 15))
        k_features = min(max_features, X.shape[1])
        
        if k_features < X.shape[1]:
            logger.info(f"Applying feature selection: {k_features} out of {X.shape[1]} features")
            
            if self.config.feature_selection_method == 'selectkbest':
                self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
                self.feature_selector.fit(X, y)
            
            # Could add other selection methods here (RFE, etc.)
        
    def _get_feature_names(self) -> List[str]:
        """Get feature names from pipeline"""
        # This is a simplified version - in practice, you'd need to track
        # feature names through all transformations
        return [f"feature_{i}" for i in range(self.pipeline.transform(pd.DataFrame()).shape[1])]
    
    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters"""
        return {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'pipeline_steps': list(self.pipeline.named_steps.keys()) if self.pipeline else []
        }

# Usage Example
def create_feature_engineering_pipeline(config: FeatureEngineeringConfig) -> FeatureEngineeringPipeline:
    """Factory function to create pipeline"""
    return FeatureEngineeringPipeline(config)

# Example usage:
if __name__ == "__main__":
    # Create configuration
    config = FeatureEngineeringConfig(
        lag_periods=[1, 2, 3],
        rolling_windows=[3, 5, 7],
        max_features=15,
        feature_selection_method='selectkbest',
        scale_features=True,
        validate_features=True
    )
    
    # Create and use pipeline
    pipeline = create_feature_engineering_pipeline(config)
    
    # Training
    # X_train_transformed = pipeline.fit_transform(X_train, y_train)
    
    # Prediction
    # X_test_transformed = pipeline.transform(X_test)