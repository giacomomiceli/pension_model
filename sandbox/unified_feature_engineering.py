import pandas as pd

# General use tools
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def feature_engineering(self, data: pd.DataFrame = None, is_training: bool = True):
    """
    Unified feature engineering for both training and prediction phases
    
    Args:
        data: Input DataFrame. If None, uses self.df (training phase)
        is_training: Boolean flag to distinguish between training and prediction
    
    Returns:
        tuple: (X, y) for training, X for prediction
    """
    logger.info("="*60)
    logger.info(f" Feature engineering ({'Training' if is_training else 'Prediction'} phase)")
    logger.info("="*60)
    
    # Use provided data or default to training data
    df = data.copy() if data is not None else self.df.copy()
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [self.target_col, 'Annee']]
    X = df[feature_cols].copy()
    
    # Handle target variable (only for training)
    if is_training:
        y = df[self.target_col].copy()
        # Handle missing values in target
        y = y.fillna(y.median())
        # Store training medians for later use in prediction
        self.training_medians = X.median().to_dict()
    else:
        y = None
    
    # Handle missing values in features
    if is_training:
        X = X.fillna(X.median())
    else:
        # Use training medians if available, otherwise current data median
        if hasattr(self, 'training_medians'):
            for col in X.columns:
                if col in self.training_medians:
                    X[col] = X[col].fillna(self.training_medians[col])
                else:
                    X[col] = X[col].fillna(X[col].median())
        else:
            X = X.fillna(X.median())
    
    logger.info(f" Original features: {len(feature_cols)}")
    logger.info(f" Feature names: {feature_cols}")
    
    # Create lag features (only if conditions are met)
    if self._should_create_lag_features(df):
        X = self._create_lag_features(X, feature_cols, is_training)
    else:
        logger.info(" Skipping lag feature creation")
    
    # Create rolling window features (only if conditions are met)
    if self._should_create_rolling_features(df):
        X = self._create_rolling_features(X, feature_cols, is_training)
    else:
        logger.info(" Skipping rolling window feature creation")
    
    # Handle NaN values created by feature engineering
    X, y = self._handle_engineered_nans(X, y, is_training)
    
    # Apply feature selection (training: fit_transform, prediction: transform)
    X = self._apply_feature_selection(X, y, is_training)
    
    # Update instance variables for training phase
    if is_training:
        self.X = X
        self.y = y
        self.feature_names = X.columns
        self._update_max_lag()
        self._log_training_summary()
    
    logger.info(f" Final data shape: {X.shape}")
    logger.info(f" Final features: {list(X.columns)}")
    
    return (X, y) if is_training else X

def _should_create_lag_features(self, df: pd.DataFrame) -> bool:
    """Check if lag features should be created"""
    return (
        'Annee' in df.columns and 
        self.feature_engineering_params.get('lag_periods') and 
        len(self.feature_engineering_params['lag_periods']) > 0
    )

def _should_create_rolling_features(self, df: pd.DataFrame) -> bool:
    """Check if rolling window features should be created"""
    return (
        'Annee' in df.columns and 
        self.feature_engineering_params.get('rolling_windows') and 
        len(self.feature_engineering_params['rolling_windows']) > 0
    )

def _create_lag_features(self, X: pd.DataFrame, feature_cols: list, is_training: bool) -> pd.DataFrame:
    """Create lag features"""
    lag_periods = self.feature_engineering_params['lag_periods']
    logger.info(f" Creating lag features with periods: {lag_periods}")
    
    for col in feature_cols:
        for lag in lag_periods:
            lag_col = f"{col}_lag_{lag}"
            X[lag_col] = X[col].shift(lag)
            
            # Track created features only during training
            if is_training:
                self.lag_features.append(lag_col)
                self.feature_engineering_params['created_features'].append(lag_col)
    
    created_count = len(feature_cols) * len(lag_periods)
    logger.info(f" Created {created_count} lag features")
    return X

def _create_rolling_features(self, X: pd.DataFrame, feature_cols: list, is_training: bool) -> pd.DataFrame:
    """Create rolling window features"""
    rolling_windows = self.feature_engineering_params['rolling_windows']
    logger.info(f" Creating rolling window features with windows: {rolling_windows}")
    
    for col in feature_cols:
        for window in rolling_windows:
            roll_col = f"{col}_roll_mean_{window}"
            X[roll_col] = X[col].rolling(window=window).mean()
            
            # Track created features only during training
            if is_training:
                self.window_features.append(roll_col)
                self.feature_engineering_params['created_features'].append(roll_col)
    
    created_count = len(feature_cols) * len(rolling_windows)
    logger.info(f" Created {created_count} rolling window features")
    return X

def _handle_engineered_nans(self, X: pd.DataFrame, y: pd.Series, is_training: bool) -> tuple:
    """Handle NaN values created by feature engineering"""
    if is_training:
        # Training phase: remove rows with NaN values
        initial_rows = len(X)
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        removed_rows = initial_rows - len(X)
        if removed_rows > 0:
            logger.info(f" Removed {removed_rows} rows due to NaN values from feature engineering")
            
        return X, y
    else:
        # Prediction phase: impute NaN values
        nan_count_before = X.isnull().sum().sum()
        if nan_count_before > 0:
            logger.warning(f" Found {nan_count_before} NaN values after feature engineering")
            
            # Strategy 1: Forward fill then backward fill
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Strategy 2: Use training medians or current median
            remaining_nan = X.isnull().sum().sum()
            if remaining_nan > 0:
                logger.warning(f" {remaining_nan} NaN values remain. Using median imputation.")
                
                if hasattr(self, 'training_medians'):
                    for col in X.columns:
                        if col in self.training_medians:
                            X[col] = X[col].fillna(self.training_medians[col])
                        else:
                            X[col] = X[col].fillna(X[col].median())
                else:
                    X = X.fillna(X.median())
            
            final_nan_count = X.isnull().sum().sum()
            if final_nan_count > 0:
                logger.error(f" Warning: {final_nan_count} NaN values still remain")
            else:
                logger.info(" All NaN values successfully handled")
        
        return X, y

def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series, is_training: bool) -> pd.DataFrame:
    """Apply feature selection"""
    if X.shape[1] == 0:
        raise ValueError("No features available after feature engineering")
    
    if is_training:
        # Training phase: fit feature selector
        max_features = max(5, min(20, len(X) // 15))
        k_features = min(max_features, X.shape[1])
        
        if k_features < X.shape[1]:
            logger.info(f" Applying feature selection: selecting {k_features} out of {X.shape[1]} features")
            
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            
            self.feature_engineering_params['selected_features'] = selected_features
            logger.info(f" Selected features ({len(selected_features)}): {list(selected_features)}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        else:
            logger.info(" No feature selection applied (number of features <= maximum allowed)")
            self.feature_engineering_params['selected_features'] = X.columns
            self.feature_selector = None
            return X
    else:
        # Prediction phase: apply existing feature selector
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            logger.info(" Applying feature selection from training...")
            
            selected_features = self.feature_engineering_params.get('selected_features', [])
            missing_features = [f for f in selected_features if f not in X.columns]
            
            if missing_features:
                raise ValueError(f"Missing features required for prediction: {missing_features}")
            
            # Transform using the fitted selector
            X_selected = self.feature_selector.transform(X[selected_features])
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        elif hasattr(self, 'feature_names') and self.feature_names is not None:
            logger.info(" Using feature names from training...")
            
            available_features = [f for f in self.feature_names if f in X.columns]
            missing_features = [f for f in self.feature_names if f not in X.columns]
            
            if missing_features:
                logger.warning(f" Missing features: {missing_features}")
            
            if available_features:
                return X[available_features]
            else:
                raise ValueError("No training features found in new data")
        
        return X

def _update_max_lag(self):
    """Update max lag for prediction purposes"""
    if self.feature_engineering_params.get('lag_periods'):
        self.max_lag = max(self.feature_engineering_params['lag_periods'])
    else:
        self.max_lag = 0

def _log_training_summary(self):
    """Log training phase summary"""
    total_created = len(self.feature_engineering_params['created_features'])
    total_selected = len(self.feature_engineering_params['selected_features'])
    
    logger.info(f" Feature engineering completed:")
    logger.info(f"   - Created {total_created} new features")
    logger.info(f"   - Selected {total_selected} final features")
    logger.info(f"   - Max lag period: {self.max_lag}")

# Usage examples:
# Training phase:
# X, y = self.feature_engineering(is_training=True)

# Prediction phase:
# X = self.feature_engineering(data=new_data, is_training=False)