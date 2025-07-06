import pandas as pd
# General use tools
import logging

# Machine learning libraries
from sklearn.feature_selection import SelectKBest, f_regression

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the dummies
# Modify the method `feature_engineering`
# Modify the methof `_apply_feature_engineering` for the prediction phase

def create_structural_break_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dummy variables for major structural breaks/crises
    """
    df_with_dummies = df.copy()
    
    # Ensure we have year column
    if 'Annee' not in df_with_dummies.columns:
        logger.warning("Year column 'Annee' not found. Cannot create crisis dummies.")
        return df_with_dummies
    
    logger.info("Creating structural break dummy variables...")
    
    # Convert year to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_with_dummies['Annee']):
        df_with_dummies['year'] = pd.to_datetime(df_with_dummies['Annee']).dt.year
    else:
        df_with_dummies['year'] = df_with_dummies['Annee'].dt.year
    
    # 1. Financial Crisis (2008-2009)
    df_with_dummies['crisis_2008'] = ((df_with_dummies['year'] >= 2008) & 
                                     (df_with_dummies['year'] <= 2009)).astype(int)
    
    # 2. European Debt Crisis (2010-2012)
    df_with_dummies['crisis_eurozone'] = ((df_with_dummies['year'] >= 2010) & 
                                         (df_with_dummies['year'] <= 2012)).astype(int)
    
    # 3. COVID-19 Pandemic (2020-2022)
    df_with_dummies['crisis_covid'] = ((df_with_dummies['year'] >= 2020) & 
                                      (df_with_dummies['year'] <= 2022)).astype(int)
    
    # 4. Russia-Ukraine War / Energy Crisis (2022-2024)
    df_with_dummies['crisis_ukraine'] = ((df_with_dummies['year'] >= 2022) & 
                                        (df_with_dummies['year'] <= 2024)).astype(int)
    
    # 5. Post-2008 regime (permanent level shift)
    df_with_dummies['post_2008_regime'] = (df_with_dummies['year'] >= 2008).astype(int)
    
    # 6. Post-COVID regime (permanent level shift)
    df_with_dummies['post_covid_regime'] = (df_with_dummies['year'] >= 2020).astype(int)
    
    # 7. Create trend break variables
    self.trend_base_year = df_with_dummies['year'].min()  # Store for prediction consistency
    df_with_dummies['trend'] = df_with_dummies['year'] - self.trend_base_year
    df_with_dummies['trend_post_2008'] = (df_with_dummies['post_2008_regime'] * 
                                         df_with_dummies['trend'])
    df_with_dummies['trend_post_covid'] = (df_with_dummies['post_covid_regime'] * 
                                          df_with_dummies['trend'])
    
    # 8. Volatility regime dummies (periods of high uncertainty)
    df_with_dummies['high_volatility'] = ((df_with_dummies['crisis_2008'] == 1) | 
                                         (df_with_dummies['crisis_eurozone'] == 1) |
                                         (df_with_dummies['crisis_covid'] == 1) |
                                         (df_with_dummies['crisis_ukraine'] == 1)).astype(int)
    
    # Clean up temporary column
    df_with_dummies.drop('year', axis=1, inplace=True)
    
    # Store dummy variable names for later use
    self.dummy_variables = [
        'crisis_2008', 'crisis_eurozone', 'crisis_covid', 'crisis_ukraine',
        'post_2008_regime', 'post_covid_regime', 'trend', 'trend_post_2008',
        'trend_post_covid', 'high_volatility'
    ]
    
    logger.info(f"Created {len(self.dummy_variables)} structural break dummy variables")
    logger.info(f"Dummy variables: {self.dummy_variables}")
    
    return df_with_dummies

def feature_engineering(self):
    """
    Enhanced feature engineering with structural break dummies
    """
    logger.info("="*60)
    logger.info(" Feature engineering with structural breaks")
    logger.info("="*60)
    
    # Create structural break dummies first
    self.df = self.create_structural_break_dummies(self.df)
    
    # Separate features and target (exclude dummy variables from lag/rolling creation)
    base_feature_cols = [col for col in self.df.columns 
                        if col not in [self.target_col, 'Annee'] + 
                        getattr(self, 'dummy_variables', [])]
    
    dummy_cols = getattr(self, 'dummy_variables', [])
    
    X = self.df[base_feature_cols + dummy_cols].copy()
    y = self.df[self.target_col].copy()
    
    # Handle missing values
    X[base_feature_cols] = X[base_feature_cols].fillna(X[base_feature_cols].median())
    y = y.fillna(y.median())
    
    logger.info(f"Base features: {len(base_feature_cols)}")
    logger.info(f"Dummy variables: {len(dummy_cols)}")
    logger.info(f"Total features before engineering: {len(base_feature_cols) + len(dummy_cols)}")
    
    # Initialize feature tracking
    self.lag_features = []
    self.window_features = []
    self.feature_engineering_params.setdefault('created_features', [])
    
    # Create lag features (only for base features, not dummies)
    if ('Annee' in self.df.columns and 
        self.feature_engineering_params.get('lag_periods') and 
        len(self.feature_engineering_params['lag_periods']) > 0):
        
        logger.info("Creating lag features...")
        for col in base_feature_cols:
            for lag in self.feature_engineering_params['lag_periods']:
                lag_col = f"{col}_lag_{lag}"
                X[lag_col] = X[col].shift(lag)
                self.lag_features.append(lag_col)
                self.feature_engineering_params['created_features'].append(lag_col)
    
    # Create rolling window features (only for base features, not dummies)
    if ('Annee' in self.df.columns and 
        self.feature_engineering_params.get('rolling_windows') and 
        len(self.feature_engineering_params['rolling_windows']) > 0):
        
        logger.info("Creating rolling window features...")
        for col in base_feature_cols:
            for window in self.feature_engineering_params['rolling_windows']:
                roll_col = f"{col}_roll_mean_{window}"
                X[roll_col] = X[col].rolling(window=window).mean()
                self.window_features.append(roll_col)
                self.feature_engineering_params['created_features'].append(roll_col)
    
    # Handle max lag
    if self.feature_engineering_params.get('lag_periods'):
        self.max_lag = max(self.feature_engineering_params['lag_periods'])
    else:
        self.max_lag = 0
    
    # Clean NaN values
    initial_rows = len(X)
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Features after engineering: {X.shape[1]}")
    logger.info(f"Samples after cleaning: {len(X)}")
    
    # Feature selection (keeping dummy variables)
    if X.shape[1] > 0:
        max_features = max(10, min(25, len(X) // 12))  # Slightly more features due to dummies
        k_features = min(max_features, X.shape[1])
        
        if k_features < X.shape[1]:
            # Separate dummy variables to ensure they're kept
            dummy_mask = [col in dummy_cols for col in X.columns]
            n_dummies = sum(dummy_mask)
            
            if n_dummies > 0:
                # Select features excluding dummies first
                non_dummy_features = X.loc[:, ~pd.Series(dummy_mask, index=X.columns)]
                k_non_dummy = max(5, k_features - n_dummies)
                
                if k_non_dummy > 0 and k_non_dummy < non_dummy_features.shape[1]:
                    selector = SelectKBest(score_func=f_regression, k=k_non_dummy)
                    X_selected_non_dummy = selector.fit_transform(non_dummy_features, y)
                    selected_non_dummy_features = non_dummy_features.columns[selector.get_support()]
                    
                    # Combine selected features with all dummy variables
                    X_final = pd.concat([
                        pd.DataFrame(X_selected_non_dummy, columns=selected_non_dummy_features, index=X.index),
                        X[dummy_cols]
                    ], axis=1)
                    
                    selected_features = list(selected_non_dummy_features) + dummy_cols
                else:
                    X_final = X
                    selected_features = X.columns
            else:
                selector = SelectKBest(score_func=f_regression, k=k_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
                X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.feature_engineering_params['selected_features'] = selected_features
            self.X = X_final
        else:
            self.X = X
            self.feature_engineering_params['selected_features'] = X.columns
    else:
        raise ValueError("No features available after feature engineering")
    
    self.y = y
    self.feature_names = self.X.columns
    
    # Store training medians
    self._store_training_medians(self.X)
    
    logger.info(f"Feature engineering completed:")
    logger.info(f"  - Total features: {len(self.X.columns)}")
    logger.info(f"  - Dummy variables: {len([f for f in self.X.columns if f in dummy_cols])}")
    logger.info(f"  - Created features: {len(self.feature_engineering_params['created_features'])}")
    
    return self.X, self.y

def _apply_feature_engineering(self, new_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as during training, including structural break dummies"""
    
    logger.info(" Applying feature engineering to new data...")
    
    df = new_data.copy()
    
    # Get feature engineering parameters
    lag_periods = self.feature_engineering_params.get('lag_periods', [])
    rolling_windows = self.feature_engineering_params.get('rolling_windows', [])
    
    logger.info(f" Original new data shape: {df.shape}")
    
    # Step 1: Create structural break dummies (same as training)
    df = self._create_structural_break_dummies_for_prediction(df)
    
    # Step 2: Separate base features (exclude target, year, and dummy variables)
    dummy_cols = getattr(self, 'dummy_variables', [])
    base_feature_cols = [col for col in df.columns 
                        if col not in [self.target_col, 'Annee'] + dummy_cols]
    
    logger.info(f" Base feature columns: {base_feature_cols}")
    logger.info(f" Dummy variables: {dummy_cols}")
    
    # Step 3: Create lag features (only for base features, not dummies)
    if lag_periods and len(lag_periods) > 0:
        logger.info(f" Creating lag features with periods: {lag_periods}")
        
        for col in base_feature_cols:
            for lag in lag_periods:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)
        
        logger.info(f" Created {len(base_feature_cols) * len(lag_periods)} lag features")
    else:
        logger.info(" Skipping lag feature creation (lag_periods is empty)")
    
    # Step 4: Create rolling window features (only for base features, not dummies)
    if rolling_windows and len(rolling_windows) > 0:
        logger.info(f" Creating rolling window features with windows: {rolling_windows}")
        
        for col in base_feature_cols:
            for window in rolling_windows:
                roll_col = f"{col}_roll_mean_{window}"
                df[roll_col] = df[col].rolling(window=window).mean()
        
        logger.info(f" Created {len(base_feature_cols) * len(rolling_windows)} rolling window features")
    else:
        logger.info(" Skipping rolling window feature creation (rolling_windows is empty)")
    
    logger.info(f" Data shape after feature engineering: {df.shape}")
    
    # Step 5: Handle NaN values for predictions
    nan_count_before = df.isnull().sum().sum()
    if nan_count_before > 0:
        logger.warning(f" Found {nan_count_before} NaN values after feature engineering")
        
        # Strategy 1: Forward fill then backward fill
        df_filled = df.fillna(method='ffill').fillna(method='bfill')
        
        # Strategy 2: If still NaN, use median of available data
        remaining_nan = df_filled.isnull().sum().sum()
        if remaining_nan > 0:
            logger.warning(f" {remaining_nan} NaN values remain after forward/backward fill. Using median imputation.")
            
            # Use median from training data if available, otherwise current data
            if hasattr(self, 'training_medians'):
                for col in df_filled.columns:
                    if col in self.training_medians:
                        df_filled[col] = df_filled[col].fillna(self.training_medians[col])
                    else:
                        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            else:
                # Don't impute dummy variables with median - they should be 0 or 1
                non_dummy_cols = [col for col in df_filled.columns if col not in dummy_cols]
                df_filled[non_dummy_cols] = df_filled[non_dummy_cols].fillna(df_filled[non_dummy_cols].median())
                
                # Set dummy variables to 0 if they're NaN (shouldn't happen, but safety check)
                df_filled[dummy_cols] = df_filled[dummy_cols].fillna(0)
        
        df = df_filled
        final_nan_count = df.isnull().sum().sum()
        
        if final_nan_count > 0:
            logger.error(f" Warning: {final_nan_count} NaN values still remain after all imputation strategies")
            logger.error(" Consider providing more historical data or reviewing the feature engineering pipeline")
        else:
            logger.info(" All NaN values successfully handled")
    
    # Step 6: Apply feature selection if it was used during training
    if hasattr(self, 'feature_selector') and self.feature_selector is not None:
        logger.info(" Applying feature selection from training...")
        
        # Get selected features from training
        selected_features = self.feature_engineering_params.get('selected_features', [])
        
        # Check if all selected features are available
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features required for prediction: {missing_features}")
        
        # Select only the features that were selected during training
        df = df[selected_features]
        logger.info(f" Applied feature selection: {len(selected_features)} features selected")
    
    elif hasattr(self, 'feature_names') and self.feature_names is not None:
        # Fallback: use feature names from training if available
        logger.info(" Using feature names from training...")
        
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]
        
        if missing_features:
            logger.warning(f" Missing features: {missing_features}")
        
        if available_features:
            df = df[available_features]
            logger.info(f" Selected {len(available_features)} features based on training")
        else:
            raise ValueError("No training features found in new data")
    
    logger.info(f" Final data shape for prediction: {df.shape}")
    logger.info(f" Final features: {list(df.columns)}")
    
    return df