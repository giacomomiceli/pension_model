import pandas as pd
# General use tools
import logging

# Machine learning libraries
from sklearn.feature_selection import SelectKBest, f_regression

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    df_with_dummies['trend'] = df_with_dummies['year'] - df_with_dummies['year'].min()
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