
import numpy as np
import pandas as pd
import pickle
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

# General use tools
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Statistical tests and analysis
from scipy import stats
from scipy.stats import jarque_bera, shapiro
from statsmodels.tsa.stattools import adfuller

# Machine learning libraries
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesPredictor:
    """
    Advanced Time Series Predictor with production-ready features
    """
    
    def __init__(self, target_col='T', model_dir='models'):
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        
        # Enhanced attributes for production readiness
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Store preprocessing pipeline components
        self.preprocessing_pipeline = {}
        self.feature_engineering_params = {}
        self.data_schema = {}
        self.model_metadata = {}
        
        # For lag features and windowing
        self.lag_features = []
        self.window_features = []
        self.max_lag = 0
        
        # Original data for reference
        self.original_data = None
        self.training_data = None

    def load_and_explore_data(self, df_input):
        """
        Enhanced data loading with validation
        """
        logger.info("="*60)
        logger.info(" Loading and exploring data...")
        logger.info("="*60)

        # Store original data
        self.original_data = df_input.copy()

        # Validate required columns
        required_cols = [self.target_col, 'Annee'] #+ [col for col in df.columns if col.startswith('F')] # check for other needed features, F
        missing_cols = [col for col in required_cols if col not in df_input.columns]
        if missing_cols:
            raise ValueError(f" Missing required columns: {missing_cols}")
        
        # Store data schema
        self.data_schema = {
            'columns': list(df_input.columns),
            'dtypes': df_input.dtypes.to_dict(),
            'shape': df_input.shape,
            'date_range': (df_input['Annee'].min(), df_input['Annee'].max()) if 'Annee' in df_input.columns else None
        }

        logger.info(f" Data loaded: {df_input.shape[0]} rows, {df_input.shape[1]} columns")
        
        self.df = df_input.copy()
        
        # Basic info
        logger.info(f" Data loaded: {df_input.shape[0]} rows, {df_input.shape[1]} columns")
        logger.info(f" Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            logger.info(" Missing values detected:")
            logger.info(missing_data[missing_data > 0])
        else:
            logger.info(" ✓ No missing values detected")
            
        # Data types
        logger.info(f" Data types:\n{self.df.dtypes}")
        
        # Statistical summary
        target_stats = self.df[self.target_col].describe()
        logger.info(f" Target variable ({self.target_col}) statistics:\n{target_stats}")
        
        # Detect potential outliers using IQR method
        Q1 = self.df[self.target_col].quantile(0.25)
        Q3 = self.df[self.target_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[(self.df[self.target_col] < Q1 - 1.5*IQR) | 
                          (self.df[self.target_col] > Q3 + 1.5*IQR)]
        logger.info(f" Potential outliers in target: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
        
        return self.df

    def visualize_data(self):
        """
        Data visualization and pattern analysis
        """

        logger.info("="*60)
        logger.info("Data visualization and pattern analysis...")
        logger.info("="*60)

        # Set up the plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time series plot
        if 'Annee' in self.df.columns:
            axes[0,0].plot(self.df['Annee'], self.df[self.target_col], 'b-', linewidth=2)
            axes[0,0].set_title(f'{self.target_col} Over time')
            axes[0,0].set_xlabel('Year')
            axes[0,0].set_ylabel(self.target_col)
            axes[0,0].grid(True, alpha=0.3)
        
        # Distribution of target variable
        axes[0,1].hist(self.df[self.target_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title(f'Distribution of {self.target_col}')
        axes[0,1].set_xlabel(self.target_col)
        axes[0,1].set_ylabel('Frequency')
        
        # Q-Q plot for normality check
        stats.probplot(self.df[self.target_col], dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot (Normality check)')
        
        # Box plot for outlier detection
        axes[1,1].boxplot(self.df[self.target_col])
        axes[1,1].set_title(f'Box plot of {self.target_col}')
        axes[1,1].set_ylabel(self.target_col)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [x for x in numeric_cols if x != 'Annee']
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

        logger.info(f" Carried out data visualization and returning correlation matrix.")
        
        return correlation_matrix

    def statistical_tests(self):
        """
        Statistical tests and assumptions
        """
        
        logger.info("="*60)
        logger.info(" Perform statistical tests and verify assumptions...")
        logger.info("="*60)
        
        target_data = self.df[self.target_col].dropna()
        
        # Normality tests
        logger.info(" Normality tests:")
        jb_stat, jb_p = jarque_bera(target_data)
        sw_stat, sw_p = shapiro(target_data)
        
        logger.info(f" Jarque-Bera Test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
        logger.info(f" Shapiro-Wilk Test: statistic={sw_stat:.4f}, p-value={sw_p:.4f}")
        
        if jb_p > 0.05 and sw_p > 0.05:
            logger.info(" ✓ Data appears to be normally distributed")
        else:
            logger.info(" ⚠ Data may not be normally distributed")
            
        # Stationarity tests (if time series)
        if len(target_data) > 12:  # Minimum observations for meaningful test
            logger.info(f" Stationary tests:")
            adf_result = adfuller(target_data)
            logger.info(f" Augmented Dickey-Fuller Test:")
            logger.info(f"    ADF Statistic: {adf_result[0]:.4f}")
            logger.info(f"    p-value: {adf_result[1]:.4f}")
            logger.info(f"    Critical Values: {adf_result[4]}")
            
            if adf_result[1] <= 0.05:
                logger.info(" ✓ Series is stationary (ADF test)")
            else:
                logger.info(" ⚠ Series may be non-stationary (ADF test)")

    def feature_engineering(self):
        """
        Feature engineering and selection
        """
        logger.info("="*60)
        logger.info(" Feature engineering and selection")
        logger.info("="*60)

        # Store feature engineering parameters
        self.feature_engineering_params = {
            'lag_periods': [1, 2],     # Example lag periods
            'rolling_windows': [3, 6], # Example rolling windows
            'created_features': [],
            'selected_features': [],
        }
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col not in [self.target_col, 'Annee']]
        X = self.df[feature_cols].copy()
        y = self.df[self.target_col].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        logger.info(f" Original features: {len(feature_cols)}")
        logger.info(f" Feature names: {feature_cols}")
        
        # Create lag features for time series
        if 'Annee' in self.df.columns:
            logger.info(" Creating lag features...")
            for col in feature_cols:  # Create lags for all features to avoid overfitting
                for lag in self.feature_engineering_params['lag_periods']:
                    lag_col = f"{col}_lag_{lag}"
                    X[lag_col] = X[col].shift(lag)
                    self.lag_features.append(lag_col)
                    self.feature_engineering_params['created_features'].append(lag_col)
            
            # Create rolling windows features
            for col in feature_cols:  # Rolling stats for all features
                for window in self.feature_engineering_params['rolling_windows']:
                    roll_col = f"{col}_roll_mean_{window}"
                    X[roll_col] = X[col].rolling(window=window).mean()
                    self.window_features.append(roll_col)
                    self.feature_engineering_params['created_features'].append(roll_col)

        # Update max lag for prediction purposes
        self.max_lag = max(self.feature_engineering_params['lag_periods'])
        
        # Remove rows with NaN values created by lag/rolling features
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        logger.info(f" Features after engineering: {X.shape[1]}")
        logger.info(f" Samples after cleaning: {len(X)}")
        
        # Feature selection using statistical tests
        selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1])) # 15 is currently a magic number, Harrell's rule of thumb maybe better (to be improved)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        self.feature_engineering_params['selected_features'] = selected_features
        
        logger.info(f" Selected features ({len(selected_features)}): {list(selected_features)}")
        
        self.X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        self.y = y
        self.feature_names = selected_features

        logger.info(f" Feature engineering completed.\nCreated {len(self.feature_engineering_params['created_features'])} new features\nSelected {len(self.feature_engineering_params['selected_features'])} features")
        
        return self.X, self.y
    
    def prepare_data_for_modeling(self):
        """
        Data preparation and scaling
        """
        logger.info("="*60)
        logger.info(" Data preparation and scaling...")
        logger.info("="*60)
        
        # Time series split (maintain temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        self.cv_splitter = tscv
        
        # Scaling - using RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index)

        # Store feature names and preprocessing components
        self.feature_names = self.X.columns
        self.preprocessing_pipeline = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_engineering_params': self.feature_engineering_params
        }
        
        logger.info(f" Applied RobustScaler to features")
        logger.info(f" Set up TimeSeriesSplit with 5 folds")

        logger.info(f" Data preparation completed. Features: {len(self.feature_names)}")
        
        return self.X_scaled, self.y
    
    def train_models(self):
        """
        Model training and comparison
        """
        logger.info("="*60)
        logger.info(" Model training and comparison")
        logger.info("="*60)
        
        # Define models with hyperparameter tuning
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Elastic Net': ElasticNet(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            #'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
            #'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            #'SVR': SVR(),
            #'Neural Network': MLPRegressor(random_state=42, max_iter=1000)
        }
        
        # Hyperparameter grids for key models
        param_grids = {
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            # 'XGBoost': {
            #     'n_estimators': [50, 100, 200],
            #     'max_depth': [3, 6, 9],
            #     'learning_rate': [0.01, 0.1, 0.2]
            # }
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f" Training {name}...")
            
            try:
                # Hyperparameter tuning for selected models
                if name in param_grids:
                    logger.info(f"    Performing hyperparameter tuning...")
                    grid_search = GridSearchCV(
                        model, param_grids[name], 
                        cv=self.cv_splitter, 
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    grid_search.fit(self.X_scaled, self.y)
                    best_model = grid_search.best_estimator_
                    logger.info(f"    Best parameters: {grid_search.best_params_}")
                else:
                    logger.info(f"    Set best model, no hyperparameter tuning")
                    best_model = model
                
                # Cross-validation
                logger.info(f"    Performing cross-validation...")
                cv_scores = cross_val_score(
                    best_model, self.X_scaled, self.y, 
                    cv=self.cv_splitter, scoring='neg_mean_squared_error'
                )
                
                cv_rmse = np.sqrt(-cv_scores)
                results[name] = {
                    'model': best_model,
                    'cv_rmse_mean': cv_rmse.mean(),
                    'cv_rmse_std': cv_rmse.std(),
                    'cv_scores': cv_scores
                }
                
                logger.info(f"    CV RMSE: {cv_rmse.mean():.6f} (+/- {cv_rmse.std()*2:.6f})")
                
            except Exception as e:
                logger.info(f"    Error training {name}: {str(e)}")
                continue
        
        self.results = results
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_rmse_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"  🏆 Champion model: {best_model_name}")
        logger.info(f"   CV RMSE: {results[best_model_name]['cv_rmse_mean']:.6f}")
        
        return results

    def evaluate_model(self):
        """
        Model evaluation and diagnostics
        """
        logger.info("="*60)
        logger.info(" Model evaluation and diagnostics...")
        logger.info("="*60)
        
        # Train best model on full dataset for final evaluation
        self.best_model.fit(self.X_scaled, self.y)
        y_pred = self.best_model.predict(self.X_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        mae = mean_absolute_error(self.y, y_pred)
        r2 = r2_score(self.y, y_pred)
        
        logger.info(f" Final Model Performance ({self.best_model_name}):")
        logger.info(f"    RMSE: {rmse:.6f}")
        logger.info(f"    MAE:  {mae:.6f}")
        logger.info(f"    R-squared:   {r2:.6f}")
        
        # Residual analysis
        residuals = self.y - y_pred
        
        # Plot diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Fitted
        axes[0,0].scatter(y_pred, residuals, alpha=0.6)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Fitted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Fitted')
        
        # Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Normal Q-Q Plot of Residuals')
        
        # Histogram of residuals
        axes[1,0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Residuals')
        
        # Actual vs Predicted
        axes[1,1].scatter(self.y, y_pred, alpha=0.6)
        axes[1,1].plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', lw=2)
        axes[1,1].set_xlabel('Actual Values')
        axes[1,1].set_ylabel('Predicted Values')
        axes[1,1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.show()
            
            print("\nTop 5 Most Important Features:")
            print(feature_imp.head().to_string(index=False))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'residuals': residuals,
            'predictions': y_pred
        }
    
    def model_comparison_summary(self):
        """
        Model comparison summary
        """
        logger.info("="*60)
        logger.info(" Model comparison summary...")
        logger.info("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV_RMSE_Mean': [self.results[model]['cv_rmse_mean'] for model in self.results.keys()],
            'CV_RMSE_Std': [self.results[model]['cv_rmse_std'] for model in self.results.keys()]
        }).sort_values('CV_RMSE_Mean')
        
        logger.info(f" Model performance ranking:\n{comparison_df.to_string(index=False)}")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        models = comparison_df['Model']
        means = comparison_df['CV_RMSE_Mean']
        stds = comparison_df['CV_RMSE_Std']
        
        bars = plt.bar(range(len(models)), means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation RMSE')
        plt.title('Model Performance Comparison')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # Highlight best model
        bars[0].set_color('gold')
        bars[0].set_edgecolor('orange')
        bars[0].set_linewidth(2)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df

    def save_best_model(self, model_name: Optional[str] = None, include_metadata: bool = True) -> str:
        """
        Save the best model with all preprocessing components and metadata
        
        Args:
            model_name: Custom name for the model. If None, uses timestamp
            include_metadata: Whether to save model metadata
            
        Returns:
            str: Path to saved model directory
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run train_models() and evaluate_model() first")
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"best_model_{timestamp}"
        
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        try:
            # Save the model
            model_file = model_path / "model.pkl"
            joblib.dump(self.best_model, model_file)
            logger.info(f" Model saved to {model_file}")
            
            # Save preprocessing pipeline
            pipeline_file = model_path / "preprocessing_pipeline.pkl"
            joblib.dump(self.preprocessing_pipeline, pipeline_file)
            logger.info(f" Preprocessing pipeline saved to {pipeline_file}")
            
            # Save metadata if requested
            if include_metadata:
                metadata = {
                    'model_name': model_name,
                    'best_model_name': self.best_model_name,
                    'target_column': self.target_col,
                    'feature_names': self.feature_names,
                    'data_schema': self.data_schema,
                    'feature_engineering_params': self.feature_engineering_params,
                    'model_results': self.results,
                    'max_lag': self.max_lag,
                    'created_at': datetime.now().isoformat(),
                    'python_version': f"{pd.__version__}|{np.__version__}",
                }
                
                metadata_file = model_path / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f" Metadata saved to {metadata_file}")
            
            logger.info(f" Complete model package saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f" Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """
        Load a saved model with all preprocessing components
        
        Args:
            model_path: Path to the saved model directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f" Model path does not exist: {model_path}")
        
        try:
            # Load the model
            model_file = model_path / "model.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f" Model file not found: {model_file}")
            
            self.best_model = joblib.load(model_file)
            logger.info(f" Model loaded from {model_file}")
            
            # Load preprocessing pipeline
            pipeline_file = model_path / "preprocessing_pipeline.pkl"
            if pipeline_file.exists():
                self.preprocessing_pipeline = joblib.load(pipeline_file)
                
                # Restore components from pipeline
                self.scaler = self.preprocessing_pipeline.get('scaler')
                self.feature_names = self.preprocessing_pipeline.get('feature_names')
                self.feature_engineering_params = self.preprocessing_pipeline.get(
                    'feature_engineering_params', {}
                )
                
                logger.info(f" Preprocessing pipeline loaded from {pipeline_file}")
            else:
                logger.warning(" No preprocessing pipeline found")
            
            # Load metadata if available
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.target_col = metadata.get('target_column', self.target_col)
                self.best_model_name = metadata.get('best_model_name')
                self.data_schema = metadata.get('data_schema', {})
                self.max_lag = metadata.get('max_lag', 0)
                
                logger.info(f"Metadata loaded from {metadata_file}")
            else:
                logger.warning("No metadata found")
            
            logger.info("Model loading completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict_new_data(self, new_data: pd.DataFrame, return_confidence: bool = False) -> Union[np.ndarray, tuple]:
        """
        Predict on new data with proper preprocessing handling
        
        Args:
            new_data: DataFrame with same structure as training data
            return_confidence: Whether to return confidence intervals (if supported)
            
        Returns:
            Predictions array or (predictions, confidence_intervals) tuple
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Use load_model() or train a model first")
        
        if self.scaler is None or self.feature_names is None:
            raise ValueError("Preprocessing pipeline not available. Load complete model package")
        
        logger.info(f" Predicting on {len(new_data)} samples")
        
        try:
            # Validate input data
            self._validate_prediction_data(new_data)
            
            # Apply feature engineering
            processed_data = self._apply_feature_engineering(new_data)
            
            # Select and order features
            if not all(feature in processed_data.columns for feature in self.feature_names):
                missing_features = [f for f in self.feature_names if f not in processed_data.columns]
                raise ValueError(f" Missing features in processed data: {missing_features}")
            
            X_new = processed_data[self.feature_names]
            
            # Apply scaling
            X_scaled = self.scaler.transform(X_new)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X_new.index)
            
            # Make predictions
            predictions = self.best_model.predict(X_scaled)
            
            # Handle confidence intervals if requested and supported
            if return_confidence:
                if hasattr(self.best_model, 'predict_proba'):
                    # For classification models
                    probabilities = self.best_model.predict_proba(X_scaled)
                    confidence = np.max(probabilities, axis=1)
                    return predictions, confidence
                elif hasattr(self.best_model, 'predict') and hasattr(self.best_model, 'score'):
                    # For regression models - return predictions with dummy confidence
                    confidence = np.ones(len(predictions)) * 0.95  # Placeholder
                    return predictions, confidence
                else:
                    logger.warning("Model doesn't support confidence intervals")
                    return predictions, None
            
            logger.info(f" Predictions completed for {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            logger.error(f" Error during prediction: {str(e)}")
            raise

    def _validate_prediction_data(self, new_data: pd.DataFrame):
        """Validate input data for prediction"""
        # Check required columns
        required_base_cols = ['Annee'] + [col for col in self.data_schema.get('columns', [])]
        missing_cols = [col for col in required_base_cols if col not in new_data.columns]
        
        if missing_cols:
            raise ValueError(f" Missing required columns: {missing_cols}")
        
        # Check data types
        for col in new_data.columns:
            if col in self.data_schema.get('dtypes', {}):
                expected_dtype = self.data_schema['dtypes'][col]
                if new_data[col].dtype != expected_dtype:
                    logger.warning(f" Column {col} has dtype {new_data[col].dtype}, "
                                    f"expected {expected_dtype}")
        
        logger.info(" Input data validation passed")
    
    def _apply_feature_engineering(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as during training"""
        df = new_data.copy()
        
        # Get feature engineering parameters
        lag_periods = self.feature_engineering_params.get('lag_periods', [])
        rolling_windows = self.feature_engineering_params.get('rolling_windows', [])
        
        # Create lag features
        feature_cols = [col for col in df.columns]
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
            logger.warning(" NaN values found after feature engineering. "
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
            "required_base_columns": [col for col in self.data_schema.get('columns', [])],
            "max_lag_required": self.max_lag,
            "minimum_rows_needed": max(self.max_lag, 
                                        max(self.feature_engineering_params.get('rolling_windows', [1]))),
            "feature_engineering_applied": {
                "lag_periods": self.feature_engineering_params.get('lag_periods'),
                "rolling_windows": self.feature_engineering_params.get('rolling_windows')
            },
            "total_features_created_and_selected": len(self.feature_names)
        }

if __name__ == "__main__":

    dtypes = {
        "Annee": "int64",
        "D_PIB": "float64"
    }

    raw_df = pd.read_csv("data/dataset_COR_norm.csv", index_col=False, dtype=dtypes)
    raw_df = raw_df.loc[:, ~raw_df.columns.str.contains("^Unnamed")]
    
    # Instantiate the predictor object
    predictor = TimeSeriesPredictor(target_col='D_PIB')

    # Step 1: Load and explore data
    df_clean = predictor.load_and_explore_data(raw_df)

    # # Step 2: Visualize data
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
    predictor.save_best_model()