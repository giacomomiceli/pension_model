# Advanced Time series predictive modeling pipeline
# Author: Giacomo Miceli
# Goal: Predict D_PIB using multivariate time series features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# # Advanced models
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.neural_network import MLPRegressor

# Statistical tests and analysis
from scipy import stats
from scipy.stats import jarque_bera, shapiro
# from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
# from statsmodels.tsa.seasonal import seasonal_decompose
# import statsmodels.api as sm

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TimeSeriesPredictor:
    """
    Advanced time series prediction class following ML best practices
    """
    
    def __init__(self, target_col='D_PIB'):
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.scaler = None
        self.feature_names = None
        self.best_model = None
        
    def load_and_explore_data(self, df):
        """
        Data loading and exploratory analysis
        """
        print("="*60)
        print("Data loading and exploratory analysis")
        print("="*60)
        
        self.df = df.copy()
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print("\nMissing values detected:")
            print(missing_data[missing_data > 0])
        else:
            print("\n✓ No missing values detected")
            
        # Data types
        print(f"\nData types:\n{self.df.dtypes}")
        
        # Statistical summary
        print(f"\nTarget variable ({self.target_col}) statistics:")
        target_stats = self.df[self.target_col].describe()
        print(target_stats)
        
        # Detect potential outliers using IQR method
        Q1 = self.df[self.target_col].quantile(0.25)
        Q3 = self.df[self.target_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[(self.df[self.target_col] < Q1 - 1.5*IQR) | 
                          (self.df[self.target_col] > Q3 + 1.5*IQR)]
        print(f"\nPotential outliers in target: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def visualize_data(self):
        """
        Data visualization and pattern analysis
        """
        print("\n" + "="*60)
        print("Data visualization and pattern analysis")
        print("="*60)
        
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
        
        return correlation_matrix
    

    def statistical_tests(self):
        """
        Statistical tests and assumptions
        """
        print("\n" + "="*60)
        print("Statistical tests and assumptions")
        print("="*60)
        
        target_data = self.df[self.target_col].dropna()
        
        # Normality tests
        print("Normality tests:")
        jb_stat, jb_p = jarque_bera(target_data)
        sw_stat, sw_p = shapiro(target_data)
        
        print(f"Jarque-Bera Test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
        print(f"Shapiro-Wilk Test: statistic={sw_stat:.4f}, p-value={sw_p:.4f}")
        
        if jb_p > 0.05 and sw_p > 0.05:
            print("✓ Data appears to be normally distributed")
        else:
            print("⚠ Data may not be normally distributed")
            
        # Stationarity tests (if time series)
        if len(target_data) > 12:  # Minimum observations for meaningful test
            print(f"\nStationary tests:")
            adf_result = adfuller(target_data)
            print(f"Augmented Dickey-Fuller Test:")
            print(f"  ADF Statistic: {adf_result[0]:.4f}")
            print(f"  p-value: {adf_result[1]:.4f}")
            print(f"  Critical Values: {adf_result[4]}")
            
            if adf_result[1] <= 0.05:
                print("✓ Series is stationary (ADF test)")
            else:
                print("⚠ Series may be non-stationary (ADF test)")


    def feature_engineering(self):
        """
        Feature engineering and selection
        """
        print("\n" + "="*60)
        print("Feature engineering and selection")
        print("="*60)
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col not in [self.target_col, 'Annee']]
        X = self.df[feature_cols].copy()
        y = self.df[self.target_col].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        print(f"Original features: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        # Create lag features for time series
        if 'Annee' in self.df.columns:
            print("\nCreating lag features...")
            for col in feature_cols[:5]:  # Create lags for first 5 features to avoid overfitting
                X[f'{col}_lag1'] = X[col].shift(1)
                X[f'{col}_lag2'] = X[col].shift(2)
            
            # Create rolling statistics
            for col in feature_cols[:3]:  # Rolling stats for top 3 features
                X[f'{col}_roll3'] = X[col].rolling(window=3).mean()
                X[f'{col}_roll5'] = X[col].rolling(window=5).mean()
        
        # Remove rows with NaN values created by lag/rolling features
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Features after engineering: {X.shape[1]}")
        print(f"Samples after cleaning: {len(X)}")
        
        # Feature selection using statistical tests
        selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        print(f"Selected features ({len(selected_features)}): {list(selected_features)}")
        
        self.X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        self.y = y
        self.feature_names = selected_features
        
        return self.X, self.y
    
    def prepare_data_for_modeling(self):
        """
        Data preparation and scaling
        """
        print("\n" + "="*60)
        print("Data preparation and scaling")
        print("="*60)
        
        # Time series split (maintain temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        self.cv_splitter = tscv
        
        # Scaling - using RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index)
        
        print(f"Applied RobustScaler to features")
        print(f"Set up TimeSeriesSplit with 5 folds")
        
        return self.X_scaled, self.y


    def train_models(self):
        """
        Model training and comparison
        """
        print("\n" + "="*60)
        print("Model training and comparison")
        print("="*60)
        
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
            print(f"\nTraining {name}...")
            
            try:
                # Hyperparameter tuning for selected models
                if name in param_grids:
                    print(f"  Performing hyperparameter tuning...")
                    grid_search = GridSearchCV(
                        model, param_grids[name], 
                        cv=self.cv_splitter, 
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    grid_search.fit(self.X_scaled, self.y)
                    best_model = grid_search.best_estimator_
                    print(f"  Best parameters: {grid_search.best_params_}")
                else:
                    best_model = model
                
                # Cross-validation
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
                
                print(f"  CV RMSE: {cv_rmse.mean():.6f} (+/- {cv_rmse.std()*2:.6f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        self.results = results
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_rmse_mean'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Champion model: {best_model_name}")
        print(f"   CV RMSE: {results[best_model_name]['cv_rmse_mean']:.6f}")
        
        return results
    
    def evaluate_model(self):
        """
        Model evaluation and diagnostics
        """
        print("\n" + "="*60)
        print("SModel evaluation and diagnostics")
        print("="*60)
        
        # Train best model on full dataset for final evaluation
        self.best_model.fit(self.X_scaled, self.y)
        y_pred = self.best_model.predict(self.X_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        mae = mean_absolute_error(self.y, y_pred)
        r2 = r2_score(self.y, y_pred)
        
        print(f"Final Model Performance ({self.best_model_name}):")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")
        
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
        print("\n" + "="*60)
        print("Model comparison summary")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV_RMSE_Mean': [self.results[model]['cv_rmse_mean'] for model in self.results.keys()],
            'CV_RMSE_Std': [self.results[model]['cv_rmse_std'] for model in self.results.keys()]
        }).sort_values('CV_RMSE_Mean')
        
        print("Model performance ranking:")
        print(comparison_df.to_string(index=False))
        
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
    
    def predict_new_data(self, new_data):
        """
        Make predictions on new data
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Ensure new_data has the same features
        new_data_scaled = self.scaler.transform(new_data[self.feature_names])
        predictions = self.best_model.predict(new_data_scaled)
        
        return predictions
    

# Usage Example
def run_full_pipeline(df):
    """
    Execute the complete modeling pipeline
    """
    predictor = TimeSeriesPredictor(target_col='D_PIB')
    
    # Step 1: Load and explore data
    df_clean = predictor.load_and_explore_data(df)
    
    # Step 2: Visualize data
    correlation_matrix = predictor.visualize_data()
    
    # Step 3: Statistical tests
    predictor.statistical_tests()
    
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
    
    return predictor, results, evaluation, comparison
    

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

     # Step 2: Visualize data
    correlation_matrix = predictor.visualize_data()
    
    # # Step 3: Statistical tests
    # predictor.statistical_tests()
    
    # # Step 4: Feature engineering
    # X, y = predictor.feature_engineering()
    
    # # Step 5: Prepare data
    # X_scaled, y = predictor.prepare_data_for_modeling()
    
    # # Step 6: Train models
    # results = predictor.train_models()
    
    # # Step 7: Evaluate best model
    # evaluation = predictor.evaluate_model()
    
    # # Step 8: Model comparison
    # comparison = predictor.model_comparison_summary()