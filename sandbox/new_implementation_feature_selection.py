# Advanced Time Series Predictive Modeling Pipeline
# Author: Data Science Best Practices Implementation
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

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

# Statistical tests and analysis
from scipy import stats
from scipy.stats import jarque_bera, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TimeSeriesPredictor:
    """
    Advanced Time Series Prediction Class following ML best practices
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
        Step 1: Data Loading and Exploratory Analysis
        """
        print("="*60)
        print("STEP 1: DATA EXPLORATION AND ANALYSIS")
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
        Step 2: Data Visualization and Pattern Analysis
        """
        print("\n" + "="*60)
        print("STEP 2: DATA VISUALIZATION")
        print("="*60)
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Time series plot
        if 'Annee' in self.df.columns:
            axes[0,0].plot(self.df['Annee'], self.df[self.target_col], 'b-', linewidth=2)
            axes[0,0].set_title(f'{self.target_col} Over Time')
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
        axes[1,0].set_title('Q-Q Plot (Normality Check)')
        
        # Box plot for outlier detection
        axes[1,1].boxplot(self.df[self.target_col])
        axes[1,1].set_title(f'Box Plot of {self.target_col}')
        axes[1,1].set_ylabel(self.target_col)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
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
        Step 3: Statistical Tests and Assumptions
        """
        print("\n" + "="*60)
        print("STEP 3: STATISTICAL TESTS")
        print("="*60)
        
        target_data = self.df[self.target_col].dropna()
        
        # Normality tests
        print("NORMALITY TESTS:")
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
            print(f"\nSTATIONARITY TESTS:")
            adf_result = adfuller(target_data)
            print(f"Augmented Dickey-Fuller Test:")
            print(f"  ADF Statistic: {adf_result[0]:.4f}")
            print(f"  p-value: {adf_result[1]:.4f}")
            print(f"  Critical Values: {adf_result[4]}")
            
            if adf_result[1] <= 0.05:
                print("✓ Series is stationary (ADF test)")
            else:
                print("⚠ Series may be non-stationary (ADF test)")
    
    def comprehensive_feature_analysis(self):
        """
        Comprehensive feature analysis to identify most significant predictors
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE FEATURE ANALYSIS")
        print("="*60)
        
        feature_cols = [col for col in self.df.columns if col not in [self.target_col, 'Annee']]
        X_analysis = self.df[feature_cols].fillna(self.df[feature_cols].median())
        y_analysis = self.df[self.target_col].fillna(self.df[self.target_col].median())
        
        analysis_results = {}
        
        # 1. Correlation Analysis
        correlations = X_analysis.corrwith(y_analysis).abs().sort_values(ascending=False)
        analysis_results['correlations'] = correlations
        
        # 2. Mutual Information (captures non-linear relationships)
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X_analysis, y_analysis, random_state=42)
        mi_results = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
        analysis_results['mutual_info'] = mi_results
        
        # 3. F-statistics (linear relationships)
        from sklearn.feature_selection import f_regression
        f_stats, f_pvalues = f_regression(X_analysis, y_analysis)
        f_results = pd.DataFrame({
            'Feature': feature_cols,
            'F_Statistic': f_stats,
            'P_Value': f_pvalues
        }).sort_values('F_Statistic', ascending=False)
        analysis_results['f_statistics'] = f_results
        
        # 4. Univariate Linear Regression R²
        from sklearn.linear_model import LinearRegression
        univariate_r2 = {}
        for col in feature_cols:
            lr = LinearRegression()
            lr.fit(X_analysis[[col]], y_analysis)
            r2 = lr.score(X_analysis[[col]], y_analysis)
            univariate_r2[col] = r2
        
        univariate_r2_series = pd.Series(univariate_r2).sort_values(ascending=False)
        analysis_results['univariate_r2'] = univariate_r2_series
        
        # 5. Create comprehensive ranking
        # Normalize all scores to 0-1 scale for comparison
        norm_corr = (correlations - correlations.min()) / (correlations.max() - correlations.min())
        norm_mi = (mi_results - mi_results.min()) / (mi_results.max() - mi_results.min())
        norm_f = (f_results.set_index('Feature')['F_Statistic'] - f_results['F_Statistic'].min()) / (f_results['F_Statistic'].max() - f_results['F_Statistic'].min())
        norm_r2 = (univariate_r2_series - univariate_r2_series.min()) / (univariate_r2_series.max() - univariate_r2_series.min())
        
        # Composite score (equal weighting)
        composite_score = (norm_corr + norm_mi + norm_f + norm_r2) / 4
        composite_ranking = composite_score.sort_values(ascending=False)
        analysis_results['composite_ranking'] = composite_ranking
        
        # Create summary table
        summary_df = pd.DataFrame({
            'Feature': feature_cols,
            'Correlation': [correlations[col] for col in feature_cols],
            'Mutual_Info': [mi_results[col] for col in feature_cols],
            'F_Statistic': [f_results[f_results['Feature']==col]['F_Statistic'].iloc[0] for col in feature_cols],
            'P_Value': [f_results[f_results['Feature']==col]['P_Value'].iloc[0] for col in feature_cols],
            'Univariate_R2': [univariate_r2_series[col] for col in feature_cols],
            'Composite_Score': [composite_score[col] for col in feature_cols]
        }).sort_values('Composite_Score', ascending=False)
        
        print("FEATURE IMPORTANCE RANKING:")
        print("="*50)
        print(summary_df.round(4).to_string(index=False))
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Correlation plot
        top_10_corr = correlations.head(10)
        axes[0,0].barh(range(len(top_10_corr)), top_10_corr.values)
        axes[0,0].set_yticks(range(len(top_10_corr)))
        axes[0,0].set_yticklabels(top_10_corr.index)
        axes[0,0].set_title('Top 10 Features by Correlation')
        axes[0,0].set_xlabel('Absolute Correlation')
        
        # Mutual Information plot
        top_10_mi = mi_results.head(10)
        axes[0,1].barh(range(len(top_10_mi)), top_10_mi.values)
        axes[0,1].set_yticks(range(len(top_10_mi)))
        axes[0,1].set_yticklabels(top_10_mi.index)
        axes[0,1].set_title('Top 10 Features by Mutual Information')
        axes[0,1].set_xlabel('Mutual Information Score')
        
        # F-statistic plot (significant features only)
        significant_f = f_results[f_results['P_Value'] < 0.05].head(10)
        if len(significant_f) > 0:
            axes[1,0].barh(range(len(significant_f)), significant_f['F_Statistic'])
            axes[1,0].set_yticks(range(len(significant_f)))
            axes[1,0].set_yticklabels(significant_f['Feature'])
            axes[1,0].set_title('Top 10 Significant Features (F-test, p<0.05)')
            axes[1,0].set_xlabel('F-Statistic')
        else:
            axes[1,0].text(0.5, 0.5, 'No significant features\n(p<0.05)', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('F-Statistics')
        
        # Composite ranking
        top_10_composite = composite_ranking.head(10)
        axes[1,1].barh(range(len(top_10_composite)), top_10_composite.values)
        axes[1,1].set_yticks(range(len(top_10_composite)))
        axes[1,1].set_yticklabels(top_10_composite.index)
        axes[1,1].set_title('Top 10 Features by Composite Score')
        axes[1,1].set_xlabel('Composite Score')
        
        plt.tight_layout()
        plt.show()
        
        # Recommendations
        print(f"\n🎯 FEATURE SELECTION RECOMMENDATIONS:")
        print("="*50)
        top_5_features = composite_ranking.head(5).index.tolist()
        print(f"Top 5 most predictive features: {top_5_features}")
        
        significant_features = f_results[f_results['P_Value'] < 0.05]['Feature'].tolist()
        print(f"Statistically significant features (p<0.05): {len(significant_features)} out of {len(feature_cols)}")
        
        if len(significant_features) > 0:
            print(f"Significant features: {significant_features}")
        
        return analysis_results, summary_df
    
    def detect_and_create_dummy_variables(self):
        """
        Analyze data for potential dummy variable creation
        """
        print("\n" + "="*60)
        print("DUMMY VARIABLE ANALYSIS")
        print("="*60)
    
        dummy_recommendations = []
        
        # Check for categorical variables that might be encoded as numbers
        for col in self.df.columns:
            if col not in [self.target_col, 'Annee']:
                unique_vals = self.df[col].nunique()
                data_range = self.df[col].max() - self.df[col].min() if self.df[col].dtype in ['int64', 'float64'] else None
                
                # Potential categorical if few unique values
                if unique_vals <= 10 and self.df[col].dtype in ['int64', 'float64']:
                    dummy_recommendations.append({
                        'variable': col,
                        'unique_values': unique_vals,
                        'values': sorted(self.df[col].unique()),
                        'reason': 'Few unique numeric values - might be categorical',
                        'recommendation': 'Consider creating dummy variables'
                    })
        
        # Time-based dummies from 'Annee' if present
        if 'Annee' in self.df.columns:
            year_range = self.df['Annee'].max() - self.df['Annee'].min()
            if year_range > 5:  # If data spans multiple years
                dummy_recommendations.append({
                    'variable': 'Annee',
                    'unique_values': self.df['Annee'].nunique(),
                    'values': f"Range: {self.df['Annee'].min()} - {self.df['Annee'].max()}",
                    'reason': 'Time series data - could benefit from period dummies',
                    'recommendation': 'Create decade/period/crisis dummies or cyclical features'
                })
        
        # Economic data specific recommendations
        economic_indicators = ['PROD', 'CHO', 'MI', 'FEC', 'ESP_F', 'ESP_H', 'PIB_VAL', 'PIB_VOL']
        for indicator in economic_indicators:
            if indicator in self.df.columns:
                # Check for structural breaks or regime changes
                # Look for periods of high volatility vs stability
                rolling_std = self.df[indicator].rolling(window=12).std()
                if rolling_std.std() > rolling_std.mean() * 0.5:  # High variation in volatility
                    dummy_recommendations.append({
                        'variable': f'{indicator}_regime',
                        'unique_values': 'N/A',
                        'values': 'High/Low volatility periods',
                        'reason': f'{indicator} shows varying volatility - potential regime changes',
                        'recommendation': 'Create volatility regime dummies (high/low volatility periods)'
                    })
        
        print("DUMMY VARIABLE RECOMMENDATIONS:")
        print("="*40)
        if dummy_recommendations:
            for i, rec in enumerate(dummy_recommendations, 1):
                print(f"{i}. {rec['variable']}:")
                print(f"   Unique values: {rec['unique_values']}")
                print(f"   Values: {rec['values']}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Recommendation: {rec['recommendation']}")
                print()
        else:
            print("No clear candidates for dummy variables detected.")
            print("All variables appear to be continuous.")
        
        # Economic crisis periods (example for your time range 2002-2024)
        if 'Annee' in self.df.columns:
            print("SUGGESTED ECONOMIC PERIOD DUMMIES:")
            print("="*40)
            print("Based on your data range (2002-2024), consider these period dummies:")
            print("• Financial Crisis (2007-2009): Market turbulence period")
            print("• European Debt Crisis (2010-2012): Sovereign debt issues")
            print("• Recovery Period (2013-2019): Economic expansion")
            print("• COVID-19 Period (2020-2022): Pandemic economic disruption") 
            print("• Post-COVID Recovery (2023-2024): Recovery and inflation period")
        
        return dummy_recommendations
        """
        Step 4: Feature Engineering and Selection
        """
        print("\n" + "="*60)
        print("STEP 4: FEATURE ENGINEERING")
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
        
        # Create lag features for time series - select based on correlation with target
        if 'Annee' in self.df.columns:
            print("\nAnalyzing feature importance for lag creation...")
            
            # Calculate correlation with target to select most relevant features for lags
            correlations = abs(X.corrwith(y)).sort_values(ascending=False)
            print(f"Feature correlations with target:\n{correlations}")
            
            # Select top correlated features (but limit to avoid overfitting)
            # Rule: Use sqrt(n_features) as a heuristic, max 5 for datasets < 200 samples
            max_lag_features = min(5, int(np.sqrt(len(feature_cols))), len(feature_cols))
            top_features_for_lags = correlations.head(max_lag_features).index.tolist()
            
            print(f"\nCreating lag features for top {max_lag_features} correlated features:")
            print(f"Selected features: {top_features_for_lags}")
            
            for col in top_features_for_lags:
                X[f'{col}_lag1'] = X[col].shift(1)
                X[f'{col}_lag2'] = X[col].shift(2)
            
            # Create rolling statistics for most correlated features
            # Rule: Use top 3 most correlated features for rolling stats
            top_features_for_rolling = correlations.head(3).index.tolist()
            print(f"Creating rolling statistics for: {top_features_for_rolling}")
            
            for col in top_features_for_rolling:
                X[f'{col}_roll3'] = X[col].rolling(window=3).mean()
                X[f'{col}_roll5'] = X[col].rolling(window=5).mean()
                X[f'{col}_roll_std3'] = X[col].rolling(window=3).std()  # Add volatility measure
        
        # Remove rows with NaN values created by lag/rolling features
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Features after engineering: {X.shape[1]}")
        print(f"Samples after cleaning: {len(X)}")
        
        # Feature selection using statistical tests
        # Rule: Select features based on sample size ratio (Harrell's rule of thumb)
        # For regression: max features ≈ n_samples / 15, but ensure at least 5 and max 20
        max_features = max(5, min(20, len(X) // 15))
        k_features = min(max_features, X.shape[1])
        
        print(f"\nFeature selection strategy:")
        print(f"Sample size: {len(X)}, Total features: {X.shape[1]}")
        print(f"Selecting {k_features} features (based on n_samples/15 rule)")
        
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_[selector.get_support()]
        
        # Show feature selection results
        feature_ranking = pd.DataFrame({
            'Feature': selected_features,
            'F_Score': feature_scores
        }).sort_values('F_Score', ascending=False)
        
        print(f"Selected features ({len(selected_features)}):")
        print(feature_ranking.to_string(index=False))
        
        self.X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        self.y = y
        self.feature_names = selected_features
        
        return self.X, self.y
    
    def prepare_data_for_modeling(self):
        """
        Step 5: Data Preparation and Scaling
        """
        print("\n" + "="*60)
        print("STEP 5: DATA PREPARATION")
        print("="*60)
        
        # Time series split - choose folds based on data size and temporal structure
        # Rule: For time series, use n_splits = min(10, max(3, n_samples//20))
        # This ensures each fold has sufficient data while maintaining temporal order
        optimal_splits = min(10, max(3, len(self.X) // 20))
        print(f"Optimal CV splits for {len(self.X)} samples: {optimal_splits}")
        
        tscv = TimeSeriesSplit(n_splits=optimal_splits)
        self.cv_splitter = tscv
        
        # Scaling - using RobustScaler for better outlier handling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index)
        
        print(f"✓ Applied RobustScaler to features")
        print(f"✓ Set up TimeSeriesSplit with {optimal_splits} folds")
        
        return self.X_scaled, self.y
    
    def train_models(self):
        """
        Step 6: Model Training and Comparison
        """
        print("\n" + "="*60)
        print("STEP 6: MODEL TRAINING AND COMPARISON")
        print("="*60)
        
        # Define models with hyperparameter tuning
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Elastic Net': ElasticNet(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'SVR': SVR(),
            'Neural Network': MLPRegressor(random_state=42, max_iter=1000)
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
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
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
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   CV RMSE: {results[best_model_name]['cv_rmse_mean']:.6f}")
        
        return results
    
    def evaluate_model(self):
        """
        Step 7: Model Evaluation and Diagnostics
        """
        print("\n" + "="*60)
        print("STEP 7: MODEL EVALUATION")
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
        Step 8: Model Comparison Summary
        """
        print("\n" + "="*60)
        print("STEP 8: MODEL COMPARISON SUMMARY")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'CV_RMSE_Mean': [self.results[model]['cv_rmse_mean'] for model in self.results.keys()],
            'CV_RMSE_Std': [self.results[model]['cv_rmse_std'] for model in self.results.keys()]
        }).sort_values('CV_RMSE_Mean')
        
        print("Model Performance Ranking:")
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
    
    # NEW: Comprehensive feature analysis
    feature_analysis, feature_summary = predictor.comprehensive_feature_analysis()
    
    # NEW: Dummy variable analysis
    dummy_recommendations = predictor.detect_and_create_dummy_variables()
    
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
    
    return predictor, results, evaluation, comparison, feature_analysis, feature_summary, dummy_recommendations

# Instructions for usage:
print("="*60)
print("TIME SERIES PREDICTIVE MODELING PIPELINE")
print("="*60)
print("\nTo use this pipeline:")
print("1. Load your data into a pandas DataFrame")
print("2. Ensure your data has the columns mentioned in your summary")
print("3. Run: predictor, results, evaluation, comparison = run_full_pipeline(your_dataframe)")
print("\nThe pipeline will:")
print("✓ Perform comprehensive data analysis")
print("✓ Test multiple advanced ML models")
print("✓ Apply proper time series validation")
print("✓ Prevent overfitting through cross-validation")
print("✓ Provide detailed model diagnostics")
print("✓ Select the best performing model")
