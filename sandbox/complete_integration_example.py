import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLRegressionPipeline:
    """Complete ML regression pipeline with feature engineering"""
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        self.config = config or FeatureEngineeringConfig()
        self.feature_pipeline = None
        self.model = None
        self.is_fitted = False
        self.performance_metrics = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            test_size: float = 0.2, 
            model_params: dict = None):
        """
        Fit the complete pipeline
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Test set size for evaluation
            model_params: Parameters for the ML model
        """
        logger.info("Starting ML pipeline training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=self.config.random_state
        )
        
        # Initialize and fit feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(self.config)
        X_train_transformed = self.feature_pipeline.fit_transform(X_train, y_train)
        
        # Transform test set
        X_test_transformed = self.feature_pipeline.transform(X_test)
        
        # Initialize and fit model
        model_params = model_params or {}
        self.model = RandomForestRegressor(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            **model_params
        )
        
        logger.info("Training ML model...")
        self.model.fit(X_train_transformed, y_train)
        
        # Evaluate model
        self._evaluate_model(X_test_transformed, y_test)
        
        self.is_fitted = True
        logger.info("ML pipeline training completed!")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Transform features
        X_transformed = self.feature_pipeline.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_transformed)
        
        return predictions
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """Make predictions with uncertainty estimates (for ensemble models)"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Transform features
        X_transformed = self.feature_pipeline.transform(X)
        
        # Get predictions from all trees
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                tree.predict(X_transformed) for tree in self.model.estimators_
            ])
            
            predictions = np.mean(tree_predictions, axis=0)
            uncertainty = np.std(tree_predictions, axis=0)
            
            return predictions, uncertainty
        else:
            predictions = self.model.predict(X_transformed)
            return predictions, np.zeros_like(predictions)
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        self.performance_metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': np.mean(np.abs(y_test - y_pred))
        }
        
        logger.info(f"Model Performance:")
        for metric, value in self.performance_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names from the pipeline
            feature_names = self.feature_pipeline.pipeline.named_steps['missing_final'].get_feature_names_out()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """Save the complete pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        model_data = {
            'config': self.config,
            'feature_pipeline': self.feature_pipeline,
            'model': self.model,
            'performance_metrics': self.performance_metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLRegressionPipeline':
        """Load a saved pipeline"""
        model_data = joblib.load(filepath)
        
        pipeline = cls(model_data['config'])
        pipeline.feature_pipeline = model_data['feature_pipeline']
        pipeline.model = model_data['model']
        pipeline.performance_metrics = model_data['performance_metrics']
        pipeline.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return pipeline

# Comprehensive example usage
def run_complete_example():
    """Run a complete example with synthetic data"""
    
    # Create synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate features with time patterns
    data = pd.DataFrame({
        'Annee': range(n_samples),
        'feature_1': np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 0.1, n_samples),
        'feature_2': np.cos(np.arange(n_samples) * 0.05) + np.random.normal(0, 0.1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.exponential(1, n_samples),
    })
    
    # Create target with realistic patterns
    data['target'] = (
        2 * data['feature_1'] + 
        1.5 * data['feature_2'] + 
        0.5 * data['feature_3'] + 
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data.loc[missing_indices, 'feature_3'] = np.nan
    
    print("=== Complete ML Pipeline Example ===")
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    # 1. Configuration Management
    print("\n1. Setting up configuration...")
    config_manager = ConfigurationManager()
    
    # Create custom configuration
    config = config_manager.create_custom_config(
        base_preset='advanced_timeseries',
        **{
            'target_column': 'target',
            'time_series.lag_periods': [1, 2, 3],
            'time_series.rolling_windows': [3, 5],
            'feature_selection.max_features': 15,
            'preprocessing.handle_outliers': True,
            'validation.validate_features': True
        }
    )
    
    # Save configuration for reproducibility
    config.save_to_file('ml_config.yaml')
    print(f"Configuration saved. Steps: {config.get_feature_engineering_steps()}")
    
    # 2. Pipeline Training
    print("\n2. Training pipeline...")
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Initialize pipeline
    ml_pipeline = MLRegressionPipeline(config)
    
    # Train with custom model parameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    }
    
    ml_pipeline.fit(X, y, test_size=0.2, model_params=model_params)
    
    # 3. Model Analysis
    print("\n3. Model analysis...")
    
    # Feature importance
    importance_df = ml_pipeline.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Performance metrics
    print(f"\nModel Performance:")
    for metric, value in ml_pipeline.performance_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # 4. Predictions
    print("\n4. Making predictions...")
    
    # Create new data for prediction
    new_data = pd.DataFrame({
        'Annee': range(n_samples, n_samples + 10),
        'feature_1': np.sin(np.arange(n_samples, n_samples + 10) * 0.1),
        'feature_2': np.cos(np.arange(n_samples, n_samples + 10) * 0.05),
        'feature_3': np.random.normal(0, 1, 10),
        'feature_4': np.random.exponential(1, 10),
    })
    
    # Make predictions
    predictions = ml_pipeline.predict(new_data)
    predictions_with_uncertainty = ml_pipeline.predict_with_uncertainty(new_data)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5]}")
    print(f"First 5 uncertainties: {predictions_with_uncertainty[1][:5]}")
    
    # 5. Model Persistence
    print("\n5. Saving and loading model...")
    ml_pipeline.save_model('complete_ml_model.pkl')
    
    # Load model
    loaded_pipeline = MLRegressionPipeline.load_model('complete_ml_model.pkl')
    
    # Verify loaded model works
    loaded_predictions = loaded_pipeline.predict(new_data)
    print(f"Loaded model predictions match: {np.allclose(predictions, loaded_predictions)}")
    
    # 6. Configuration Versioning
    print("\n6. Configuration versioning...")
    
    # Create different configurations for experimentation
    configs = {
        'basic': config_manager.get_preset('basic_timeseries'),
        'minimal': config_manager.get_preset('minimal'),
        'advanced': config_manager.get_preset('advanced_timeseries')
    }
    
    # Save all configurations
    for name, cfg in configs.items():
        cfg.target_column = 'target'  # Set target column
        cfg.save_to_file(f'config_{name}.yaml')
    
    print("All configurations saved for experimentation")
    
    return ml_pipeline

# Advanced usage patterns
class ExperimentTracker:
    """Track experiments with different configurations"""
    
    def __init__(self):
        self.experiments = []
    
    def run_experiment(self, config_name: str, config: FeatureEngineeringConfig, 
                      X: pd.DataFrame, y: pd.Series):
        """Run an experiment with a specific configuration"""
        logger.info(f"Running experiment: {config_name}")
        
        # Create and train pipeline
        pipeline = MLRegressionPipeline(config)
        pipeline.fit(X, y)
        
        # Store results
        experiment_result = {
            'config_name': config_name,
            'config': config,
            'performance': pipeline.performance_metrics,
            'n_features': len(pipeline.feature_pipeline.pipeline.named_steps),
            'steps': config.get_feature_engineering_steps()
        }
        
        self.experiments.append(experiment_result)
        return experiment_result
    
    def get_best_experiment(self, metric: str = 'r2') -> dict:
        """Get the best experiment based on a metric"""
        if not self.experiments:
            raise ValueError("No experiments run yet")
        
        best_exp = max(self.experiments, key=lambda x: x['performance'][metric])
        return best_exp
    
    def compare_experiments(self) -> pd.DataFrame:
        """Compare all experiments"""
        if not self.experiments:
            return pd.DataFrame()
        
        comparison_data = []
        for exp in self.experiments:
            row = {
                'config_name': exp['config_name'],
                'n_features': exp['n_features'],
                'n_steps': len(exp['steps']),
                **exp['performance']
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

# Example of experiment tracking
def run_experiment_comparison():
    """Run comparison of different configurations"""
    
    # Generate sample data (same as before)
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'Annee': range(n_samples),
        'feature_1': np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 0.1, n_samples),
        'feature_2': np.cos(np.arange(n_samples) * 0.05) + np.random.normal(0, 0.1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples)
    })
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Set up experiment tracker
    tracker = ExperimentTracker()
    config_manager = ConfigurationManager()
    
    # Run experiments with different configurations
    configs_to_test = ['basic_timeseries', 'advanced_timeseries', 'minimal']
    
    for config_name in configs_to_test:
        config = config_manager.get_preset(config_name)
        config.target_column = 'target'
        tracker.run_experiment(config_name, config, X, y)
    
    # Compare results
    comparison_df = tracker.compare_experiments()
    print("\nExperiment Comparison:")
    print(comparison_df)
    
    # Get best experiment
    best_exp = tracker.get_best_experiment('r2')
    print(f"\nBest experiment: {best_exp['config_name']}")
    print(f"Best R2 score: {best_exp['performance']['r2']:.4f}")

if __name__ == "__main__":
    # Run the complete example
    pipeline = run_complete_example()
    
    print("\n" + "="*50)
    print("Running experiment comparison...")
    run_experiment_comparison()
    
    print("\nComplete integration example finished!")