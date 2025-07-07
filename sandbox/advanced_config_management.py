from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import yaml
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class FeatureSelectionMethod(Enum):
    """Enum for feature selection methods"""
    NONE = "none"
    SELECT_K_BEST = "selectkbest"
    RFE = "rfe"
    LASSO = "lasso"
    TREE_BASED = "tree_based"

class ScalerType(Enum):
    """Enum for scaler types"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"

class MissingValueStrategy(Enum):
    """Enum for missing value strategies"""
    MEDIAN = "median"
    MEAN = "mean"
    FORWARD_FILL = "forward_fill"
    INTERPOLATE = "interpolate"
    DROP = "drop"

@dataclass
class TimeSeriesConfig:
    """Configuration for time series specific features"""
    time_column: str = 'Annee'
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 7])
    seasonal_periods: List[int] = field(default_factory=list)
    trend_features: bool = False
    
    def __post_init__(self):
        """Validate time series configuration"""
        if self.lag_periods and min(self.lag_periods) <= 0:
            raise ValueError("Lag periods must be positive integers")
        if self.rolling_windows and min(self.rolling_windows) <= 1:
            raise ValueError("Rolling windows must be greater than 1")

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    method: FeatureSelectionMethod = FeatureSelectionMethod.SELECT_K_BEST
    max_features: int = 20
    min_features: int = 5
    auto_tune: bool = True
    selection_threshold: float = 0.05
    
    # Method-specific parameters
    rfe_step: float = 0.1
    lasso_alpha: float = 0.01
    tree_importance_threshold: float = 0.01
    
    def get_max_features(self, n_samples: int) -> int:
        """Calculate max features based on sample size"""
        if self.auto_tune:
            # Harrell's rule of thumb: n_samples / 15
            calculated_max = max(self.min_features, min(self.max_features, n_samples // 15))
            return calculated_max
        return self.max_features

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps"""
    missing_value_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN
    scaler_type: ScalerType = ScalerType.STANDARD
    scale_features: bool = True
    handle_outliers: bool = False
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0

@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced feature engineering"""
    polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_features: bool = False
    max_interaction_degree: int = 2
    
    # Domain-specific features
    financial_features: bool = False  # RSI, MACD, etc.
    statistical_features: bool = False  # skewness, kurtosis, etc.
    
    # Text features (if applicable)
    text_features: bool = False
    text_vectorizer: str = 'tfidf'  # 'tfidf', 'count', 'word2vec'

@dataclass
class ValidationConfig:
    """Configuration for feature validation"""
    validate_features: bool = True
    feature_drift_threshold: float = 0.1
    missing_value_threshold: float = 0.1
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    
    # Logging and monitoring
    log_feature_stats: bool = True
    save_feature_reports: bool = False
    report_path: Optional[str] = None

@dataclass
class FeatureEngineeringConfig:
    """Main configuration class for feature engineering"""
    target_column: str = 'target'
    
    # Sub-configurations
    time_series: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    advanced: AdvancedFeatureConfig = field(default_factory=AdvancedFeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Pipeline configuration
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the entire configuration"""
        # Validate target column
        if not self.target_column:
            raise ValueError("Target column must be specified")
        
        # Validate feature selection
        if (self.feature_selection.max_features < self.feature_selection.min_features):
            raise ValueError("max_features must be >= min_features")
        
        # Validate time series config
        if (self.time_series.lag_periods and 
            self.time_series.rolling_windows and 
            max(self.time_series.lag_periods) >= min(self.time_series.rolling_windows)):
            logger.warning("Large lag periods with small rolling windows may cause issues")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureEngineeringConfig':
        """Create configuration from dictionary"""
        # Handle nested configurations
        if 'time_series' in config_dict:
            config_dict['time_series'] = TimeSeriesConfig(**config_dict['time_series'])
        
        if 'feature_selection' in config_dict:
            # Convert string enums back to enum objects
            fs_dict = config_dict['feature_selection']
            if 'method' in fs_dict and isinstance(fs_dict['method'], str):
                fs_dict['method'] = FeatureSelectionMethod(fs_dict['method'])
            config_dict['feature_selection'] = FeatureSelectionConfig(**fs_dict)
        
        if 'preprocessing' in config_dict:
            prep_dict = config_dict['preprocessing']
            if 'missing_value_strategy' in prep_dict and isinstance(prep_dict['missing_value_strategy'], str):
                prep_dict['missing_value_strategy'] = MissingValueStrategy(prep_dict['missing_value_strategy'])
            if 'scaler_type' in prep_dict and isinstance(prep_dict['scaler_type'], str):
                prep_dict['scaler_type'] = ScalerType(prep_dict['scaler_type'])
            config_dict['preprocessing'] = PreprocessingConfig(**prep_dict)
        
        if 'advanced' in config_dict:
            config_dict['advanced'] = AdvancedFeatureConfig(**config_dict['advanced'])
        
        if 'validation' in config_dict:
            config_dict['validation'] = ValidationConfig(**config_dict['validation'])
        
        return cls(**config_dict)
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to file (JSON or YAML)"""
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        # Convert enums to strings for serialization
        self._convert_enums_to_strings(config_dict)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'FeatureEngineeringConfig':
        """Load configuration from file"""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    def _convert_enums_to_strings(self, config_dict: Dict[str, Any]):
        """Convert enum values to strings for serialization"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._convert_enums_to_strings(value)
            elif isinstance(value, Enum):
                config_dict[key] = value.value
    
    def get_feature_engineering_steps(self) -> List[str]:
        """Get list of feature engineering steps that will be applied"""
        steps = []
        
        if self.time_series.lag_periods:
            steps.append("lag_features")
        
        if self.time_series.rolling_windows:
            steps.append("rolling_features")
        
        if self.time_series.seasonal_periods:
            steps.append("seasonal_features")
        
        if self.advanced.polynomial_features:
            steps.append("polynomial_features")
        
        if self.advanced.interaction_features:
            steps.append("interaction_features")
        
        if self.preprocessing.handle_outliers:
            steps.append("outlier_handling")
        
        if self.preprocessing.scale_features:
            steps.append("scaling")
        
        if self.feature_selection.method != FeatureSelectionMethod.NONE:
            steps.append("feature_selection")
        
        return steps

class ConfigurationManager:
    """Manage multiple configurations and presets"""
    
    def __init__(self):
        self.presets = {}
        self._load_default_presets()
    
    def _load_default_presets(self):
        """Load default configuration presets"""
        # Basic time series configuration
        self.presets['basic_timeseries'] = FeatureEngineeringConfig(
            time_series=TimeSeriesConfig(
                lag_periods=[1, 2, 3],
                rolling_windows=[3, 5],
                trend_features=False
            ),
            feature_selection=FeatureSelectionConfig(
                method=FeatureSelectionMethod.SELECT_K_BEST,
                max_features=15
            ),
            preprocessing=PreprocessingConfig(
                missing_value_strategy=MissingValueStrategy.MEDIAN,
                scaler_type=ScalerType.STANDARD
            )
        )
        
        # Advanced time series configuration
        self.presets['advanced_timeseries'] = FeatureEngineeringConfig(
            time_series=TimeSeriesConfig(
                lag_periods=[1, 2, 3, 6, 12],
                rolling_windows=[3, 5, 7, 12],
                seasonal_periods=[12, 24],
                trend_features=True
            ),
            feature_selection=FeatureSelectionConfig(
                method=FeatureSelectionMethod.RFE,
                max_features=25,
                auto_tune=True
            ),
            preprocessing=PreprocessingConfig(
                missing_value_strategy=MissingValueStrategy.INTERPOLATE,
                scaler_type=ScalerType.ROBUST,
                handle_outliers=True
            ),
            advanced=AdvancedFeatureConfig(
                polynomial_features=True,
                polynomial_degree=2,
                statistical_features=True
            )
        )
        
        # Minimal configuration
        self.presets['minimal'] = FeatureEngineeringConfig(
            time_series=TimeSeriesConfig(
                lag_periods=[1],
                rolling_windows=[3]
            ),
            feature_selection=FeatureSelectionConfig(
                method=FeatureSelectionMethod.NONE
            ),
            preprocessing=PreprocessingConfig(
                scale_features=False
            )
        )
    
    def get_preset(self, name: str) -> FeatureEngineeringConfig:
        """Get a preset configuration"""
        if name not in self.presets:
            raise ValueError(f"Preset '{name}' not found. Available presets: {list(self.presets.keys())}")
        return self.presets[name]
    
    def add_preset(self, name: str, config: FeatureEngineeringConfig):
        """Add a new preset configuration"""
        self.presets[name] = config
    
    def list_presets(self) -> List[str]:
        """List available presets"""
        return list(self.presets.keys())
    
    def create_custom_config(self, base_preset: str = 'basic_timeseries', **kwargs) -> FeatureEngineeringConfig:
        """Create a custom configuration based on a preset"""
        base_config = self.get_preset(base_preset)
        config_dict = base_config.to_dict()
        
        # Update with custom parameters
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys like 'time_series.lag_periods'
                parts = key.split('.')
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return FeatureEngineeringConfig.from_dict(config_dict)

# Usage Examples
if __name__ == "__main__":
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Use a preset
    config = config_manager.get_preset('basic_timeseries')
    print(f"Steps: {config.get_feature_engineering_steps()}")
    
    # Create custom configuration
    custom_config = config_manager.create_custom_config(
        base_preset='basic_timeseries',
        **{
            'time_series.lag_periods': [1, 2, 3, 6],
            'feature_selection.max_features': 20,
            'preprocessing.handle_outliers': True
        }
    )
    
    # Save and load configuration
    config.save_to_file('config.yaml')
    loaded_config = FeatureEngineeringConfig.load_from_file('config.yaml')
    
    print("Configuration management example completed!")