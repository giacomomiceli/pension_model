
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.genmod.families import Gamma, Tweedie, Gaussian
from statsmodels.genmod.families.links import Log, Identity
import statsmodels.api as sm
import warnings


class GLMWrapper(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for statsmodels GLM
    This allows GLM to work with GridSearchCV
    """
    
    def __init__(self, family='gamma', link='log', alpha=0.01, fit_intercept=True):
        self.family = family
        self.link = link
        self.alpha = alpha  # Regularization parameter
        self.fit_intercept = fit_intercept
        self.model_ = None
        self.fitted_model_ = None
        
    def _get_family_link(self):
        """Convert string parameters to statsmodels family objects"""
        families = {
            'gamma': Gamma,
            'gaussian': Gaussian,
            'tweedie_1.1': lambda: Tweedie(var_power=1.1),
            'tweedie_1.5': lambda: Tweedie(var_power=1.5),
            'tweedie_1.9': lambda: Tweedie(var_power=1.9)
        }
        
        links = {
            'log': Log,
            'identity': Identity
        }
        
        family_class = families.get(self.family, Gamma)
        link_class = links.get(self.link, Log)
        
        if 'tweedie' in self.family:
            return family_class()(link=link_class())
        else:
            return family_class(link=link_class())
    
    def fit(self, X, y):
        """Fit the GLM model"""
        try:
            # Add intercept if requested
            if self.fit_intercept:
                X_with_const = sm.add_constant(X)
            else:
                X_with_const = X
            
            # Get family and link
            family_link = self._get_family_link()
            
            # Create and fit GLM
            self.model_ = sm.GLM(y, X_with_const, family=family_link)
            
            # Fit with regularization if alpha > 0
            if self.alpha > 0:
                # Use regularized GLM (elastic net penalty)
                self.fitted_model_ = self.model_.fit_regularized(
                    alpha=self.alpha, 
                    L1_wt=0.5  # 50% L1, 50% L2 penalty
                )
            else:
                self.fitted_model_ = self.model_.fit()
            
            return self
            
        except Exception as e:
            # Fallback to simple linear regression if GLM fails
            warnings.warn(f"GLM fitting failed ({str(e)}), falling back to linear regression")
            from sklearn.linear_model import LinearRegression
            self.fallback_model_ = LinearRegression()
            self.fallback_model_.fit(X, y)
            return self
    
    def predict(self, X):
        """Make predictions"""
        try:
            if hasattr(self, 'fallback_model_'):
                return self.fallback_model_.predict(X)
            
            # Add intercept if it was used during fitting
            if self.fit_intercept:
                X_with_const = sm.add_constant(X)
            else:
                X_with_const = X
            
            predictions = self.fitted_model_.predict(X_with_const)
            
            # Ensure predictions are positive for expense data
            return np.maximum(predictions, 0.01)
            
        except Exception as e:
            warnings.warn(f"GLM prediction failed: {str(e)}")
            # Return mean as fallback
            return np.full(len(X), np.mean(self.fitted_model_.fittedvalues))
    
    def get_params(self, deep=True):
        """Get parameters for GridSearch"""
        return {
            'family': self.family,
            'link': self.link,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept
        }
    
    def set_params(self, **params):
        """Set parameters for GridSearch"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    

def create_enhanced_models_with_glm():
    """
    Enhanced model dictionary including GLM models for expense prediction
    """
    # Your existing models
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
        'Neural Network': MLPRegressor(random_state=42, max_iter=1000),
        
        # GLM Models - Perfect for expense data
        'GLM_Gamma_Log': GLMWrapper(family='gamma', link='log'),
        'GLM_Gamma_Identity': GLMWrapper(family='gamma', link='identity'),
        'GLM_Gaussian_Log': GLMWrapper(family='gaussian', link='log'),
        'GLM_Tweedie_1.5': GLMWrapper(family='tweedie_1.5', link='log'),
        'GLM_Tweedie_1.9': GLMWrapper(family='tweedie_1.9', link='log'),
    }
    
    return models

def create_enhanced_param_grids():
    """
    Enhanced parameter grids including GLM hyperparameters
    """
    param_grids = {
        # Your existing grids
        'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
        'Elastic Net': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        },
        
        # GLM-specific parameter grids
        'GLM_Gamma_Log': {
            'alpha': [0.0, 0.01, 0.1, 1.0],  # Regularization strength
            'fit_intercept': [True, False]
        },
        'GLM_Gamma_Identity': {
            'alpha': [0.0, 0.01, 0.1, 1.0],
            'fit_intercept': [True, False]
        },
        'GLM_Gaussian_Log': {
            'alpha': [0.0, 0.01, 0.1, 1.0],
            'fit_intercept': [True, False]
        },
        'GLM_Tweedie_1.5': {
            'alpha': [0.0, 0.01, 0.1, 1.0],
            'fit_intercept': [True, False]
        },
        'GLM_Tweedie_1.9': {
            'alpha': [0.0, 0.01, 0.1, 1.0],
            'fit_intercept': [True, False]
        }
    }
    
    return param_grids