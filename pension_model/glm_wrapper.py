
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
        
def analyze_target_distribution(df, target_col):
    """Analyze target distribution to recommend GLM family"""
    
    target_data = df[target_col].dropna()
    
    target_analysis = {
        'min_value': target_data.min(),
        'max_value': target_data.max(),
        'mean': target_data.mean(),
        'std': target_data.std(),
        'cv': target_data.std() / target_data.mean(),
        'skewness': target_data.skew(),
        'has_zeros': (target_data == 0).sum(),
        'percent_zeros': (target_data == 0).mean() * 100
    }
    
    # GLM recommendations
    recommendations = []
    if target_analysis['has_zeros'] == 0 and target_analysis['cv'] > 1:
        recommendations.append('GLM_Gamma_Log')
    elif target_analysis['has_zeros'] > 0:
        recommendations.append('GLM_Tweedie_1.5')
    else:
        recommendations.append('GLM_Gaussian_Log')
        
    target_analysis['glm_recommendations'] = recommendations
    
    return target_analysis
    
def create_glm_recommendations(target_analysis):
    """
    Create GLM model recommendations based on target analysis
    """
    recommendations = []
    
    cv = target_analysis['cv']
    zeros_pct = target_analysis['percent_zeros']
    skewness = target_analysis['skewness']
    
    if zeros_pct == 0:  # No zeros
        if cv > 1.0:  # High variability
            recommendations.extend([
                'GLM_Gamma_Log',  # Best for right-skewed, positive data
                'GLM_Tweedie_1.9'  # Close to Gamma
            ])
        else:  # Moderate variability
            recommendations.extend([
                'GLM_Gaussian_Log',  # Log-normal assumption
                'GLM_Gamma_Log'
            ])
    else:  # Has zeros
        recommendations.extend([
            'GLM_Tweedie_1.5',  # Good middle ground
            'GLM_Tweedie_1.1'   # Closer to Poisson
        ])
    
    return recommendations