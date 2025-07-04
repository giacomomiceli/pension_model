from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# LGM Wrapper
from glm_wrapper import GLMWrapper

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
        #'SVR': SVR(),
        #'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        # 'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
        # 'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        # 'Neural Network': MLPRegressor(random_state=42, max_iter=1000),
        
        # # GLM Models - Perfect for expense data
        # 'GLM_Gamma_Log': GLMWrapper(family='gamma', link='log'),
        # 'GLM_Gamma_Identity': GLMWrapper(family='gamma', link='identity'),
        # 'GLM_Gaussian_Log': GLMWrapper(family='gaussian', link='log'),
        # 'GLM_Tweedie_1.5': GLMWrapper(family='tweedie_1.5', link='log'),
        # 'GLM_Tweedie_1.9': GLMWrapper(family='tweedie_1.9', link='log'),
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
        # 'SVR': {
        #     'C': [0.1, 1, 10, 100],
        #     'gamma': ['scale', 'auto', 0.001, 0.01],
        #     'kernel': ['rbf', 'linear']
        # },
        # 'XGBoost': {
        #     'n_estimators': [50, 100, 200],
        #     'max_depth': [3, 6, 9],
        #     'learning_rate': [0.01, 0.1, 0.2]
        # },
        # 'LightGBM': {
        #     'n_estimators': [50, 100, 200],
        #     'max_depth': [3, 6, 9],
        #     'learning_rate': [0.01, 0.1, 0.2]
        # },
        # 'Neural Network': {
        #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        #     'alpha': [0.0001, 0.001, 0.01],
        #     'learning_rate_init': [0.001, 0.01]
        # },
        
        # # GLM-specific parameter grids
        # 'GLM_Gamma_Log': {
        #     'alpha': [0.0, 0.01, 0.1, 1.0],  # Regularization strength
        #     'fit_intercept': [True, False]
        # },
        # 'GLM_Gamma_Identity': {
        #     'alpha': [0.0, 0.01, 0.1, 1.0],
        #     'fit_intercept': [True, False]
        # },
        # 'GLM_Gaussian_Log': {
        #     'alpha': [0.0, 0.01, 0.1, 1.0],
        #     'fit_intercept': [True, False]
        # },
        # 'GLM_Tweedie_1.5': {
        #     'alpha': [0.0, 0.01, 0.1, 1.0],
        #     'fit_intercept': [True, False]
        # },
        # 'GLM_Tweedie_1.9': {
        #     'alpha': [0.0, 0.01, 0.1, 1.0],
        #     'fit_intercept': [True, False]
        # }
    }
    
    return param_grids