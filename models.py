from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np

class Models:
    DEFAULT_CONFIGS = {
        'lgbm': {
            'objective': 'poisson',
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.6,
            'colsample_bytree': 0.5,
            'min_data_in_leaf': 100,
            'random_state': 42,
            'verbosity': -1
        },
        'xgb': {
            'objective': 'reg:tweedie',
            'num_parallel_tree': 20,
            'n_estimators': 175,
            'max_depth': 3,
            'learning_rate': 0.03,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.003,
            'reg_lambda': 0.002,
            'tweedie_variance_power': 1.2,
            'random_state': 42,
            'verbosity': 0
        },
        'catboost': {
            'objective': 'RMSE',
            'iterations': 250,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 0.09,
            'bagging_temperature': 0.3,
            'random_strength': 3.5,
            'min_data_in_leaf': 60,
            'random_state': 42,
            'verbose': 0
        },
        'gaussian_nb': {
            'var_smoothing': 1e-9,
            'priors': np.array([0.58, 0.26, 0.13, 0.03])
        },
        'logistic_regression': {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'multi_class': 'multinomial',
            'class_weight': 'balanced'
        }
    }

    def __init__(self, model_type='lgbm', **kwargs):
        """
        Initialize the specified model
        Args:
            model_type (str): Type of model ('lgbm', 'xgb', 'catboost', 'gaussian_nb', 'logistic_regression')
            **kwargs: Parameters to override default configuration
        """
        if model_type not in self.DEFAULT_CONFIGS:
            raise ValueError(f"Model type '{model_type}' not found. Available types: {list(self.DEFAULT_CONFIGS.keys())}")

        # Get default config and update with provided parameters
        model_params = self.DEFAULT_CONFIGS[model_type].copy()
        
        # Only update with kwargs that are relevant to the specific model
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_params}
        model_params.update(filtered_kwargs)

        # Initialize the appropriate model with its specific parameters
        if model_type == 'lgbm':
            self.model = LGBMRegressor(**model_params)
        elif model_type == 'xgb':
            self.model = XGBRegressor(**model_params)
        elif model_type == 'catboost':
            self.model = CatBoostRegressor(**model_params)
        elif model_type == 'gaussian_nb':
            self.model = GaussianNB(**model_params)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**model_params)
