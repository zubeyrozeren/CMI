from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BayesianRidge

class Imputers:
    def __init__(self, imputer_type='simple', strategy='mean', n_neighbors=5, max_iter=10, random_state=42):
        """
        Initialize and return the specified imputer
        Args:
            imputer_type (str): Type of imputer ('simple', 'knn', 'iterative')
            strategy (str): Strategy for SimpleImputer ('mean', 'median', 'most_frequent', 'constant')
            n_neighbors (int): Number of neighbors for KNNImputer
            max_iter (int): Maximum iterations for IterativeImputer
            random_state (int): Random state for IterativeImputer
        """
        imputers = {
            'simple': SimpleImputer(strategy=strategy),
            'knn': KNNImputer(n_neighbors=n_neighbors),
            'iterative': IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter, random_state=random_state),
            'linear_regression': LinearRegression()
        }
        
        if imputer_type not in imputers:
            raise ValueError(f"Imputer type '{imputer_type}' not found. Available types: {list(imputers.keys())}")
        
        return imputers[imputer_type]

