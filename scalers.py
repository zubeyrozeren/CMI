from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class Scalers:
    def __init__(self, scaler_type='standard'):
        """
        Initialize the specified scaler
        Args:
            scaler_type (str): Type of scaler ('standard', 'minmax', 'robust')
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Scaler type '{scaler_type}' not found. Available types: {list(scalers.keys())}")
        
        self.scaler = scalers[scaler_type]
    
    def fit_transform(self, X):
        """Convenience method to directly fit and transform the data"""
        return self.scaler.fit_transform(X)
    
    def fit(self, X, y=None):
        """Convenience method to fit the scaler"""
        return self.scaler.fit(X, y)

    def transform(self, X):
        """Convenience method to transform the data"""
        return self.scaler.transform(X)
