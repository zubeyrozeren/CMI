from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class Scalers:
    def __init__(self):
        """
        Initialize scalers with default parameters
        """
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
    
    def get_scaler(self, scaler_type='standard'):
        """
        Get the specified scaler
        Args:
            scaler_type (str): Type of scaler ('standard', 'minmax', 'robust')
        Returns:
            Selected scaler object
        """
        if scaler_type not in self.scalers:
            raise ValueError(f"Scaler type '{scaler_type}' not found. Available types: {list(self.scalers.keys())}")
        return self.scalers[scaler_type]
