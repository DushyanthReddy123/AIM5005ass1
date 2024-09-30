import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None  
        self.scale = None  
    
    def fit(self, X):
        X = np.array(X)  # Ensure X is converted to a NumPy array
        self.mean = np.mean(X, axis=0)  # Calculate the mean along columns (features)
        self.scale = np.std(X, axis=0)  # Calculate the standard deviation along columns
        return self
    
    def transform(self, X):
        X = np.array(X)  # Ensure X is converted to a NumPy array
        return (X - self.mean) / self.scale  # Apply the scaling formula
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class LabelEncoder:
    def __init__(self):
        self.classes_ = None
    
    def fit(self, y):
        self.classes_ = np.unique(y)
        return self
    
    def transform(self, y):
        return np.array([np.where(self.classes_ == label)[0][0] for label in y])
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)