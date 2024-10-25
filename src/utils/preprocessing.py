"""Data preprocessing utilities for neural networks."""

import numpy as np
from typing import Tuple, Optional


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            X: Training data to compute statistics on
            
        Returns:
            self: Fitted scaler instance
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features using previously computed statistics.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the standardization transformation.
        
        Args:
            X: Standardized data to reverse
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """Scale features to a given range, typically [0, 1]."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute the minimum and range to be used for later scaling.
        
        Args:
            X: Training data to compute statistics on
            
        Returns:
            self: Fitted scaler instance
        """
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        data_range = data_max - data_min
        
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        self.min_ = feature_min - data_min * self.scale_
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using previously computed statistics.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.
        
        Args:
            X: Scaled data to reverse
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return (X - self.min_) / self.scale_


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Args:
        X: Input features
        y: Target values
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split arrays
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def to_categorical(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert class vector to binary class matrix (one-hot encoding).
    
    Args:
        y: Class vector to convert
        num_classes: Total number of classes (inferred if None)
        
    Returns:
        Binary matrix representation of input
    """
    y = np.array(y, dtype=int)
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    categorical = np.zeros((len(y), num_classes))
    categorical[np.arange(len(y)), y] = 1
    return categorical


def add_noise(X: np.ndarray, noise_factor: float = 0.1, 
             random_state: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise to data for data augmentation.
    
    Args:
        X: Input data
        noise_factor: Standard deviation of noise relative to data std
        random_state: Random seed for reproducibility
        
    Returns:
        Noisy data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    noise = np.random.normal(0, noise_factor * np.std(X), X.shape)
    return X + noise


def normalize_data(X: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize data to unit norm.
    
    Args:
        X: Data to normalize
        axis: Axis along which to normalize (None for global norm)
        
    Returns:
        Normalized data
    """
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    norm[norm == 0] = 1  # Avoid division by zero
    return X / norm


def shuffle_data(X: np.ndarray, y: np.ndarray, 
                random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle data arrays in unison.
    
    Args:
        X: Input features
        y: Target values
        random_state: Random seed for reproducibility
        
    Returns:
        Shuffled X and y arrays
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
