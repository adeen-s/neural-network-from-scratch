"""
Data loading and preprocessing utilities.
"""
import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def normalize(X, method='standard'):
    """Normalize input data."""
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (X - mean) / std, mean, std
    
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero
        return (X - min_val) / range_val, min_val, max_val
    
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")


def to_categorical(y, num_classes=None):
    """Convert integer labels to one-hot encoded vectors."""
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    categorical = np.zeros((len(y), num_classes))
    categorical[np.arange(len(y)), y] = 1
    return categorical


def batch_generator(X, y, batch_size=32, shuffle=True):
    """Generate batches of data for training."""
    n_samples = X.shape[0]
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        yield X[i:end_idx], y[i:end_idx]


class DataLoader:
    """Simple data loader for batch processing."""
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
        else:
            X_shuffled = self.X
            y_shuffled = self.y
        
        for i in range(0, self.n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.n_samples)
            yield X_shuffled[i:end_idx], y_shuffled[i:end_idx]
    
    def __len__(self):
        return self.n_batches
