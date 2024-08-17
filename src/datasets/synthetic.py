"""
Synthetic dataset generators for testing neural networks.
"""
import numpy as np


def make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, 
                       random_state=None):
    """Generate a random n-class classification problem."""
    if random_state is not None:
        np.random.seed(random_state)
    
    X = []
    y = []
    
    for class_idx in range(n_classes):
        for cluster in range(n_clusters_per_class):
            # Random center for this cluster
            center = np.random.randn(n_features) * 2
            
            # Generate samples around this center
            samples_per_cluster = n_samples // (n_classes * n_clusters_per_class)
            cluster_X = np.random.randn(samples_per_cluster, n_features) + center
            cluster_y = np.full(samples_per_cluster, class_idx)
            
            X.append(cluster_X)
            y.append(cluster_y)
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def make_circles(n_samples=100, noise=0.1, factor=0.8, random_state=None):
    """Generate a 2D dataset with two concentric circles."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Outer circle
    theta_out = np.random.uniform(0, 2 * np.pi, n_samples_out)
    X_out = np.column_stack([np.cos(theta_out), np.sin(theta_out)])
    y_out = np.zeros(n_samples_out)
    
    # Inner circle
    theta_in = np.random.uniform(0, 2 * np.pi, n_samples_in)
    X_in = factor * np.column_stack([np.cos(theta_in), np.sin(theta_in)])
    y_in = np.ones(n_samples_in)
    
    # Combine
    X = np.vstack([X_out, X_in])
    y = np.hstack([y_out, y_in])
    
    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def make_moons(n_samples=100, noise=0.1, random_state=None):
    """Generate a 2D dataset with two interleaving half circles."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # Outer moon
    theta_out = np.linspace(0, np.pi, n_samples_out)
    X_out = np.column_stack([np.cos(theta_out), np.sin(theta_out)])
    y_out = np.zeros(n_samples_out)
    
    # Inner moon
    theta_in = np.linspace(0, np.pi, n_samples_in)
    X_in = np.column_stack([1 - np.cos(theta_in), 1 - np.sin(theta_in) - 0.5])
    y_in = np.ones(n_samples_in)
    
    # Combine
    X = np.vstack([X_out, X_in])
    y = np.hstack([y_out, y_in])
    
    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def make_regression(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """Generate a random regression problem."""
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create a non-linear relationship
    if n_features == 1:
        y = 0.5 * X[:, 0]**2 + 0.3 * X[:, 0] + 0.1
    else:
        # For multiple features, create a more complex relationship
        y = np.sum(X**2, axis=1) + 0.1 * np.sum(X, axis=1)
    
    # Add noise
    if noise > 0:
        y += np.random.normal(0, noise, y.shape)
    
    return X, y


def make_spiral(n_samples=100, noise=0.1, random_state=None):
    """Generate a 2D spiral dataset."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples_per_class = n_samples // 2
    
    # Generate spiral data
    theta = np.linspace(0, 4 * np.pi, n_samples_per_class)
    r = np.linspace(0.1, 1, n_samples_per_class)
    
    # First spiral
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y1 = np.zeros(n_samples_per_class)
    
    # Second spiral (rotated)
    X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    y2 = np.ones(n_samples_per_class)
    
    # Combine
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def make_xor(n_samples=100, noise=0.1, random_state=None):
    """Generate XOR dataset with noise."""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Base XOR points
    base_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    base_y = np.array([0, 1, 1, 0])
    
    # Replicate and add noise
    X = []
    y = []
    
    samples_per_point = n_samples // 4
    for i in range(4):
        cluster_X = np.random.normal(base_X[i], noise, (samples_per_point, 2))
        cluster_y = np.full(samples_per_point, base_y[i])
        X.append(cluster_X)
        y.append(cluster_y)
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
