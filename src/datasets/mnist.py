"""
MNIST dataset loader and utilities.
"""
import numpy as np
import gzip
import os
from urllib.request import urlretrieve


def download_mnist(data_dir='data'):
    """Download MNIST dataset if not already present."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urlretrieve(base_url + file, file_path)
            print(f"Downloaded {file}")


def load_mnist_images(filename):
    """Load MNIST images from compressed file."""
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        
        return images


def load_mnist_labels(filename):
    """Load MNIST labels from compressed file."""
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels


def load_mnist(data_dir='data', subset='train'):
    """Load MNIST dataset."""
    # Download if necessary
    download_mnist(data_dir)
    
    if subset == 'train':
        images_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    elif subset == 'test':
        images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    else:
        raise ValueError("subset must be 'train' or 'test'")
    
    # Load data
    X = load_mnist_images(images_file)
    y = load_mnist_labels(labels_file)
    
    return X, y


def load_mnist_subset(data_dir='data', subset='train', num_samples=1000, classes=None):
    """Load a subset of MNIST for faster experimentation."""
    X, y = load_mnist(data_dir, subset)
    
    if classes is not None:
        # Filter by classes
        mask = np.isin(y, classes)
        X = X[mask]
        y = y[mask]
        
        # Remap labels to 0, 1, 2, ...
        for i, cls in enumerate(classes):
            y[y == cls] = i
    
    # Sample random subset
    if num_samples < len(X):
        indices = np.random.choice(len(X), num_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X, y


def visualize_mnist_samples(X, y, num_samples=10):
    """Visualize MNIST samples."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image = X[idx].reshape(28, 28)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {y[idx]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def preprocess_mnist(X, y, flatten=True, normalize=True, one_hot=False):
    """Preprocess MNIST data."""
    if normalize and X.max() > 1.0:
        X = X.astype(np.float32) / 255.0
    
    if not flatten and X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    elif flatten and X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    
    if one_hot:
        from ..utils.data_loader import to_categorical
        y = to_categorical(y, 10)
    
    return X, y


# Example usage
if __name__ == "__main__":
    # Load a small subset for testing
    X_train, y_train = load_mnist_subset(num_samples=100, classes=[0, 1, 2])
    print(f"Loaded {len(X_train)} training samples")
    print(f"Image shape: {X_train[0].shape}")
    print(f"Labels: {np.unique(y_train)}")
    
    # Visualize some samples
    visualize_mnist_samples(X_train, y_train)
