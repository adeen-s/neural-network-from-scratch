"""
MNIST digit classification example using neural network from scratch.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Softmax
from src.core.losses import CategoricalCrossentropy
from src.core.optimizers import Adam
from src.datasets.mnist import load_mnist_subset, visualize_mnist_samples
from src.utils.data_loader import train_test_split, to_categorical
from src.utils.metrics import accuracy
from src.utils.visualization import plot_training_history


def main():
    """Train a neural network on MNIST digit classification."""
    print("MNIST Digit Classification - Neural Network from Scratch")
    print("=" * 60)
    
    # Load a subset of MNIST for faster training
    print("Loading MNIST dataset...")
    try:
        X, y = load_mnist_subset(num_samples=2000, classes=[0, 1, 2, 3, 4])
        print(f"Loaded {len(X)} samples with {len(np.unique(y))} classes")
        print(f"Image shape: {X[0].shape}")
        
        # Visualize some samples
        print("\nVisualizing sample images...")
        visualize_mnist_samples(X, y, num_samples=10)
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Using synthetic data instead...")
        
        # Fallback to synthetic data
        from src.datasets.synthetic import make_classification
        X, y = make_classification(n_samples=1000, n_features=784, n_classes=5, random_state=42)
        X = np.abs(X)  # Make sure values are positive like images
        X = X / np.max(X)  # Normalize to [0, 1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {num_classes}")
    print(f"Input features: {X_train.shape[1]}")
    
    # Create neural network
    print("\nBuilding neural network...")
    network = NeuralNetwork()
    
    # Input layer -> Hidden layer 1
    network.add(Dense(X_train.shape[1], 128))
    network.add(ReLU())
    
    # Hidden layer 1 -> Hidden layer 2
    network.add(Dense(128, 64))
    network.add(ReLU())
    
    # Hidden layer 2 -> Output layer
    network.add(Dense(64, num_classes))
    network.add(Softmax())
    
    print("Network architecture:")
    print(f"Input ({X_train.shape[1]}) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense({num_classes}) -> Softmax")
    
    # Compile network
    network.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy()
    )
    
    # Train network
    print("\nTraining network...")
    history = network.fit(
        X_train, y_train_onehot, 
        epochs=50, 
        verbose=True
    )
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_predictions = network.predict(X_train)
    train_accuracy = accuracy(y_train_onehot, train_predictions)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = network.predict(X_test)
    test_accuracy = accuracy(y_test_onehot, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(10, len(X_test))):
        pred_class = np.argmax(test_predictions[i])
        actual_class = y_test[i]
        confidence = test_predictions[i][pred_class]
        print(f"Sample {i+1}: Predicted={pred_class}, Actual={actual_class}, Confidence={confidence:.3f}")
    
    # Plot training history
    try:
        plot_training_history(history, save_path='mnist_training_history.png')
    except:
        print("Could not generate training history plot")
    
    # Final results
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Training Loss: {history['loss'][-1]:.6f}")
    
    if test_accuracy > 0.8:
        print("🎉 Great performance! The network learned to classify digits well.")
    elif test_accuracy > 0.6:
        print("👍 Good performance! The network shows decent learning.")
    else:
        print("🤔 The network might need more training or architecture changes.")


if __name__ == "__main__":
    main()
