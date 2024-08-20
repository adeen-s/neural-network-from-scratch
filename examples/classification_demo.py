"""
Classification demonstration with various synthetic datasets.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Sigmoid, Softmax
from src.core.losses import MeanSquaredError, CategoricalCrossentropy
from src.core.optimizers import Adam
from src.datasets.synthetic import make_circles, make_moons, make_classification
from src.utils.data_loader import train_test_split, to_categorical
from src.utils.metrics import accuracy
from src.utils.visualization import plot_decision_boundary


def test_binary_classification(dataset_name, X, y):
    """Test binary classification on a dataset."""
    print(f"\n{dataset_name} Dataset")
    print("-" * 30)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create network
    network = NeuralNetwork()
    network.add(Dense(2, 8))
    network.add(ReLU())
    network.add(Dense(8, 4))
    network.add(ReLU())
    network.add(Dense(4, 1))
    network.add(Sigmoid())
    
    # Compile
    network.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=MeanSquaredError()
    )
    
    # Train
    print("Training...")
    history = network.fit(X_train, y_train.reshape(-1, 1), epochs=300, verbose=False)
    
    # Evaluate
    train_pred = network.predict(X_train)
    test_pred = network.predict(X_test)
    
    train_acc = accuracy(y_train, (train_pred > 0.5).astype(int))
    test_acc = accuracy(y_test, (test_pred > 0.5).astype(int))
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Final Loss: {history['loss'][-1]:.6f}")
    
    # Plot decision boundary
    try:
        plot_decision_boundary(network, X, y, save_path=f'{dataset_name.lower()}_boundary.png')
    except:
        print("Could not generate decision boundary plot")
    
    return test_acc


def test_multiclass_classification():
    """Test multiclass classification."""
    print("\nMulticlass Classification")
    print("-" * 30)
    
    # Generate data
    X, y = make_classification(n_samples=300, n_features=2, n_classes=3, 
                              n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to one-hot
    y_train_onehot = to_categorical(y_train, 3)
    y_test_onehot = to_categorical(y_test, 3)
    
    # Create network
    network = NeuralNetwork()
    network.add(Dense(2, 10))
    network.add(ReLU())
    network.add(Dense(10, 6))
    network.add(ReLU())
    network.add(Dense(6, 3))
    network.add(Softmax())
    
    # Compile
    network.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=CategoricalCrossentropy()
    )
    
    # Train
    print("Training...")
    history = network.fit(X_train, y_train_onehot, epochs=400, verbose=False)
    
    # Evaluate
    train_pred = network.predict(X_train)
    test_pred = network.predict(X_test)
    
    train_acc = accuracy(y_train_onehot, train_pred)
    test_acc = accuracy(y_test_onehot, test_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Final Loss: {history['loss'][-1]:.6f}")
    
    return test_acc


def main():
    """Run classification demonstrations."""
    print("Neural Network Classification Demo")
    print("=" * 40)
    
    results = {}
    
    # Test binary classification datasets
    print("\nBinary Classification Tasks:")
    
    # Circles dataset
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, random_state=42)
    results['Circles'] = test_binary_classification("Circles", X_circles, y_circles)
    
    # Moons dataset
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    results['Moons'] = test_binary_classification("Moons", X_moons, y_moons)
    
    # Random classification
    X_random, y_random = make_classification(n_samples=200, n_features=2, n_classes=2, 
                                           random_state=42)
    results['Random'] = test_binary_classification("Random", X_random, y_random)
    
    # Multiclass classification
    results['Multiclass'] = test_multiclass_classification()
    
    # Summary
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    for dataset, acc in results.items():
        print(f"{dataset:12}: {acc:.4f}")
    
    avg_acc = np.mean(list(results.values()))
    print(f"{'Average':12}: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
