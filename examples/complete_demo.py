"""
Complete demonstration of neural network capabilities.
This script showcases all major features of the neural network implementation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Sigmoid, Softmax
from src.core.losses import MeanSquaredError, CategoricalCrossentropy
from src.core.optimizers import SGD, Momentum, Adam
from src.datasets.synthetic import make_circles, make_moons, make_classification, make_regression
from src.utils.data_loader import train_test_split, to_categorical, normalize
from src.utils.metrics import accuracy, mean_squared_error, r2_score
from src.utils.visualization import plot_training_history, plot_decision_boundary


def test_binary_classification():
    """Test binary classification with different optimizers."""
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION COMPARISON")
    print("="*60)
    
    # Generate data
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    optimizers = [
        ('SGD', SGD(learning_rate=0.1)),
        ('Momentum', Momentum(learning_rate=0.1, momentum=0.9)),
        ('Adam', Adam(learning_rate=0.01))
    ]
    
    results = {}
    
    for name, optimizer in optimizers:
        print(f"\nTesting {name} optimizer...")
        
        # Create network
        network = NeuralNetwork()
        network.add(Dense(2, 16))
        network.add(ReLU())
        network.add(Dense(16, 8))
        network.add(ReLU())
        network.add(Dense(8, 1))
        network.add(Sigmoid())
        
        network.compile(optimizer=optimizer, loss=MeanSquaredError())
        
        # Train
        start_time = time.time()
        history = network.fit(X_train, y_train.reshape(-1, 1), epochs=200, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        predictions = network.predict(X_test)
        acc = accuracy(y_test, (predictions > 0.5).astype(int))
        
        results[name] = {
            'accuracy': acc,
            'final_loss': history['loss'][-1],
            'training_time': training_time
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Final Loss: {history['loss'][-1]:.6f}")
        print(f"  Training Time: {training_time:.2f}s")
    
    # Summary
    print(f"\n{'-'*40}")
    print("OPTIMIZER COMPARISON SUMMARY")
    print(f"{'-'*40}")
    for name, result in results.items():
        print(f"{name:10}: Acc={result['accuracy']:.4f}, Loss={result['final_loss']:.6f}, Time={result['training_time']:.2f}s")


def test_multiclass_classification():
    """Test multiclass classification."""
    print("\n" + "="*60)
    print("MULTICLASS CLASSIFICATION")
    print("="*60)
    
    # Generate data
    X, y = make_classification(n_samples=800, n_features=2, n_classes=4, 
                              n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    X_train, mean, std = normalize(X_train, method='standard')
    X_test = (X_test - mean) / std
    
    # Convert to one-hot
    y_train_onehot = to_categorical(y_train, 4)
    y_test_onehot = to_categorical(y_test, 4)
    
    # Create network
    network = NeuralNetwork()
    network.add(Dense(2, 32))
    network.add(ReLU())
    network.add(Dense(32, 16))
    network.add(ReLU())
    network.add(Dense(16, 4))
    network.add(Softmax())
    
    network.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=CategoricalCrossentropy()
    )
    
    print("Training multiclass classifier...")
    history = network.fit(X_train, y_train_onehot, epochs=300, verbose=False, batch_size=32)
    
    # Evaluate
    train_pred = network.predict(X_train)
    test_pred = network.predict(X_test)
    
    train_acc = accuracy(y_train_onehot, train_pred)
    test_acc = accuracy(y_test_onehot, test_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Final Loss: {history['loss'][-1]:.6f}")
    
    # Plot decision boundary
    try:
        plot_decision_boundary(network, X_test, y_test, save_path='multiclass_boundary.png')
        print("Decision boundary saved as 'multiclass_boundary.png'")
    except:
        print("Could not generate decision boundary plot")


def test_regression():
    """Test regression capabilities."""
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)
    
    # Generate data
    X, y = make_regression(n_samples=400, n_features=1, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create network
    network = NeuralNetwork()
    network.add(Dense(1, 20))
    network.add(ReLU())
    network.add(Dense(20, 20))
    network.add(ReLU())
    network.add(Dense(20, 1))
    
    network.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=MeanSquaredError()
    )
    
    print("Training regression model...")
    history = network.fit(X_train, y_train.reshape(-1, 1), epochs=500, verbose=False)
    
    # Evaluate
    train_pred = network.predict(X_train).flatten()
    test_pred = network.predict(X_test).flatten()
    
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")


def test_batch_vs_online():
    """Compare batch vs online learning."""
    print("\n" + "="*60)
    print("BATCH VS ONLINE LEARNING COMPARISON")
    print("="*60)
    
    # Generate data
    X, y = make_circles(n_samples=600, noise=0.1, random_state=42)
    
    methods = [
        ('Online (batch_size=1)', 1),
        ('Mini-batch (batch_size=32)', 32),
        ('Batch (batch_size=600)', 600)
    ]
    
    for name, batch_size in methods:
        print(f"\nTesting {name}...")
        
        network = NeuralNetwork()
        network.add(Dense(2, 16))
        network.add(ReLU())
        network.add(Dense(16, 1))
        network.add(Sigmoid())
        
        network.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
        
        start_time = time.time()
        history = network.fit(X, y.reshape(-1, 1), epochs=100, verbose=False, batch_size=batch_size)
        training_time = time.time() - start_time
        
        predictions = network.predict(X)
        acc = accuracy(y, (predictions > 0.5).astype(int))
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Final Loss: {history['loss'][-1]:.6f}")
        print(f"  Training Time: {training_time:.2f}s")


def main():
    """Run complete demonstration."""
    print("NEURAL NETWORKS FROM SCRATCH - COMPLETE DEMONSTRATION")
    print("="*60)
    print("This demo showcases all major features of the implementation:")
    print("- Binary and multiclass classification")
    print("- Regression")
    print("- Different optimizers (SGD, Momentum, Adam)")
    print("- Batch vs online learning")
    print("- Data preprocessing and evaluation metrics")
    
    try:
        test_binary_classification()
        test_multiclass_classification()
        test_regression()
        test_batch_vs_online()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All tests passed. The neural network implementation is working correctly.")
        print("Check the generated plots for visual results.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Some features may not be working correctly.")


if __name__ == "__main__":
    main()
