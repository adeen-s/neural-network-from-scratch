"""
Simple regression example using neural network from scratch.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU
from src.core.losses import MeanSquaredError
from src.core.optimizers import Adam
from src.utils.metrics import mean_squared_error, r2_score


def generate_data(n_samples=100):
    """Generate synthetic regression data."""
    np.random.seed(42)
    X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
    y = 0.5 * X**2 + 0.3 * X + 0.1 + np.random.normal(0, 0.1, X.shape)
    return X.flatten(), y.flatten()


def main():
    """Train a neural network for regression."""
    print("Regression Example - Neural Network from Scratch")
    print("=" * 50)
    
    # Generate data
    X, y = generate_data(100)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create network
    network = NeuralNetwork()
    network.add(Dense(1, 10))
    network.add(ReLU())
    network.add(Dense(10, 10))
    network.add(ReLU())
    network.add(Dense(10, 1))
    
    # Compile network
    network.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=MeanSquaredError()
    )
    
    print("\nNetwork architecture:")
    print("Input (1) -> Dense(10) -> ReLU -> Dense(10) -> ReLU -> Dense(1) -> Output")
    
    # Train network
    print("\nTraining...")
    history = network.fit(X, y, epochs=500, verbose=True)
    
    # Make predictions
    predictions = network.predict(X)
    predictions = predictions.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"\nResults:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    # Plot results
    try:
        plt.figure(figsize=(10, 6))
        
        # Sort for plotting
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        pred_sorted = predictions[sort_idx]
        
        plt.scatter(X, y, alpha=0.6, label='True data', color='blue')
        plt.plot(X_sorted, pred_sorted, 'r-', label='Predictions', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Neural Network Regression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'regression_results.png'")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
