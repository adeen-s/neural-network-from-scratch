"""
XOR problem demonstration using neural network from scratch.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Sigmoid
from src.core.losses import MeanSquaredError
from src.core.optimizers import Adam
from src.utils.metrics import accuracy


def generate_xor_data():
    """Generate XOR dataset."""
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float32)

    return X, y


def main():
    """Train a neural network to solve the XOR problem."""
    print("XOR Problem - Neural Network from Scratch")
    print("=" * 40)

    # Generate data
    X, y = generate_xor_data()
    print(f"Training data shape: {X.shape}")
    print(f"Target data shape: {y.shape}")

    # Create network
    network = NeuralNetwork()
    network.add(Dense(2, 4))
    network.add(ReLU())
    network.add(Dense(4, 1))
    network.add(Sigmoid())

    # Compile network
    network.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=MeanSquaredError()
    )

    print("\nNetwork architecture:")
    print("Input (2) -> Dense(4) -> ReLU -> Dense(1) -> Sigmoid -> Output")

    # Train network
    print("\nTraining...")
    history = network.fit(X, y, epochs=1000, verbose=True, validation_split=0.0)

    # Test network
    print("\nTesting XOR function:")
    predictions = network.predict(X)

    for i in range(len(X)):
        pred = predictions[i][0]
        actual = y[i][0]
        print(f"Input: {X[i]} -> Predicted: {pred.item():.4f}, Actual: {actual.item()}")

    # Calculate accuracy (using 0.5 threshold)
    binary_preds = (predictions > 0.5).astype(int)
    acc = accuracy(y, binary_preds)
    print(f"\nAccuracy: {acc:.4f}")

    # Final loss
    final_loss = history['loss'][-1]
    print(f"Final loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()
