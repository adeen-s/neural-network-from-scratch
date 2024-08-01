# Neural Networks from Scratch

A Python implementation of neural networks built from the ground up using only NumPy.

## Overview

This project demonstrates the fundamental concepts of neural networks by implementing them from scratch without using high-level machine learning frameworks like TensorFlow or PyTorch.

## Features

- Dense/Linear layers
- Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Various loss functions (MSE, Cross-entropy)
- Optimizers (SGD, Momentum, Adam)
- Training and evaluation utilities
- Example implementations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Softmax

# Create a simple network
network = NeuralNetwork()
network.add(Dense(784, 128))
network.add(ReLU())
network.add(Dense(128, 10))
network.add(Softmax())

# Train the network
network.fit(X_train, y_train, epochs=100)
```

## Project Structure

```
src/
├── core/           # Core neural network components
├── utils/          # Utility functions
└── datasets/       # Dataset loaders

examples/           # Example implementations
tests/             # Unit tests
notebooks/         # Jupyter notebooks with tutorials
```

## Examples

- XOR Problem
- MNIST Digit Classification
- Simple Regression
- Binary Classification

## License

MIT License
