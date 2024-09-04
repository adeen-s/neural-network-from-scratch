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

## Quick Start

```python
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Softmax
from src.core.optimizers import Adam
from src.core.losses import CategoricalCrossentropy

# Create a simple network
network = NeuralNetwork()
network.add(Dense(784, 128))
network.add(ReLU())
network.add(Dense(128, 10))
network.add(Softmax())

# Compile the network
network.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy()
)

# Train the network
history = network.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)

# Make predictions
predictions = network.predict(X_test)
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

Run the example scripts to see the neural network in action:

```bash
# XOR Problem - Classic non-linear classification
python examples/xor_problem.py

# MNIST Digit Classification - Real-world dataset
python examples/mnist_example.py

# Regression Example - Function approximation
python examples/regression_example.py

# Classification Demo - Multiple synthetic datasets
python examples/classification_demo.py
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_layers.py -v
```

## Jupyter Notebooks

Explore the interactive tutorials in the `notebooks/` directory:

- `01_basic_concepts.ipynb` - Introduction to neural network concepts
- `02_building_first_network.ipynb` - Step-by-step network construction
- `03_advanced_examples.ipynb` - Advanced techniques and examples

## License

MIT License
