# Neural Networks from Scratch

A Python implementation of neural networks built from the ground up using only NumPy.

## Overview

This project demonstrates the fundamental concepts of neural networks by implementing them from scratch without using high-level machine learning frameworks like TensorFlow or PyTorch. The implementation includes robust shape handling, numerically stable gradient computation, and comprehensive testing.

## Features

### Core Components
- **Dense/Linear layers** with proper gradient computation
- **Multiple activation functions** (ReLU, Sigmoid, Tanh, Softmax) with stable backpropagation
- **Various loss functions** (MSE, Categorical Cross-entropy) with numerically stable gradients
- **Optimizers** (SGD, Momentum, Adam) with parameter update mechanisms
- **Robust shape handling** for both single samples and batch processing
- **Training and evaluation utilities** with comprehensive metrics

### Key Improvements
- ✅ **Fixed shape mismatch errors** in gradient computation
- ✅ **Numerically stable** Softmax + Categorical Cross-entropy combination
- ✅ **Proper gradient flow** through all layer types
- ✅ **Comprehensive testing** with 19 passing unit tests
- ✅ **Example implementations** demonstrating various use cases

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
# Expected: Perfect accuracy (1.0000) on XOR logic gate

# MNIST Digit Classification - Real-world dataset
python examples/mnist_example.py
# Expected: High accuracy on digit classification (>95% on synthetic data)

# Regression Example - Function approximation
python examples/regression_example.py
# Expected: Low MSE on function approximation tasks

# Classification Demo - Multiple synthetic datasets
python examples/classification_demo.py
# Expected: Good performance on circles, moons, and random datasets
```

### Example Output
```
MNIST Digit Classification - Neural Network from Scratch
============================================================
Training network...
Epoch 1/50, Loss: 0.645851
Epoch 11/50, Loss: 0.000305
...
Epoch 50/50, Loss: 0.000017

Training Accuracy: 1.0000
Test Accuracy: 1.0000
🎉 Great performance! The network learned to classify digits well.
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

## Technical Details

### Architecture
- **Modular design** with separate components for layers, activations, losses, and optimizers
- **Consistent API** following common deep learning framework patterns
- **NumPy-only implementation** for educational clarity and minimal dependencies

### Key Technical Improvements
1. **Shape Handling**: All layers now properly handle both 1D and 2D input shapes
2. **Gradient Computation**: Fixed matrix multiplication errors in backpropagation
3. **Numerical Stability**: Improved Softmax and loss function implementations
4. **Memory Efficiency**: Optimized gradient accumulation for batch processing

### Supported Operations
- **Forward Pass**: Efficient matrix operations for prediction
- **Backward Pass**: Automatic gradient computation through all layers
- **Batch Processing**: Support for mini-batch training with gradient accumulation
- **Parameter Updates**: Integration with various optimization algorithms

## Troubleshooting

### Common Issues

**Shape Mismatch Errors**:
- ✅ **Fixed**: The implementation now handles shape mismatches automatically
- All gradients are properly reshaped to column vectors when needed

**Numerical Instability**:
- ✅ **Fixed**: Softmax uses numerical stability techniques (subtracting max)
- Loss functions include clipping to prevent log(0) and division by 0

**Poor Convergence**:
- Try different learning rates (0.001 - 0.01 work well)
- Use Adam optimizer for better convergence
- Ensure proper data normalization

### Performance Tips
- Use batch processing for larger datasets (`batch_size=32` or higher)
- Normalize input data to [0, 1] or [-1, 1] range
- Start with smaller networks and gradually increase complexity

## License

MIT License
