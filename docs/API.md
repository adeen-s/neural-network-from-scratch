# API Documentation

## Core Components

### NeuralNetwork

The main class for building and training neural networks.

```python
from src.core.network import NeuralNetwork

# Create a network
network = NeuralNetwork()
```

#### Methods

**`add(layer)`**
- Adds a layer to the network
- Parameters: `layer` - Any layer object (Dense, ReLU, Softmax, etc.)

**`compile(optimizer='sgd', loss='mse', learning_rate=0.01)`**
- Compiles the network with optimizer and loss function
- Parameters:
  - `optimizer`: String ('sgd', 'adam', 'momentum') or optimizer object
  - `loss`: String ('mse', 'categorical_crossentropy') or loss object
  - `learning_rate`: Float, learning rate for string optimizers

**`fit(x_train, y_train, epochs=100, verbose=False, batch_size=32)`**
- Trains the network
- Parameters:
  - `x_train`: Training input data (numpy array)
  - `y_train`: Training target data (numpy array)
  - `epochs`: Number of training epochs
  - `verbose`: Whether to print training progress
  - `batch_size`: Size of mini-batches for training
- Returns: Dictionary with training history

**`predict(input_data)`**
- Makes predictions on input data
- Parameters: `input_data` - Input data (numpy array)
- Returns: Predictions as numpy array

### Dense Layer

Fully connected layer implementation.

```python
from src.core.layers import Dense

# Create a dense layer
layer = Dense(input_size=784, output_size=128)
```

#### Parameters
- `input_size`: Number of input features
- `output_size`: Number of output neurons

#### Methods
- `forward(input_data)`: Forward pass through the layer
- `backward(output_gradient, learning_rate=None)`: Backward pass for gradient computation

### Activation Functions

#### ReLU
```python
from src.core.activations import ReLU
relu = ReLU()
```

#### Sigmoid
```python
from src.core.activations import Sigmoid
sigmoid = Sigmoid()
```

#### Tanh
```python
from src.core.activations import Tanh
tanh = Tanh()
```

#### Softmax
```python
from src.core.activations import Softmax
softmax = Softmax()
```

All activation functions have:
- `forward(input_data)`: Apply activation function
- `backward(output_gradient, learning_rate=None)`: Compute gradients

### Loss Functions

#### Mean Squared Error
```python
from src.core.losses import MeanSquaredError
mse = MeanSquaredError()
```

#### Categorical Cross-entropy
```python
from src.core.losses import CategoricalCrossentropy
cce = CategoricalCrossentropy()
```

Loss functions have:
- `forward(y_true, y_pred)`: Compute loss value
- `backward(y_true, y_pred)`: Compute loss gradients

### Optimizers

#### SGD (Stochastic Gradient Descent)
```python
from src.core.optimizers import SGD
sgd = SGD(learning_rate=0.01)
```

#### Adam
```python
from src.core.optimizers import Adam
adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
```

#### Momentum
```python
from src.core.optimizers import Momentum
momentum = Momentum(learning_rate=0.01, momentum=0.9)
```

All optimizers have:
- `update(params, grads)`: Update parameters using gradients

## Utility Functions

### Data Loading
```python
from src.utils.data_loader import train_test_split, to_categorical

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to one-hot encoding
y_onehot = to_categorical(y, num_classes=10)
```

### Metrics
```python
from src.utils.metrics import accuracy

# Calculate accuracy
acc = accuracy(y_true, y_pred)
```

### Visualization
```python
from src.utils.visualization import plot_training_history

# Plot training history
plot_training_history(history, save_path='training.png')
```

## Example Usage

### Basic Classification
```python
import numpy as np
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Softmax
from src.core.losses import CategoricalCrossentropy
from src.core.optimizers import Adam

# Create network
network = NeuralNetwork()
network.add(Dense(784, 128))
network.add(ReLU())
network.add(Dense(128, 64))
network.add(ReLU())
network.add(Dense(64, 10))
network.add(Softmax())

# Compile
network.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy()
)

# Train
history = network.fit(X_train, y_train, epochs=50, verbose=True)

# Predict
predictions = network.predict(X_test)
```

### Binary Classification
```python
# For binary classification, use sigmoid activation and binary crossentropy
from src.core.activations import Sigmoid
from src.core.losses import BinaryCrossentropy

network = NeuralNetwork()
network.add(Dense(input_size, 64))
network.add(ReLU())
network.add(Dense(64, 1))
network.add(Sigmoid())

network.compile(
    optimizer='adam',
    loss=BinaryCrossentropy()
)
```

### Regression
```python
# For regression, use linear output and MSE loss
from src.core.losses import MeanSquaredError

network = NeuralNetwork()
network.add(Dense(input_size, 64))
network.add(ReLU())
network.add(Dense(64, 32))
network.add(ReLU())
network.add(Dense(32, 1))  # No activation for regression

network.compile(
    optimizer='adam',
    loss=MeanSquaredError()
)
```
