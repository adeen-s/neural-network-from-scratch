"""Tests for the neural network class."""

import pytest
import numpy as np
from src.core.network import NeuralNetwork
from src.core.layers import Dense
from src.core.activations import ReLU, Sigmoid
from src.core.losses import MeanSquaredError
from src.core.optimizers import SGD


class TestNeuralNetwork:
    """Test cases for NeuralNetwork class."""

    def test_network_initialization(self):
        """Test network initialization."""
        network = NeuralNetwork()
        assert len(network.layers) == 0
        assert network.loss_function is None
        assert network.optimizer is None
        assert not network.compiled

    def test_add_layer(self):
        """Test adding layers to network."""
        network = NeuralNetwork()
        layer = Dense(10, 5)
        network.add(layer)
        assert len(network.layers) == 1
        assert network.layers[0] == layer

    def test_add_invalid_layer(self):
        """Test adding invalid layer raises error."""
        network = NeuralNetwork()
        with pytest.raises(TypeError):
            network.add("not a layer")

    def test_compile_network(self):
        """Test network compilation."""
        network = NeuralNetwork()
        network.compile(optimizer='sgd', loss='mse', learning_rate=0.01)
        assert network.compiled
        assert isinstance(network.optimizer, SGD)
        assert isinstance(network.loss_function, MeanSquaredError)

    def test_predict_before_compile(self):
        """Test prediction before compilation raises error."""
        network = NeuralNetwork()
        network.add(Dense(2, 1))
        with pytest.raises(RuntimeError):
            network.predict(np.array([[1, 2]]))

    def test_fit_before_compile(self):
        """Test training before compilation raises error."""
        network = NeuralNetwork()
        network.add(Dense(2, 1))
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1], [0]])
        with pytest.raises(RuntimeError):
            network.fit(X, y, epochs=1)

    def test_simple_network_training(self):
        """Test simple network can train without errors."""
        # Create simple network
        network = NeuralNetwork()
        network.add(Dense(2, 3))
        network.add(ReLU())
        network.add(Dense(3, 1))
        network.add(Sigmoid())
        
        # Compile network
        network.compile(optimizer='sgd', loss='mse', learning_rate=0.1)
        
        # Create simple training data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Train for a few epochs
        history = network.fit(X, y, epochs=5, verbose=False)
        
        # Check that training completed
        assert len(history) == 5
        assert all(isinstance(loss, (int, float)) for loss in history)

    def test_prediction_output_shape(self):
        """Test prediction output has correct shape."""
        network = NeuralNetwork()
        network.add(Dense(2, 1))
        network.compile(optimizer='sgd', loss='mse')
        
        X = np.array([[1, 2], [3, 4]])
        predictions = network.predict(X)
        
        assert predictions.shape == (2, 1)
        assert isinstance(predictions, np.ndarray)

    def test_network_with_multiple_layers(self):
        """Test network with multiple layers."""
        network = NeuralNetwork()
        network.add(Dense(4, 8))
        network.add(ReLU())
        network.add(Dense(8, 4))
        network.add(ReLU())
        network.add(Dense(4, 1))
        
        network.compile(optimizer='sgd', loss='mse')
        
        X = np.random.randn(10, 4)
        predictions = network.predict(X)
        
        assert predictions.shape == (10, 1)
