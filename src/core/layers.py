"""Neural network layer implementations."""

import numpy as np


class Layer:
    """Base class for all neural network layers."""
    
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """Forward pass through the layer."""
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass through the layer."""
        raise NotImplementedError


class Dense(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.zeros((output_size, 1))
    
    def forward(self, input_data):
        """Forward pass: output = weights * input + bias"""
        self.input = input_data
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass: compute gradients and update weights"""
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        return input_gradient
