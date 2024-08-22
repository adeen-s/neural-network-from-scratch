"""Activation function implementations."""

import numpy as np
from .layers import Layer


class Activation(Layer):
    """Base class for activation functions."""

    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class ReLU(Activation):
    """Rectified Linear Unit activation function."""

    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    """Sigmoid activation function."""

    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        def sigmoid_prime(x):
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):
    """Hyperbolic tangent activation function."""

    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Softmax(Layer):
    """Softmax activation function."""

    def forward(self, input_data):
        self.input = input_data
        # Subtract max for numerical stability
        exp_values = np.exp(input_data - np.max(input_data))
        self.output = exp_values / np.sum(exp_values)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        # For softmax, the gradient computation is more complex
        # This is a simplified version that works with categorical crossentropy
        return output_gradient
