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
            # Optimized ReLU with better memory efficiency
            return np.maximum(0, x, out=np.zeros_like(x))

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
    """
    Softmax activation function.

    Applies the softmax function to convert logits to probabilities.
    Uses numerical stability techniques to prevent overflow.

    The softmax function is defined as:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Note:
        When used with categorical cross-entropy loss, the combined gradient
        simplifies to (y_pred - y_true), which is handled in the loss function.
    """

    def forward(self, input_data):
        """
        Forward pass: apply softmax activation

        Args:
            input_data (numpy.ndarray): Input logits

        Returns:
            numpy.ndarray: Softmax probabilities (sum to 1)
        """
        self.input = input_data
        # Subtract max for numerical stability
        exp_values = np.exp(input_data - np.max(input_data))
        self.output = exp_values / np.sum(exp_values)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """
        Backward pass: compute softmax gradients

        Args:
            output_gradient (numpy.ndarray): Gradient from the next layer
            learning_rate (float, optional): Learning rate (unused)

        Returns:
            numpy.ndarray: Gradient w.r.t. input

        Note:
            For softmax + categorical cross-entropy, the gradient computation
            is simplified and handled in the loss function.
        """
        # For softmax with categorical crossentropy, the gradient simplifies
        # When used with categorical crossentropy, the combined gradient is just (y_pred - y_true)
        # This is handled in the loss function, so we just pass through the gradient
        # but ensure proper shape
        if output_gradient.ndim == 1:
            output_gradient = output_gradient.reshape(-1, 1)
        return output_gradient


class LeakyReLU(Activation):
    """Leaky ReLU activation function."""

    def __init__(self, alpha=0.01):
        """
        Initialize Leaky ReLU with leak parameter.

        Args:
            alpha (float): Leak parameter for negative values
        """
        self.alpha = alpha

        def leaky_relu(x):
            return np.where(x > 0, x, alpha * x)

        def leaky_relu_prime(x):
            return np.where(x > 0, 1, alpha)

        super().__init__(leaky_relu, leaky_relu_prime)


class ELU(Activation):
    """Exponential Linear Unit activation function."""

    def __init__(self, alpha=1.0):
        """
        Initialize ELU with alpha parameter.

        Args:
            alpha (float): Scale parameter for negative values
        """
        self.alpha = alpha

        def elu(x):
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))

        def elu_prime(x):
            return np.where(x > 0, 1, alpha * np.exp(x))

        super().__init__(elu, elu_prime)
