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
    """
    Fully connected (dense) layer.

    Implements a linear transformation: output = weights * input + bias
    Includes proper gradient computation for backpropagation.

    Args:
        input_size (int): Number of input features
        output_size (int): Number of output neurons

    Attributes:
        weights (numpy.ndarray): Weight matrix with shape (output_size, input_size)
        biases (numpy.ndarray): Bias vector with shape (output_size, 1)
        dweights (numpy.ndarray): Gradients w.r.t. weights (computed during backward pass)
        dbiases (numpy.ndarray): Gradients w.r.t. biases (computed during backward pass)

    Example:
        >>> layer = Dense(784, 128)  # 784 inputs, 128 outputs
        >>> output = layer.forward(input_data)
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize weights with small random values
        self.weights = np.random.randn(output_size, input_size) * 0.1
        # Initialize biases to zero
        self.biases = np.zeros((output_size, 1))
        # Gradients (computed during backward pass)
        self.dweights = None
        self.dbiases = None

    def forward(self, input_data):
        """
        Forward pass: output = weights * input + bias

        Args:
            input_data (numpy.ndarray): Input data with shape (input_size,) or (input_size, 1)

        Returns:
            numpy.ndarray: Output with shape (output_size, 1)
        """
        self.input = input_data.reshape(-1, 1) if input_data.ndim == 1 else input_data
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate=None):
        """
        Backward pass: compute gradients and return input gradient

        Args:
            output_gradient (numpy.ndarray): Gradient from the next layer
            learning_rate (float, optional): Learning rate (unused, kept for API consistency)

        Returns:
            numpy.ndarray: Gradient w.r.t. input for the previous layer

        Note:
            This method computes and stores gradients in self.dweights and self.dbiases
            for use by the optimizer during parameter updates.
        """
        # Ensure output_gradient has proper shape
        if output_gradient.ndim == 1:
            output_gradient = output_gradient.reshape(-1, 1)

        # Store gradients for optimizer
        self.dweights = np.dot(output_gradient, self.input.T)
        self.dbiases = output_gradient

        # Return input gradient for previous layer
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient
