"""Loss function implementations."""

import numpy as np


class Loss:
    """Base class for loss functions."""

    def forward(self, y_true, y_pred):
        """Calculate the loss."""
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        """Calculate the gradient of the loss."""
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean Squared Error loss function."""

    def forward(self, y_true, y_pred):
        """Calculate MSE loss."""
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        """Calculate MSE gradient."""
        return 2 * (y_pred - y_true) / np.size(y_true)


class CategoricalCrossentropy(Loss):
    """
    Categorical cross-entropy loss function.

    Used for multi-class classification problems where each sample belongs
    to exactly one class. Expects one-hot encoded target labels.

    The loss is computed as: -sum(y_true * log(y_pred))

    Note:
        When used with softmax activation, the gradient computation simplifies
        to (y_pred - y_true), which is more numerically stable.
    """

    def forward(self, y_true, y_pred):
        """
        Calculate categorical cross-entropy loss.

        Args:
            y_true (numpy.ndarray): True labels (one-hot encoded)
            y_pred (numpy.ndarray): Predicted probabilities

        Returns:
            float: Loss value
        """
        # Ensure proper shapes
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred))

    def backward(self, y_true, y_pred):
        """
        Calculate categorical cross-entropy gradient.

        Args:
            y_true (numpy.ndarray): True labels (one-hot encoded)
            y_pred (numpy.ndarray): Predicted probabilities

        Returns:
            numpy.ndarray: Gradient w.r.t. predictions

        Note:
            For softmax + categorical cross-entropy, this returns (y_pred - y_true)
            which is the simplified and numerically stable gradient.
        """
        # Ensure proper shapes
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        # For softmax + categorical crossentropy, the gradient simplifies to (y_pred - y_true)
        # This is more numerically stable than -y_true / y_pred
        return y_pred - y_true


class BinaryCrossentropy(Loss):
    """Binary cross-entropy loss function."""

    def forward(self, y_true, y_pred):
        """Calculate binary cross-entropy loss."""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        """Calculate binary cross-entropy gradient."""
        # Clip predictions to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


# Legacy function implementations for backward compatibility
def mse(y_true, y_pred):
    """Mean Squared Error loss function."""
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """Derivative of Mean Squared Error."""
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy loss function."""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_crossentropy_prime(y_true, y_pred):
    """Derivative of binary cross-entropy."""
    # Clip predictions to prevent division by 0
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
