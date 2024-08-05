"""Loss function implementations."""

import numpy as np


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
