"""
Optimization algorithms for neural network training.
"""
import numpy as np


class Optimizer:
    """Base class for all optimizers."""

    def update(self, params, grads):
        """Update parameters using gradients."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """Update parameters using SGD."""
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class Momentum(Optimizer):
    """SGD with momentum optimizer."""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must be in range [0, 1)")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = []

    def update(self, params, grads):
        """Update parameters using momentum."""
        if not self.velocities:
            self.velocities = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
            param += self.velocities[i]


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []  # First moment estimates
        self.v = []  # Second moment estimates
        self.t = 0   # Time step

    def update(self, params, grads):
        """Update parameters using Adam."""
        self.t += 1

        if not self.m:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
