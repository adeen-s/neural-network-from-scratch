"""Regularization techniques for neural networks."""

import numpy as np
from .layers import Layer


class Dropout(Layer):
    """
    Dropout layer for regularization.
    
    Randomly sets a fraction of input units to 0 during training,
    which helps prevent overfitting.
    """
    
    def __init__(self, rate: float = 0.5):
        """
        Initialize dropout layer.
        
        Args:
            rate: Fraction of input units to drop (between 0 and 1)
        """
        super().__init__()
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        self.rate = rate
        self.training = True
        self.mask = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through dropout layer.
        
        Args:
            input_data: Input data
            
        Returns:
            Output with dropout applied (if training)
        """
        self.input = input_data
        
        if self.training:
            # Create random mask
            self.mask = np.random.binomial(1, 1 - self.rate, input_data.shape)
            # Scale by 1/(1-rate) to maintain expected value
            return input_data * self.mask / (1 - self.rate)
        else:
            # During inference, return input unchanged
            return input_data
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass through dropout layer.
        
        Args:
            output_gradient: Gradient from next layer
            learning_rate: Learning rate (unused)
            
        Returns:
            Gradient for previous layer
        """
        if self.training and self.mask is not None:
            return output_gradient * self.mask / (1 - self.rate)
        else:
            return output_gradient
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training


class L1Regularizer:
    """L1 (Lasso) regularization."""
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        Initialize L1 regularizer.
        
        Args:
            lambda_reg: Regularization strength
        """
        self.lambda_reg = lambda_reg
    
    def __call__(self, weights: np.ndarray) -> float:
        """
        Compute L1 regularization loss.
        
        Args:
            weights: Weight matrix
            
        Returns:
            L1 regularization loss
        """
        return self.lambda_reg * np.sum(np.abs(weights))
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute L1 regularization gradient.
        
        Args:
            weights: Weight matrix
            
        Returns:
            L1 regularization gradient
        """
        return self.lambda_reg * np.sign(weights)


class L2Regularizer:
    """L2 (Ridge) regularization."""
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        Initialize L2 regularizer.
        
        Args:
            lambda_reg: Regularization strength
        """
        self.lambda_reg = lambda_reg
    
    def __call__(self, weights: np.ndarray) -> float:
        """
        Compute L2 regularization loss.
        
        Args:
            weights: Weight matrix
            
        Returns:
            L2 regularization loss
        """
        return 0.5 * self.lambda_reg * np.sum(weights ** 2)
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute L2 regularization gradient.
        
        Args:
            weights: Weight matrix
            
        Returns:
            L2 regularization gradient
        """
        return self.lambda_reg * weights


class ElasticNetRegularizer:
    """Elastic Net regularization (combination of L1 and L2)."""
    
    def __init__(self, lambda_reg: float = 0.01, l1_ratio: float = 0.5):
        """
        Initialize Elastic Net regularizer.
        
        Args:
            lambda_reg: Overall regularization strength
            l1_ratio: Ratio of L1 to L2 regularization (0 = pure L2, 1 = pure L1)
        """
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio
        self.l1_reg = L1Regularizer(lambda_reg * l1_ratio)
        self.l2_reg = L2Regularizer(lambda_reg * (1 - l1_ratio))
    
    def __call__(self, weights: np.ndarray) -> float:
        """
        Compute Elastic Net regularization loss.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Elastic Net regularization loss
        """
        return self.l1_reg(weights) + self.l2_reg(weights)
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute Elastic Net regularization gradient.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Elastic Net regularization gradient
        """
        return self.l1_reg.gradient(weights) + self.l2_reg.gradient(weights)


class BatchNormalization(Layer):
    """
    Batch normalization layer.
    
    Normalizes inputs to have zero mean and unit variance,
    then applies learnable scale and shift parameters.
    """
    
    def __init__(self, epsilon: float = 1e-8, momentum: float = 0.99):
        """
        Initialize batch normalization layer.
        
        Args:
            epsilon: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.training = True
        
        # Learnable parameters (initialized after first forward pass)
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        
        # Running statistics for inference
        self.running_mean = None
        self.running_var = None
        
        # Cache for backward pass
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through batch normalization.
        
        Args:
            input_data: Input data
            
        Returns:
            Normalized and scaled output
        """
        self.input = input_data
        
        # Initialize parameters on first forward pass
        if self.gamma is None:
            self.gamma = np.ones(input_data.shape[-1])
            self.beta = np.zeros(input_data.shape[-1])
            self.running_mean = np.zeros(input_data.shape[-1])
            self.running_var = np.ones(input_data.shape[-1])
        
        if self.training:
            # Compute batch statistics
            self.batch_mean = np.mean(input_data, axis=0)
            self.batch_var = np.var(input_data, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize using batch statistics
            self.normalized = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        else:
            # Normalize using running statistics
            self.normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        return self.gamma * self.normalized + self.beta
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float = None) -> np.ndarray:
        """
        Backward pass through batch normalization.
        
        Args:
            output_gradient: Gradient from next layer
            learning_rate: Learning rate (unused)
            
        Returns:
            Gradient for previous layer
        """
        if not self.training:
            # Simplified gradient for inference mode
            return output_gradient * self.gamma / np.sqrt(self.running_var + self.epsilon)
        
        batch_size = self.input.shape[0]
        
        # Gradients w.r.t. gamma and beta
        self.dgamma = np.sum(output_gradient * self.normalized, axis=0)
        self.dbeta = np.sum(output_gradient, axis=0)
        
        # Gradient w.r.t. normalized input
        dnormalized = output_gradient * self.gamma
        
        # Gradient w.r.t. variance
        dvar = np.sum(dnormalized * (self.input - self.batch_mean), axis=0) * \
               (-0.5) * (self.batch_var + self.epsilon) ** (-1.5)
        
        # Gradient w.r.t. mean
        dmean = np.sum(dnormalized * (-1) / np.sqrt(self.batch_var + self.epsilon), axis=0) + \
                dvar * np.sum(-2 * (self.input - self.batch_mean), axis=0) / batch_size
        
        # Gradient w.r.t. input
        dx = dnormalized / np.sqrt(self.batch_var + self.epsilon) + \
             dvar * 2 * (self.input - self.batch_mean) / batch_size + \
             dmean / batch_size
        
        return dx
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
