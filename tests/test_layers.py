"""Tests for neural network layers."""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.layers import Dense


class TestDense(unittest.TestCase):
    """Test cases for Dense layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.layer = Dense(3, 2)
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.weights.shape, (2, 3))
        self.assertEqual(self.layer.biases.shape, (2, 1))
        self.assertIsNone(self.layer.dweights)
        self.assertIsNone(self.layer.dbiases)
    
    def test_forward_pass(self):
        """Test forward propagation."""
        input_data = np.array([1.0, 2.0, 3.0])
        output = self.layer.forward(input_data)
        
        self.assertEqual(output.shape, (2, 1))
        self.assertIsNotNone(self.layer.input)
    
    def test_backward_pass(self):
        """Test backward propagation."""
        # Forward pass first
        input_data = np.array([1.0, 2.0, 3.0])
        self.layer.forward(input_data)
        
        # Backward pass
        output_gradient = np.array([[0.5], [0.3]])
        input_gradient = self.layer.backward(output_gradient)
        
        self.assertEqual(input_gradient.shape, (3, 1))
        self.assertIsNotNone(self.layer.dweights)
        self.assertIsNotNone(self.layer.dbiases)
        self.assertEqual(self.layer.dweights.shape, (2, 3))
        self.assertEqual(self.layer.dbiases.shape, (2, 1))
    
    def test_gradient_shapes(self):
        """Test that gradients have correct shapes."""
        input_data = np.array([1.0, 2.0, 3.0])
        self.layer.forward(input_data)
        
        output_gradient = np.array([[0.1], [0.2]])
        self.layer.backward(output_gradient)
        
        self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)
        self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)


if __name__ == '__main__':
    unittest.main()
