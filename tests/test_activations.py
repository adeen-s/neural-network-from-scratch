"""Tests for activation functions."""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.activations import ReLU, Sigmoid, Tanh, Softmax


class TestActivations(unittest.TestCase):
    """Test cases for activation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_input = np.array([[-2.0], [0.0], [2.0]])
        self.small_input = np.array([[0.1], [0.5], [0.9]])
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        relu = ReLU()
        output = relu.forward(self.test_input)
        
        expected = np.array([[0.0], [0.0], [2.0]])
        np.testing.assert_array_equal(output, expected)
    
    def test_relu_backward(self):
        """Test ReLU backward pass."""
        relu = ReLU()
        relu.forward(self.test_input)
        
        grad_output = np.array([[1.0], [1.0], [1.0]])
        grad_input = relu.backward(grad_output)
        
        expected = np.array([[0.0], [0.0], [1.0]])
        np.testing.assert_array_equal(grad_input, expected)
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        sigmoid = Sigmoid()
        output = sigmoid.forward(self.test_input)
        
        # Check output is in valid range [0, 1]
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
        
        # Check specific values
        self.assertAlmostEqual(output[1, 0], 0.5, places=5)  # sigmoid(0) = 0.5
    
    def test_sigmoid_backward(self):
        """Test Sigmoid backward pass."""
        sigmoid = Sigmoid()
        output = sigmoid.forward(self.small_input)
        
        grad_output = np.ones_like(output)
        grad_input = sigmoid.backward(grad_output)
        
        # Gradient should be positive for all inputs
        self.assertTrue(np.all(grad_input > 0))
    
    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        tanh = Tanh()
        output = tanh.forward(self.test_input)
        
        # Check output is in valid range [-1, 1]
        self.assertTrue(np.all(output >= -1))
        self.assertTrue(np.all(output <= 1))
        
        # Check specific values
        self.assertAlmostEqual(output[1, 0], 0.0, places=5)  # tanh(0) = 0
    
    def test_tanh_backward(self):
        """Test Tanh backward pass."""
        tanh = Tanh()
        tanh.forward(self.small_input)
        
        grad_output = np.ones_like(self.small_input)
        grad_input = tanh.backward(grad_output)
        
        # Gradient should be positive for small inputs
        self.assertTrue(np.all(grad_input > 0))
    
    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        softmax = Softmax()
        test_input = np.array([[1.0], [2.0], [3.0]])
        output = softmax.forward(test_input)
        
        # Check output sums to 1
        self.assertAlmostEqual(np.sum(output), 1.0, places=5)
        
        # Check all values are positive
        self.assertTrue(np.all(output > 0))
        
        # Check monotonicity (larger input -> larger output)
        self.assertTrue(output[0, 0] < output[1, 0] < output[2, 0])
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability with large inputs."""
        softmax = Softmax()
        large_input = np.array([[1000.0], [1001.0], [1002.0]])
        output = softmax.forward(large_input)
        
        # Should not contain NaN or inf
        self.assertFalse(np.any(np.isnan(output)))
        self.assertFalse(np.any(np.isinf(output)))
        
        # Should still sum to 1
        self.assertAlmostEqual(np.sum(output), 1.0, places=5)
    
    def test_activation_shapes(self):
        """Test that activations preserve input shapes."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        for activation in activations:
            output = activation.forward(self.test_input)
            self.assertEqual(output.shape, self.test_input.shape)
    
    def test_gradient_shapes(self):
        """Test that gradients have correct shapes."""
        activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
        
        for activation in activations:
            activation.forward(self.test_input)
            grad_output = np.ones_like(self.test_input)
            grad_input = activation.backward(grad_output)
            self.assertEqual(grad_input.shape, self.test_input.shape)


if __name__ == '__main__':
    unittest.main()
