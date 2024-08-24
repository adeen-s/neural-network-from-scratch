"""Tests for optimizer implementations."""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.optimizers import SGD, Momentum, Adam


class TestOptimizers(unittest.TestCase):
    """Test cases for optimizers."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.params = [np.random.randn(3, 2), np.random.randn(3, 1)]
        self.grads = [np.random.randn(3, 2), np.random.randn(3, 1)]
        self.initial_params = [p.copy() for p in self.params]
    
    def test_sgd_update(self):
        """Test SGD optimizer update."""
        optimizer = SGD(learning_rate=0.1)
        
        # Update parameters
        optimizer.update(self.params, self.grads)
        
        # Check that parameters changed
        for initial, updated in zip(self.initial_params, self.params):
            self.assertFalse(np.array_equal(initial, updated))
        
        # Check update direction (should be opposite to gradient)
        expected_param_0 = self.initial_params[0] - 0.1 * self.grads[0]
        np.testing.assert_array_almost_equal(self.params[0], expected_param_0)
    
    def test_momentum_update(self):
        """Test Momentum optimizer update."""
        optimizer = Momentum(learning_rate=0.1, momentum=0.9)
        
        # First update
        optimizer.update(self.params, self.grads)
        
        # Check that velocities are initialized
        self.assertEqual(len(optimizer.velocities), len(self.params))
        for vel, param in zip(optimizer.velocities, self.params):
            self.assertEqual(vel.shape, param.shape)
        
        # Parameters should have changed
        for initial, updated in zip(self.initial_params, self.params):
            self.assertFalse(np.array_equal(initial, updated))
    
    def test_adam_update(self):
        """Test Adam optimizer update."""
        optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        
        # First update
        optimizer.update(self.params, self.grads)
        
        # Check that moment estimates are initialized
        self.assertEqual(len(optimizer.m), len(self.params))
        self.assertEqual(len(optimizer.v), len(self.params))
        self.assertEqual(optimizer.t, 1)
        
        for m, v, param in zip(optimizer.m, optimizer.v, self.params):
            self.assertEqual(m.shape, param.shape)
            self.assertEqual(v.shape, param.shape)
        
        # Parameters should have changed
        for initial, updated in zip(self.initial_params, self.params):
            self.assertFalse(np.array_equal(initial, updated))
    
    def test_adam_multiple_updates(self):
        """Test Adam optimizer with multiple updates."""
        optimizer = Adam(learning_rate=0.001)
        
        # Multiple updates
        for _ in range(5):
            optimizer.update(self.params, self.grads)
        
        # Time step should be updated
        self.assertEqual(optimizer.t, 5)
        
        # Moment estimates should be non-zero
        for m, v in zip(optimizer.m, optimizer.v):
            self.assertTrue(np.any(m != 0))
            self.assertTrue(np.any(v != 0))
    
    def test_optimizer_convergence(self):
        """Test that optimizers can minimize a simple quadratic function."""
        # Simple quadratic: f(x) = x^2, gradient = 2x
        x = np.array([[2.0]])  # Start at x=2
        
        optimizers = [
            SGD(learning_rate=0.1),
            Momentum(learning_rate=0.1, momentum=0.9),
            Adam(learning_rate=0.1)
        ]
        
        for optimizer in optimizers:
            x_opt = np.array([[2.0]])  # Reset starting point
            
            # Run optimization steps
            for _ in range(100):
                grad = 2 * x_opt  # Gradient of x^2
                optimizer.update([x_opt], [grad])
            
            # Should converge close to 0
            self.assertLess(abs(x_opt[0, 0]), 0.1, 
                          f"{optimizer.__class__.__name__} failed to converge")


if __name__ == '__main__':
    unittest.main()
