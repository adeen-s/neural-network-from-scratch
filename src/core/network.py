"""Neural network implementation."""

import numpy as np


class NeuralNetwork:
    """A simple neural network implementation."""
    
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    
    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def use(self, loss, loss_prime):
        """Set the loss function for the network."""
        self.loss = loss
        self.loss_prime = loss_prime
    
    def predict(self, input_data):
        """Make predictions on input data."""
        samples = len(input_data)
        result = []
        
        for i in range(samples):
            # Forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        
        return result
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        """Train the neural network."""
        samples = len(x_train)
        
        for epoch in range(epochs):
            err = 0
            for j in range(samples):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)
                
                # Compute loss
                err += self.loss(y_train[j], output)
                
                # Backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            
            # Average error
            err /= samples
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Error: {err:.6f}')
