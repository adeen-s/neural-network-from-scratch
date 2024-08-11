"""Neural network implementation."""

import numpy as np


class NeuralNetwork:
    """A simple neural network implementation."""

    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.compiled = False

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def compile(self, optimizer='sgd', loss='mse', learning_rate=0.01):
        """Compile the network with optimizer and loss function."""
        # Set optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'sgd':
                from .optimizers import SGD
                self.optimizer = SGD(learning_rate)
            elif optimizer.lower() == 'adam':
                from .optimizers import Adam
                self.optimizer = Adam(learning_rate)
            elif optimizer.lower() == 'momentum':
                from .optimizers import Momentum
                self.optimizer = Momentum(learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer

        # Set loss function
        if isinstance(loss, str):
            if loss.lower() in ['mse', 'mean_squared_error']:
                from .losses import MeanSquaredError
                self.loss_function = MeanSquaredError()
            elif loss.lower() in ['categorical_crossentropy', 'crossentropy']:
                from .losses import CategoricalCrossentropy
                self.loss_function = CategoricalCrossentropy()
            else:
                raise ValueError(f"Unknown loss function: {loss}")
        else:
            self.loss_function = loss

        self.compiled = True

    def predict(self, input_data):
        """Make predictions on input data."""
        if isinstance(input_data, list):
            input_data = np.array(input_data)

        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        result = []
        for i in range(len(input_data)):
            # Forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return np.array(result)

    def fit(self, x_train, y_train, epochs=100, verbose=False, validation_split=0.0):
        """Train the neural network."""
        if not self.compiled:
            raise ValueError("Model must be compiled before training")

        if isinstance(x_train, list):
            x_train = np.array(x_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        history = {'loss': []}
        samples = len(x_train)

        for epoch in range(epochs):
            epoch_loss = 0

            for j in range(samples):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                loss_value = self.loss_function.forward(y_train[j], output)
                epoch_loss += loss_value

                # Backward propagation
                error = self.loss_function.backward(y_train[j], output)

                # Collect parameters and gradients
                params = []
                grads = []

                for layer in reversed(self.layers):
                    error = layer.backward(error, 0.01)  # learning rate not used here
                    if hasattr(layer, 'weights'):
                        params.extend([layer.weights, layer.biases])
                        grads.extend([layer.dweights, layer.dbiases])

                # Update parameters using optimizer
                if params and grads:
                    self.optimizer.update(params, grads)

            # Average loss
            epoch_loss /= samples
            history['loss'].append(epoch_loss)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}')

        return history
