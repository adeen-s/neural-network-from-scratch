"""Neural network implementation."""

import numpy as np


class NeuralNetwork:
    """
    A neural network implementation from scratch.

    This class provides a simple interface for building, training, and using
    neural networks with various architectures, optimizers, and loss functions.

    Attributes:
        layers (list): List of network layers
        loss_function: Loss function for training
        optimizer: Optimization algorithm for parameter updates
        compiled (bool): Whether the network has been compiled
    """

    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.compiled = False

    def add(self, layer):
        """Add a layer to the network."""
        # Validate layer type
        if not hasattr(layer, 'forward') or not hasattr(layer, 'backward'):
            raise TypeError("Layer must implement forward() and backward() methods")
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
        """
        Make predictions on input data.

        Args:
            input_data (array-like): Input data for prediction. Can be:
                - 1D array for single sample
                - 2D array for multiple samples
                - List that will be converted to numpy array

        Returns:
            numpy.ndarray: Predictions with shape (n_samples, n_outputs).
                          Outputs are flattened for easier handling.

        Raises:
            ValueError: If network is not compiled before prediction.

        Example:
            >>> predictions = network.predict(X_test)
            >>> print(predictions.shape)  # (n_samples, n_classes)
        """
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
            # Flatten output to 1D array for easier handling
            result.append(output.flatten())

        return np.array(result)

    def fit(self, x_train, y_train, epochs=100, verbose=False, validation_split=0.0, batch_size=32):
        """
        Train the neural network using backpropagation.

        Args:
            x_train (array-like): Training input data
            y_train (array-like): Training target data
            epochs (int): Number of training epochs (default: 100)
            verbose (bool): Whether to print training progress (default: False)
            validation_split (float): Fraction of data for validation (currently unused)
            batch_size (int): Size of mini-batches for training (default: 32)

        Returns:
            dict: Training history containing loss values for each epoch

        Raises:
            ValueError: If network is not compiled before training

        Example:
            >>> history = network.fit(X_train, y_train, epochs=50, verbose=True)
            >>> print(f"Final loss: {history['loss'][-1]:.6f}")
        """
        if not self.compiled:
            raise ValueError("Model must be compiled before training")

        if isinstance(x_train, list):
            x_train = np.array(x_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        history = {'loss': []}
        samples = len(x_train)

        # Use batch processing if batch_size is specified and > 1
        if batch_size > 1 and samples > batch_size:
            return self._fit_batched(x_train, y_train, epochs, verbose, batch_size, history)

        # Original sample-by-sample training
        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle data each epoch
            indices = np.random.permutation(samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for j in range(samples):
                # Forward propagation
                output = x_shuffled[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                loss_value = self.loss_function.forward(y_shuffled[j], output)
                epoch_loss += loss_value

                # Backward propagation
                error = self.loss_function.backward(y_shuffled[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward(error, 0.01)  # learning rate not used here

                # Collect parameters and gradients from all trainable layers
                params = []
                grads = []

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        params.extend([layer.weights, layer.biases])
                        grads.extend([layer.dweights, layer.dbiases])

                # Update parameters using optimizer
                if params and grads:
                    self.optimizer.update(params, grads)

            # Average loss
            epoch_loss /= samples
            history['loss'].append(epoch_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}')

        return history

    def _fit_batched(self, x_train, y_train, epochs, verbose, batch_size, history):
        """Train using mini-batches for better performance."""
        samples = len(x_train)

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            # Shuffle data each epoch
            indices = np.random.permutation(samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Process in batches
            for i in range(0, samples, batch_size):
                end_idx = min(i + batch_size, samples)
                batch_x = x_shuffled[i:end_idx]
                batch_y = y_shuffled[i:end_idx]

                batch_loss = 0

                # Accumulate gradients over batch
                accumulated_grads = {}

                for j in range(len(batch_x)):
                    # Forward propagation
                    output = batch_x[j]
                    for layer in self.layers:
                        output = layer.forward(output)

                    # Compute loss
                    loss_value = self.loss_function.forward(batch_y[j], output)
                    batch_loss += loss_value

                    # Backward propagation
                    error = self.loss_function.backward(batch_y[j], output)

                    for layer in reversed(self.layers):
                        error = layer.backward(error, 0.01)

                    # Accumulate gradients
                    for layer in self.layers:
                        if hasattr(layer, 'weights'):
                            layer_id = id(layer)
                            if layer_id not in accumulated_grads:
                                accumulated_grads[layer_id] = {
                                    'dweights': np.zeros_like(layer.weights),
                                    'dbiases': np.zeros_like(layer.biases)
                                }
                            accumulated_grads[layer_id]['dweights'] += layer.dweights
                            accumulated_grads[layer_id]['dbiases'] += layer.dbiases

                # Average gradients and update parameters
                params = []
                grads = []

                for layer in self.layers:
                    if hasattr(layer, 'weights'):
                        layer_id = id(layer)
                        avg_dweights = accumulated_grads[layer_id]['dweights'] / len(batch_x)
                        avg_dbiases = accumulated_grads[layer_id]['dbiases'] / len(batch_x)

                        params.extend([layer.weights, layer.biases])
                        grads.extend([avg_dweights, avg_dbiases])

                if params and grads:
                    self.optimizer.update(params, grads)

                epoch_loss += batch_loss
                num_batches += 1

            # Average loss
            epoch_loss /= samples
            history['loss'].append(epoch_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}')

        return history
