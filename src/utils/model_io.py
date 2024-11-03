"""Model serialization and checkpointing utilities."""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self, filepath: str, monitor: str = 'loss', 
                 save_best_only: bool = True, mode: str = 'min', verbose: bool = True):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save model checkpoints
            monitor: Metric to monitor for saving
            save_best_only: Only save when metric improves
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            verbose: Print messages when saving
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'min':
            self.best_value = float('inf')
            self.is_better = lambda current, best: current < best
        else:
            self.best_value = float('-inf')
            self.is_better = lambda current, best: current > best
    
    def __call__(self, network, epoch: int, logs: Dict[str, float]):
        """
        Check if model should be saved and save if necessary.
        
        Args:
            network: Neural network to save
            epoch: Current epoch number
            logs: Dictionary of metrics from current epoch
        """
        current_value = logs.get(self.monitor)
        if current_value is None:
            if self.verbose:
                print(f"Warning: Metric '{self.monitor}' not found in logs")
            return
        
        if not self.save_best_only or self.is_better(current_value, self.best_value):
            if self.save_best_only:
                self.best_value = current_value
            
            # Format filepath with epoch and metric value
            filepath = self.filepath.format(epoch=epoch, **logs)
            save_model(network, filepath)
            
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current_value:.6f}, "
                      f"saving model to {filepath}")


def save_model(network, filepath: str):
    """
    Save a neural network model to disk.
    
    Args:
        network: Neural network to save
        filepath: Path to save the model
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare model data
    model_data = {
        'layers': [],
        'loss_function': None,
        'optimizer': None,
        'compiled': network.compiled,
        'metadata': {
            'save_time': time.time(),
            'save_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_layers': len(network.layers)
        }
    }
    
    # Save layer information
    for layer in network.layers:
        layer_data = {
            'type': layer.__class__.__name__,
            'module': layer.__class__.__module__
        }
        
        # Save layer parameters
        if hasattr(layer, 'weights'):
            layer_data['weights'] = layer.weights.tolist()
        if hasattr(layer, 'biases'):
            layer_data['biases'] = layer.biases.tolist()
        if hasattr(layer, 'alpha'):  # For LeakyReLU, ELU
            layer_data['alpha'] = layer.alpha
        if hasattr(layer, 'rate'):  # For Dropout
            layer_data['rate'] = layer.rate
        if hasattr(layer, 'epsilon'):  # For BatchNorm
            layer_data['epsilon'] = layer.epsilon
            layer_data['momentum'] = layer.momentum
        if hasattr(layer, 'gamma'):  # For BatchNorm
            if layer.gamma is not None:
                layer_data['gamma'] = layer.gamma.tolist()
                layer_data['beta'] = layer.beta.tolist()
                layer_data['running_mean'] = layer.running_mean.tolist()
                layer_data['running_var'] = layer.running_var.tolist()
        
        model_data['layers'].append(layer_data)
    
    # Save loss function info
    if network.loss_function:
        model_data['loss_function'] = {
            'type': network.loss_function.__class__.__name__,
            'module': network.loss_function.__class__.__module__
        }
    
    # Save optimizer info
    if network.optimizer:
        optimizer_data = {
            'type': network.optimizer.__class__.__name__,
            'module': network.optimizer.__class__.__module__,
            'learning_rate': network.optimizer.learning_rate
        }
        
        # Save optimizer-specific parameters
        if hasattr(network.optimizer, 'momentum'):
            optimizer_data['momentum'] = network.optimizer.momentum
        if hasattr(network.optimizer, 'beta1'):
            optimizer_data['beta1'] = network.optimizer.beta1
            optimizer_data['beta2'] = network.optimizer.beta2
            optimizer_data['epsilon'] = network.optimizer.epsilon
        
        model_data['optimizer'] = optimizer_data
    
    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(filepath: str):
    """
    Load a neural network model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded neural network
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Import necessary modules
    from ..core.network import NeuralNetwork
    from ..core.layers import Dense
    from ..core.activations import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU
    from ..core.losses import MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy
    from ..core.optimizers import SGD, Momentum, Adam
    from ..core.regularization import Dropout, BatchNormalization
    
    # Create network
    network = NeuralNetwork()
    
    # Reconstruct layers
    for layer_data in model_data['layers']:
        layer_type = layer_data['type']
        
        if layer_type == 'Dense':
            # Reconstruct Dense layer
            weights = np.array(layer_data['weights'])
            biases = np.array(layer_data['biases'])
            input_size, output_size = weights.shape[1], weights.shape[0]
            
            layer = Dense(input_size, output_size)
            layer.weights = weights
            layer.biases = biases
            
        elif layer_type == 'ReLU':
            layer = ReLU()
        elif layer_type == 'Sigmoid':
            layer = Sigmoid()
        elif layer_type == 'Tanh':
            layer = Tanh()
        elif layer_type == 'Softmax':
            layer = Softmax()
        elif layer_type == 'LeakyReLU':
            layer = LeakyReLU(alpha=layer_data.get('alpha', 0.01))
        elif layer_type == 'ELU':
            layer = ELU(alpha=layer_data.get('alpha', 1.0))
        elif layer_type == 'Dropout':
            layer = Dropout(rate=layer_data.get('rate', 0.5))
        elif layer_type == 'BatchNormalization':
            layer = BatchNormalization(
                epsilon=layer_data.get('epsilon', 1e-8),
                momentum=layer_data.get('momentum', 0.99)
            )
            if 'gamma' in layer_data:
                layer.gamma = np.array(layer_data['gamma'])
                layer.beta = np.array(layer_data['beta'])
                layer.running_mean = np.array(layer_data['running_mean'])
                layer.running_var = np.array(layer_data['running_var'])
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        network.add(layer)
    
    # Reconstruct loss function
    if model_data['loss_function']:
        loss_type = model_data['loss_function']['type']
        if loss_type == 'MeanSquaredError':
            network.loss_function = MeanSquaredError()
        elif loss_type == 'CategoricalCrossentropy':
            network.loss_function = CategoricalCrossentropy()
        elif loss_type == 'BinaryCrossentropy':
            network.loss_function = BinaryCrossentropy()
    
    # Reconstruct optimizer
    if model_data['optimizer']:
        opt_data = model_data['optimizer']
        opt_type = opt_data['type']
        
        if opt_type == 'SGD':
            network.optimizer = SGD(learning_rate=opt_data['learning_rate'])
        elif opt_type == 'Momentum':
            network.optimizer = Momentum(
                learning_rate=opt_data['learning_rate'],
                momentum=opt_data.get('momentum', 0.9)
            )
        elif opt_type == 'Adam':
            network.optimizer = Adam(
                learning_rate=opt_data['learning_rate'],
                beta1=opt_data.get('beta1', 0.9),
                beta2=opt_data.get('beta2', 0.999),
                epsilon=opt_data.get('epsilon', 1e-8)
            )
    
    network.compiled = model_data['compiled']
    return network


def export_model_summary(network, filepath: str):
    """
    Export a human-readable model summary.
    
    Args:
        network: Neural network to summarize
        filepath: Path to save the summary
    """
    summary = {
        'model_info': {
            'num_layers': len(network.layers),
            'compiled': network.compiled,
            'total_parameters': 0
        },
        'layers': []
    }
    
    total_params = 0
    for i, layer in enumerate(network.layers):
        layer_info = {
            'index': i,
            'type': layer.__class__.__name__,
            'parameters': 0
        }
        
        if hasattr(layer, 'weights'):
            layer_info['input_size'] = layer.weights.shape[1]
            layer_info['output_size'] = layer.weights.shape[0]
            layer_params = layer.weights.size + layer.biases.size
            layer_info['parameters'] = layer_params
            total_params += layer_params
        
        summary['layers'].append(layer_info)
    
    summary['model_info']['total_parameters'] = total_params
    
    # Save as JSON for readability
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
