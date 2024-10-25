"""Performance benchmarking utilities for neural networks."""

import time
import numpy as np
from typing import Dict, List, Callable, Any


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time


class NetworkBenchmark:
    """Benchmark neural network performance."""
    
    def __init__(self, network):
        """
        Initialize benchmark with a neural network.
        
        Args:
            network: Compiled neural network to benchmark
        """
        self.network = network
        self.results = {}
    
    def benchmark_forward_pass(self, X: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark forward pass performance.
        
        Args:
            X: Input data for forward pass
            num_runs: Number of runs to average over
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        for _ in range(num_runs):
            with Timer() as timer:
                _ = self.network.predict(X)
            times.append(timer.elapsed_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times),
            'num_runs': num_runs
        }
    
    def benchmark_training(self, X: np.ndarray, y: np.ndarray, 
                          epochs: int = 10, num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark training performance.
        
        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            num_runs: Number of training runs to average over
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        
        for _ in range(num_runs):
            # Reset network weights for fair comparison
            for layer in self.network.layers:
                if hasattr(layer, 'weights'):
                    layer.weights = np.random.randn(*layer.weights.shape) * 0.1
                if hasattr(layer, 'biases'):
                    layer.biases = np.zeros_like(layer.biases)
            
            with Timer() as timer:
                _ = self.network.fit(X, y, epochs=epochs, verbose=False)
            times.append(timer.elapsed_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times),
            'epochs': epochs,
            'num_runs': num_runs,
            'time_per_epoch': np.mean(times) / epochs
        }
    
    def memory_usage_estimate(self, X: np.ndarray) -> Dict[str, int]:
        """
        Estimate memory usage for network operations.
        
        Args:
            X: Sample input data
            
        Returns:
            Dictionary with memory usage estimates in bytes
        """
        total_params = 0
        total_activations = 0
        
        # Calculate parameter memory
        for layer in self.network.layers:
            if hasattr(layer, 'weights'):
                total_params += layer.weights.size
            if hasattr(layer, 'biases'):
                total_params += layer.biases.size
        
        # Estimate activation memory (rough approximation)
        batch_size = X.shape[0] if X.ndim > 1 else 1
        current_size = X.size
        
        for layer in self.network.layers:
            if hasattr(layer, 'weights'):
                # Dense layer output size
                current_size = layer.weights.shape[0] * batch_size
            total_activations += current_size
        
        # Assume float64 (8 bytes per number)
        bytes_per_float = 8
        
        return {
            'parameters_bytes': total_params * bytes_per_float,
            'activations_bytes': total_activations * bytes_per_float,
            'total_bytes': (total_params + total_activations) * bytes_per_float,
            'parameters_mb': (total_params * bytes_per_float) / (1024 * 1024),
            'activations_mb': (total_activations * bytes_per_float) / (1024 * 1024),
            'total_mb': ((total_params + total_activations) * bytes_per_float) / (1024 * 1024)
        }
    
    def run_full_benchmark(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            Complete benchmark results
        """
        print("Running forward pass benchmark...")
        forward_results = self.benchmark_forward_pass(X)
        
        print("Running training benchmark...")
        training_results = self.benchmark_training(X, y)
        
        print("Calculating memory usage...")
        memory_results = self.memory_usage_estimate(X)
        
        return {
            'forward_pass': forward_results,
            'training': training_results,
            'memory': memory_results,
            'data_shape': X.shape,
            'target_shape': y.shape
        }


def compare_optimizers(network_factory: Callable, optimizers: List[str], 
                      X: np.ndarray, y: np.ndarray, epochs: int = 50) -> Dict[str, Dict]:
    """
    Compare performance of different optimizers.
    
    Args:
        network_factory: Function that creates a fresh network instance
        optimizers: List of optimizer names to compare
        X: Training input data
        y: Training target data
        epochs: Number of training epochs
        
    Returns:
        Dictionary with results for each optimizer
    """
    results = {}
    
    for optimizer in optimizers:
        print(f"Benchmarking {optimizer} optimizer...")
        
        # Create fresh network
        network = network_factory()
        network.compile(optimizer=optimizer, loss='mse')
        
        # Benchmark training
        benchmark = NetworkBenchmark(network)
        training_results = benchmark.benchmark_training(X, y, epochs=epochs, num_runs=3)
        
        # Get final loss
        history = network.fit(X, y, epochs=epochs, verbose=False)
        final_loss = history[-1] if history else float('inf')
        
        results[optimizer] = {
            'timing': training_results,
            'final_loss': final_loss,
            'convergence_rate': (history[0] - final_loss) / history[0] if history else 0
        }
    
    return results
