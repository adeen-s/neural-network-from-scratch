"""Learning rate scheduling utilities."""

import numpy as np
from typing import Callable, Optional


class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, initial_lr: float):
        """
        Initialize scheduler with initial learning rate.
        
        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def __call__(self, epoch: int) -> float:
        """
        Get learning rate for given epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Learning rate for this epoch
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr


class StepDecay(LearningRateScheduler):
    """Step decay learning rate scheduler."""
    
    def __init__(self, initial_lr: float, drop_rate: float = 0.5, epochs_drop: int = 10):
        """
        Initialize step decay scheduler.
        
        Args:
            initial_lr: Initial learning rate
            drop_rate: Factor by which to reduce learning rate
            epochs_drop: Number of epochs between drops
        """
        super().__init__(initial_lr)
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop
    
    def __call__(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        self.current_lr = self.initial_lr * (self.drop_rate ** (epoch // self.epochs_drop))
        return self.current_lr


class ExponentialDecay(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""
    
    def __init__(self, initial_lr: float, decay_rate: float = 0.96, decay_steps: int = 100):
        """
        Initialize exponential decay scheduler.
        
        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay factor
            decay_steps: Number of steps between decay applications
        """
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def __call__(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        self.current_lr = self.initial_lr * (self.decay_rate ** (epoch / self.decay_steps))
        return self.current_lr


class CosineAnnealing(LearningRateScheduler):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, initial_lr: float, min_lr: float = 0.0, T_max: int = 100):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            T_max: Maximum number of epochs for one cycle
        """
        super().__init__(initial_lr)
        self.min_lr = min_lr
        self.T_max = T_max
    
    def __call__(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                         (1 + np.cos(np.pi * (epoch % self.T_max) / self.T_max)) / 2
        return self.current_lr


class ReduceLROnPlateau(LearningRateScheduler):
    """Reduce learning rate when metric has stopped improving."""
    
    def __init__(self, initial_lr: float, factor: float = 0.5, patience: int = 10,
                 min_lr: float = 1e-7, mode: str = 'min', threshold: float = 1e-4):
        """
        Initialize ReduceLROnPlateau scheduler.
        
        Args:
            initial_lr: Initial learning rate
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement to wait
            min_lr: Minimum learning rate
            mode: 'min' for metrics that should decrease, 'max' for increase
            threshold: Threshold for measuring improvement
        """
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.is_better = self._get_comparison_fn()
    
    def _get_comparison_fn(self) -> Callable:
        """Get comparison function based on mode."""
        if self.mode == 'min':
            return lambda current, best: current < best - self.threshold
        else:
            return lambda current, best: current > best + self.threshold
    
    def step(self, metric: float) -> float:
        """
        Update learning rate based on metric.
        
        Args:
            metric: Current metric value
            
        Returns:
            Updated learning rate
        """
        if self.is_better(metric, self.best_metric):
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.wait = 0
        
        return self.current_lr
    
    def __call__(self, epoch: int) -> float:
        """Get current learning rate (for compatibility)."""
        return self.current_lr


class WarmupScheduler(LearningRateScheduler):
    """Learning rate warmup scheduler."""
    
    def __init__(self, initial_lr: float, target_lr: float, warmup_epochs: int):
        """
        Initialize warmup scheduler.
        
        Args:
            initial_lr: Starting learning rate (usually small)
            target_lr: Target learning rate after warmup
            warmup_epochs: Number of epochs for warmup
        """
        super().__init__(initial_lr)
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
    
    def __call__(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                             (epoch / self.warmup_epochs)
        else:
            self.current_lr = self.target_lr
        
        return self.current_lr


class CyclicLR(LearningRateScheduler):
    """Cyclic learning rate scheduler."""
    
    def __init__(self, initial_lr: float, max_lr: float, step_size: int = 2000,
                 mode: str = 'triangular', gamma: float = 1.0):
        """
        Initialize cyclic learning rate scheduler.
        
        Args:
            initial_lr: Base learning rate
            max_lr: Maximum learning rate
            step_size: Number of training iterations in half cycle
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: Decay factor for 'exp_range' mode
        """
        super().__init__(initial_lr)
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.iteration = 0
    
    def __call__(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_fn = 1.0
        elif self.mode == 'triangular2':
            scale_fn = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_fn = self.gamma ** self.iteration
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        self.current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * \
                         np.maximum(0, (1 - x)) * scale_fn
        
        self.iteration += 1
        return self.current_lr


class PolynomialDecay(LearningRateScheduler):
    """Polynomial decay learning rate scheduler."""
    
    def __init__(self, initial_lr: float, final_lr: float = 0.0001, 
                 decay_steps: int = 1000, power: float = 1.0):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            initial_lr: Initial learning rate
            final_lr: Final learning rate
            decay_steps: Number of steps to decay over
            power: Power of polynomial decay
        """
        super().__init__(initial_lr)
        self.final_lr = final_lr
        self.decay_steps = decay_steps
        self.power = power
    
    def __call__(self, epoch: int) -> float:
        """Calculate learning rate for given epoch."""
        if epoch >= self.decay_steps:
            self.current_lr = self.final_lr
        else:
            decay_factor = (1 - epoch / self.decay_steps) ** self.power
            self.current_lr = (self.initial_lr - self.final_lr) * decay_factor + self.final_lr
        
        return self.current_lr


def get_scheduler(name: str, initial_lr: float, **kwargs) -> LearningRateScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        name: Name of scheduler
        initial_lr: Initial learning rate
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler instance
    """
    schedulers = {
        'step': StepDecay,
        'exponential': ExponentialDecay,
        'cosine': CosineAnnealing,
        'plateau': ReduceLROnPlateau,
        'warmup': WarmupScheduler,
        'cyclic': CyclicLR,
        'polynomial': PolynomialDecay
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")
    
    return schedulers[name](initial_lr, **kwargs)
