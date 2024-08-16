"""
Visualization utilities for neural networks.
"""
import numpy as np


def plot_training_history(history, save_path=None):
    """Plot training history."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy if available
        if 'accuracy' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], 'b-', label='Training Accuracy')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], 'r-', label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")


def plot_decision_boundary(network, X, y, resolution=100, save_path=None):
    """Plot decision boundary for 2D classification problems."""
    try:
        import matplotlib.pyplot as plt
        
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting only supports 2D input")
        
        # Create a mesh
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = network.predict(mesh_points)
        
        if Z.ndim > 1 and Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        else:
            Z = (Z > 0.5).astype(int).flatten()
        
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        if y.ndim > 1:
            y_plot = np.argmax(y, axis=1)
        else:
            y_plot = y
        
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y_plot, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Decision boundary plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")


def plot_weights_distribution(network, save_path=None):
    """Plot distribution of weights in the network."""
    try:
        import matplotlib.pyplot as plt
        
        weights = []
        layer_names = []
        
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'weights'):
                weights.append(layer.weights.flatten())
                layer_names.append(f'Layer {i+1}')
        
        if not weights:
            print("No trainable layers found")
            return
        
        plt.figure(figsize=(12, 4))
        
        for i, (w, name) in enumerate(zip(weights, layer_names)):
            plt.subplot(1, len(weights), i+1)
            plt.hist(w, bins=30, alpha=0.7, density=True)
            plt.title(f'{name} Weights')
            plt.xlabel('Weight Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Weights distribution plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")


def plot_activations(network, X_sample, save_path=None):
    """Plot activations for each layer given a sample input."""
    try:
        import matplotlib.pyplot as plt
        
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        activations = []
        layer_names = []
        
        # Forward pass to collect activations
        output = X_sample[0]
        for i, layer in enumerate(network.layers):
            output = layer.forward(output)
            if hasattr(layer, 'weights') or 'activation' in layer.__class__.__name__.lower():
                activations.append(output.flatten())
                layer_names.append(f'Layer {i+1}')
        
        if not activations:
            print("No activations to plot")
            return
        
        plt.figure(figsize=(15, 3))
        
        for i, (act, name) in enumerate(zip(activations, layer_names)):
            plt.subplot(1, len(activations), i+1)
            plt.bar(range(len(act)), act)
            plt.title(f'{name} Activations')
            plt.xlabel('Neuron')
            plt.ylabel('Activation')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Activations plot saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")
