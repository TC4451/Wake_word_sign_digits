import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and inspect model
#model = load_model("./wake_word_model.keras")
#model.summary()
#print(f"Input shape: {model.input_shape}")

# Visualizes the distribution of weights across layers
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    weights = []
    names = []
    
    for layer in model.layers:
        for weight in layer.weights:
            if len(weight.shape) > 1:
                weights.append(weight.numpy().flatten())
                names.append(weight.name)

    num_plots = len(weights)
    if num_plots == 0:
        print("No multi-dimensional weights found in the model.")
        return

    rows = min(3, (num_plots + 2) // 3)
    cols = min(3, num_plots)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    if num_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.ravel()

    for i, (weight, name) in enumerate(zip(weights, names)):
        ax = axes[i]
        if count_nonzero_only:
            weight = weight[weight != 0]
        ax.hist(weight, bins=bins, density=True, color='blue', alpha=0.5)
        ax.set_xlabel(name)
        ax.set_ylabel('density')

    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()

# Outputs weights, biases, and sparsity stats for each layer
def print_cnn_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if weights:
                print(f"Layer: {layer.name}")
                for i, weight_array in enumerate(weights):
                    if len(weight_array.shape) > 1:
                        print(f"  Weights (Shape: {weight_array.shape}):")
                        print(weight_array)
                        zero_count = np.sum(weight_array == 0)
                        nonzero_count = np.sum(weight_array != 0)
                        print(f"  Number of zero weights: {zero_count}")
                        print(f"  Number of non-zero weights: {nonzero_count}")
                    else:
                        print(f"  Bias (Shape: {weight_array.shape}):")
                        print(weight_array)
                        zero_count = np.sum(weight_array == 0)
                        nonzero_count = np.sum(weight_array != 0)
                        print(f"  Number of zero biases: {zero_count}")
                        print(f"  Number of non-zero biases: {nonzero_count}")

"""
def print_cnn_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):  # Check if the layer has weights
            weights = layer.get_weights()
            if weights:  # Check if the layer has actual weights (not just biases)
                print(f"Layer: {layer.name}")
                for i, weight_array in enumerate(weights):
                    if len(weight_array.shape) > 1:  # filter out bias weights, which are usually 1D
                        print(f"  Weights (Shape: {weight_array.shape}):")
                        print(weight_array)  # Or, print summary statistics instead of the whole array
                    else:
                        print(f"  Bias (Shape: {weight_array.shape}):")
                        print(weight_array)
"""

# Displays weights and biases for a specific named layer
def print_layer_weights(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            if hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                if weights:
                    print(f"Weights for layer: {layer_name}")
                    for i, weight_array in enumerate(weights):
                        if len(weight_array.shape) > 1:
                            print(f"  Weights (Shape: {weight_array.shape}):")
                            print(weight_array)
                        else:
                            print(f"  Bias (Shape: {weight_array.shape}):")
                            print(weight_array)
                    return
            else:
                print(f"Layer '{layer_name}' has no weights.")
                return
    print(f"Layer '{layer_name}' not found in the model.")

# Returns all layer names in the model
def get_cnn_layer_names(model):
    layer_names = [layer.name for layer in model.layers]
    return layer_names

# Example usage (replace with your model)
#model = tf.keras.applications.MobileNetV2()
#plot_weight_distribution(model)
#print_cnn_weights(model)

# Displays 3x3 kernel weights from a Conv2D layer
def print_conv2d_3x3_weights(layer):
    """
    Prints each 3x3 weight matrix from a Conv2D layer with shape (3, 3, input_channels, output_channels).

    Args:
        layer: A TensorFlow Keras Conv2D layer.
    """
    if not isinstance(layer, tf.keras.layers.Conv2D):
        print("Error: Input layer is not a Conv2D layer.")
        return

    weights = layer.get_weights()
    if not weights:
        print(f"Layer '{layer.name}' has no weights.")
        return

    kernel_weights = weights[0]
    if len(kernel_weights.shape) != 4:
        print(f"Error: Layer '{layer.name}' weights do not have the expected 4D shape.")
        return
    if kernel_weights.shape[0] != 3 or kernel_weights.shape[1] != 3:
        print(f"Error: Layer '{layer.name}' weights are not 3x3.")
        return

    input_channels = kernel_weights.shape[2]
    output_channels = kernel_weights.shape[3]

    print(f"Weights for layer: {layer.name}")
    for output_channel in range(output_channels):
        print(f"Output Channel/Filter {output_channel + 1}:")
        for input_channel in range(input_channels):
            print(f"  Input Channel {input_channel + 1}:")
            print(kernel_weights[:, :, input_channel, output_channel])

# Counts all zero-values weights in the model
def count_zero_weights(model):
    zero_weight_count = 0

    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            for weight_array in weights:
                zero_weight_count += np.sum(weight_array == 0)

    return zero_weight_count

# Calculates overall weight sparsity (fraction of weights that are zero)
def get_cnn_weight_sparsity(model):
    total_weights = 0
    zero_weights = 0

    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            for weight_array in weights:
                total_weights += weight_array.size
                zero_weights += np.sum(weight_array == 0)

    if total_weights == 0:
        return 0.0
    return zero_weights / total_weights

# Example usage:
#print(get_cnn_layer_names(model))
#print_layer_weights(model, 'conv2d')
#print(model.get_layer('conv2d'))
#print_conv2d_3x3_weights(model.get_layer('conv2d'))
#print(f"Number of zero weights: {count_zero_weights(model)}")
#sparsity = get_cnn_weight_sparsity(model)
#print(f"Weight sparsity (no zeros): {sparsity:.4f}")
