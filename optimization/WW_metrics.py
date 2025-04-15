import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

#model = load_model("./wake_word_model.keras")

#model.summary()

#input_shape = model.input_shape
#print(f"Input shape (using input_shape): {input_shape}")

def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    """
    Plots the distribution of weights in a TensorFlow Keras model.

    Args:
        model: A TensorFlow Keras model.
        bins: Number of bins for the histogram.
        count_nonzero_only: If True, only plots the distribution of non-zero weights.
    """

    weights = []
    names = []
    for layer in model.layers:
        for weight in layer.weights:
            if len(weight.shape) > 1:  # Only plot weights with dimensions > 1 (e.g., Conv2D, Dense)
                weights.append(weight.numpy().flatten())
                names.append(weight.name)

    num_plots = len(weights)
    if num_plots == 0:
        print("No weights with dimensions > 1 found in the model.")
        return

    rows = min(3, (num_plots + 2) // 3)  # Dynamically adjust rows
    cols = min(3, num_plots) # Dynamically adjust columns

    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    if num_plots == 1:
        axes = np.array([axes]) #make axes iterable if only one subplot
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

def print_cnn_weights(model):
    """
    Prints the weights and biases for each layer in a TensorFlow CNN model,
    including the count of zero and non-zero weights and biases.

    Args:
        model: A TensorFlow Keras CNN model.
    """
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):  # Check if the layer has weights
            weights = layer.get_weights()
            if weights:  # Check if the layer has actual weights (not just biases)
                print(f"Layer: {layer.name}")
                for i, weight_array in enumerate(weights):
                    if len(weight_array.shape) > 1:  # filter out bias weights, which are usually 1D
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
def print_layer_weights(model, layer_name):
    """
    Prints the weights of a specific layer in a TensorFlow Keras model.

    Args:
        model: A TensorFlow Keras CNN model.
        layer_name: The name of the layer whose weights to print.
    """
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
                    return  # Exit the function after printing the weights
            else:
                print(f"Layer '{layer_name}' has no weights.")
                return
    print(f"Layer '{layer_name}' not found in the model.")

def get_cnn_layer_names(model):
    """
    Returns a list of layer names in a TensorFlow Keras CNN model.

    Args:
        model: A TensorFlow Keras CNN model.

    Returns:
        A list of strings, where each string is the name of a layer.
    """
    layer_names = [layer.name for layer in model.layers]
    return layer_names

def print_conv2d_layer_weights(layer):
    """
    Prints the 3x3 weight matrices for a Conv2D layer.

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

    kernel_weights = weights[0]  # The kernel weights are the first element
    if len(kernel_weights.shape) != 4:
        print(f"Error: Layer '{layer.name}' weights do not have the expected 4D shape.")
        return
    if kernel_weights.shape[0] != 3 or kernel_weights.shape[1] != 3:
        print(f"Error: Layer '{layer.name}' weights are not 3x3.")
        return

    num_filters = kernel_weights.shape[3]
    input_channels = kernel_weights.shape[2]

    print(f"Weights for layer: {layer.name}")
    for filter_index in range(num_filters):
        print(f"Filter {filter_index + 1}:")
        for input_channel in range(input_channels):
            print(f"  Input Channel {input_channel + 1}:")
            print(kernel_weights[:, :, input_channel, filter_index])

# Example usage (replace with your model)
#model = tf.keras.applications.MobileNetV2()
#plot_weight_distribution(model)
#print_cnn_weights(model)

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

    kernel_weights = weights[0]  # The kernel weights are the first element
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

def count_zero_weights(model):
    """
    Counts the number of zero weights in a TensorFlow Keras model.

    Args:
        model: A TensorFlow Keras model.

    Returns:
        The number of zero weights as an integer.
    """
    zero_weight_count = 0

    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            for weight_array in weights:
                zero_weight_count += np.sum(weight_array == 0)

    return zero_weight_count
def get_cnn_weight_sparsity(model):
    """
    Calculates the weight sparsity of a TensorFlow Keras CNN model.

    Args:
        model: A TensorFlow Keras CNN model.

    Returns:
        The weight sparsity as a float (0.0 to 1.0).
    """
    total_weights = 0
    zero_weights = 0

    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            for weight_array in weights:
                total_weights += weight_array.size
                zero_weights += np.sum(weight_array == 0)

    if total_weights == 0:
        return 0.0  # Handle the case where there are no weights
    return zero_weights / total_weights

#print(get_cnn_layer_names(model))
#print_layer_weights(model, 'conv2d')
#print(model.get_layer('conv2d'))
#print_conv2d_3x3_weights(model.get_layer('conv2d'))
#print(f"Number of zero weights: {count_zero_weights(model)}")
#sparsity = get_cnn_weight_sparsity(model)
#print(f"Weight sparsity (no zeros): {sparsity:.4f}")


