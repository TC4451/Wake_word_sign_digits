import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Plots histogram(s) of weight values
def plot_tflite_weight_distribution(tflite_file, bins=256, count_nonzero_only=False):
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()

        tensor_details = interpreter.get_tensor_details()

        weights = []
        names = []

        for tensor in tensor_details:
            if len(tensor['shape']) > 1 and tensor['dtype'] in [np.float32, np.int8, np.int16, np.int32, np.int64]:
                try:
                    tensor_data = interpreter.get_tensor(tensor['index']).flatten()
                    weights.append(tensor_data)
                    names.append(tensor['name'])
                except Exception as e:
                    print(f"Error getting tensor '{tensor['name']}': {e}")

        num_plots = len(weights)
        if num_plots == 0:
            print("No weights with dimensions > 1 found in the TFLite model.")
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

        fig.suptitle('Histogram of TFLite Weights')
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()

    except FileNotFoundError:
        print(f"Error: TFLite file '{tflite_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

import tensorflow as tf
import numpy as np

# Prints weight tensors
def print_tflite_weights(tflite_file):
    """
    Prints the actual weights of a TensorFlow Lite model to the screen.

    Args:
        tflite_file: Path to the TFLite model file.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_file)
        interpreter.allocate_tensors()

        tensor_details = interpreter.get_tensor_details()

        for tensor in tensor_details:
            if len(tensor['shape']) > 1 and tensor['dtype'] in [np.float32, np.int8, np.int16, np.int32, np.int64]:
                try:
                    tensor_data = interpreter.get_tensor(tensor['index'])
                    print(f"Tensor Name: {tensor['name']}, Shape: {tensor['shape']}, Dtype: {tensor['dtype']}")
                    print(tensor_data)  # Print the actual weights
                    print("-" * 20)

                except Exception as e:
                    print(f"Error getting tensor '{tensor['name']}': {e}")

    except FileNotFoundError:
        print(f"Error: TFLite file '{tflite_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usages:
# Replace 'your_model.tflite' with the actual path to your TFLite model file
plot_tflite_weight_distribution('wake_word_dr_quantized.tflite')

# Replace 'your_model.tflite' with the actual path to your TFLite model file
print_tflite_weights('wake_word_dr_quantized.tflite')
tf.lite.experimental.Analyzer.analyze(model_content="wake_word_dr_quantized.tflite")
