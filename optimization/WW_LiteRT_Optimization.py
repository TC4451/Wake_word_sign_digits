import tensorflow as tf
import numpy as np
import librosa
import os
#import Evaluate_Optimitizied_tfile as eval_tf
#import WW_tflite_Metrics as met_tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained Keras model
keras_model_path = "../training/wake_word_model.keras"
model = tf.keras.models.load_model(keras_model_path)
model.summary()

## Various quantization modes (kept for reference below)

# Post-training dynamic range quantization
"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tf.lite.experimental.Analyzer.analyze(model_content=tflite_quant_model)

with open("wake_word_dr_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)
#eval_tf.evaluate_tflite_model_on_directory(tflite_quant_model, data_dir = "./Dataset")
"""

# Post-training integer quantization (uint8 I/O, int8 ops)
"""
padded_features = np.load('padded_features.npy')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(padded_features).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model_quant)

with open("wake_word_ALL_INT_quantized.tflite", "wb") as f:
    f.write(tflite_model_quant)
"""

# Post-training float16 quantization
"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16_model = converter.convert()
tf.lite.experimental.Analyzer.analyze(model_content=tflite_fp16_model)

with open("wake_word_FP16_quantized.tflite", "wb") as f:
    f.write(tflite_fp16_model)
"""

# Post-training integer quantization with int16 activations / int8 weights
padded_features = np.load('padded_features.npy')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

# Generate representative dataset for calibration
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(padded_features).batch(1).take(100):
    yield [input_value]
converter.representative_dataset = representative_data_gen
tflite_16x8_model = converter.convert()

# Inspect and save the quantized model
tf.lite.experimental.Analyzer.analyze(model_content=tflite_16x8_model)

with open("wake_word_INT8_16_quantized.tflite", "wb") as f:
    f.write(tflite_16x8_model)
