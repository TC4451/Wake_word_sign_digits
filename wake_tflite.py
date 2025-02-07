import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

model_path = "."
loaded_model = tf.keras.models.load_model("wake_word_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()
open("wake_word.tflite", "wb").write(tflite_model)