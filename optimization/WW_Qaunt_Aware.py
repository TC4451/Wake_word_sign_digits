import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
import os

from tensorflow_model_optimization.python.core.keras.compat import keras
import WW_Metrics as wwm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


#from tensorflow.keras import layers
random.seed(0)
np.random.seed(0)

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
numerical_labels = 2
epochs = 4
validation_split = 0.1 # 10% of training set will be used for validation set.
model = keras.Sequential(
    [
        keras.Input(shape=[50,13,1]),  # Input shape (max_len, n_mfcc, 1) for 2D CNN
        #keras.layers.InputLayer(batch_input_shape=(None, 50, 13, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),  # Add dropout for regularization
        keras.layers.Dense(2, activation="softmax"),  # Output layer (softmax for multi-class)
    ]
)
print(type(model))
padded_features = np.load('padded_features.npy')
print(f"The size of padded_labels is: {padded_features.size}")
numerical_labels = np.load('numerical_labels.npy')
print(f"The size of numerical_labels is: {numerical_labels.size}")

X_train, X_val, y_train, y_val = train_test_split(
    padded_features, numerical_labels, test_size=0.2, random_state=42)

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
epochs = 12
batch_size = 32

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

quantize_model = tfmot.quantization.keras.quantize_model

q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

q_aware_model.summary()

history = q_aware_model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))
#q_aware_model.fit(X_train, X_val,
#                  batch_size=128, epochs=1, validation_split=0.1)

wwm.print_cnn_weights(q_aware_model)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

tf.lite.experimental.Analyzer.analyze(model_content=quantized_tflite_model)

# Save the quantized TFLite model to a file (optional)
with open("wake_word_QAT.tflite", "wb") as f:
    f.write(quantized_tflite_model)
