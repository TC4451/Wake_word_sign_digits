import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
import os
import WW_Metrics as wwm
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

# Pruning configuration
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params_2_by_4 = {
    'sparsity_m_by_n': (2, 4),
}

pruning_params_sparsity_0_5 = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
        target_sparsity=0.5,
        begin_step=0,
        frequency=100)
}

model = keras.Sequential(
    [
        keras.Input(shape=[50,13,1]),
        #keras.layers.InputLayer(batch_input_shape=(None, 50, 13, 1)),
        
        prune_low_magnitude(
                keras.layers.Conv2D(
                    2, kernel_size=(3, 3), activation='relu', name="pruning_sparsity_0_5"
                ),
                **pruning_params_sparsity_0_5
        ),
        #keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        prune_low_magnitude(
                keras.layers.Conv2D(
                    64, kernel_size=(3, 3), activation="relu", name="structural_pruning"
                ),
                **pruning_params_2_by_4
        ),
        #keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        
        prune_low_magnitude(
                keras.layers.Dense(
                    2, activation="softmax", name="structural_pruning_dense"
                ),
                **pruning_params_2_by_4
        ),
        #keras.layers.Dense(2, activation="softmax"),  # Output layer (softmax for multi-class)
    ]
)

model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

# Load training data
padded_features = np.load('padded_features.npy')
print(f"The size of padded_labels is: {padded_features.size}")
labels = np.load('labels.npy')
print(f"The size of labels is: {labels.size}")

X_train, X_val, y_train, y_val = train_test_split(
    padded_features, labels, test_size=0.2, random_state=42)

batch_size = 32
epochs = 20

# Train model with pruning
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=tfmot.sparsity.keras.UpdatePruningStep(), validation_data=(X_val, y_val))
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

model = tfmot.sparsity.keras.strip_pruning(model)

wwm.print_cnn_weights(model)

conv_layer = model.get_layer('structural_pruning')
weights = conv_layer.get_weights()
kernel_weights = weights[0]
print("Kernel weights shape:", kernel_weights.shape)
print(kernel_weights)
