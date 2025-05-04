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

# Basic training setup
initial_batch_size = 128
num_classes = 2
epochs = 4
val_split = 0.1

model = keras.Sequential(
    [
        keras.Input(shape=[50,13,1]),
        #keras.layers.InputLayer(batch_input_shape=(None, 50, 13, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation="softmax"),
    ]
)

print(f"Model type: {type(model)}")

# Load input data
padded_features = np.load('padded_features.npy')
print(f"The size of padded_labels is: {padded_features.size}")
num_classes = np.load('num_classes.npy')
print(f"The size of num_classes is: {num_classes.size}")

X_train, X_val, y_train, y_val = train_test_split(
    padded_features, num_classes, test_size=0.2, random_state=42)

model.summary()

# Compile and train the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
training_epochs = 12
batch_size = 32

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=training_epochs, validation_data=(X_val, y_val))

# Evaluate on validation data
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

#model.save("wake_word_model.keras")

# Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

#batch_size = 128
pruning_epochs = 5

num_images = padded_features.shape[0] * (1 - val_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * pruning_epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

# Wrap the model with pruning
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the pruned model
model_for_pruning.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model_for_pruning.summary()

# Setup pruning callbacks
log_dir = tempfile.mkdtemp()
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
]

model_for_pruning.fit(
    padded_features, num_classes,
    batch_size=batch_size,
    epochs=pruning_epochs,
    val_split=val_split,
    callbacks=callbacks)

wwm.print_cnn_weights(model_for_pruning)

model_for_pruning.save("wake_word_model_for_pruning.keras")
