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

batch_size = 128
labels = 2
epochs = 4
validation_split = 0.1

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

# Load data
padded_features = np.load('padded_features.npy')
print(f"The size of padded_labels is: {padded_features.size}")
labels = np.load('labels.npy')
print(f"The size of labels is: {labels.size}")

X_train, X_val, y_train, y_val = train_test_split(
    padded_features, labels, test_size=0.2, random_state=42)

model.summary()

# Compile and train the model
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

wwm.print_cnn_weights(model)

# Clustering setup
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInit = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
  'number_of_clusters': 16,
  'cluster_centroids_init': CentroidInit.LINEAR
}

# Apply weight clustering
clustered_model = cluster_weights(model, **clustering_params)

# Fine-tune clustered model
opt = keras.optimizers.Adam(learning_rate=1e-5)

clustered_model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(),
  optimizer=opt,
  metrics=['accuracy'])

clustered_model.summary()

history = clustered_model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))

wwm.print_cnn_weights(clustered_model)

if clustered_model is None:
    print("Error: Clustering failed. Check clustering parameters and model compatibility.")
else:
    # Print layer names
    print("Layer Names in Clustered Model:")
    for layer in clustered_model.layers:
        print(layer.name)

    #Or, if you want to also see the type of layer.
    print("\nLayer Names and Types in Clustered Model:")
    for layer in clustered_model.layers:
        print(f"{layer.name} ({layer.__class__.__name__})")

# View weights of a specific clustered layer
clustered_layer = clustered_model.get_layer('cluster_conv2d_1')
underlying_layer = clustered_layer.layer
weights = underlying_layer.get_weights()

print(f"Weights of layer 'cluster_conv2d_1':")
for i, weight_array in enumerate(weights):
    print(f"  Weight array {i} (Shape: {weight_array.shape}):")
    print(weight_array)

# Remove clustering wrappers
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

# Save stripped models
keras.models.save_model(final_model, "clustered_keras_file",
                           include_optimizer=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_clustered_model = converter.convert()

with open("clustered_tflite_file", 'wb') as f:
  f.write(tflite_clustered_model)
print('Saved clustered TFLite model to:', clustered_tflite_file)


# Get gzipped model size
def get_gzipped_model_size(file):
  # It returns the size of the gzipped model in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped clustered Keras model: %.2f bytes" % (get_gzipped_model_size("clustered_keras_file")))
print("Size of gzipped clustered TFlite model: %.2f bytes" % (get_gzipped_model_size("clustered_tflite_file")))
