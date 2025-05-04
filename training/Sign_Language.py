import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

random_seed = 42
val_split = 0.1
batch_size = 4
num_epochs = 10
lr = 3e-5
train_new = True

print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

dataset_path = "../Dataset/Sign_language/"
x_path = os.path.join(dataset_path, "X.npy")
y_path = os.path.join(dataset_path, "Y.npy")

x = np.load(x_path)
y = np.load(y_path)

# Shuffle data and labels
np.random.seed(random_seed)
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Split dataset
split_number = int(val_split * x.shape[0])
val_x = tf.convert_to_tensor(x[:split_number])
test_x = tf.convert_to_tensor(x[split_number:2 * split_number])
train_x = tf.convert_to_tensor(x[2 * split_number:])

val_y_labels = tf.convert_to_tensor(y[:split_number])
test_y_labels = tf.convert_to_tensor(y[split_number:2 * split_number])
train_y_labels = tf.convert_to_tensor(y[2 * split_number:])

# Label mapping
label_dict = {}
for number, label in enumerate(np.unique(train_y_labels)):
    label_dict[number] = label

print(label_dict, x.shape)

reverse_label_dict = {}
for key in label_dict.keys():
    reverse_label_dict[label_dict[key]] = key

print(reverse_label_dict)

np_train_y = np.zeros_like(train_y_labels)
np_val_y = np.zeros_like(val_y_labels)
np_test_y = np.zeros_like(test_y_labels)

for ii in range(np_train_y.shape[0]):
    np_train_y[ii] = reverse_label_dict[train_y_labels[ii].numpy()[0]]
for ii in range(np_val_y.shape[0]):
    np_val_y[ii] = reverse_label_dict[val_y_labels[ii].numpy()[0]]
for ii in range(np_test_y.shape[0]):
    np_test_y[ii] = reverse_label_dict[test_y_labels[ii].numpy()[0]]

# Convert string labels to numerical indices
train_y = tf.convert_to_tensor(np_train_y.reshape(-1), dtype=tf.int32)
val_y = tf.convert_to_tensor(np_val_y.reshape(-1), dtype=tf.int32)
test_y = tf.convert_to_tensor(np_test_y.reshape(-1), dtype=tf.int32)

num_classes = len(label_dict.keys())
input_shape = train_x.shape[1:]   # (height, width, channels)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)), 
    Flatten(),
    Dropout(0.25),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Prime the model
_ = model(train_x[0:1])
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Learning rate adjustment function
def lr_schedule(my_lr):
    def scheduler(epoch, lr):
        if epoch <= 1:
            return my_lr / 10.0
        elif epoch == 2:
            return my_lr * 10.0
        else:
            return lr * 0.9
    return scheduler

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs",
    write_graph=True,
    update_freq='epoch',
)

scheduler = lr_schedule(lr)
lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

save_path = "sign_language_try.keras"
tflite_path = os.path.join(".", "sign_language.tflite")

if train_new:
    # Training loop
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(val_x, val_y),
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[tensorboard_callback, lr_scheduler_callback]
    )

    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Export to TensorFlow Lite
    loaded_model = tf.keras.models.load_model(save_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)
    print("Tensorflow Lite Model Saved to {tflite_path}")

else:
    model = tf.keras.models.load_model(save_path)
    print(f"Model loaded from {save_path}")
