import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


my_seed = 42
split_proportion = 0.1
batch_size = 4
number_epochs = 10
lr = 3e-5

figure_count = 0
#figure_dir = os.path.join("..", "Dataset")
#if not os.path.exists(figure_dir):
#    os.mkdir(figure_dir)

train_new = True

print(f"Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

input_dir = "../Dataset/Sign_language/"

x_filename = os.path.join(input_dir, "X.npy")
y_filename = os.path.join(input_dir, "Y.npy")

print(x_filename)
x = np.load(x_filename)
y = np.load(y_filename)

# Shuffle and split the data
split_number = int(split_proportion * x.shape[0])

np.random.seed(my_seed)
np.random.shuffle(x)
val_x = tf.convert_to_tensor(x[:split_number])
test_x = tf.convert_to_tensor(x[split_number:2 * split_number])
train_x = tf.convert_to_tensor(x[2 * split_number:])

np.random.seed(my_seed)
np.random.shuffle(y)
val_y_labels = tf.convert_to_tensor(y[:split_number])
test_y_labels = tf.convert_to_tensor(y[split_number:2 * split_number])
train_y_labels = tf.convert_to_tensor(y[2 * split_number:])

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

train_y = tf.convert_to_tensor(np_train_y.reshape(-1), dtype=tf.int32)
val_y = tf.convert_to_tensor(np_val_y.reshape(-1), dtype=tf.int32)
test_y = tf.convert_to_tensor(np_test_y.reshape(-1), dtype=tf.int32)

number_classes = len(label_dict.keys())

input_shape = train_x.shape[1:]  # (height, width, channels)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),  # Start with a convolutional layer
    MaxPooling2D((2, 2)),  # Pooling to reduce spatial dimensions
    Conv2D(64, (3, 3), activation='relu', padding='same'),  # Another convolutional layer
    MaxPooling2D((2, 2)),  # Another pooling layer
    Conv2D(128, (3, 3), activation='relu', padding='same'),  # Convolutional layer
    MaxPooling2D((2, 2)), # Pooling layer
    Flatten(),  # Flatten before dense layers
    Dropout(0.25),  # Dropout for regularization
    Dense(32, activation='relu'),  # Dense layers
    Dense(32, activation='relu'),
    Dense(number_classes, activation='softmax')  # Output layer
])

# Warm up the model by making an initial forward pass
_ = model(train_x[0:1])
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


def make_scheduler(my_lr):
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

scheduler = make_scheduler(lr)
lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# save_model_dir = os.path.join(".", "mobilenet_sign.keras")
save_model_dir = "sign_language_try.keras"
tf_lite_model_filename = os.path.join(".", "sign_language.tflite")

if train_new:
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(val_x, val_y),
        batch_size=batch_size,
        epochs=number_epochs,
        callbacks=[tensorboard_callback, lr_scheduler_callback]
    )

    # ---- SAVE THE MODEL AFTER TRAINING ----
    model.save(save_model_dir)
    print(f"Model saved to {save_model_dir}")

    loaded_model = tf.keras.models.load_model("sign_language_try.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    tflite_model = converter.convert()
    open("./sign_language.tflite", "wb").write(tflite_model)
    print("Tensorflow Lite Model Saved")

else:
    model = tf.keras.models.load_model(save_model_dir)
    print(f"Model loaded from {save_model_dir}")
