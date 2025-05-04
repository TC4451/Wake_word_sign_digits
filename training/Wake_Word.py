import librosa
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Look at 1 wav file --> extract Mel-frequency Cepstral Coefficient (MFCC)

# Get audio file
audio_file = "../Dataset/Wake_Word/House/00b01445_nohash_0.wav"
mfcc_dim = 13
max_len = 50
sample_rate = 22050

audio_signal, sr = librosa.load(audio_file)

# Normalize audio signal to [-1,1]
audio_signal = audio_signal /np.max(np.abs(audio_signal))

print(f"Audio Signal : {audio_signal}")
print(len(audio_signal))
print(f"Sampling Rate : {sr}")

plt.figure(figsize=(10, 4))
plt.plot(audio_signal)
plt.show()

# Compute MFCCs from the audio signal
mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=mfcc_dim)
print("Shape of MFCC:", mfccs.shape)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()

mfccs = mfccs.T

padded_mfccs = tf.keras.preprocessing.sequence.pad_sequences(
    [mfccs], padding='post', dtype='float32', maxlen=50)

output = padded_mfccs[0]
print(len(output))
plt.imshow(mfccs.T, cmap='viridis', origin='lower', aspect='auto', extent=[0, mfccs.shape[0], 0, mfcc_dim])
plt.show()

padded_mfccs = padded_mfccs.reshape(-1, max_len, mfcc_dim, 1)
label_array = np.array(['House'])


def convert_wav_to_mfcc(audio_file):
    audio_signal, sr = librosa.load(audio_file)
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=22050, n_mfcc=13)
    return mfccs.T

def prepare_label(data_dir, labels):
    features = []
    all_labels = []
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    file_path = os.path.join(label_dir, filename)
                    mfccs = convert_wav_to_mfcc(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        all_labels.append(label)
    return features, all_labels

data_directory = ("../Dataset/Wake_word/")
labels = ["House" , "BG_Noise"]

features, labels_list = prepare_label(data_directory, labels)

padded_features = pad_sequences(features, maxlen=max_len, padding='post', dtype='float32')

le = LabelEncoder()
encoded_labels = le.fit_transform(labels_list)
print(encoded_labels)

padded_features = padded_features.reshape(-1, max_len, 13, 1)

print("Padded features shape:", padded_features.shape)
print("Encoded labels shape:", encoded_labels.shape)

# print("\nFirst 2 samples, first 5 time steps, all MFCCs and channels (if 2D):")
# print(padded_features[:2, :5, :, :])

# print("\nOr for 1D:")
# print(padded_features[:2, :5, :])

# Save processed features and labels
np.save('padded_features.npy', padded_features)
np.save('encoded_labels.npy', encoded_labels)
print("padded_features.npy and encoded_labels.npy files saved")

# Load and split for training
padded_features = np.load('padded_features.npy')
encoded_labels = np.load('encoded_labels.npy')

X_train, X_val, y_train, y_val = train_test_split(
    padded_features, encoded_labels, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.Input(shape=[50,13,1]),   # model expects input shape (time, mfcc, channels)
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(encoded_labels)), activation="softmax"),
])

model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 15
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

model.save("wake_word_model.keras")
keras.models.save_model(model, "keras_file.h5", include_optimizer=False)

loaded_model = tf.keras.models.load_model("wake_word_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()
open("./wake_word.tflite", "wb").write(tflite_model)
