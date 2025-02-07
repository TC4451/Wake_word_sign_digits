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

# Look at 1 wav file to MCFF

# get audio file
audio_file = "./House/00b01445_nohash_0.wav"
mfcc_dimensionality = 13
max_len = 50
sample_rate = 22050

Audio_Amp_Data, Sample_Rate = librosa.load(audio_file)

#normailize ass valie between -1,1
Audio_Amp_Data = Audio_Amp_Data /np.max(np.abs(Audio_Amp_Data))

print(f"Audio_Amp_Data : {Audio_Amp_Data}")  # Will print the NumPy array of audio samples
print(len(Audio_Amp_Data))
print(f"Sampling rate : {Sample_Rate}")  # Will print(the original sampling rate)

plt.figure(figsize=(10, 4))  # Adjust figure size as needed
plt.plot(Audio_Amp_Data)  # Plot the waveform
plt.show()

# Computes the Mel-Frequency Cepstral Coefficients (MFCCs) of an audio signal
mfccs = librosa.feature.mfcc(y=Audio_Amp_Data, sr=Sample_Rate, n_mfcc=mfcc_dimensionality)
print("Shape of mfcc:", mfccs.shape)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=Sample_Rate, x_axis='time')
plt.show()

# transpose so Rows: Time frames and Columns: MFCC coefficients.
mfccs = mfccs.T

padded_mfccs = tf.keras.preprocessing.sequence.pad_sequences(
    [mfccs], padding='post', dtype='float32', maxlen=50)

output = padded_mfccs[0]
print(len(output))
plt.imshow(mfccs.T, cmap='viridis', origin='lower', aspect='auto', extent=[0, mfccs.shape[0], 0, mfcc_dimensionality])
plt.show()

padded_mfccs = padded_mfccs.reshape(-1, max_len, mfcc_dimensionality, 1)

print(len(output))
plt.imshow(mfccs.T, cmap='viridis', origin='lower', aspect='auto', extent=[0, mfccs.shape[0], 0, mfcc_dimensionality])
plt.show()

label_array = np.array(['House'])

def convert_wav_to_mcff(audio_file):
    print(audio_file)
    Audio_Amp_Data, Sample_Rate = librosa.load(audio_file)
    Audio_Amp_Data = Audio_Amp_Data / np.max(np.abs(Audio_Amp_Data))
    mfccs = librosa.feature.mfcc(y=Audio_Amp_Data, sr=22050, n_mfcc=13)
    return mfccs.T

def prepare_label(data_dir, labels):
    features = []
    all_labels = []
    for label in labels:  # Iterate through the labels
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    file_path = os.path.join(label_dir, filename)
                    mfccs = convert_wav_to_mcff(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        all_labels.append(label)  # Append the label

    return features, all_labels

data_directory = ("./")
labels = ["House" , "BG_Noise"]

features, labels_list = prepare_label(data_directory, labels)

padded_features = pad_sequences(features, maxlen=max_len, padding='post', dtype='float32')

le = LabelEncoder()
numerical_labels = le.fit_transform(labels_list)

padded_features = padded_features.reshape(-1, max_len, 13, 1)

print("Padded features shape:", padded_features.shape)
print("Numerical labels shape:", numerical_labels.shape)

# Print a few MFCCs for verification (adjust indices as needed)
print("\nFirst 2 samples, first 5 time steps, all MFCCs and channels (if 2D):")
print(padded_features[:2, :5, :, :])  # If 2D

# Or for 1D:
# print(padded_features[:2, :5, :])  # If 1D

print("\nNumerical labels:")
print(numerical_labels)

#8. Save to numpy array
np.save('padded_features.npy', padded_features)
np.save('numerical_labels.npy', numerical_labels)
print("padded_features.npy and numerical_labels.npy files saved")

## Split and train and test

padded_features = np.load('padded_features.npy')
numerical_labels = np.load('numerical_labels.npy')

X_train, X_val, y_train, y_val = train_test_split(
    padded_features, numerical_labels, test_size=0.2, random_state=42)

model = keras.Sequential(
    [
        keras.Input(shape=X_train.shape[1:]),  # Input shape (max_len, n_mfcc, 1) for 2D CNN
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Add dropout for regularization
        layers.Dense(len(np.unique(numerical_labels)), activation="softmax"),  # Output layer (softmax for multi-class)
    ]
)

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
epochs = 10
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