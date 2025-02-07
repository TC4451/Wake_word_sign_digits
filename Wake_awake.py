import time
import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

loaded_model = tf.keras.models.load_model("wake_word_model.keras")

# params
fs = 22050
seconds = 1
n_mfcc = 13
max_len = 50


def record_audio():
    print("Recording...")
    recording = sd.rec(
        int(seconds * fs),
        samplerate=fs,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("Done recording.")
    return recording.flatten()


def preprocess_audio(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    mfccs = librosa.feature.mfcc(y=audio_data, sr=fs, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    
    mfccs_padded = pad_sequences([mfccs], maxlen=max_len, padding='post', dtype='float32')
    
    mfccs_padded = mfccs_padded.reshape(-1, max_len, n_mfcc, 1)
    return mfccs_padded


def predict_wake_word():
    audio_data = record_audio()
    features = preprocess_audio(audio_data)
    predictions = loaded_model.predict(features)
    print("Predictions:", predictions)
    
    if predictions[0, 1] > 0.9: 
        print(">>> Wake word detected! <<<")


def main_loop():
    while True:
        predict_wake_word()
        # Tiny sleep to prevent CPU overload
        time.sleep(0.1)


if __name__ == "__main__":
    main_loop()
