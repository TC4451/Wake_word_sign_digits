import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = "."
loaded_model = tf.keras.models.load_model("wake_word_model.keras")

# Convert the Keras model to a TensorFlow Lite model
#converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
#tflite_model = converter.convert()

# Save the TensorFlow Lite model
#with open("wake_word_model.tflite", "wb") as f:
#  f.write(tflite_model)

#print("TensorFlow Lite model saved as wake_word_model.tflite")

def convert_wav_to_mcff_single(audio_file):
    Audio_Amp_Data, Sample_Rate = librosa.load(audio_file)
    Audio_Amp_Data = Audio_Amp_Data / np.max(np.abs(Audio_Amp_Data))
    mfccs = librosa.feature.mfcc(y=Audio_Amp_Data, sr=22050, n_mfcc=13)
    mfccs = mfccs.T
    padded_mfccs = pad_sequences([mfccs], maxlen=50, padding='post', dtype='float32')
    padded_mfccs = padded_mfccs.reshape(-1, max_len, 13, 1)
    return padded_mfccs


for layer in loaded_model.layers:
    print(f"Layer: {layer.name}")
    weights = layer.get_weights()
    if weights:  # Check if the layer has weights (e.g., Conv2D, Dense)
        for i, w in enumerate(weights):
            print(f"  Weight {i+1} shape: {w.shape}")
            # If the weight matrix isn't too large, you can print it:
            # print(w) # Be cautious with large weight matrices

new_audio_file = "./me_House.wav"  # Path to the new WAV file
n_mfcc = 13  # from training
max_len = 50  # Example. Change to the value used during training
new_audio_data = convert_wav_to_mcff_single(new_audio_file)
predictions = loaded_model.predict(new_audio_data)
print(predictions)
