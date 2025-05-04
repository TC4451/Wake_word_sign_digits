import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="./wake_word.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

max_len = 50


def convert_wav_to_mfcc_single(audio_file):
    audio_signal, sr = librosa.load(audio_file)
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=22050, n_mfcc=13)
    mfccs = mfccs.T
    padded_mfccs = pad_sequences([mfccs], maxlen=max_len, padding='post', dtype='float32')
    padded_mfccs = padded_mfccs.reshape(-1, max_len, 13, 1)
    return padded_mfccs


def prepare_input_for_tflite(audio_file):
    """Prepares audio data for inference with the TFLite model."""
    mfccs = convert_wav_to_mfcc_single(audio_file)
    input_shape = input_details[0]['shape']
    return mfccs

# Convert the WAV file
input_data = prepare_input_for_tflite("./output1.wav")

# Set the tensor to the input data
interpreter.set_tensor(input_details[0]['index'], input_data)

num_inferences = 50
inference_times = []

print(f"Running inference {num_inferences} times...")

for i in range(num_inferences):
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    inference_time_seconds = end_time - start_time
    inference_time_ms = int(inference_time_seconds * 1000)
    inference_times.append(inference_time_ms)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(f"Inference {i+1} Output: {output_data}") # Uncomment to print each output

print("\nInference Times (milliseconds):")
print(inference_times)

average_inference_time = np.mean(inference_times)
print(f"\nAverage Inference Time: {average_inference_time:.2f} milliseconds")
