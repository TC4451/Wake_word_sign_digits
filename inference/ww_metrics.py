import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# 1. Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="./wake_word.tflite")  # Replace with your model path
interpreter.allocate_tensors()

# 2. Get input and output details
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
max_len = 50

def convert_wav_to_mfcc_single(audio_file):
    Audio_Amp_Data, Sample_Rate = librosa.load(audio_file)
    Audio_Amp_Data = Audio_Amp_Data / np.max(np.abs(Audio_Amp_Data))
    mfccs = librosa.feature.mfcc(y=Audio_Amp_Data, sr=22050, n_mfcc=13)
    mfccs = mfccs.T
    padded_mfccs = pad_sequences([mfccs], maxlen=max_len, padding='post', dtype='float32')
    padded_mfccs = padded_mfccs.reshape(-1, max_len, 13, 1)
    return padded_mfccs

def prepare_input_for_tflite(audio_file):
    """Prepares audio data for inference with the TFLite model."""
    mfccs = convert_wav_to_mfcc_single(audio_file)
    input_shape = input_details[0]['shape']
    return mfccs

# Convert the WAV file once
input_data = prepare_input_for_tflite("./output1.wav")

# Set the tensor to the input data (do this only once before the loop)
interpreter.set_tensor(input_details[0]['index'], input_data)

num_inferences = 50
inference_times = []

print(f"Running inference {num_inferences} times...")

for i in range(num_inferences):
    # Record the start time
    start_time = time.time()

    # Run inference
    interpreter.invoke()

    # Record the end time
    end_time = time.time()

    # Calculate the inference time in seconds
    inference_time_seconds = end_time - start_time

    # Convert to milliseconds
    inference_time_ms = int(inference_time_seconds * 1000)
    inference_times.append(inference_time_ms)

    # Get the output tensor (optional: you might not need to print it every time)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(f"Inference {i+1} Output: {output_data}") # Uncomment to print each output

# Print the inference times
print("\nInference Times (milliseconds):")
print(inference_times)

# Calculate and print the average inference time
average_inference_time = np.mean(inference_times)
print(f"\nAverage Inference Time: {average_inference_time:.2f} milliseconds")
