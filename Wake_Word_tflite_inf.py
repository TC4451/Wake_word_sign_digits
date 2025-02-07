import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="./Infer/wake_word.tflite")  # Replace with your model path
interpreter.allocate_tensors()

# 2. Get input and output details
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
max_len=50

def convert_wav_to_mfcc_single(audio_file):
    Audio_Amp_Data, Sample_Rate = librosa.load(audio_file)
    Audio_Amp_Data = Audio_Amp_Data / np.max(np.abs(Audio_Amp_Data))
    mfccs = librosa.feature.mfcc(y=Audio_Amp_Data, sr=22050, n_mfcc=13)
    mfccs = mfccs.T
    padded_mfccs = pad_sequences([mfccs], maxlen=max_len, padding='post', dtype='float32') # Use max_len here
    padded_mfccs = padded_mfccs.reshape(-1, max_len, 13, 1) # Use max_len here
    return padded_mfccs


def prepare_input_for_tflite(audio_file):
    """Prepares audio data for inference with the TFLite model."""
    mfccs = convert_wav_to_mfcc_single(audio_file)

    # The input needs to be a NumPy array with the correct shape and type
    # Get the expected input shape from the model
    input_shape = input_details[0]['shape']
    return mfccs

input_data = prepare_input_for_tflite("./me_House.wav")

# Set the tensor to the input data
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)