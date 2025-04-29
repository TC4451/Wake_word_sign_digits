import time
import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from LCD import LCD
import capture_images
import Image_results

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="./wake_word.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)


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

    mfccs_padded = mfccs_padded.reshape(-1, max_len, n_mfcc, 1).astype(np.float32) # ensure float32 for tflite
    return mfccs_padded


def predict_wake_word():
    audio_data = record_audio()
    features = preprocess_audio(audio_data)

    # Set the input tensor.
    interpreter.set_tensor(input_details[0]['index'], features)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    predictions = interpreter.get_tensor(output_details[0]['index'])

    print("Predictions:", predictions)
    print(predictions[0, 1])

    if predictions[0, 1] > 0.95:
        print(">>> Wake word detected! <<<")
      # time.sleep(1)
        return True # Return True if the condition is met.
    else:
        return False # Return False if the condition is not met.


def main_loop():
    # Initialize the LCD with specific parameters: Raspberry Pi revision, I2C address, and backlight status
    lcd = LCD(2, 0x27, True)
    lcd.message("Locked", 1)
    while True:
        if predict_wake_word():
            break
        # Tiny sleep to prevent CPU overload
        time.sleep(0.1)
    lcd.message("Enter Code", 1)
    capture_images.capture_images()
    entered_code = Image_results.load_sl_model()
    print(entered_code)
    if entered_code == [1, 2, 1, 2]:
        lcd.message("Unlocked", 1)
        
    
    
# 
if __name__ == "__main__":
    main_loop()
