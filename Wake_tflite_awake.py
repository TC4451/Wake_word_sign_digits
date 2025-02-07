import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import deque

interpreter = tf.lite.Interpreter(model_path="wake_word.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

fs = 22050
n_mfcc = 13 
max_len = 50   
THRESHOLD = 0.9 

# We want a 1-second window of audio that updates every 0.5s
WINDOW_SIZE = fs
STEP_SIZE = fs // 2


audio_buffer = deque(maxlen=WINDOW_SIZE * 2)

new_data_ready = False
lock = False

def audio_callback(indata, frames, time_info, status):

    global new_data_ready, lock

    if status:
        print(status, flush=True)

    mono_data = indata[:, 0]

    if not lock:
        lock = True
        # Add to ring buffer
        audio_buffer.extend(mono_data)
        lock = False

    new_data_ready = True


def audio_to_mfcc(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    mfccs = librosa.feature.mfcc(y=audio_data, sr=fs, n_mfcc=n_mfcc)
    mfccs = mfccs.T  # shape => (time, n_mfcc)

    padded_mfccs = pad_sequences([mfccs], maxlen=max_len, padding='post', dtype='float32')
    padded_mfccs = padded_mfccs.reshape(-1, max_len, n_mfcc, 1)
    return padded_mfccs

def run_inference(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def main():
    global new_data_ready, lock

    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=1024
    )

    with stream:
        print("Recording via callback. Press Ctrl+C to stop.")
        last_inference_time = time.time()

        while True:
            time.sleep(0.1)

            if time.time() - last_inference_time >= 0.5:
                if not lock:
                    lock = True
                    if len(audio_buffer) >= WINDOW_SIZE:
                        window_data = list(audio_buffer)[-WINDOW_SIZE:]
                        lock = False

                        window_data = np.array(window_data, dtype='float32')
                        mfcc_input = audio_to_mfcc(window_data)
                        print(time.time() - last_inference_time)
                        print(len(window_data))
                        print(">>> Prediction Happens <<<")

                        predictions = run_inference(mfcc_input)

                        wake_prob = predictions[0][1] if predictions.shape[1] >= 2 else predictions[0][0]

                        print(f"Wake prob: {wake_prob:.3f}")
                        if wake_prob > THRESHOLD:
                            print(">>> Wake word detected <<<")
                    else:
                        lock = False

                last_inference_time = time.time()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
