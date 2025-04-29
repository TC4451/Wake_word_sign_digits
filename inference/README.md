<h1>Inference of the Raspberry PI</h1>

The Wake Word Model and Sign Language Detection Model are saved as tflite after training and run on the Raspberry Pi

<h2>Hardware Used</h2>

Raspberry Pi 5 with 2.4Ghz 64-bit quad-core CPU (8GB RAM)

Raspberry Pi Camera Module V2-8 Megapixel,1080p (RPI-CAM-V2)

USB 2.0 Mini Microphone

LAFVIN Kit (4 Digit 7-segment Display, LCD, LED’s, etc)

<h2>Lite RT Optimization Code</h2>

All code can be found in [Whole System.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/inference/Whole%20System.py) , 
[capture_images.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/inference/capture_images.py), 
[image_results.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/inference/Image_results.py), 
[LCD.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/inference/LCD.py)


<h2>Inference Process Summary</h2>

0. LCD is inititialize and displays the word "Locked"
1. Load the wake_word.tflite and sign_language.tfile
2. Record Audio
3. send to wake_word model for detetection
4. If detected turn on camera
5. if not detected goto #2
6. Take 4 photo every 3 secs
7. send photos to sign_language model for inference
8. if detected write "Unlocked" to LCD
9. if incorrect stay in locked state - return to #2


<h2></h2> Detailed Wake Word Inference Process </h2>

Audio Recording & Buffering​

Open a continuous audio stream at 22050 Hz.​

Capture mono audio data via a callback.​

Append samples to a ring buffer (deque) with thread-safety using a lock.​

Audio Preprocessing​

Use a 1-second audio window that updates every 0.5 seconds.​

Normalize audio and extract MFCC features with librosa.​

Pad and reshape MFCCs for consistent model input.​

Inference Process​

Feed preprocessed MFCC data into the TFLite model.​

Retrieve prediction probabilities for wake word detection.​

Trigger a wake word detection alert if probability exceeds 0.9.​

Continuous Monitoring​

Run inference every 0.5 seconds in a loop.​

Ensure smooth and real-time detection with overlapping audio windows.​

Concurrency Considerations​

Use a lock to synchronize audio buffering and processing.


