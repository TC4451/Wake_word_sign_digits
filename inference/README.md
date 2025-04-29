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

![image](https://github.com/user-attachments/assets/f0219675-0c6d-4fc2-8397-ee54eafc5e79)

<h2> Detailed Sign Language Inference Process </h2>

![image](https://github.com/user-attachments/assets/85e50d53-2d9d-4523-9917-0e9a620941a7)

<h2> Results </h2>

The Wake Word Model is 79K. THe following code was written to determine the speed in ms and the power used on the Raspberry Pi in mw.

Code Used --> [ww_metrics.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/inference/ww_metrics.py)

1. Convert a wav file to MFCC for inference
2. start_time
3. run inference on the wake_word
4. end time
5. repeat #2-4 fifty times
6. print out Average Time
7. Monitor the Power Meter

   






