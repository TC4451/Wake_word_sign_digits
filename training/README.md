<h1>Wake Word and Sign Language CNN’s Training</h1>

For the wake word detector we used the Google Speech Command Dataset

[https://www.kaggle.com/datasets/neehakurelli/google-speech-commands](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands)

 "Warden P. Speech Commands: A public dataset for single-word
speech recognition, 2017. Available from
[http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz]{http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz}"

The Wake word used was “House”. The data set used to train the WakeWord CNN used 1750 one sec .wav files of the word “House” and 1750 one sec .wav files (combined) other various words (cat, dog, happy, Marvin, zero, etc),


<h2><ins>Wake Word Training Code Location</ins></h2>

[Training/Wake_Word.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/training/Wake_Word.py) contains all the code to train the Wake Word CNN.


<h2><ins>Wake Word Preprocessing</ins></h2>

•	Wav file loaded using librosa.load

•	Normalize samples to be between 1 & -1

![image](https://github.com/user-attachments/assets/49c34b67-2361-4327-87dc-6bd428a3d67e)

•	Get the Mel-frequency cepstral coefficient, a representation of the short-term power spectrum of a sound 

![image](https://github.com/user-attachments/assets/240a7c63-a344-423d-a2ef-9e215efea301)

•	Transpose and add padding so max_len is 50

![image](https://github.com/user-attachments/assets/265a63f7-8d36-4d55-b516-e7c46e8fbe00)

•	All wavs file stored as NP Arrays after processed

<h2><ins>Wake Word Training</ins></h2>

•	Split Train/Val dataset 80%/20%

CNN Summary 

![image](https://github.com/user-attachments/assets/1a375578-f641-46ad-a32d-725e0037d089)

<h2><ins>Wakw Word Results</ins></h2>

![image](https://github.com/user-attachments/assets/eb86e3fa-71d0-4d69-bafb-38252161a7fc)

![image](https://github.com/user-attachments/assets/cd38cc12-4348-4fe8-8966-2c1fb000c96d)

•	Save output for inference to a Keras Model and TensorflowLite Model.

<h1>Sign Language CNN</h1>

<h2><ins>Dataset</ins></h2>

The dataset to train the sign language CNN can be found here.
[27 Class Sign Language Dataset](https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset/discussion?sort=undefined)
The dataset consists of 22801 images 128x128 RGB.
There are 27 different images but we only used 10 (Digits 0-9) as the door code.
The CNN does recognize 27 outputs but anything not 0-9 is assumed a non-signal.

<h2><ins>Sign Language Training Code Location</ins></h2>

[training/Sign_Language.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/training/Sign_Language.py) contains all the code to train the sign language detection model.

<h2><ins>Sign Language Preprocessing</ins></h2>

•	Load the NumPy Array

•	Shuffle and split the data into training, validation, and test sets

<h2><ins>Sign Language Training</ins></h2>

CNN Model

![image](https://github.com/user-attachments/assets/2a4549ba-ce70-447a-824f-8fd0d666bb88)

Trainable Parameters 1,143,804 (4.36MB)

<h2><ins>Sign Language Results</ins></h2>

Training Accuracy = 99.4% | Testing Accuracy = 96.4% | Validation Accuracy = 93.5%





