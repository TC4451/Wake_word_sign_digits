import os
import cv2
import numpy as np
import tensorflow as tf
import time
from LCD import LCD

def load_sl_model():
    tflite_model_path = "./sign_language.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TensorFlow Lite model loaded successfully!")

    y = np.load("./Y.npy")
    train_y_labels = tf.convert_to_tensor(y)

    label_dict = {}
    for number, label in enumerate(np.unique(train_y_labels)):
        label_dict[number] = label

    reverse_label_dict = {}
    for key in label_dict.keys():
        reverse_label_dict[label_dict[key]] = key

    print(label_dict)
    print(reverse_label_dict)
    
    image_files = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
    predicted_labels = []
    lcd = LCD(2, 0x27, True)
    predicted_labels_string = ""
    
    for filename in image_files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (128, 128))
        img_resized = img_resized / 255.0
        img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])

        pred_idx = np.argmax(predictions, axis=1)
        if pred_idx[0] == 10:
            pred_idx[0]= 9
        confidence = predictions[0, pred_idx[0]] * 100
        predicted_label = label_dict[pred_idx[0]]

        print(f"Image: {filename}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Predicted class index: {pred_idx[0]}")
        print(f"Predicted label: {predicted_label}\n")
        predicted_labels.append(pred_idx[0])
        #predicted_labels_string += predicted_label + ", "
        #lcd.message(predicted_labels_string, 1)
        
    return predicted_labels
