import os
import cv2
import numpy as np
import tensorflow as tf

# 1. Load your trained model
save_model_dir = "mobilenet_sign.keras"
model = tf.keras.models.load_model(save_model_dir)
print("Model loaded successfully!")

y = np.load("data/Y.npy")
train_y_labels = tf.convert_to_tensor(y)

label_dict = {}
for number, label in enumerate(np.unique(train_y_labels)):
    label_dict[number] = label

reverse_label_dict = {}
for key in label_dict.keys():
    reverse_label_dict[label_dict[key]] = key
    
print(label_dict)
print(reverse_label_dict)

image_path = "test4.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (128, 128))
img_resized = img_resized / 255.0 
img_batch = np.expand_dims(img_resized, axis=0)

predictions = model.predict(img_batch)
pred_idx = np.argmax(predictions, axis=1)

print(pred_idx)
print("Predicted class index:", pred_idx[0])
print("Predicted label:", label_dict[pred_idx[0]])
