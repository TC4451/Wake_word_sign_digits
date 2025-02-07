import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('my_model.keras')

#

#Get the weights of the model
weights = model.get_weights()
image_size = (32, 32)

infer_dataset = tf.keras.utils.image_dataset_from_directory(
    "./Infer",
    image_size=image_size,
    batch_size=5,
    color_mode='grayscale',
    shuffle=False,
    seed=42
)

y_predicted = model.predict(infer_dataset)
print(y_predicted[0])
max_val = np.max(y_predicted[0])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[0])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')

print(y_predicted[1])
max_val = np.max(y_predicted[1])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[1])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')

print(y_predicted[2])
max_val = np.max(y_predicted[2])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[2])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')

print(y_predicted[3])
max_val = np.max(y_predicted[3])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[3])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')

print(y_predicted[4])
max_val = np.max(y_predicted[4])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[4])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')

print(y_predicted[5])
max_val = np.max(y_predicted[5])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[5])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')

print(y_predicted[6])
max_val = np.max(y_predicted[6])
max_val_percentage = max_val * 100
max_val_position = np.argmax(y_predicted[6])
print(f'Prediction: {max_val_position}')
print(f'Percentage: {max_val_percentage:.2f}%')