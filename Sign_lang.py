import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(dataset, num_images=25, images_per_row=5):
    """Visualizes a batch of images and labels from a tf.data.Dataset."""
    for images, labels in dataset.take(1):  # Take just one batch
        num_to_display = min(num_images, images.shape[0])  # Adjust if batch is smaller
        num_rows = (num_to_display + images_per_row - 1) // images_per_row
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows))
        image_count = 0

        for i in range(num_to_display):
            row = i // images_per_row
            col = i % images_per_row

            image = tf.squeeze(images[i]).numpy() # Get image as numpy array
            label_index = tf.argmax(labels[i]).numpy() # Get index of the label
            label_name = dataset.class_names[label_index] # Get the label name

            if num_rows > 1:  # Handle cases with multiple rows
                ax = axes[row, col] if isinstance(axes, type(np.ndarray)) else axes # Fix for single row case
            else:
                 ax = axes[col] if isinstance(axes, type(np.ndarray)) else axes

            ax.imshow(image, cmap='gray')
            ax.set_title(label_name)
            ax.axis('off')
            image_count += 1

        # Hide any unused subplots
        for j in range(image_count, num_rows * images_per_row):
            row = j // images_per_row
            col = j % images_per_row
            if num_rows > 1:
                ax = axes[row, col] if isinstance(axes, type(np.ndarray)) else axes # Fix for single row case
            else:
                 ax = axes[col] if isinstance(axes, type(np.ndarray)) else axes
            ax.axis('off')

        plt.tight_layout()
        plt.show()




image_size = (32, 32)
batch_size = 32  # You can adjust this

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "Dataset",  # Your data directory
    labels='inferred',  # Infer labels from directory names
    label_mode='int',  # Or 'binary' if you have only two classes
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale', # Convert to Grayscale
    shuffle=True,  # Shuffle the data (good practice for training)
    validation_split=0.2, # Optional: Create a validation set
    subset="training",  # Optional: Specify 'training' or 'validation'
    seed=42  # Optional: For reproducibility
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "Dataset",
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42
)



class_names = train_dataset.class_names
print(class_names)

def display_images(dataset, num_images):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

# Display 25 images and labels
display_images(val_dataset, 25)

class_names = train_dataset.class_names  # Get class names
print(train_dataset)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),  # Grayscale: 1 input channel
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Output layer with softmax
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 20  # Adjust as needed
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)


# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Plot training history (optional)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('my_model.keras')

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



print(y_predicted[2])
print(y_predicted[3])
print(y_predicted[4])

