import tensorflow as tf
import numpy as np

# 1. Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="./Infer/sign_lang_img_det.tflite")  # Replace with your model path
interpreter.allocate_tensors()

# 2. Get input and output details
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)

# 3. Process the image data (consistent with training)

infer_dataset = tf.keras.utils.image_dataset_from_directory(
    "./Infer",
    image_size=(32, 32),  # Explicitly set image size
    batch_size=1,  # Or 1 if you want to process one image at a time
    color_mode='grayscale',
    shuffle=False,
    seed=42
)


for images, labels in infer_dataset:  # Iterate through the dataset (even if it's just one image)
    for image in images:  # Process each image in the batch.
        # Ensure correct shape and type.  Important for grayscale!
        input_data = np.array(image, dtype=np.float32)  # Convert to NumPy and float32
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension. Crucial!

        # Check if the model expects a batch or single image. Adjust as needed.
        if len(input_details[0]['shape']) == 4 and input_details[0]['shape'][0] == 1:  # expecting batch
            pass  # input_data already in correct format
        elif len(input_details[0]['shape']) == 3:  # expecting single image. Remove batch dim.
            input_data = np.squeeze(input_data, axis=0)
        else:
            raise ValueError("Unexpected input shape from model")

        # 4. Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 5. Run inference
        interpreter.invoke()

        # 6. Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 7. Process the output
        print(output_data)  # Or do something else with the predictions
