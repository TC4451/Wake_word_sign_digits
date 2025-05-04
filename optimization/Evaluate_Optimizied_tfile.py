input_data = padded_features[:]
expected_output = numerical_labels[:]

correct_predictions = 0
total_predictions = len(input_data)

for i in range(total_predictions):
    # Set the input tensor
    input_tensor = input_data[i:i+1].astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_tensor)

    if predicted_label == expected_output[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions

print(f"TFLite Accuracy: {accuracy:.4f}")

tf.lite.experimental.Analyzer.analyze(model_content=interpreter)
