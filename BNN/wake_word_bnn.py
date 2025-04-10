X_train, X_val, y_train, y_val = train_test_split(
    padded_features, numerical_labels, test_size=0.2, random_state=42)

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
model.add(lq.layers.QuantConv2D(32, (3, 3),
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=(50,13,1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization(scale=False))
#model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
#model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Flatten())

#model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
#model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(lq.layers.QuantDense(2, use_bias=False, **kwargs))
model.add(tf.keras.layers.BatchNormalization(scale=False))
model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=5)

#test_loss, test_acc = model.evaluate(test_images, test_labels)

model.summary()
lq.models.summary(model)

def print_larq_model_weights(model):
    """
    Prints the weights of a Larq BNN model.

    Args:
        model: A Larq BNN model (tf.keras.Model).
    """
    for layer in model.layers:
        if hasattr(layer, "weights"):  # Check if the layer has weights
            print(f"Layer: {layer.name}")
            for weight in layer.weights:
                print(f"  Weight: {weight.name}")
                print(f"    Shape: {weight.shape}")
                print(f"    Value: {weight.numpy()}") #get the numpy value
                if "quantizer" in weight.name: #check if it's a quantized weight.
                    print(f"    Quantizer: {layer.get_quantizers()}") # print the quantizers.
                print("-" * 20)

#print_larq_model_weights(model)
model.save("wake_word_precision_model.h5")  # save full precision latent weights
fp_weights = model.get_weights()  # get latent weights

with lq.context.quantized_scope(True):
    model.save("wake_word.binary_model.h5")  # save binary weights
    weights = model.get_weights()  # get binary weights]
    print(weights)
