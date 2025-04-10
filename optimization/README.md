<h1>Optimization with Tensorflow Model Optimization Toolkit and Lite-RT</h1>

The Wake Word Model will now be optimized first with Lite-RT (Tensorflow Lite)

<h2>Lite-RT (Tensorflow Lite</h2>

Documentation at - [LiteRT overview](https://ai.google.dev/edge/litert)

<h2>Lite RT Optimization Code</h2>

All code can be found in [/optimization/WW_LiteRT_Optimization.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/optimization/WW_LiteRT_Optimization.py) & [/optimization/Evaluate_Optmitizied_tfile.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/optimization/Evaluate_Optimizied_tfile.py)

<h2>Process and Results</h2>

The original Wake Word CNN was 79K

![image](https://github.com/user-attachments/assets/7dc80651-928f-4402-920c-0715224d003c)

![image](https://github.com/user-attachments/assets/6cecf19d-491d-494e-8769-741b7c290391)


<h2>Post-training dynamic range quantization</h2>

post training DR quantization documentation --> [Post-training dynamic range quantization](https://ai.google.dev/edge/litert/models/post_training_quant)

•	Took the previously trained Keras model

•	applied the optimization flag --> converter.optimizations = [tf.lite.Optimize.DEFAULT]

•	Convert the model to a tflite model

This reduce the size to 

![image](https://github.com/user-attachments/assets/417b9194-ee50-455d-b667-7e20f76f0045)

Looking at some model layer output weights of the model you see FP16 and INT8

T#2(arith.constant1) shape:[32, 3, 3, 1], type:FLOAT32

T#6(tfl.pseudo_qconst) shape:[64, 3, 3, 32], type:INT8

T#7(tfl.pseudo_qconst1) shape:[2, 704], type:INT8

| Model    | Size     | Accuracy |
| -------- | -------- | -------- |
| Tflite Model – No Optimization   | 79.1K  | 93.62%  |
|Tflite – Dynamic Range Optimization  | 21.5K   | 93.65%   |

<h2>Post-training integer quantization</h2>

documentation --> [Post-training integer quantization](https://ai.google.dev/edge/litert/models/post_training_integer_quant)

Integer quantization is an optimization strategy that converts 32-bit floating-point numbers (such as weights and activation outputs) to the nearest 8-bit fixed-point numbers
Now the input and output tensors are INT8 using the commands

converter.inference_input_type = tf.uint8, converter.inference_output_type = tf.uint8

This severly reduced accuracy

| Model    | Size     | Accuracy |
| -------- | -------- | -------- |
| Tflite Model – No Optimization   | 79.1K  | 93.62%  |
|Tflite – Dynamic Range Optimization  | 21.5K   | 93.65%   |
| Tflite – Integer quantization | 20.6K | 49.7% |

<h2>Post-training float16 quantization</h2>

This sets the model to use FP16 instead of FP32

converter.target_spec.supported_types = [tf.float16]

| Model    | Size     | Accuracy |
| -------- | -------- | -------- |
| Tflite Model – No Optimization   | 79.1K  | 93.62%  |
|Tflite – Dynamic Range Optimization  | 21.5K   | 93.65%   |
| Tflite – Integer quantization | 20.6K | 49.7% |
| Tflite – FP16 | 40.5K | 93.62% | 

<h2>Post-training integer quantization with int16 activations</h2>

command used --> converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

Subgraph#0 main(T#15) -> [T#16]
  Op#0 QUANTIZE(T#15) -> [T#0]
  Op#1 CONV_2D(T#0, T#6, T#7) -> [T#8]
  Op#2 MAX_POOL_2D(T#8) -> [T#9]
  Op#3 CONV_2D(T#9, T#2, T#1) -> [T#10]
  Op#4 MAX_POOL_2D(T#10) -> [T#11]
  Op#5 RESHAPE(T#11, T#5[-1, 704]) -> [T#12]
  Op#6 FULLY_CONNECTED(T#12, T#3, T#4) -> [T#13]
  Op#7 SOFTMAX(T#13) -> [T#14]
  Op#8 DEQUANTIZE(T#14) -> [T#16]

Tensors of Subgraph#0
  T#0(serving_default_input_layer:0_int16) shape_signature:[-1, 50, 13, 1], type:INT16
  
  T#1(arith.constant) shape:[64], type:INT64 RO 512 bytes, buffer: 2, data:[??, ??, ??, ??, ??, ...]
  
  T#2(arith.constant1) shape:[64, 3, 3, 32], type:INT8 RO 18432 bytes, buffer: 3, data:[8, ., ., ., ., ...]
  
  T#3(arith.constant2) shape:[2, 704], type:INT8 RO 1408 bytes, buffer: 4, data:[., ., ., ., ., ...]
  
  T#4(arith.constant3) shape:[2], type:INT64 RO 16 bytes, buffer: 5, data:[??, ??, ??, ??, ??, ...]
  
  T#5(arith.constant4) shape:[2], type:INT32 RO 8 bytes, buffer: 6, data:[-1, 704]
  
  T#6(sequential_1/conv2d_1/convolution) shape:[32, 3, 3, 1], type:INT8 RO 288 bytes, buffer: 7, data:[., ., ., ., z, ...]
  
  T#7(sequential_1/conv2d_1/Relu;sequential_1/conv2d_1/BiasAdd;sequential_1/conv2d_1/convolution;) shape:[32], type:INT64 RO 256 bytes, buffer: 8, data:[??, ??, ??, ??, ??, ...]
  
  T#8(sequential_1/conv2d_1/Relu;sequential_1/conv2d_1/BiasAdd;sequential_1/conv2d_1/convolution;1) shape_signature:[-1, 48, 11, 32], type:INT16
  
  T#9(sequential_1/max_pooling2d_1/MaxPool2d) shape_signature:[-1, 24, 5, 32], type:INT16
  
  T#10(sequential_1/conv2d_1_2/Relu;sequential_1/conv2d_1_2/BiasAdd;sequential_1/conv2d_1_2/convolution;sequential_1/conv2d_1_2/Squeeze) shape_signature:[-1, 22, 3, 64], type:INT16
  
  T#11(sequential_1/max_pooling2d_1_2/MaxPool2d) shape_signature:[-1, 11, 1, 64], type:INT16
  
  T#12(sequential_1/flatten_1/Reshape) shape_signature:[-1, 704], type:INT16
  
  T#13(sequential_1/dense_1/MatMul;sequential_1/dense_1/BiasAdd) shape_signature:[-1, 2], type:INT16
  
  T#14(StatefulPartitionedCall_1:0_int16) shape_signature:[-1, 2], type:INT16
  
  T#15(serving_default_input_layer:0) shape_signature:[-1, 50, 13, 1], type:FLOAT32
  
  T#16(StatefulPartitionedCall_1:0) shape_signature:[-1, 2], type:FLOAT32
  

| Model    | Size     | Accuracy |
| -------- | -------- | -------- |
| Tflite Model – No Optimization   | 79.1K  | 93.62%  |
|Tflite – Dynamic Range Optimization  | 21.5K   | 93.65%   |
| Tflite – Integer quantization | 20.6K | 49.7% |
| Tflite – FP16 | 40.5K | 93.62% | 
| Tflite- Int8-Int16 | 21.0K | 93.57% |

<h1>Get started with TensorFlow model optimization</h1>

The wake word model will now be optimized using the techniques found in the tensorflow model optimization guide located at [TensorFlow model optimization](https://www.tensorflow.org/model_optimization/guide)

The toolkit supports post-training quantization, quantization aware training, pruning, and clustering. The toolkit also provides experimental support for collaborative optimization to combine various techniques.

<h2>Pruning</h2>

All code can be found at  /Optimization/WW_TFOptimization.py

Apply pruning to the whole model. Start the model with 50% sparsity (50% zeros in weights) and end with 80% sparsity. Use the command

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)

<h3>Results</h3>

| Original    | Training params – 20226 (79.01KB)        |Accuracy – 96.1 |
| -------- | -------- | -------- |
| Layer Conv2d   | 320  |   |
|Layer Conv2d_1  | 18496   |   |
| Dense | 1410 | |

| Pruned    | Training params non-zero – 4063 (16.2KB)          |Accuracy – 93,7 |
| -------- | -------- | -------- |
| Layer Conv2d   | 58 (non-zero weights) - 230 (zero weights) - 32 (non-zero biases) |   |
|Layer Conv2d_1  | 3721 (non-zero weights) - 14711 (Zero Weights) - 64 (non-zero bias)|   |
| Dense | 284 (non-zero weights)- 1124 (Zero Weights) - 2 (non-zero biases)| |

<h2>Structural pruning of weights</h2>

Structural pruning systematically zeroes out model weights at the beginning of the training process. Apply pruning to each layer

All code can be found at  /Optimization/WW_TFOptimization_Structured_Pruning.

Apply Pruning to each layer

Model = keras.Sequential(
    [
        keras.Input(shape=[50,13,1]),  # Input shape (max_len, n_mfcc, 1) for 2D CNN
        
        #keras.layers.InputLayer(batch_input_shape=(None, 50, 13, 1)),
        
        prune_low_magnitude(
        
                keras.layers.Conv2D(
                
                    2, kernel_size=(3, 3), activation='relu',
                    
                    name="pruning_sparsity_0_5"),
                    
                **pruning_params_sparsity_0_5),
                
        #keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        prune_low_magnitude(
        
                keras.layers.Conv2D(
                
                    64, kernel_size=(3, 3), activation="relu",
                    
                    name="structural_pruning"),
                    
                **pruning_params_2_by_4),
                
        #keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Flatten(),
        
        keras.layers.Dropout(0.5),  # Add dropout for regularization
        
        prune_low_magnitude(
        
                keras.layers.Dense(
                
                    2, activation="softmax",
                    
                    name="structural_pruning_dense"),
                    
                **pruning_params_2_by_4),
                
        #keras.layers.Dense(2, activation="softmax"),  # Output layer (softmax for multi-class)
        

Trainable params: 2646 (10.34 KB)

![image](https://github.com/user-attachments/assets/251c97cc-9e35-42d8-afcd-11caf04b9f26)


Training Accuracy = 92.8
      
Val Accuracy = 94.5

<h1>Quantization aware training</h1>

Apply quantize_model = tfmot.quantization.keras.quantize_model to the model.

All code can be found at  /Optimization/WW_Qaunt_Aware.py

During training, quantize_layer inserts quantization and dequantization operations into the forward pass of the layer. 
This simulates the effects of quantization (rounding and clamping) on the weights and activations, but the actual weights are still stored as floating-point numbers.   

This allows the model to learn weights that are more robust to quantization.

Convert this to a tensorflow lite model - converter.optimizations = [tf.lite.Optimize.DEFAULT]

<h2>Results</h2>
Subgraph#0 main(T#0) -> [T#16]

  Op#0 QUANTIZE(T#0) -> [T#5]
  
  Op#1 CONV_2D(T#5, T#6, T#3[21, -2, -37, 22, 6, ...]) -> [T#7]
  
  Op#2 MAX_POOL_2D(T#7) -> [T#8]
  
  Op#3 CONV_2D(T#8, T#9, T#2[-190, -470, -463, -841, 22, ...]) -> [T#10]
  
  Op#4 MAX_POOL_2D(T#10) -> [T#11]
  
  Op#5 RESHAPE(T#11, T#1[-1, 704]) -> [T#12]
  
  Op#6 FULLY_CONNECTED(T#12, T#13, T#4[282, -282]) -> [T#14]
  
  Op#7 SOFTMAX(T#14) -> [T#15]
  
  Op#8 DEQUANTIZE(T#15) -> [T#16]

Tensors of Subgraph#0
  T#0(serving_default_input_1:0) shape_signature:[-1, 50, 13, 1], type:FLOAT32
  
  T#1(arith.constant) shape:[2], type:INT32 RO 8 bytes, buffer: 2, data:[-1, 704]
  
  T#2(tfl.pseudo_qconst) shape:[64], type:INT32 RO 256 bytes, buffer: 3, data:[-190, -470, -463, -841, 22, ...]
  
  T#3(tfl.pseudo_qconst1) shape:[32], type:INT32 RO 128 bytes, buffer: 4, data:[21, -2, -37, 22, 6, ...]
  
  T#4(tfl.pseudo_qconst2) shape:[2], type:INT32 RO 8 bytes, buffer: 5, data:[282, -282]
  
  T#5(sequential/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars;) shape_signature:[-1, 50, 13, 1], type:INT8
  
  T#6(sequential/quant_conv2d/Conv2D;sequential/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel) shape:[32, 3, 3, 1], type:INT8 RO 288 bytes, buffer: 7, data:[., ., ., ., ., ...]
  
  T#7(sequential/quant_conv2d/Relu;sequential/quant_conv2d/BiasAdd;sequential/quant_conv2d/Conv2D;) shape_signature:[-1, 48, 11, 32], type:INT8
  
  T#8(sequential/quant_max_pooling2d/MaxPool) shape_signature:[-1, 24, 5, 32], type:INT8
  
  T#9(sequential/quant_conv2d_1/Conv2D;sequential/quant_conv2d_1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel) shape:[64, 3, 3, 32], type:INT8 RO 18432 bytes, buffer: 10, data:[., ., ., ., ., ...]
  T#10(sequential/quant_conv2d_1/Relu;sequential/quant_conv2d_1/BiasAdd;sequential/quant_conv2d_1/Conv2D;sequential/quant_conv2d_1/BiasAdd/ReadVariableOp) shape_signature:[-1, 22, 3, 64], type:INT8
  
  T#11(sequential/quant_max_pooling2d_1/MaxPool) shape_signature:[-1, 11, 1, 64], type:INT8
  
  T#12(sequential/quant_flatten/Reshape) shape_signature:[-1, 704], type:INT8
  
  T#13(sequential/quant_dense/MatMul;sequential/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars) shape:[2, 704], type:INT8 RO 1408 bytes, buffer: 14, data:[., L, ., ., ., ...]
  
  T#14(sequential/quant_dense/MatMul;sequential/quant_dense/BiasAdd) shape_signature:[-1, 2], type:INT8
  
  T#15(sequential/quant_dense/Softmax) shape_signature:[-1, 2], type:INT8
  
  T#16(StatefulPartitionedCall:0) shape_signature:[-1, 2], type:FLOAT32

  Model Size 20.6K

  Accuracy 92.6

  <h1>Clustering</h1>
