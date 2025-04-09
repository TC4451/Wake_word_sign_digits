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



