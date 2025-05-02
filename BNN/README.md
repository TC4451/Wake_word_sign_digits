<h1>Binary Neural Networks</h1>

In this section of the project we will binarize the wake word neural network.

Tutorial's on BNN can be found here -->

[Binary Neural Network Survey](https://arxiv.org/pdf/2004.03333)

[Comprehensive review of BNN](https://arxiv.org/pdf/2110.06804)

[Larq](https://docs.larq.dev/larq/) is an open-source Python library for training neural networks with extremely low-precision weights and activations, such as Binarized Neural Networks.

One issue is that larq only works with older version of python. This is an excerpt from the documentation

![image](https://github.com/user-attachments/assets/8620398a-4837-4169-8482-29ab03221ade)

For this project found out we can use

python version 3.10.16

larq 13.3

tensorflow 2.10

We will follow the tutorial flow provided in the [LARQ documentation](https://docs.larq.dev/larq/tutorials/binarynet_cifar10/)

<h2>DataSet</h2>

For the wake word detector we used the same data as before (Google Speech Command Dataset)

[https://www.kaggle.com/datasets/neehakurelli/google-speech-commands](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands)

<h2>Binary Model</h2>

All code can be found at [/bnn/wake_word_bnn.py](https://github.com/TC4451/Wake_word_sign_digits/blob/main/BNN/wake_word_bnn.py)


The non-BNN for the Wake Word had the form of 

![image](https://github.com/user-attachments/assets/27a9f88c-af78-449e-bbfa-19e8f8a5ccac)

We will keep these same number of layers. Therefore the BNN model will be described with the following syntax

model = tf.keras.models.Sequential()


model.add(lq.layers.QuantConv2D(32, (3, 3),
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False,
                                input_shape=(50,13,1)))
                                
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))

model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.BatchNormalization(scale=False))
93
model.add(tf.keras.layers.Flatten())

model.add(lq.layers.QuantDense(2, use_bias=False, **kwargs))

model.add(tf.keras.layers.BatchNormalization(scale=False))

model.add(tf.keras.layers.Activation("softmax"))

<h2>Results</h2>

Lyrq output shosw a non-BNN model of this structure would be 79.3K but the BNN is 3.22K

![image](https://github.com/user-attachments/assets/511d6f38-67c3-4c5b-a8c4-dbef0d80d533)

The BNN is not a 100% BNN. To maintain accuracy the input, output, and batch normalization layers are FP32.

![image](https://github.com/user-attachments/assets/3b28994c-c95e-413e-bcbb-8652784472a7)

After training of 5 epoch this model has an accuracy of 95.3%

If I could change the kernel from 3x3 to 2x2, I would save many 1-bit MACs'. Below are the results.

![image](https://github.com/user-attachments/assets/1beb64f1-cada-4e04-8597-43d10286bd08)

Accuracy - 93.2%

You can see some of the weigtht of the model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs)) layer


![image](https://github.com/user-attachments/assets/75f9c1a2-450e-4f73-b457-d1dcb8a115d6)

