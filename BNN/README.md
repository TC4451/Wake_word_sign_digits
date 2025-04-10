<h1>Binary Neural Networks</h1>

In this section of the project we will binarize the wake work.

Tutorial on BNN can be found here -->

[Binary Neural Network Survey](https://arxiv.org/pdf/2004.03333)
[Comprehensive review of BNN](https://arxiv.org/pdf/2110.06804)

[Larq](https://docs.larq.dev/larq/) is an open-source Python library for training neural networks with extremely low-precision weights and activations, such as Binarized Neural Networks.

One issue is the larg only works with older version of python

![image](https://github.com/user-attachments/assets/8620398a-4837-4169-8482-29ab03221ade)

For this project we use

python version 3.10.16
larq 13.3
tensorflow 2.10

We will follow the tutorial flow provided in the [LARQ documentation](https://docs.larq.dev/larq/tutorials/binarynet_cifar10/)

<h2>DataSet</h2>

For the wake word detector we used the same data as before (Google Speech Command Dataset)

[https://www.kaggle.com/datasets/neehakurelli/google-speech-commands](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands)

<h2>Binary Model</h2>



