# NN-Data-Classification

[![made-with-python](https://img.shields.io/badge/Made_with-Python_3.6-1f425f?style=plastic)](https://www.python.org/) ![dev-status](https://img.shields.io/badge/Dev_Status-97.5%25-yellowgreen?style=plastic)

**Note for GitHub users:**  
1. Code and results in rework.
2. Repository size: 248.4 MiB ('rs' folder: 248 MiB).

## How to run?

### Install requirements

#### For using datasets
```
emnist==0.0
scikit-learn==0.24.1
tensorflow==2.5.0
```

#### For mlp lib
```
numpy==1.19.2
bidict==0.21.2
```

#### For main_example.py and mnist_example.py
```
matplotlib==3.0.2
seaborn==0.9.0
```

### Run
Main example:
```
$ python3.6 main_example.py
```
MNIST example:
```
$ python3.6 mnist_example.py
```
XOR example:
```
$ python3.6 xor_example.py
```

## Description

The program is designed for data classification by a neural network.  
All data is first converted to a numeric type. If the data are string identifiers, they are converted to numeric identifiers (sequentially starting with 0).

Neural networks used:
1. ![MLP-done](https://img.shields.io/badge/Multi--layer_perceptron_with_back_propagation_(MLPClassifier)-done-success?style=flat-square)
    * Layers:
        * Dense
        * ReluLayer
        * LeakyReluLayer
        * SwishLayer
        * Dropout
    * Optimizers:
        * Gradient descent
        * Gradient descent with momentum
        * AdaGrad (Adaptive Gradient Algorithm)
        * AdaDelta (Adaptive Delta)
        * RMSProp (Root Mean Square Propagation)
        * Adam (Adaptive Moment estimation)
        * AdaMax
    * Activation functions:
        * Linear
        * Sigmoid
        * ReLU
        * Tanh
        * SoftMax
        * HardLim
2. ![CPN-done](https://img.shields.io/badge/Kohonen--Grossberg_counter_propagation_network_(CPClassifier)-done-success?style=flat-square)
    * Layers:
        * Kohonen
        * Grossberg
    * Distance functions:
        * Euclidean
        * Octile
        * Manhattan (L1)
        * Chebyshev

<br>

![weights-and-biases-initializers-done](https://img.shields.io/badge/Weights_and_biases_initializers-done-success?style=flat-square)
* Zeros
* Ones
* Full (user value)
* Standard normal
* Xavier Glorot normal
* Xavier Glorot normal normalized
* Xavier Glorot uniform
* Xavier Glorot uniform normalized
* Kaiming He normal
* Kaiming He uniform

<br>

![losses-done](https://img.shields.io/badge/Losses-done-success?style=flat-square)
* MSE (Mean Squared Error)
* SSE (Sum Squared Error)
* MAE (Mean Absolute Error)
* SAE (Sum Absolute Error)
* SMCE (SoftMax Cross Entropy)

## Results

### MLPClassifier Results («Net 1»)
Best choice: test_accuracy

|     Dataset     |                              Dataset info                              | Best Train Accuracy | Best Test Accuracy | Epoch | Time |         Path        |
|:---------------:|:----------------------------------------------------------------------:|:-------------------:|:------------------:|:-----:|:----:|:-------------------:|
| EMNIST Balanced | Train: 112.800<br>Test: 18.800<br>Classes: 47<br>Features: 784 (28x28) |        92.26%       |       86.44%       |   39  |  4m  | rs/EMNIST Balanced/ |
|  EMNIST Letters | Train: 124.800<br>Test: 20.800<br>Classes: 26<br>Features: 784 (28x28) |        96.56%       |       92.70%       |   36  |  4m  |  rs/EMNIST Letters/ |
|  EMNIST Digits  | Train: 240.000<br>Test: 40.000<br>Classes: 10<br>Features: 784 (28x28) |        99.94%       |       99.38%       |   63  |  11m |  rs/EMNIST Digits/  |
|       Iris      |                 Train: 150<br>Classes: 3<br>Features: 4                |         100%        |          -         |  4822 |  3s  |       rs/Iris/      |

### CPClassifier Results («Net 2»)
Best choice: train_accuracy

|     Dataset     |                              Dataset info                              | Best Train Accuracy | Best Test Accuracy | Epoch |  Time  |    Path    |
|:---------------:|:----------------------------------------------------------------------:|:-------------------:|:------------------:|:-----:|:------:|:----------:|
|       Wine      |                 Train: 178<br>Classes: 3<br>Features: 13               |         100%        |          -         |   35  |  0.4s  |  rs/Wine/  |
|       Iris      |                 Train: 150<br>Classes: 3<br>Features: 4                |         100%        |          -         |    1  |  0.04s |  rs/Iris/  |

In the case where the number of neurons of the counter-propagation network coincides with the number of samples, one epoch is enough for an accuracy of 100% (however, such training lasts a very long time; 6 and a half hours per MNIST dataset, size of this network in json ~466.5 MB; in experiments, such a network is not considered).  
<img src='./MNIST.%2060000%20neurons.png' alt='MNIST-60000-neurons-result-100-97.5' width='800px'>

### Experiments

![experiments-in-process](https://img.shields.io/badge/Experiments-in_process-yellowgreen?style=flat-square)

1. **Experiment #1**  
   Study of the effect of image deviation from the reference (at different points of the image) on recognition quality. One of the reference images is taken as the original recognized image and adjusted so that there is no one, two, etc., pixels in the image of the reference. The experiment is conducted for several cases (no pixels in different areas of the image) for all standards. In this case, different neural networks are compared.
   
   <img src='./rs/Experiments/experiment_1.png' alt='experiment-1-result' width='500px' />
2. **Experiment #2**  
   Study of the effect of deviations in the form of noise of one, two, three, etc., pixels in the image on the quality of recognition. One of the references is taken as the original recognizable image and one or more noise pixels are added. The experiment is conducted for different noise locations and for different standards. The experiment also compares different neural networks.
   
   <img src='./rs/Experiments/experiment_2.png' alt='experiment-2-result' width='500px' />
3. **Experiment #3**  
   Study of the effect of noise and deviations in the form of one, two, three, etc., pixels in the image on the quality of recognition. One of the references is taken as the original recognizable image. A multi-pixel noise is introduced into the image of the image being recognized and several pixels in the character image are deleted. The experiment is repeated for different locations of noise and deviations, and for different standards on different types of neural networks. This experiment is a combination of the first two.
   
   <img src='./rs/Experiments/experiment_3.png' alt='experiment-3-result' width='500px' />
4. **Experiment #4**  
   Investigate the effect of having a black row or column in the image (as interference in the image) on recognition quality. One of the references is taken as the original recognizable image. The image is filled with a black row or column. The experiment is repeated for different position of row or column in image and for different standards. During the experiment, various neural networks are compared.

   <img src='./rs/Experiments/experiment_4.png' alt='experiment-4-result' width='500px' />
5. **Experiment #5**  
   Investigate the effect of the presence of a white row or column in the image (as interference in the image) on the quality of recognition. One of the references is taken as the original recognizable image. The image is filled with a white row or column. Experiment is repeated for different position of row or column in image and for different standards on different neural networks.

   <img src='./rs/Experiments/experiment_5.png' alt='experiment-5-result' width='500px' />
6. **Experiment #6**  
   Investigation of the effect of the number of neurons in layers on recognition quality. The number of neurons in the layer varies from two to some sufficient number. The experiment is repeated on various neural networks.

   <img src='./rs/Experiments/experiment_6.png' alt='experiment-6-result' width='500px' />
7. **Experiment #7**  
   Investigation of the effect of number of neurons in layers and number of benchmarks on network learning rate. During the experiment, the number of neurons in the layer varies from two to a certain sufficient number, and the number of standards (within the selected range) also varies. In each case, the number of iterations, and the training time are recorded. The experiment is repeated on various neural networks.

   <img src='./rs/Experiments/experiment_7.png' alt='experiment-7-result' width='500px' />
8. **Experiment #8**  
   Investigate the impact of input pattern drawing on recognition quality. One of the references is taken as the original recognizable image. The standard is modified (bold/slanted/underlined). The experiment is repeated on various neural networks for various modifications.

   <img src='./rs/Experiments/experiment_8.png' alt='experiment-8-result' width='500px' />

Note: When using color in an image, experiments #4 and #5 should be performed for the background color and image color. In experiment #4, the image color column or row is taken instead of the black row or column. Experiment #5 takes a background color row or column instead of a white row or column.
