# Activation Function

## Activation Functions

Activation function or transfer function is used to determine the output of a neuron or node. It is a mathematical gate in between the input feeding the current neuron and its output going to the next layer.

![](../../.gitbook/assets/image%20%2878%29.png)

In deep learning, we commonly use non-linear Activation Function include

* Sigmoid:  Output limit to \[0 1\]. But gives gradient vanishing problem, not used anymore 
* ReLU\(rectified linear unit\): most commonly used in CNN \(hidden layers\) 
* Others: Tanh, Leaky ReLU, Maxout...

![Image from MIT Deeplearning Lecture](../../.gitbook/assets/image%20%2883%29.png)

![Cheat sheet of commonly used Activation Function ](../../.gitbook/assets/image%20%2877%29.png)

## Output Activation Function

These functions are transformations we apply to vectors coming out from CNNs \( s \) before the loss computation. \[ [reference ](https://gombru.github.io/2018/05/23/cross_entropy_loss/)\]

**Sigmoid**

It squashes a vector in the range \(0, 1\). It is applied independently to each element of ss sisi. It’s also called **logistic function**.

![](https://gombru.github.io/assets/cross_entropy_loss/sigmoid.png)[![](https://latex.codecogs.com/gif.latex?f%28s_{i}%29&space;=&space;\frac{1}{1&space;+&space;e^{-s_{i}}})](https://www.codecogs.com/eqnedit.php?latex=f%28s_{i}%29&space;=&space;\frac{1}{1&space;+&space;e^{-s_{i}}})

**Softmax**

Softmax it’s a function, not a loss. It squashes a vector in the range \(0, 1\) and all the resulting elements add up to 1. It is applied to the output scores ss. As elements represent a class, they can be interpreted as class probabilities.  
The Softmax function cannot be applied independently to each sisi, since it depends on all elements of ss. For a given class sisi, the Softmax function can be computed as:[![](https://latex.codecogs.com/gif.latex?f%28s%29_{i}&space;=&space;\frac{e^{s_{i}}}{\sum_{j}^{C}&space;e^{s_{j}}})](https://www.codecogs.com/eqnedit.php?latex=f%28s%29_{i}&space;=&space;\frac{e^{s_{i}}}{\sum_{j}^{C}&space;e^{s_{j}}})

Where sjsj are the scores inferred by the net for each class in CC. Note that the Softmax activation for a class sisi depends on all the scores in ss.

### 

