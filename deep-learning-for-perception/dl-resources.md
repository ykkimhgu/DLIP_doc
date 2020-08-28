# DL resources



## Techniques

### Hyperparameter Tuning

{% embed url="https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/" %}

### Batch Normalization

{% embed url="https://towardsdatascience.com/batch-normalization-the-greatest-breakthrough-in-deep-learning-77e64909d81d" %}

Batch normalization greatly reduces the variation in the loss landscape, gradient productiveness, and β-smoothness, making the task of navigating the terrain to find the global error minima much easier.

* _More freedom in setting the initial learning rate_. Large initial learning rates will not result in missing out on the minimum during optimization, and can lead to quicker convergence.
* _Accelerate the learning rate decay_. 
* _Remove dropout_. One can get away with not using dropout layers when using batch normalization, since dropout can provide damage and/or slow down the training process. Batch normalization introduces an additional form of resistance to overfitting.
* _Reduce L2 weight regularization._ 
* _Solving the vanishing gradient problem_.
* _Solving the exploding gradient problem._ 

### Effect of Regularization

Regularization refers to the practice of constraining /regularizing the model from learning complex concepts, thereby reducing the risk of overfitting.

#### Regularization Methods

* Dropout Regularization
* L2 Regularization
* L1 Regularization

#### Effects of Methods

* Dropout has the best performance among other regularizers. Dropout has both weight regularization effect and induces sparsity.
* L1 Regularization has a tendency to produce sparse weights whereas L2 Regularization produces small weights
* Regularization hyper parameters for CONV and FC layers should tuned separately.

{% embed url="https://medium.com/deep-learning-experiments/science-behind-regularization-in-neural-net-training-9a3e0529ab80" %}



### Effect of Batch size

We use mini-batches because it tends to converge more quickly,  allow us to **parallelize computations** 

#### What is Batch Size

Neural networks are trained to minimize a loss function of the following form:

![Image for post](https://miro.medium.com/max/183/1*XA9OkVLg3q7AfL_zQxN_ZA.gif)

Figure 1: Loss function. Adapted from Keskar et al \[1\].

Stochastic gradient descent computes the gradient on _a **subset** of the training data, B\_k, as opposed to the entire training dataset_.

![Image for pFigure 2: Stochastic gradient descent update equation. Adapted from Keskar et al \[1\].ost](https://miro.medium.com/max/339/1*yOhYIBLlKh0OMS1PVBVydA.png)

**Usually small Batch size perform better**

![Figure 5: Training and validation loss curves for different batch sizes](https://miro.medium.com/max/640/1*z5UEgD9eBRWa03uQLj9haA.png)

![Figure 23: Training and validation loss for different batch sizes, with adjusted learning rates for post](https://miro.medium.com/max/524/1*_5vEKoUO-cxnwRReVPakQA.png)

Training with small batch sizes tends to converge to **flat minimizers** that vary only slightly within a small neighborhood of the minimizer, whereas large batch sizes converge to **sharp minimizers**, which vary sharply \[1\]

* Small batch sizes perform best with smaller learning rates, while large batch sizes do best on larger learning rates. 
* Linear scaling rule: when the minibatch size is multiplied by k, multiply the learning rate by k.
*  When the right learning rate is chosen, larger batch sizes can train faster, especially when parallelized.

{% embed url="https://medium.com/deep-learning-experiments/effect-of-batch-size-on-neural-net-training-c5ae8516e57" %}



### Methods of Efficient Inference

* **Pruning**
* Weight Sharing
* Quantization
* Low-Rank Approximation
* Binary / Ternary Net
* Winograd Transformation

### Pruning Deep Network

![\[Lecun et al. NIPS&#x2019;89\] \[Han et al. NIPS&#x2019;15\]](../.gitbook/assets/image%20%2810%29.png)

#### Weight pruning

* Set individual weights in the weight matrix to zero. This corresponds to deleting connections as in the figure above.

**Unit/Neuron pruning**

* Set entire columns to zero in the weight matrix to zero, in effect deleting the corresponding output neuron



![Image for post](https://miro.medium.com/max/791/1*pQeZG3Dp91OZ8WWV-VJ9Mw.png)

{% embed url="https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505" %}



### 1x1D CNN

![Image for Source: Inception v3 paper, image free to share.post](https://miro.medium.com/max/2521/1*whVu6bmbDi9HtPIjSYPoWg.png)

* Read: Inception paper   __[_“Going deeper with convolutions”_](https://arxiv.org/pdf/1409.4842.pdf)
* 1×1 convolutions are an essential part of the Inception module.
* A 1×1 convolution returns an output image with the same dimensions as the input image.
* Colored images have three dimensions, or channels. 1×1 convolutions compress these channels at little cost, leaving a two-dimensional image to perform expensive 3×3 and 5×5 convolutions on.
* Convolutional layers learn many filters to identify attributes of images. 1×1 convolutions can be placed as ‘bottlenecks’ to help compress a high number of filters into just the amount of information that is necessary for a classification.

### Code Template - PyTorch

* Template 1:  the template is [here](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template/tree/master)

{% embed url="https://towardsdatascience.com/pytorch-deep-learning-template-6e638fc2fe64" %}



