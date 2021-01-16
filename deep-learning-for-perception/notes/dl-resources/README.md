# DL Techniques

## Techniques

### Hyperparameter Tuning

{% embed url="https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/" caption="" %}

### Methods of Efficient Inference

* **Pruning**
* Weight Sharing
* Quantization
* Low-Rank Approximation
* Binary / Ternary Net
* Winograd Transformation

### Pruning Deep Network

![\[Lecun et al. NIPS&#x2019;89\] \[Han et al. NIPS&#x2019;15\]](https://github.com/ykkimhgu/DLIP_doc/tree/b285a6df5d496b0b481f8f4bba36710a4dfd1914/deep-learning-for-perception/images/image%20%2810%29.png)

#### Weight pruning

* Set individual weights in the weight matrix to zero. This corresponds to deleting connections as in the figure above.

**Unit/Neuron pruning**

* Set entire columns to zero in the weight matrix to zero, in effect deleting the corresponding output neuron

![Image for post](https://miro.medium.com/max/791/1*pQeZG3Dp91OZ8WWV-VJ9Mw.png)

{% embed url="https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505" caption="" %}

### 1x1D CNN

![Image for Source: Inception v3 paper, image free to share.post](https://miro.medium.com/max/2521/1*whVu6bmbDi9HtPIjSYPoWg.png)

* Read: Inception paper   _\_\[_“Going deeper with convolutions”\_\]\([https://arxiv.org/pdf/1409.4842.pdf](https://arxiv.org/pdf/1409.4842.pdf)\)
* 1×1 convolutions are an essential part of the Inception module.
* A 1×1 convolution returns an output image with the same dimensions as the input image.
* Colored images have three dimensions, or channels. 1×1 convolutions compress these channels at little cost, leaving a two-dimensional image to perform expensive 3×3 and 5×5 convolutions on.
* Convolutional layers learn many filters to identify attributes of images. 1×1 convolutions can be placed as ‘bottlenecks’ to help compress a high number of filters into just the amount of information that is necessary for a classification.

### Code Template - PyTorch

* Template 1:  the template is [here](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template/tree/master)

{% embed url="https://towardsdatascience.com/pytorch-deep-learning-template-6e638fc2fe64" caption="" %}

