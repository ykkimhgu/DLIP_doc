# CNN Overview

## Introduction

[Read this: Convolution Neural Network by cs231n](https://cs231n.github.io/convolutional-networks/)

* [image from here](https://developersbreach.com/convolution-neural-network-deep-learning/)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28225%29.png)

Example of simple CNN architecture

![VGG-19](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28232%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28224%29.png)

LeNet-5 \(1998\): [image ](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)[by Raimi Karim](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28230%29.png)

AlexNet\(2012\)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28226%29.png)

VGG-16\(2014\)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%28233%29.png)

## Convolution

## Activation Function

## Pooling

A problem with the output feature maps is that they are sensitive to the location of the features in the input. One approach to address this sensitivity is to down sample the feature maps. This has the effect of making the resulting down sampled feature maps more robust to changes in the position of the feature in the image, referred to by the technical phrase “_local translation invariance_.”

[Read this: Introduction to Pooling Layers by Machine Learning Mastery](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

## Fully Connected Layer

## LeNet-5

[Read here](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)

LeNet-5 is one of the simplest architectures. It has 2 convolutional and 3 fully-connected layers \(hence “5” — it is very common for the names of neural networks to be derived from the number of _convolutional_ and _fully connected_ layers that they have\). The average-pooling layer as we know it now was called a _sub-sampling layer_ and it had trainable weights \(which isn’t the current practice of designing CNNs nowadays\). This architecture has about **60,000 parameters**.

**⭐️What’s novel?**

This architecture has become the standard ‘template’: stacking convolutions with activation function, and pooling layers, and ending the network with one or more fully-connected layers.

**📝Publication**

* Paper: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/index.html#lecun-98)
* Authors: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
* Published in: _\*\*_Proceedings of the IEEE \(1998\)

