# VGG

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556), Simonyan, Karen, and Andrew Zisserman. 2015 ICLR & preprint arXiv (2014)

![VGG architecture image from https://neurohive.io/en/popular-networks/vgg16/](<../../images/image (247).png>)

![](<../../images/image (254).png>)

VGGNet is invented by VGG ([Visual Geometry Group](http://www.robots.ox.ac.uk/\~vgg/)) from University of Oxford. Though VGGNet is the **1st runner-up**, of **ILSVRC 2014 in the classification task.** Accuracy error less than 10%

> GoogLeNet is the winner of ILSVLC 2014. **VGGNet beats the GoogLeNet and won the localization task in ILSVRC 2014.**
>
> Improved from AlexNet(-8) 16.4% to VGG-16 7.3%)

![](<../../images/image (249).png>)

A very important paper on CNN. It uses only 3x3 CONV and many networks are based on VGG architecture.

## **The Use of 3×3 Filters**

* Instead of large-size filters (such as 11×11, 7×7) as in AlexNet
* Repeats 3 layers of 3x3 CONV instead of 1 time of 7x7. They both cover 7x7 receptive field
* VGG has fewer parameters, more non-linearity.

![](<../../.gitbook/assets/image (253).png>)

![a) Feature Map from 7x7 CONV has 7x7 receptive field. b) Feature Map from 3 layers of 3x3 also has 7x7 receptive field. Image from https://medium.com/@msmapark2/](<../../images/image (245).png>)

## **Ablation Study: VGG-16, 19**

1. VGG-16 (Conv1) obtains 9.4% error rate, which means the additional three 1×1 conv layers help the classification accuracy. 1×1 conv actually helps to increase non-linearlity of the decision function. Without changing the dimensions of input and output, 1×1 conv is doing the projection mapping in the same high dimensionality.
2. VGG-16 obtains 8.8% error rate which means the deep learning network is still improving by adding number of layers.
3. VGG-19 obtains 9.0% error rate which means the deep learning network is NOT improving by adding number of layers. Thus, authors stop adding layers.

![](<../../images/image (248).png>)

## **Multi-Scale Training and Testing**

Rescale from 224 to 256\~512px. Then crop to 224px which contains the object fully or partially. Has the effect of data augmentation with scaling and translation, which helps to reduce overfitting.

![](<../../.gitbook/assets/image (250).png>)

![](<../../images/image (246).png>)

## Training

* Dataset: ImageNet of 256x256x3
* Input: 224×224×3 image with data augmentation from multi-scaling
* Batch normalization
* Multinomial logistic regression loss
* Mini-batch GD with momentum
  * Batch size: 256
  * Momentum v: 0.9
  * Weight Decay: 0.0005
  * Learning rate : 0.01 decreased factor of 10
