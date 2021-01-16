# ShuffleNet

## ShuffleNet

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

{% embed url="https://towardsdatascience.com/review-shufflenet-v1-light-weight-model-image-classification-5b253dfe982f" caption="" %}

ShuffleNet\(CVPR 2018\) pursues **the best accuracy in very limited computational budgets at tens or hundreds of MFLOPs**, focusing on common mobile platforms such as **drones**, **robots**, and **smartphones**. By shuffling the channels, ShuffleNet outperforms [MobileNetV1](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69). In ARM device, ShuffleNet achieves 13× actual speedup over [AlexNet](https://medium.com/coinmonks/paper-review-of-alexnet-caffenet-winner-in-ilsvrc-2012-image-classification-b93598314160) while maintaining comparable accuracy.

Need to understand the concept of 'Grouped Convolution'

"If we allow group convolution to obtain input data from different groups \(as shown in Fig 1 \(b\)\), the input and output channels will be fully related.This can be efficiently and elegantly implemented by a channel shuffle operation \(Fig 1 \(c\)\): suppose a convolutional layer with g groups whose output has g × n channels; we first reshape the output channel dimension into \(g, n\), transposing and then flattening it back as the input of next layer. Note that the operation still takes effect even if the two convolutions have different numbers of groups. Moreover, channel shuffle is also differentiable, which means it can be embedded into network structures for end-to-end training."

![\(a\) Two Stacked Group Convolutions \(GConv1 &amp; GConv2\), \(b\) Shuffle the channels before convolution, \(c\) Equivalent implementation of \(b\)](../../.gitbook/assets/image%20%28197%29.png)

### **ShuffleNet Unit**

![\(a\) bottleneck unit with depthwise convolution \(DWConv\)of ResNet,InceptionV1, MobileNetv2, \(b\) ShuffleNet unit with pointwise group convolution \(GConv\) and channel shuffle, \(c\) ShuffleNet unit with stride = 2.](../../.gitbook/assets/image%20%28196%29.png)

* **\(a\) Bottleneck Unit**: This is a standard residual bottleneck unit, but with depthwise convolution used. It can be also treated as a bottleneck type of depthwise separable convolution used in [MobileNetV2](https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c).

> Even though depthwise convolution usually has very low theoretical complexity, we find it difficult to efficiently implement on lowpower mobile devices, which may result from a worse computation/memory access ratio compared with other dense operations.
>
> In ShuffleNet units, we intentionally use depthwise convolution only on bottleneck in order to prevent overhead as much as possible.

* **\(b\) ShuffleNet Unit**: The first and second 1×1 convolutions are replaced by group convolutions. A channel shuffle is applied after the first 1×1 convolution.
* **\(c\) ShuffleNet Unit with Stride=2:** When stride is applied, a 3×3 average pooling on the shortcut path is added.  \(Stride=2 reduces the image size by half\). Also, the element-wise addition is replaced with channel **concatenation**, which makes it easy to enlarge channel dimension with little extra computation cost.

Given the input _c_×_h_×_w_, and bottleneck channels _m_, [ResNet](https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8) unit requires _hw_\(2_cm_+9_m_²\) FLOPs and [ResNeXt](https://towardsdatascience.com/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac) requires _hw_\(2_cm_+9_m_²/_g_\) FLOPs, while **ShuffleNet only requires** _**hw**_**\(2**_**cm**_**/**_**g**_**+9**_**m**_**\) FLOPs** where g is the number of group convolutions. Given a computational budget, **ShuffleNet can use wider feature maps**

### **Results**

![](../../.gitbook/assets/image%20%28198%29.png)

* With _g_ = 1, i.e. no pointwise group convolution.
* Models with group convolutions \(_g_ &gt; 1\) consistently perform better than the counterparts without pointwise group convolutions \(_g_ = 1\).

![](../../.gitbook/assets/image%20%28195%29.png)

* With similar accuracy, ShuffleNet is much more efficient than [VGGNet](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11), [GoogLeNet](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7), [AlexNet](https://medium.com/coinmonks/paper-review-of-alexnet-caffenet-winner-in-ilsvrc-2012-image-classification-b93598314160) and [SqueezeNet](https://towardsdatascience.com/review-squeezenet-image-classification-e7414825581a).

![](../../.gitbook/assets/image%20%28200%29.png)

* Compared with [AlexNet](https://medium.com/coinmonks/paper-review-of-alexnet-caffenet-winner-in-ilsvrc-2012-image-classification-b93598314160), ShuffleNet 0.5× model still achieves ~13× actual speedup under comparable classification accuracy \(the theoretical speedup is 18×\).

