# Convolution

There are different types of Convolution in CNN such as

* 2D Conv, 3D Conv
* 1x1 Conv, BottleNeck
* Spatially Separable
* Depthwise Separable
* Grouped Convolution
* Shuffled Grouped

Read the following blog for more detailed explanations on types of convolution

{% embed url="https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215" %}

## Convolution: single channel

It is the element-wise multiplication and addition with window sliding.

Read the followings for more detailed information

* [Intuitive Understanding of Convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
* [Guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)

![](<../../.gitbook/assets/image (181).png>)

![Convolution for a single channel. Image is adopted from medium@IrhumShafkat](<../../.gitbook/assets/image (188).png>)

Using 3x3 kernel. from 5x5=25 input features --> 3x3=9 output.

Common techniques in convolution

* Padding: pad the edges with '0','1' or other values
* With padding: WxHxC --> WxHxC
* Without padding: WxHxC -> (W-w+1)x(H-h+1)xC

![Two-dimensional cross-correlation with padding](https://d2l.ai/\_images/conv-pad.svg)

image from[ here](https://d2l.ai/chapter\_convolutional-neural-networks/padding-and-strides.html)

![Same padding\[1\]](<../../.gitbook/assets/image (186).png>)

* Striding: skip some of the slide locations
* ⌊(nh−kh+ph+sh)/sh⌋×⌊(nw−kw+pw+sw)/sw⌋.
* With padding: WxHxC  (W+S-1)/S x (H+S-1)/S x C Without padding: WxHxC  (W-w+S)/S x (H-h+S)/S xC

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](<../../.gitbook/assets/image (234).png>)

![A stride 2 convolution w/o padding \[1\]](<../../.gitbook/assets/image (179).png>)

### Filter vs Kernel

For 2D convolution, kernel and filter are the same

For 3D convolution, a filter is the collection of the stacked kernels

![Image fDifference between “layer” (“filter”) and “channel” (“kernel”)or post](https://miro.medium.com/max/1524/1\*NCDUVdTGF3hu6zrzdYFqJA.png)

## 2D Convolution: multiple channel

The filter has the same depth (channel) as the input matrix.

The output is 2D matrix.

Example: Input is 5x5x3 matrix. Filter is 3x3x3 matrix.

![The first step of 2D convolution for multi-channels: each of the kernels in the filter are applied to three channels in the input layer, separately. The image is adopted from this link.](<../../.gitbook/assets/image (191).png>)

Then, three channels are summed by element-wise addition to form one single channel (3x3x1)

![](<../../.gitbook/assets/image (185).png>)

![Another way to think about 2D convolution: thinking of the process as sliding a 3D filter matrix through the input layer. Notice that the input layer and the filter have the same depth (channel number = kernel number). The 3D filter moves only in 2-direction, height & width of the image (That’s why such operation is called as 2D convolution although a 3D filter is used to process 3D volumetric data). The output is a one-layer matrix](<../../.gitbook/assets/image (177).png>)

![](<../../.gitbook/assets/image (229).png>)

## 3D Convolution

A general form of convolution but the filter _kernel size < channel size. The filter moves in three directions: height, width, channel_

The output is 3D matrix.

![In 3D convolution, a 3D filter can move in all 3-direction (height, width, channel of the image). At each position, the element-wise multiplication and addition provide one number. Since the filter slides through a 3D space, the output numbers are arranged in a 3D space as well. The output is then a 3D data.](<../../.gitbook/assets/image (176).png>)

## 1D Convolution

Input: HxWxD. Filtering with 1x1xD produces the Output' HxWx1

![1 x 1 convolution, where the filter size is 1 x 1 x D.](<../../.gitbook/assets/image (182).png>)

Initially, proposed in 'Network-in-Network (2013)' . Widely used after introduced in 'Inception (2014)'

* Dimensionality reduction for efficient computations
  * HxWxD --> HxWx1
*   Efficient low dimensional embedding, or feature pooling

    \*
* Applying nonlinearity again after convolution
  * after 1x1 conv, non-linear activation(ReLU etc) can be added

## Cost of Convolution

Calculation cost for a convolution depends on:

1. Input size: i\*i\*D
2. Kernel Size: k\*k\*D
3. Stride: s
4. Padding: p

The output image ( o\*o\*1 ) then becomes

![For output o\*o.](<../../.gitbook/assets/image (183).png>)

The required operations are

* o\*o repetition of { (k\*k) multiplications and (k\*k-1) additions}

In terms of multiplications

* For input of size H x W x D, 2D convolution (stride=1, padding=0) with Nc kernels of size h x h x D, where h is even
* Total multiplications: Nc x h x h x D x (H-h+1) x (W-h+1)

## Separable Convolution

Used in [MobileNet(2017)](https://arxiv.org/abs/1704.04861), [Xception(2016)](https://arxiv.org/abs/1610.02357) for efficient processing.

### Spatially Separable Convolution

Not used much in deep learning. It is decomposing a convolution into two separate operations

Example: A Sobel kernel can be divided into a 3 x 1 and a 1 x 3 kernel.

![](<../../.gitbook/assets/image (180).png>)

![Spatially separable convolution with 1 channel.](<../../.gitbook/assets/image (184).png>)

### **Depthwise Separable Convolution**

Commonly used in Deep Learning such as MobileNet and Xception. It is two steps of (1) Depthwise convolution (2) 1x1 convolution

For example: Input 7\*7\*3 --> 128 of 3\*3\*3 filters --> 5\*5\*128 output

* Step1: Depthwise convolution
  * Each layer of a single filter is separated into kernels. (e.g. 3 of 3x3x1)
  * Each kernel convoles with 1 channel(only) layer input : (5\*5\*1) for each kernel
  * Then, stack the maps to get the final output e.g. (5\*5\*3)

![Depthwise separable convolution — first step: Instead of using a single filter of size 3 x 3 x 3 in 2D convolution, we used 3 kernels, separately. Each filter has size 3 x 3 x 1. Each kernel convolves with 1 channel of the input layer (1 channel only, not all channels!). Each of such convolution provides a map of size 5 x 5 x 1. We then stack these maps together to create a 5 x 5 x 3 image. After this, we have the output with size 5 x 5 x 3.](<../../.gitbook/assets/image (178).png>)

* Step 2: 1\*1 Convolution
  * Apply 1\*1 convolution with 1\*1\*3 kernels to get 5\*5\*1 map.
  * Apply 128 of 1x1 convolutions to get 5\*5\*128 map

![](<../../.gitbook/assets/image (193).png>)

![Depthwise separable convolution — second step: apply multiple 1 x 1 convolutions to modify depth.](<../../.gitbook/assets/image (175).png>)

* Standard 2D convolution vs Depthwise Convolution

![Standard 2D convolution](<../../.gitbook/assets/image (190).png>)

![The overall process of depthwise separable convolution.](<../../.gitbook/assets/image (189).png>)

Calculation comparison

* Standard: 128\*(3\*3\*3)\*(5\*5) multiplications
  * 128\*(3\*3\*3)\*(5\*5) =86,400
  * Nc x h x h x D x (H-h+1) x (W-h+1)
* Separable: 3\*(3\*3\*1)\*(5\*5)+128\*(1\*1\*3)\*(5\*5) multiplications
  * \=675+9600=10,275 (12%)
  * D x h x h x 1 x (H-h+1) x (W-h+1) + Nc x 1 x 1 x D x (H-h+1) x (W-h+1) = (h x h + Nc) x D x (H-h+1) x (W-h+1)
* The ratio of multiplication is

![](<../../.gitbook/assets/image (187).png>)

* If Nc>>h, then it is approx. 1/(h^2). for 5x5 filters, 25 times more multiplications

## Grouped Convolution

Introduced in AlexNet(2012), to do parallel convolutions. The filters are separated into different groups. Each group is responsible for standard 2D conv with certain depth. Then the each outputs are concetenated in depth-wise

![Grouped convolution with 2 filter groups](<../../.gitbook/assets/image (192).png>)

* Model-Parallelization for efficient training
  * each group can be handled by different GPUs
  * Better than data parallelization using batches
* Efficient Computation
  * Standard: h x w x Din x Dout
  * Grouped: 2\*(h x w x Din/2 x Dout/2)= (1/2)\*(h x w x Din x Dout)

## Shuffled Grouped Convolution

Introduced by[ ShuffleNet(2017)](https://arxiv.org/abs/1707.01083) for computation -efficient convolution. The idea is mixing up the information from different filter groups to connect the information flow between the channel groups.

Read [this blog ](https://towardsdatascience.com/review-shufflenet-v1-light-weight-model-image-classification-5b253dfe982f)for the paper explanations

![Channel shuffle.](<../../.gitbook/assets/image (199).png>)

## Pointwise grouped convolution

The group operation is performed on the 3x3 spatial convolution, but not on 1 x 1 convolution. The ShuffleNet suggested 1x1 convolution on Group convolution

> Group convolution of 1x1 filters instead of NxN filters (N>1).
