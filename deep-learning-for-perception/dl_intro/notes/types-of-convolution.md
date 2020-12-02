# Types of Convolution

There are different types of Convolution in CNN such as

* 2D Conv, 3D Conv
* 1x1 Conv, BottleNeck
* Spatially Separable
* Depthwise Separable
* Grouped Convolution
* Shuffled Grouped

Read the following for more detailed explanations

{% embed url="https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215" %}



### Convolution: single channel

It is the element-wise multiplication and addition.

Read the followings for more

* [Intuitive Understanding of Convolution](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1) 
* [Guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)



![](../../../.gitbook/assets/image%20%28176%29.png)

![Convolution for a single channel. Image is adopted from medium@IrhumShafkat](../../../.gitbook/assets/image%20%28179%29.png)

Using 3x3 kernel.  from 5x5=25 input features --&gt; 3x3=9 output.

Common techniques in convolution

* Padding: pad the edges with '0','1' or other values 
* 
![Same padding\[1\]](../../../.gitbook/assets/image%20%28178%29.png)

* Striding: skip some of the slide locations
* 
![A stride 2 convolution\[1\]](../../../.gitbook/assets/image%20%28175%29.png)

#### Filter vs Kernel

For 2D convolution, kernel and filter are the same

For 3D convolution, a filter is the collection of the stacked kernels

![Image fDifference between &#x201C;layer&#x201D; \(&#x201C;filter&#x201D;\) and &#x201C;channel&#x201D; \(&#x201C;kernel&#x201D;\)or post](https://miro.medium.com/max/1524/1*NCDUVdTGF3hu6zrzdYFqJA.png)

### 3D Convolution: multiple channel

Example: Input is 5x5x3 matrix.   Filter is 3x3x3 matrix.

![The first step of 2D convolution for multi-channels: each of the kernels in the filter are applied to three channels in the input layer, separately. The image is adopted from this link.](../../../.gitbook/assets/image%20%28180%29.png)

Then, three channels are summed by element-wise addition to form one single channel \(3x3x1\)

![](../../../.gitbook/assets/image%20%28177%29.png)

