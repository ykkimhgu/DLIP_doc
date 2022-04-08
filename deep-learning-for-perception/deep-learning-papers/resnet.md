# ResNet

## Background

Since AlexNet, the state-of-the-art CNN architecture is going deeper and deeper. While AlexNet had only 5 convolutional layers, the VGG network \[3] and GoogleNet (also codenamed Inception\_v1) \[4] had 19 and 22 layers respectively.

Deep networks are hard to train because of the notorious vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitively small.

![](<../../.gitbook/assets/image (278).png>)

ResNets solve is the famous known vanishing gradient. With ResNets, the **gradients can flow directly through the skip connections backwards from later layers to initial filters**.

The core idea of ResNet is introducing a so-called “identity shortcut connection” that skips one or more layers

## Residual Block

The stacked layers fit a residual mapping is easier than letting them directly fit the desired underlaying mapping. With ResNets, the **gradients can flow directly through the skip connections backwards from later layers to initial filters**.

![](<../../.gitbook/assets/image (272).png>)

![](<../../.gitbook/assets/image (273).png>)

### Blocks

![Layer 1, block 1](<../../.gitbook/assets/image (274).png>)

![Layer 1](<../../.gitbook/assets/image (279).png>)

Down sampling of the volume though the network is achieved by increasing the stride instead of a pooling operation.

Since the volume got modified we need to apply one of our down sampling strategies. The 1x1 convolution approach is shown.

![Layer 2, Block 1](https://miro.medium.com/max/1170/1\*Xd-OIT9GRwLaM3F5jdbfzQ.png)

![Layer 2](<../../.gitbook/assets/image (277).png>)

### Variant of Residual Block

There are several new architectures based on ResNet over years

![variants of residual blocks](<../../.gitbook/assets/image (275).png>)

#### Pre-activation variant of residual block \[7]

K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.

The gradients can flow through the shortcut connections to any other earlier layer unimpededly.

Performance: 1202-layer ResNet <110-layer Full Pre-activation ResNet

### ResNeXt <a href="#5ce9" id="5ce9"></a>

S. Xie, R. Girshick, P. Dollar, Z. Tu and K. He. Aggregated Residual Transformations for Deep Neural Networks. arXiv preprint arXiv:1611.05431v1,2016.

Xie et al. \[8] proposed a variant of ResNet that is codenamed ResNeXt with the following building block:

![Imaleft: a building block of \[2\], right: a building block of ResNeXt with cardinality = 32ge for post](https://miro.medium.com/max/1044/1\*7JzJ1RGh1Y4VoG1M4dseTw.png)

This may look familiar to you as it is very similar to the Inception module of \[4], they both follow the split-transform-merge paradigm, except in this variant, the outputs of different paths are merged by adding them together, while in \[4] they are depth-concatenated. Another difference is that in \[4], each path is different (1x1, 3x3 and 5x5 convolution) from each other, while in this architecture, all paths share the same topology.

This novel building block has three equivalent form as follows:

![Image for pthree equivalent formost](https://miro.medium.com/max/2097/1\*tZb5Ol72dMw\_SBB-gZ1wjA.png)

In practice, the “split-transform-merge” is usually done by pointwise grouped convolutional layer, which divides its input into groups of feature maps and perform normal convolution respectively, their outputs are depth-concatenated and then fed to a 1x1 convolutional layer.

### Densely Connected CNN <a href="#7d2a" id="7d2a"></a>

G. Huang, Z. Liu, K. Q. Weinberger and L. Maaten. Densely Connected Convolutional Networks. arXiv:1608.06993v3,2016

Huang et al. \[9] proposed a novel architecture called DenseNet that further exploits the effects of shortcut connections — it connects all layers directly with each other. In this novel architecture, the input of each layer consists of the feature maps of all earlier layer, and its output is passed to each subsequent layer. The feature maps are aggregated with depth-concatenation.

![Image for post](https://miro.medium.com/max/1056/1\*WpX\_8eCeTsEcCs8vdXtUCw.png)

Other than tackling the vanishing gradients problem, the authors of \[8] argue that this architecture also encourages feature reuse, making the network highly parameter-efficient. One simple interpretation of this is that, in \[2]\[7], the output of the identity mapping was added to the next block, which might impede information flow if the feature maps of two layers have very different distributions. Therefore, concatenating feature maps can preserve them all and increase the variance of the outputs, encouraging feature reuse

![Image for post](https://miro.medium.com/max/2130/1\*gdFcbkMGn8aT8\_iP1OpfmA.png)
