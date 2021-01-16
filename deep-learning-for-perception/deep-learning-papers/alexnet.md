# AlexNet

Krizhevsky, Sutskever, Hinton, “Imagenet classification with deep convolutional neural networks”. NIPS 2012

AlexNet is the winner of **ImageNet Large Scale Visual Recognition Challenge \(ILSVRC\) 2012**.

![](../../.gitbook/assets/image%20%28242%29.png)

Prior to ILSVRC 2012, competitors mostly used feature engineering techniques combined with a classifier \(i.e SVM\).

**AlexNet** marked a breakthrough in deep learning where a CNN was used to reduce the error rate in ILSVRC 2012 substantially and achieve the first place of the ILSVRC competition.

![](../../.gitbook/assets/image%20%28237%29.png)

**The highlights of this paper**:

1. Breakthrough in Deep Learning using CNN for image classification.
2. **ReLU \(Rectified Linear Unit\)**
3. **Multiple GPUs**
4. **Local Response Normalization**
5. **Overlapping Pooling**
6. **Data Augmentation**
7. **Dropout**
8. **Other Details of Learning Parameters**
9. **Results**

## **Architecture**

![Image for post](https://lh3.googleusercontent.com/zUOYphIXXXiCEk4Lioq6WLBom_LGLt0gSrOuV_MVSPdkrDLlDPIx0zbUHj4HBGXIm4fqoq8eh93Mg8CS-Her__rC6SjSUKKChEWw89tAuKV7OWxQ5pc_ZVioTGeyQgRif2_E7ka_)

> Note that Group convolution is applied here. Thus, from 2nd layer, number of kernels are divided by 2 for each group. e.g. 256 of 5x5x48 --&gt; \(128 of 5x5x48\) \*2

AlexNet contains **eight layers**:

Input: 224×224×3 input images

**1th: Convolutional Layer: 96 kernels of size 11×11×3  
\(stride: 4, pad: 0\)**  
55×55×96 feature maps  
Then **3×3 Overlapping Max Pooling \(stride: 2\)**  
27×27×96 feature maps  
Then **Local Response Normalization**  
27×27×96 feature maps

**2nd: Convolutional Layer: 256 kernels of size 5×5×48  
\(stride: 1, pad: 2\)**  
27×27×256 feature maps  
**\*\*Then** 3×3 Overlapping Max Pooling \(stride: 2\)  
**13×13×256 feature maps  
Then** Local Response Normalization\*\*  
13×13×256 feature maps

**3rd: Convolutional Layer: 384 kernels of size 3×3×128  
\(stride: 1, pad: 1\)**  
13×13×384 feature maps

**4th: Convolutional Layer: 384 kernels of size 3×3×192  
\(stride: 1, pad: 1\)**  
13×13×384 feature maps

**5th: Convolutional Layer: 256 kernels of size 3×3×192  
\(stride: 1, pad: 1\)**  
13×13×256 feature maps  
Then **3×3 Overlapping Max Pooling \(stride: 2\)**  
6×6×256 feature maps

**6th: Fully Connected \(Dense\) Layer of**  
4096 neurons

**7th: Fully Connected \(Dense\) Layer of**  
4096 neurons

**8th: Fully Connected \(Dense\) Layer of**  
Output: 1000 neurons \(since there are 1000 classes\)  
**Softmax** is used for calculating the loss.

In total, there are 60 million parameters need to be trained !!!

## **Learning**

* Train with Stochastic Gradient Descent with:
  * Batch size: **128**
  * Momentum: **0.9**
  * Weight Decay: **0.0005**
  * Initialize the weights in each layer from a zero-mean Gaussian distribution with std **0.01**.
  * Bias: Initialize **1** for 2nd, 4th, 5th conv layers and fully-connected layers. Initialize **0** for remaining layers.
  * Learning rate: **0.01**. Equal learning rate for all layers and diving by 10 when validation error stopped improving.
* We trained our models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005.
* We found that this small amount of weight decay was important for the model to learn. 
* We trained the network for roughly 90 cycles through the training set of 1.2 million images, which took five to six days on two NVIDIA GTX 580 3GB GPUs.
* Batch size: 128 Momentum v: 0.9 Weight Decay: 0.0005 Learning rate ϵ: 0.01, reduced by 10 manually when validation error rate stopped improving, and reduced by 3 times.
* Training set of 1.2 million images. Network is trained for roughly 90 cycles. Five to six days on two NVIDIA GTX 580 3GB GPUs.

![](https://lh4.googleusercontent.com/mTwB__7CN57xCK5C6EEtQFIMa__9Ulw0iNqklUnrDKBiIx0QBlutK_c-W-4sG6EkraSrj6qEnN-Cdb1eJI3zBprz3m0mRvGopZQsefbKxceHXJJhSx_tDY-zf4ahPPXdDaiSB6is)

![](https://lh6.googleusercontent.com/WL5rVWc0inJN8VDkpr0BK0gvywOeamVMkpHlXE-2bSJw_UiXQ6jnuQ_wA6bKcVo29MOPfGLfi5TcVA-lNxIj4kv0vz6dR5oGle77xsulbHKj40wxMYQEXOM1fj1qhiFjXRcNjjsW)

**Initialization**

* initialized the weights in each layer from a zero-mean Gaussian distribution with standard de- viation 0.01.
* Bias: 
  * constant 1 for second, fourth, and fifth convolutional layers and FC ← provide RELU with positive inputs
  * constant 0 for others

## **ReLU**

Before Alexnet, Tanh was used. ReLU is introduced in AlexNet. And ReLU is six times faster than Tanh to reach 25% training error rate.

![Image for post](https://lh4.googleusercontent.com/cBCOxaq-sFr08kwCgk6H2O1g1RVzRM01tcTca9YUoG_LJHazzV3yN6Phnq2Pt_MpfsfcpeIVsiLwjE-OwT6STqrLzMeqFJVWc5B0rIvwp2cxZNV5yWn2KAR2LEQfq4stdiYoQx3r)

## 3. **Multiple GPUs** <a id="9133"></a>

At that moment, NVIDIA GTX 580 GPU is used which only got 3GB of memory. Thus, we can see in the architecture that they split into two paths and use 2 GPUs for convolutions. Inter-communications are only occurred at one specific convolutional layer.

**Thus, using 2 GPUs, is due to memory problem, NOT for speeding up the training process.**

With the whole network **compared with a net with only half of kernels** \(only one path\), **Top-1 and top-5 error rates are reduced by 1.7% and 1.2% respectively.**

\*\*\*\*

## **Local Response Normalization**

* ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. 

![Image for post](https://lh3.googleusercontent.com/xZm3UZXa94EKkNpgUML5Tpswut5kPhysIxYABbbmyTcgRyHbWGqeyyoZt2wLW04W9wiFOPjOQ5nCmTPF9YCqzwPU94y5LmhGyNXhM4gYZyVADQYpeamZx5B9TXOg03EOCT1IHrX_)

* In AlexNet, local response normalization is used. It is different from the batch normalization as we can see in the equations. Normalization helps to speed up the convergence.
* Nowadays, batch normalization is used instead of using local response normalization.
* With local response normalization, Top-1 and top-5 error rates are reduced by 1.4% and 1.2% respectively.

\*\*\*\*

## **5. Overlapping Pooling** <a id="a33b"></a>

* Overlapping Pooling is the pooling with stride smaller than the kernel size while Non-Overlapping Pooling is the pooling with stride equal to or larger than the kernel size.
* With overlapping pooling, Top-1 and top-5 error rates are reduced by 0.4% and 0.3% respectively.

\*\*\*\*

\*\*\*\*

## **Reducing Overfitting**

**Datasets turns out to be insufficient to learn so many parameters without considerable overfitting**

### **Data Augmentation**

**The dataset using label-preserving transformations**

1. **generating image translations and horizontal reflec- tions**
2. **y extracting random 224×224 patches \(and their horizontal reflections\) from the 256×256 images**
3. **increases the size of our training set by a factor of 2048**
4. **This is the reason why the input images in Figure 2 are 224 × 224 × 3-dimensional.**
5. **altering the intensities of the RGB channels in training images**
6. **we perform PCA on the set of RGB pixel values with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1.**
7. **reduced the top-1 error rate by over 1%.**

**Dropout**

![](../../.gitbook/assets/image%20%28241%29.png)

Instead of Combining the predictions of many different models , Dropout makes the neural network samples a different architecture, but all these architectures share weights

* setting to zero the output of each hidden neuron with probability 0.5. The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in back- propagation. 
* presented, the neural network samples a different architecture, but all these architectures share weights. 
* This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to
* It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons
* We use dropout in the first two fully-connected layers of Figure 2.
* Without dropout, our network ex- hibits substantial overfitting. Dropout roughly doubles the number of iterations to converge

