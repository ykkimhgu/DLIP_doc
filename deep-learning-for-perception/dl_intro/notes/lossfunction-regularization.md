# LossFunction Regularization

## Batch Normalization

{% embed url="https://towardsdatascience.com/batch-normalization-the-greatest-breakthrough-in-deep-learning-77e64909d81d" caption="" %}

Batch normalization greatly reduces the variation in the loss landscape, gradient productiveness, and β-smoothness, making the task of navigating the terrain to find the global error minima much easier.

* _More freedom in setting the initial learning rate_. Large initial learning rates will not result in missing out on the minimum during optimization, and can lead to quicker convergence.
* _Accelerate the learning rate decay_. 
* _Remove dropout_. One can get away with not using dropout layers when using batch normalization, since dropout can provide damage and/or slow down the training process. Batch normalization introduces an additional form of resistance to overfitting.
* _Reduce L2 weight regularization._ 
* _Solving the vanishing gradient problem_.
* _Solving the exploding gradient problem._ 

### [Reducing Internal Covariance Shift ](https://arxiv.org/pdf/1502.03167v3.pdf)

For example: We train our data on only black cats’ images. So, if we now try to apply this network to data with colored cats, it is obvious; we’re not going to do well. The training set and the prediction set are both cats’ images but they differ a little bit. In other words, if an algorithm learned some X to Y mapping, and if the distribution of X changes, then we might need to retrain the learning algorithm by trying to align the distribution of X with the distribution of Y.

![Image foDeeplearning.ai: Why Does Batch Norm Work? \(C2W3L06\)](https://miro.medium.com/max/2049/1*VTNB7oSbyaxtIpZ3kXdH4A.png)

Batch normalization allows each layer of a network to learn by itself a little bit more independently of other layers.

![https://arxiv.org/pdf/1502.03167v3.pdf](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28263%29.png)

* We can use higher learning rates because batch normalization makes sure that there’s no activation that’s gone really high or really low. And by that, things that previously couldn’t get to train, it will start to train.
* It reduces overfitting because it has a slight regularization effects. Similar to dropout, it adds some noise to each hidden layer’s activations. Therefore, if we use batch normalization, we will use less dropout, which is a good thing because we are not going to lose a lot of information. However, we should not depend only on batch normalization for regularization; we should better use it together with dropout.

## Effect of Regularization

Regularization refers to the practice of constraining /regularizing the model from learning complex concepts, thereby reducing the risk of overfitting.

### Regularization Methods

* Dropout Regularization
* L2 Regularization
* L1 Regularization

### Effects of Methods

* Dropout has the best performance among other regularizers. Dropout has both weight regularization effect and induces sparsity.
* L1 Regularization has a tendency to produce sparse weights whereas L2 Regularization produces small weights
* Regularization hyper parameters for CONV and FC layers should tuned separately.

{% embed url="https://medium.com/deep-learning-experiments/science-behind-regularization-in-neural-net-training-9a3e0529ab80" caption="" %}

## Effect of Batch size

We use mini-batches because it tends to converge more quickly, allow us to **parallelize computations**

### What is Batch Size

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
* When the right learning rate is chosen, larger batch sizes can train faster, especially when parallelized.

{% embed url="https://medium.com/deep-learning-experiments/effect-of-batch-size-on-neural-net-training-c5ae8516e57" caption="" %}

