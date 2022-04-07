# Optimization

## Loss Functions

A loss function calculates the error between the prediction from the ground truth. It averages all error from all datasets.

![](<../../.gitbook/assets/image (29).png>)

**Logistic Loss** and **Multinomial Logistic Loss** are other names for **Cross-Entropy loss**.

In a **binary classification problem**, where C′=2C′=2, the Cross Entropy Loss can be defined also as

[![](https://latex.codecogs.com/gif.latex?CE\&space;=\&space;-%5Csum\_%7Bi=1%7D%5E%7BC%27=2%7Dt\_%7Bi%7D\&space;log\&space;\(s\_%7Bi%7D\)\&space;=\&space;-t\_%7B1%7D\&space;log\(s\_%7B1%7D\)\&space;-\&space;\(1\&space;-\&space;t\_%7B1%7D\)\&space;log\(1\&space;-\&space;s\_%7B1%7D\))](https://www.codecogs.com/eqnedit.php?latex=CE\&space;=\&space;-\sum\_{i=1}^{C%27=2}t\_{i}\&space;log\&space;\(s\_{i}\)\&space;=\&space;-t\_{1}\&space;log\(s\_{1}\)\&space;-\&space;\(1\&space;-\&space;t\_{1}\)\&space;log\(1\&space;-\&space;s\_{1}\))

###

### For Classification

**Softmax Loss**

Softmax with Cross-Entropy Loss is often used. If we use this loss, we will train a CNN to output a probability over the C classes for each image. It is used for **multi-class classification.**

**Note that** The Softmax function cannot be applied independently to each s i , since it depends on all elements of s . For a given class s i , the Softmax function can be computed as:

![](<../../.gitbook/assets/image (49).png>)

#### Binary Cross-Entropy Loss <a href="#binary-cross-entropy-loss" id="binary-cross-entropy-loss"></a>

Also called **Sigmoid Cross-Entropy loss**. It is a **Sigmoid activation** plus a **Cross-Entropy loss**. Unlike **Softmax loss** it is independent for each vector component (class), meaning that the loss computed for every CNN output vector component is not affected by other component values. It is used for **multi-label classification**

[![](https://latex.codecogs.com/gif.latex?CE\&space;=\&space;-%5Csum\_%7Bi=1%7D%5E%7BC%27=2%7Dt\_%7Bi%7D\&space;log\&space;\(f\(s\_%7Bi%7D\)\)\&space;=\&space;-t\_%7B1%7D\&space;log\(f\(s\_%7B1%7D\)\)\&space;-\&space;\(1\&space;-\&space;t\_%7B1%7D\)\&space;log\(1\&space;-\&space;f\(s\_%7B1%7D\)\))](https://www.codecogs.com/eqnedit.php?latex=CE\&space;=\&space;-\sum\_{i=1}^{C%27=2}t\_{i}\&space;log\&space;\(f\(s\_{i}\)\)\&space;=\&space;-t\_{1}\&space;log\(f\(s\_{1}\)\)\&space;-\&space;\(1\&space;-\&space;t\_{1}\)\&space;log\(1\&space;-\&space;f\(s\_{1}\)\))

![](https://gombru.github.io/assets/cross\_entropy\_loss/sigmoid\_CE\_pipeline.png)

### Further Reading

{% embed url="https://gombru.github.io/2018/05/23/cross_entropy_loss/" %}

{% embed url="https://cs231n.github.io/linear-classify/" %}

## Optimization

We want to get the model weights(W) to minimize the value of loss function for accurate prediction. How can we change the model parameters during training? Optimizer helps to move along the slope(gradient) for min or max point.

### Gradient Descent

Minimize objective function J(w) by updating parameter(w) in opposite direction of gradient of J(w). Following the negative gradient of the Objective Function to find the minimum value of loss. It control the step size by learning rate n

![](<../../.gitbook/assets/image (85).png>)

Finding the derivative: 1) analytical 2) numerical approach. If possible, use analytical approach for faster and accurate gradient.

![](<../../.gitbook/assets/image (47).png>)

Examples of Optimizer include

* SGD (Stochastic Gradient Descent)

{% hint style="info" %}
Often SGD is refered to Mini-batch Gradient Descent
{% endhint %}

* Adagrad
* Momentum•Adam

### Further Reading

{% embed url="https://cs231n.github.io/optimization-1/" %}

{% embed url="https://ruder.io/optimizing-gradient-descent/" %}

![Forward vs backward pass](<../../.gitbook/assets/image (50).png>)
