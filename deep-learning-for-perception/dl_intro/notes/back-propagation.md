# Back Propagation

## Activation Functions

Activation function or transfer function is used to determine the output of a neuron or node. It is a mathematical gate in between the input feeding the current neuron and its output going to the next layer.

![](../../../.gitbook/assets/image%20%2878%29.png)

In deep learning,  we commonly use non-linear Activation Function include

* Sigmoid:  Output limit to \[0 1\]. But gives gradient vanishing problem, not used anymore 
* ReLU\(rectified linear unit\): most commonly used in CNN \(hidden layers\) 
* Others: Tanh, Leaky ReLU, Maxout...

![Image from MIT Deeplearning Lecture](../../../.gitbook/assets/image%20%2883%29.png)

![Cheat sheet of commonly used Activation Function ](../../../.gitbook/assets/image%20%2877%29.png)

### Further Reading



## Loss Functions

A loss function calculates the error between the prediction from the ground truth. It averages all error from all datasets.

![](../../../.gitbook/assets/image%20%2829%29.png)

### For Classification  

Softmax with Cross-Entropy Loss is often used

![](../../../.gitbook/assets/image%20%2849%29.png)

### Further Reading

{% embed url="https://gombru.github.io/2018/05/23/cross\_entropy\_loss/" %}

{% embed url="https://cs231n.github.io/linear-classify/" %}

## Optimization

We want to get the model weights\(W\) to minimize the value of loss function for accurate prediction. How can we change the model parameters during training? Optimizer helps to move along the slope\(gradient\) for min or max point.



### Gradient Descent

Minimize objective function J\(w\) by updating parameter\(w\) in opposite direction of gradient of J\(w\).  Following the negative gradient of the Objective Function to find the minimum value of loss. It control the step size by learning rate n

![](../../../.gitbook/assets/image%20%2885%29.png)

Finding the derivative: 1\) analytical 2\) numerical approach. If possible, use analytical approach for faster and accurate gradient. 

![](../../../.gitbook/assets/image%20%2847%29.png)

Examples of Optimizer include

* SGD \(Stochastic Gradient Descent\)

{% hint style="info" %}
Often SGD is refered to  Mini-batch Gradient Descent
{% endhint %}

* Adagrad
* Momentum•Adam

### Further Reading

{% embed url="https://cs231n.github.io/optimization-1/" %}

{% embed url="https://ruder.io/optimizing-gradient-descent/" %}

![Forward vs backward pass](../../../.gitbook/assets/image%20%2850%29.png)



