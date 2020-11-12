# Back Propagation

## Loss Functions

## Optimization

We want to get the model weights\(W\) to minimize the value of loss function for accurate prediction How to change the model parameters during training? Optimizer helps to move along the slope\(gradient\) for min or max point

Examples of Optimizer include

* SGD \(Stochastic Gradient Descent\)

{% hint style="info" %}
Often SGD is refered to  Mini-batch Gradient Descent
{% endhint %}

* Adagrad
* Momentum•Adam

### Gradient Descent

Minimize objective function J\(w\) by updating parameter\(w\) in opposite direction of gradient of J\(w\).  Following the negative gradient of the Objective Function to find the minimum value of loss. It control the step size by learning rate n

![](../../../.gitbook/assets/image%20%2879%29.png)

Finding the derivative: 1\) analytical 2\) numerical approach. If possible, use analytical approach for faster and accurate gradient. 

![](../../../.gitbook/assets/image%20%2846%29.png)

### Further Reading

{% embed url="https://cs231n.github.io/optimization-1/" %}

{% embed url="https://ruder.io/optimizing-gradient-descent/" %}



