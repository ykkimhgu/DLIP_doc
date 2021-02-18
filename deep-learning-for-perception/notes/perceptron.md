---
description: updated
---

# Perceptron

## What is a Perceptron?

A Perceptron is an algorithm used for [supervised learning](https://deepai.org/machine-learning-glossary-and-terms/supervised-learning) of binary [classifiers](https://deepai.org/machine-learning-glossary-and-terms/classifier). It is a single layer neural network and a multi-layer perceptron is called Neural Networks.

* [A short description](https://deepai.org/machine-learning-glossary-and-terms/perceptron)
* [Neural Representation of AND, OR, NOT, XOR and XNOR Logic Gates](https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1) 

## How it works

Binary classifier for Predict y = 1 if Wx+b &gt; 0 , otherwise y=0

![](../../.gitbook/assets/image%20%28216%29.png)

The perceptron consists of 4 parts.

1. Input values or One input layer
2. Weights and Bias
3. Net sum
4. [Activation Function](https://medium.com/towards-data-science/activation-functions-neural-networks-1cbd9f8d91d6)

**Weights** shows the strength of the particular node.

**A bias** value allows you to shift the activation function curve up or down.

**Activation Function** scales output \(0,1\) or \(-1,1\)

![](../../.gitbook/assets/image%20%28223%29%20%284%29%20%284%29%20%284%29%20%282%29.png)

## Multi-Layer Perceptron

![](../../.gitbook/assets/image%20%28223%29%20%284%29%20%284%29%20%284%29%20%283%29.png)

## Multi-Layer Perceptron

![](../../.gitbook/assets/image%20%28219%29.png)

![](../../.gitbook/assets/image%20%28215%29.png)

![](../../.gitbook/assets/image%20%28221%29.png)

![](../../.gitbook/assets/image%20%28217%29.png)

Since we cannot express XOR with a single Perceptron, we can construct a network of Perceptron or **Multi-Layer Perceptron**

![](../../.gitbook/assets/image%20%28222%29.png)

Using the previous AND, NAND, OR gates with perceptrons, we can build **XOR**

![](../../.gitbook/assets/image%20%28220%29.png)

## Activation Function

