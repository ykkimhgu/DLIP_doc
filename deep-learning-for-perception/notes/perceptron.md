---
description: updated
---

# Perceptron

## What is a Perceptron?

A Perceptron is an algorithm used for [supervised learning](https://deepai.org/machine-learning-glossary-and-terms/supervised-learning) of binary [classifiers](https://deepai.org/machine-learning-glossary-and-terms/classifier). It is a single layer neural network and a multi-layer perceptron is called Neural Networks.

* [A short description](https://deepai.org/machine-learning-glossary-and-terms/perceptron)
* [Neural Representation of AND, OR, NOT, XOR and XNOR Logic Gates](https://medium.com/@stanleydukor/neural-representation-of-and-or-not-xor-and-xnor-logic-gates-perceptron-algorithm-b0275375fea1)

## How it works

Binary classifier for Predict y = 1 if Wx+b > 0 , otherwise y=0

![](<../../images/image (216).png>)

The perceptron consists of 4 parts.

1. Input values or One input layer
2. Weights and Bias
3. Net sum
4. [Activation Function](https://medium.com/towards-data-science/activation-functions-neural-networks-1cbd9f8d91d6)

**Weights** shows the strength of the particular node.

**A bias** value allows you to shift the activation function curve up or down.

**Activation Function** scales output (0,1) or (-1,1)

![](<../../.gitbook/assets/image (223) (4) (4) (4) (2) (2) (1) (1) (3).png>)

## Multi-Layer Perceptron

![](<../../.gitbook/assets/image (223) (4) (4) (4) (2) (2) (1) (1) (4).png>)

## Multi-Layer Perceptron

![](<../../images/image (219).png>)

![](<../../images/image (215).png>)

![](<../../images/image (221).png>)

![](<../../images/image (217).png>)

Since we cannot express XOR with a single Perceptron, we can construct a network of Perceptron or **Multi-Layer Perceptron**

![](<../../images/image (222).png>)

Using the previous AND, NAND, OR gates with perceptrons, we can build **XOR**

![](<../../.gitbook/assets/image (220).png>)

## Activation Function
