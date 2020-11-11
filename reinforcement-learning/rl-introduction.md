# RL Introduction

## Textbook: Reinforcement Learning An Introduction

 _one of the most popular Reinforcement Learning book by **Richard S. Sutton** and **Andrew G. Barto** \(_2nd Edition\)_. The book can be found here:_ [_Link_](http://incompleteideas.net/book/the-book-2nd.html)_._

{% embed url="https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf" %}



## Overview of Reinforcement Learning

RL is teaching BY experience. A good strategy for an agent would be to always choose an action that maximizes the \(discounted\) future reward

Defining a useful state space, action space and reward are hard part. Getting meaningful data fro the formalization is very hard. 

### Environment and Actions

![Deep RL lecture -MIT](../.gitbook/assets/image%20%2828%29.png)

Challenge in RL in real-world applications are how to provide the experience? One option is providing Realistic simulation + transfer learning 

#### Components of RL agent

![](../.gitbook/assets/image%20%2826%29.png)

### Maximize the reward

A good strategy for an agent would be to always choose an action that maximizes the \(discounted\) future reward

![](../.gitbook/assets/image%20%2835%29.png)

### Optimal Policy

Both Environment model and Reward structures have big impact on optimal policy



### Types of Reinforcement Learning

It can be classified either Model-based or Model-Free

* Model-based: e.g. Chess etc
* Model-free
  * Value-based: Off-Policy, can choose the best action. Example: Q-Learning
  * Policy-based: On-Policy, Directly learn the best policy

![](../.gitbook/assets/image%20%2813%29.png)

![](../.gitbook/assets/image%20%2830%29.png)

#### Q-Learning \(Deep Q-Learning Network\)

It is a Model-Free, Off-Policy, Value-based Method

![](../.gitbook/assets/image%20%2820%29.png)

A conventional method of Q-Learning, it is basically a Q-table that updates. But it is not practical with limited rows/cols of table.

Deep Q-Learning uses a neural network to approximate the Q-Function.  This does not require to know and understand the physics of the environment. 

![](../.gitbook/assets/image%20%2822%29.png)

![](../.gitbook/assets/image%20%2833%29.png)

![](../.gitbook/assets/image%20%2825%29.png)

### Policy Gradient

![](../.gitbook/assets/image%20%2817%29.png)

Vanilla Policy Gradient

![](../.gitbook/assets/image%20%2829%29.png)

#### Advantage Actor-Critic \(A2C\)

Combined DQN and REINFORCE

![](../.gitbook/assets/image%20%2814%29.png)

#### Deep Deterministic Policy Gradient \(DDPG\)

![](../.gitbook/assets/image%20%2832%29.png)

![](../.gitbook/assets/image%20%2815%29.png)

![](../.gitbook/assets/image%20%2819%29.png)

