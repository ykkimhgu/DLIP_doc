# MIT Deep RL

## MIT 6.S091: Introduction to Deep Reinforcement Learning \(Deep RL\)

RL is teaching BY experience. A good strategy for an agent would be to always choose an action that maximizes the \(discounted\) future reward

Defining a useful state space, action space and reward are hard part. Getting meaningful data fro the formalization is very hard.

{% embed url="https://www.youtube.com/watch?v=zR11FLZ-O9M" caption="" %}

### Environment and Actions

![Deep RL lecture -MIT](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2865%29.png)

Challenge in RL in real-world applications are how to provide the experience? One option is providing Realistic simulation + transfer learning

#### Components of RL agent

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2861%29.png)

### Maximize the reward

A good strategy for an agent would be to always choose an action that maximizes the \(discounted\) future reward

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2886%29.png)

### Optimal Policy

Both Environment model and Reward structures have big impact on optimal policy

### Types of Reinforcement Learning

It can be classified either Model-based or Model-Free

* Model-based: e.g. Chess etc
* Model-free
  * Value-based: Off-Policy, can choose the best action. Example: Q-Learning
  * Policy-based: On-Policy, Directly learn the best policy

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2823%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2868%29.png)

#### Q-Learning \(Deep Q-Learning Network\)

It is a Model-Free, Off-Policy, Value-based Method

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2846%29.png)

A conventional method of Q-Learning, it is basically a Q-table that updates. But it is not practical with limited rows/cols of table.

Deep Q-Learning uses a neural network to approximate the Q-Function. This does not require to know and understand the physics of the environment.

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2854%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2880%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2860%29.png)

### Policy Gradient

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2832%29.png)

Vanilla Policy Gradient

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2867%29.png)

#### Advantage Actor-Critic \(A2C\)

Combined DQN and REINFORCE

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2824%29.png)

#### Deep Deterministic Policy Gradient \(DDPG\)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2874%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2828%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%2841%29.png)

